"""
Round 5, C3: informative (MNAR) EDA missingness, as pre-specified in
prespec_round5.md (SHA-256 re-checked against results/prespec_round5_manifest.json).

Unlike the paired design of run_repeated_cv.py, each test window's six
EDA-dependent features are missing with a window-specific probability, and
every system sees the same random draw. "Natural" missingness is also applied
to the training participants' windows under a training mechanism; the
systems' own training-time augmentation (50% MCAR EDA dropout for the gated
network and MLP, 50% MCAR NaN masking for (ii)) is kept on top.

Mechanisms (overall missing rate 0.30; parameters solved on the full table):
  mcar      p = 0.30
  label_OR  p(missing | stress), p(missing | baseline) with odds ratio OR
  motion_OR logit p = a + b * z(log chest_acc_mag_sd), with the implied
            stress-vs-baseline missingness odds ratio equal to OR
Scenarios: S0 mcar->mcar; S1 X->X; S2 X->mcar; S3 mcar->X, X in
{label, motion} x OR {2, 4, 8}. Primary for inference: label_4 S1 and S2.

Systems: gated + dropout, MLP (same masking), (ii) NaN-trained LightGBM, and
the per-configuration system (complete-trained LightGBM fit on the training
windows that HAVE EDA, used when EDA is present; ECG-only LightGBM fit on all
training windows, used when it is absent).

Shortcut index (paired, raw): mean over held-out baseline windows of
p(EDA removed) - p(EDA present); also for stress windows.

Post-hoc variant (review 9, `--no-added-masking`): the systems get NO extra 50%
MCAR training-time masking (gated/MLP modality dropout p = 0, (ii) train_nan_p
= 0); the only missing EDA they see in training is the training mechanism's own
(30%). The default run adds 50% MCAR on top, which can swamp a label-missingness
association; this variant is the decisive version of the shortcut check.
Outputs get a `_nomask` suffix. Not in prespec_round5.md.

CLI:
    python run_mnar.py              # 20 repeats
    python run_mnar.py --repeats 1  # smoke test
    python run_mnar.py --no-added-masking
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_core as ec
from feature_extraction import ALL_FEATURE_NAMES, ECG_FEATURE_NAMES, EDA_FEATURE_NAMES
from run_metrics import PlattCalibrator, compute_panel
from run_operating_points import prespec_sha
from run_repeated_cv import (MODEL_SEED_BASE, N_FOLDS, SESOI, bouckaert_frank,
                             participant_level)

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "results" / "mnar.json"
RATE = 0.30
ORS = (2, 4, 8)
MNAR_KEYS = [f"{k}_{o}" for k in ("label", "motion") for o in ORS]
TRAIN_KEYS = ["mcar"] + MNAR_KEYS
SCENARIOS = ([("S0", "mcar", "mcar")]
             + [("S1", k, k) for k in MNAR_KEYS]
             + [("S2", k, "mcar") for k in MNAR_KEYS]
             + [("S3", "mcar", k) for k in MNAR_KEYS])
PRIMARY = ["S1:label_4->label_4", "S2:label_4->mcar"]
COMPARISONS = [("gated", "mlp"), ("gated", "lgbm_nan_training"), ("gated", "per_config"),
               ("mlp", "per_config"), ("lgbm_nan_training", "per_config")]
METRICS = ("brier", "auroc", "log_loss")


def scen_key(s, tr, te) -> str:
    return f"{s}:{tr}->{te}"


# ---------------------------------------------------------------------------
# Missingness mechanisms
# ---------------------------------------------------------------------------

def _odds(p):
    return p / (1 - p)


def _bisect(f, lo, hi, n=200):
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def label_probs(y: np.ndarray, odds_ratio: float, rate: float = RATE) -> tuple[float, float]:
    """(p_baseline, p_stress) with prevalence-weighted mean `rate` and the given OR."""
    pi = float(np.mean(y))

    def p1_of(p0):
        o1 = odds_ratio * _odds(p0)
        return o1 / (1 + o1)
    p0 = _bisect(lambda p0: pi * p1_of(p0) + (1 - pi) * p0 - rate, 1e-9, rate)
    return p0, p1_of(p0)


def motion_probs(z: np.ndarray, y: np.ndarray, odds_ratio: float,
                 rate: float = RATE) -> tuple[np.ndarray, dict]:
    """logit p = a + b z; a solves the overall rate, b the implied label OR."""
    def p_of(a, b):
        return 1 / (1 + np.exp(-(a + b * z)))

    def a_for(b):
        return _bisect(lambda a: p_of(a, b).mean() - rate, -30.0, 30.0)

    def implied_or(b):
        p = p_of(a_for(b), b)
        return _odds(p[y == 1].mean()) / _odds(p[y == 0].mean())
    b = _bisect(lambda b: implied_or(b) - odds_ratio, 0.0, 20.0)
    a = a_for(b)
    p = p_of(a, b)
    return p, {"a": a, "b": b, "implied_or": float(implied_or(b)), "rate": float(p.mean())}


def attach_motion(df: pd.DataFrame) -> pd.DataFrame:
    """Join chest_acc_mag_sd from the extended table (same row order within
    subject x condition as extract_extended.verify_against_frozen)."""
    ext = ec.load_extended("chest", ("baseline", "stress"))
    ext["k"] = ext.groupby(["subject_id", "condition"]).cumcount()
    out = df.assign(condition=np.where(df.label == 1, "stress", "baseline"))
    out["k"] = out.groupby(["subject_id", "condition"]).cumcount()
    out = out.merge(ext[["subject_id", "condition", "k", "chest_acc_mag_sd",
                         "hrv_mean_nn"]].rename(columns={"hrv_mean_nn": "_hr_ext"}),
                    on=["subject_id", "condition", "k"], how="left", validate="1:1")
    if out["chest_acc_mag_sd"].isna().any() or not np.allclose(out["_hr_ext"], out["hrv_mean_nn"]):
        raise RuntimeError("extended-table join does not reproduce the frozen rows")
    return out.drop(columns=["_hr_ext", "k"]).sort_values("row_id").reset_index(drop=True)


def mechanism_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-window missingness probability under every mechanism (columns p_<key>)."""
    y = df.label.values
    z = np.log(df.chest_acc_mag_sd.values)
    z = (z - z.mean()) / z.std()
    probs, params = {"mcar": np.full(len(df), RATE)}, {"mcar": {"rate": RATE}}
    for o in ORS:
        p0, p1 = label_probs(y, o)
        probs[f"label_{o}"] = np.where(y == 1, p1, p0)
        params[f"label_{o}"] = {"p_baseline": p0, "p_stress": p1}
        probs[f"motion_{o}"], params[f"motion_{o}"] = motion_probs(z, y, o)
    for k, p in probs.items():
        df[f"p_{k}"] = p
        params[k]["achieved_rate"] = float(p.mean())
        params[k]["achieved_or"] = float(_odds(p[y == 1].mean()) / _odds(p[y == 0].mean()))
    return df, params


def apply_missing(df: pd.DataFrame, key: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    miss = np.random.default_rng(seed).random(len(out)) < out[f"p_{key}"].values
    out.loc[miss, EDA_FEATURE_NAMES] = np.nan
    out["has_eda"] = ~miss
    return out


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

class PerConfig:
    """(i) on training windows with EDA, (iii) on all; route by has_eda."""

    def fit(self, train_df: pd.DataFrame, seed: int):
        self.full_ = ec.LGBMModel(ALL_FEATURE_NAMES).fit(train_df[train_df.has_eda], seed)
        self.ecg_ = ec.LGBMModel(ECG_FEATURE_NAMES).fit(train_df, seed)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        has = df.has_eda.values.astype(bool)
        p = np.empty(len(df))
        if has.any():
            p[has] = self.full_.predict(df[has])
        if (~has).any():
            p[~has] = self.ecg_.predict(df[~has])
        return p


ADDED_MASK_P = 0.5          # 0.0 in the post-hoc no-added-masking variant
SUFFIX = ""


def make_systems(p: float) -> dict:
    return {
        "gated": lambda: ec.GatedModel(p, drop_eda=p > 0),
        "mlp": lambda: ec.NonGatedMLP(p),
        "lgbm_nan_training": lambda: ec.LGBMModel(ALL_FEATURE_NAMES, train_nan_p=p),
        "per_config": PerConfig,
    }


SYSTEMS = make_systems(ADDED_MASK_P)


def _inner_platt(factory, train_df: pd.DataFrame, seed: int) -> PlattCalibrator:
    """Platt on inner subject-level OOF predictions; inner test windows keep
    the training mechanism's missingness (as calibration data from the field would)."""
    ys, ps = [], []
    for k, inner_test in enumerate(ec.subject_kfold(train_df.subject_id, ec.INNER_K,
                                                    seed=seed + 7919)):
        itr, ite = ec.split_by_subjects(train_df, inner_test)
        if itr.label.nunique() < 2 or itr[itr.has_eda].label.nunique() < 2:
            continue
        ps.append(factory().fit(itr, seed=seed + 101 * (k + 1)).predict(ite))
        ys.append(ite.label.values)
    return PlattCalibrator().fit(np.concatenate(ys), np.concatenate(ps))


def _fold_task(df: pd.DataFrame, rep: int, fold_i: int, test_subj, added_p: float = ADDED_MASK_P):
    import torch
    torch.set_num_threads(1)
    train0, test0 = ec.split_by_subjects(df, test_subj)
    seed = MODEL_SEED_BASE + rep * N_FOLDS + fold_i
    tests = {k: apply_missing(test0, k, seed + 77_777) for k in TRAIN_KEYS}
    paired = ec.paired_tiers(test0)
    fold_rows, subj_rows, sc_rows = [], [], []
    for tr_key in TRAIN_KEYS:
        train = apply_missing(train0, tr_key, seed + 55_555)
        evals = [(s, te) for s, tr, te in SCENARIOS if tr == tr_key]
        for sysname, factory in make_systems(added_p).items():
            model = factory().fit(train, seed=seed)
            platt = _inner_platt(factory, train, seed)
            # Shortcut index: the same held-out windows with EDA removed vs present.
            d = model.predict(paired["reduced"]) - model.predict(paired["complete"])
            y0 = paired["complete"].label.values
            sc_rows.append({"repeat": rep, "fold": fold_i, "system": sysname,
                            "train": tr_key, "shortcut_baseline": float(d[y0 == 0].mean()),
                            "shortcut_stress": float(d[y0 == 1].mean())})
            for s, te_key in evals:
                tdf = tests[te_key]
                y, has = tdf.label.values, tdf.has_eda.values
                p_raw = model.predict(tdf)
                p_cal = platt.predict(p_raw)
                key = scen_key(s, tr_key, te_key)
                for calib, p in (("raw", p_raw), ("platt", p_cal)):
                    pan = compute_panel(y, p, smooth=False)
                    fold_rows.append({
                        "repeat": rep, "fold": fold_i, "scenario": key, "system": sysname,
                        "calib": calib, **{m: getattr(pan, m) for m in METRICS},
                        "brier_missing_rows": float(np.mean((p[~has] - y[~has]) ** 2)),
                        "brier_present_rows": float(np.mean((p[has] - y[has]) ** 2)),
                        "frac_missing": float(np.mean(~has)),
                        "frac_missing_stress": float(np.mean(~has[y == 1])),
                        "frac_missing_baseline": float(np.mean(~has[y == 0]))})
                for s_, g in tdf.assign(_p=p_raw).groupby("subject_id"):
                    subj_rows.append({"repeat": rep, "scenario": key, "system": sysname,
                                      "subject_id": s_,
                                      "brier": float(np.mean((g._p - g.label) ** 2))})
    return fold_rows, subj_rows, sc_rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(n_repeats: int = 20, n_jobs: int = 12, added_masking: bool = True) -> dict:
    from joblib import Parallel, delayed
    global SYSTEMS, SUFFIX, OUT_PATH
    SYSTEMS = make_systems(ADDED_MASK_P if added_masking else 0.0)
    SUFFIX = "" if added_masking else "_nomask"
    OUT_PATH = HERE / "results" / f"mnar{SUFFIX}.json"
    sha_pre = prespec_sha()
    df, mech = mechanism_table(attach_motion(ec.load_wesad_frozen()))
    n_subj = df.subject_id.nunique()
    tasks, n_train, n_test = [], [], []
    for rep in range(n_repeats):
        for f, ts in enumerate(ec.subject_kfold(df.subject_id, N_FOLDS, seed=1000 + rep)):
            tasks.append((rep, f, ts))
            n_train.append(n_subj - len(ts))
            n_test.append(len(ts))
    print(f"[mnar] {len(tasks)} fold tasks x {len(TRAIN_KEYS)} training mechanisms x "
          f"{len(SYSTEMS)} systems, n_jobs={n_jobs}", flush=True)
    outs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fold_task)(df, r, f, ts, ADDED_MASK_P if added_masking else 0.0)
        for r, f, ts in tasks)
    folds = pd.DataFrame([r for o in outs for r in o[0]])
    subj = pd.DataFrame([r for o in outs for r in o[1]])
    sc = pd.DataFrame([r for o in outs for r in o[2]])
    ntr, nte = float(np.mean(n_train)), float(np.mean(n_test))

    res = {"config": {"n_repeats": n_repeats, "n_folds": N_FOLDS, "n_subjects": int(n_subj),
                      "rate": RATE, "added_mcar_training_masking": ADDED_MASK_P if added_masking else 0.0,
                      "odds_ratios": ORS, "scenarios": [scen_key(*s) for s in SCENARIOS],
                      "primary": PRIMARY, "sesoi": SESOI, "prespec_sha256": sha_pre,
                      "feature_table_sha256": json.loads(
                          ec.MANIFEST_PATH.read_text(encoding="utf-8"))["sha256"]},
           "mechanisms": mech,
           "shortcut_index": {
               tr: {s: {"baseline_mean": float(g.shortcut_baseline.mean()),
                        "stress_mean": float(g.shortcut_stress.mean()),
                        "baseline_bf": bouckaert_frank(g.shortcut_baseline.values, ntr, nte, n_subj)}
                    for s, g in sc[sc.train == tr].groupby("system")}
               for tr in TRAIN_KEYS}}
    s0 = scen_key("S0", "mcar", "mcar")
    for calib in ("raw", "platt"):
        fc = folds[folds.calib == calib]
        block = {}
        for key, gk in fc.groupby("scenario"):
            frames = {s: g.set_index(["repeat", "fold"]) for s, g in gk.groupby("system")}
            entry = {"per_system": {s: {f"{c}_mean": float(g[c].mean()) for c in
                                        (*METRICS, "brier_missing_rows", "brier_present_rows",
                                         "frac_missing", "frac_missing_stress",
                                         "frac_missing_baseline")}
                                    for s, g in gk.groupby("system")},
                     "pairwise": {}, "vs_S0": {}}
            sk = subj[subj.scenario == key] if calib == "raw" else None
            for a, b in COMPARISONS:
                e = {m: bouckaert_frank((frames[a][m] - frames[b][m]).values, ntr, nte,
                                        n_subj, SESOI.get(m)) for m in METRICS}
                if sk is not None:
                    sm = sk.groupby(["system", "subject_id"]).brier.mean()
                    e["participant_brier"] = participant_level(sm[a] - sm[b], SESOI["brier"], True)
                entry["pairwise"][f"{a}__vs__{b}"] = e
            if key != s0:
                f0 = {s: g.set_index(["repeat", "fold"])
                      for s, g in fc[fc.scenario == s0].groupby("system")}
                entry["vs_S0"] = {s: bouckaert_frank((frames[s].brier - f0[s].brier).values,
                                                     ntr, nte, n_subj, SESOI["brier"])
                                  for s in frames}
            block[key] = entry
        res[calib] = block

    OUT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
    folds.to_csv(OUT_PATH.with_name(f"mnar_per_fold{SUFFIX}.csv"), index=False)
    subj.to_csv(OUT_PATH.with_name(f"mnar_per_subject{SUFFIX}.csv"), index=False)
    sc.to_csv(OUT_PATH.with_name(f"mnar_shortcut{SUFFIX}.csv"), index=False)
    _print(res)
    _figure(folds, SUFFIX)
    print(f"\n[mnar] wrote {OUT_PATH}")
    return res


def _fmt(e: dict) -> str:
    return f"{e['mean']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"


def _print(res: dict) -> None:
    print("\nmechanisms:", json.dumps(res["mechanisms"], indent=None))
    print("\nshortcut index (baseline windows, raw): mean [B-F CI]")
    for tr, d in res["shortcut_index"].items():
        print(f"  {tr:9s} " + "  ".join(f"{s} {_fmt(v['baseline_bf'])}" for s, v in d.items()))
    for key, e in res["raw"].items():
        print(f"\n=== {key} (raw) ===")
        for s, r in e["per_system"].items():
            vs = e["vs_S0"].get(s)
            print(f"  {s:18s} Brier {r['brier_mean']:.3f} AUROC {r['auroc_mean']:.3f} "
                  f"(missing rows {r['brier_missing_rows_mean']:.3f}, present "
                  f"{r['brier_present_rows_mean']:.3f})"
                  + (f" | vs S0 {_fmt(vs)}" if vs else ""))
        for k, p in e["pairwise"].items():
            b, pb = p["brier"], p["participant_brier"]
            print(f"  {k:36s} Brier BF {_fmt(b)} TOST {b['tost_p']:.3f} | participant "
                  f"{_fmt(pb)} TOST {pb['tost_p']:.3f} ({pb['n_A_better']}/{pb['n']})")


def _figure(folds: pd.DataFrame, suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"gated": "Gated + dropout", "mlp": "MLP, same masking",
              "lgbm_nan_training": "(ii) NaN-trained LightGBM",
              "per_config": "Per-configuration (i)/(iii)"}
    if suffix:
        labels = {"gated": "Gated, no added masking", "mlp": "MLP, no added masking",
                  "lgbm_nan_training": "(ii), no added masking",
                  "per_config": "Per-configuration (i)/(iii)"}
    raw = folds[folds.calib == "raw"]
    s0 = raw[raw.scenario == scen_key("S0", "mcar", "mcar")].groupby("system").brier.mean()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    titles = {"S1": "S1: trained and tested MNAR", "S2": "S2: trained MNAR, tested MCAR",
              "S3": "S3: trained MCAR, tested MNAR"}
    for i, mech in enumerate(("label", "motion")):
        for j, s in enumerate(("S1", "S2", "S3")):
            ax = axes[i, j]
            for sysname, lab in labels.items():
                ys = [s0[sysname]]
                for o in ORS:
                    k = f"{mech}_{o}"
                    tr, te = {"S1": (k, k), "S2": (k, "mcar"), "S3": ("mcar", k)}[s]
                    g = raw[(raw.scenario == scen_key(s, tr, te)) & (raw.system == sysname)]
                    ys.append(g.brier.mean())
                ax.plot([1, *ORS], ys, marker="o", label=lab)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, *ORS], ["1 (MCAR)", *map(str, ORS)])
            ax.set_title(f"{titles[s]}\n({mech}-driven)", fontsize=9)
            if j == 0:
                ax.set_ylabel("Brier, raw (mean over 100 folds)")
            if i == 1:
                ax.set_xlabel("Missingness odds ratio, stress vs baseline")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"fig_mnar_brier{suffix}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--no-added-masking", action="store_true",
                    help="post-hoc: no extra MCAR training masking (review 9)")
    a = ap.parse_args()
    run(a.repeats, a.jobs, added_masking=not a.no_added_masking)
