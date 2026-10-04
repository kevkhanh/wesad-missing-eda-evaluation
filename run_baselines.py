"""
Section 5.2 primary panel: every model, one frozen feature table, paired tiers.

Re-implemented on eval_core after peer_review_round3.md (C1-C4, M1, M2, §3).
The pre-fix version and its output are archived in
results/archive/baselines_2026-10-01_prefix.json.

Models (see eval_core.MODELS):
  lgbm_zero_fill_at_test        original straw-man: complete-trained, EDA=0 at test
  (i)   lgbm_nan_at_test        complete-trained, NaN at test
  (ii)  lgbm_nan_training_p0.5  EDA->NaN on 50% of training rows
  (iii) lgbm_ecg_only_per_config  ECG features only
  (iv)  lgbm_mean_impute_recal  training-mean imputation, per-condition Platt
  (hr)  hr_only                 hrv_mean_nn only
  gated_dropout_p0.5 / gated_no_dropout   the modality-gated network
  mlp_nongated_dropout_p0.5     capacity-matched non-gated MLP, same masking (M2)
  gated_dropout_hr_removed / gated_dropout_hr_resid (HR dropped + within-subject
      residualisation) / lgbm_nan_training_hr_removed / lgbm_ecg_only_hr_removed

Evaluation:
  * Partition schemes: "kfold5" (one random subject partition, seed 0) and
    "loso" (15 folds; every subject's predictions come from a model that
    never saw them).
  * Every test window is scored under BOTH tiers (paired; M1).
  * Raw and Platt (inner-CV OOF, training subjects only) probabilities.
  * Per-fold panels, pooled out-of-fold panels with subject-cluster
    bootstrap 95% CIs (reduced condition, 500 resamples), within-subject AUROC, and a per-subject table with
    each subject's heart-rate response.

Round-4 (peer_reviewe_v2.md) changes:
  * minor 3: 2,000 participant-bootstrap resamples (was 500), with both
    percentile and BCa intervals (acceleration from the leave-one-
    participant-out jackknife). With 15 clusters every bootstrap CI is
    approximate.
  * M5 / minor 5: pooled CIs for the complete condition as well.
  * M2: the never-masked MLP (`mlp_nongated_no_dropout`) is included.
  * M7 / minor 2: the per-subject table carries the standardised HR response
    (delta mean NN / pooled within-condition SD) and a Hanley-McNeil SE
    for each within-subject AUROC.
  * E4: `--table wrist` runs the same panel on the wrist (E4) features.

Outputs:
  results/baselines.json
  results/predictions_<scheme>.csv   (one row per window x tier x model)
  results/per_subject_<scheme>.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_core as ec
from run_metrics import compute_panel, panel_to_dict, within_subject_auroc

RESULTS = Path(__file__).resolve().parent / "results"
OUT_PATH = RESULTS / "baselines.json"
MODEL_SEED_BASE = 200
PARTITION_SEED = 0
N_BOOT = 2000
POOLED_METRICS = ("brier", "log_loss", "auroc", "auprc", "ece_sweep",
                  "calibration_slope", "calibration_intercept", "aurc", "risk_at_cov80",
                  "risk_at_cov60", "e_aurc", "aurc_oracle", "aurc_random")
BOOT_CALIBS = ("raw", "platt_pc")
CALIBS = (("raw", "p_raw"), ("platt", "p_platt"), ("platt_pc", "p_platt_pc"))
RULE_A = "gated_dropout_p0.5"
RULE_BS = ("lgbm_nan_training_p0.5", "lgbm_ecg_only_per_config")


def _folds(df: pd.DataFrame, scheme: str) -> list[np.ndarray]:
    if scheme == "kfold5":
        return ec.subject_kfold(df["subject_id"], 5, seed=PARTITION_SEED)
    if scheme == "loso":
        return ec.loso(df["subject_id"])
    raise ValueError(scheme)


DEFAULT_MODELS = [m for m in ec.MODELS
                  if not m.startswith(("tuned_", "cap_", "hrvsub_", "maskrate_"))
                  and m != "lgbm_eda_only"]


def _fold_models(df: pd.DataFrame, fold_i: int, test_subj, name: str) -> pd.DataFrame:
    # One compute thread per worker; the persistent process pool supplies
    # the parallelism (see run_repeated_cv._fold_task).
    import torch
    torch.set_num_threads(1)
    train_df, test_df = ec.split_by_subjects(df, test_subj)
    pred = ec.fit_predict(name, train_df, test_df, seed=MODEL_SEED_BASE + fold_i)
    pred["fold"] = fold_i
    return pred


def run_predictions(df: pd.DataFrame, scheme: str, models=None, n_jobs: int = 18) -> pd.DataFrame:
    from joblib import Parallel, delayed
    models = list(models or DEFAULT_MODELS)
    folds = _folds(df, scheme)
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fold_models)(df, fold_i, ts, name)
        for fold_i, ts in enumerate(folds) for name in models)
    print(f"  [{scheme}] {len(folds)} folds x {len(models)} models done", flush=True)
    return pd.concat(out, ignore_index=True)


def _cluster_boot(sub: pd.DataFrame, pcol: str, n_boot: int, seed: int) -> dict:
    """Participant-cluster bootstrap: percentile and BCa 95% CIs.

    Resamples participants with replacement (B = n_boot). BCa uses the bias
    correction z0 from the bootstrap distribution and the acceleration from
    the leave-one-participant-out jackknife (Efron & Tibshirani 1993, ch. 14).
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(sub["subject_id"].unique()))
    arr = {s_: (g["y"].values, g[pcol].values) for s_, g in sub.groupby("subject_id")}
    metrics = POOLED_METRICS + ("within_subject_auroc",)

    def stats(pick):
        y = np.concatenate([arr[s_][0] for s_ in pick])
        p = np.concatenate([arr[s_][1] for s_ in pick])
        if len(np.unique(y)) < 2:
            return None
        pan = panel_to_dict(compute_panel(y, p, smooth=False))
        sid = np.concatenate([np.full(len(arr[s_][0]), j) for j, s_ in enumerate(pick)])
        pan["within_subject_auroc"] = within_subject_auroc(y, p, sid)
        return pan

    full = stats(subjects)
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        st = stats(rng.choice(subjects, size=len(subjects), replace=True))
        if st is None:
            continue
        for m in metrics:
            draws[m].append(st[m])
    jack = [stats(np.delete(subjects, i)) for i in range(len(subjects))]
    out = {}
    for m in metrics:
        v = np.asarray(draws[m], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 10 or not np.isfinite(full[m]):
            continue
        entry = {"percentile": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
        jv = np.asarray([j[m] for j in jack if j is not None], dtype=float)
        jv = jv[np.isfinite(jv)]
        prop = np.clip(np.mean(v < full[m]) + 0.5 * np.mean(v == full[m]), 1e-6, 1 - 1e-6)
        z0 = norm.ppf(prop)
        dj = jv.mean() - jv
        den = 6.0 * (np.sum(dj ** 2) ** 1.5)
        a = float(np.sum(dj ** 3) / den) if den > 0 else 0.0
        qs = []
        for alpha in (0.025, 0.975):
            z = norm.ppf(alpha)
            qs.append(float(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))))
        entry["bca"] = [float(np.percentile(v, 100 * qs[0])), float(np.percentile(v, 100 * qs[1]))]
        entry["n_boot_valid"] = int(len(v))
        out[m] = entry
    return out


def _summarise_model(g_model: pd.DataFrame, n_boot: int) -> dict:
    res = {}
    for calib, pcol in CALIBS:
        res[calib] = {}
        for tier, sub in g_model.groupby("tier"):
            per_fold = [
                {"fold": int(f), **panel_to_dict(compute_panel(gf["y"].values, gf[pcol].values))}
                for f, gf in sub.groupby("fold")]
            pooled = panel_to_dict(compute_panel(sub["y"].values, sub[pcol].values))
            pooled["within_subject_auroc"] = within_subject_auroc(
                sub["y"].values, sub[pcol].values, sub["subject_id"].values)
            entry = {"pooled": pooled, "per_fold": per_fold}
            # Both conditions (round-4 M5); raw and per-condition Platt only.
            if n_boot and calib in BOOT_CALIBS:
                boot = _cluster_boot(sub, pcol, n_boot, seed=17)
                entry["pooled_ci95"] = {m: v["percentile"] for m, v in boot.items()}
                entry["pooled_ci95_bca"] = {m: v["bca"] for m, v in boot.items()}
            res[calib][tier] = entry
    return res


def summarise(pred: pd.DataFrame, n_boot: int) -> dict:
    from joblib import Parallel, delayed
    names = list(dict.fromkeys(pred["model"]))
    outs = Parallel(n_jobs=min(len(names), 13))(
        delayed(_summarise_model)(pred[pred.model == n], n_boot) for n in names)
    return dict(zip(names, outs))


def per_subject_table(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    hr = df.groupby(["subject_id", "label"])[ec.HR_FEATURE].agg(["mean", "std", "count"]).unstack()
    hr_resp = (hr[("mean", 1)] - hr[("mean", 0)]).rename("delta_mean_nn_ms_stress_minus_baseline")
    # Review M7: standardised response = delta / pooled within-condition SD.
    n0, n1 = hr[("count", 0)], hr[("count", 1)]
    sp = np.sqrt(((n0 - 1) * hr[("std", 0)] ** 2 + (n1 - 1) * hr[("std", 1)] ** 2) / (n0 + n1 - 2))
    hr_std = (hr_resp / sp).rename("hr_response_standardised")
    rows = []
    for (name, tier, subj), g in pred.groupby(["model", "tier", "subject_id"]):
        auc = within_subject_auroc(g.y.values, g.p_raw.values, g.subject_id.values)
        n1_, n0_ = int(g.y.sum()), int((1 - g.y).sum())
        # Hanley & McNeil (1982) SE, ignoring autocorrelation between windows.
        q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
        se = np.sqrt((auc * (1 - auc) + (n1_ - 1) * (q1 - auc ** 2)
                      + (n0_ - 1) * (q2 - auc ** 2)) / (n1_ * n0_)) if n1_ and n0_ else np.nan
        rows.append({
            "model": name, "tier": tier, "subject_id": subj,
            "brier_raw": float(np.mean((g.p_raw - g.y) ** 2)),
            "brier_platt": float(np.mean((g.p_platt - g.y) ** 2)),
            "brier_platt_pc": float(np.mean((g.p_platt_pc - g.y) ** 2)),
            "auroc_raw": auc, "auroc_se_hanley_mcneil": float(se),
            "n_windows": int(len(g)), "n_stress": n1_,
        })
    out = (pd.DataFrame(rows).merge(hr_resp, left_on="subject_id", right_index=True)
           .merge(hr_std, left_on="subject_id", right_index=True))
    return out.sort_values(["model", "tier", "delta_mean_nn_ms_stress_minus_baseline"],
                           ascending=[True, True, False])


def falsification_check(summary: dict, calib: str) -> dict:
    """Pre-stated rule, evaluated on pooled reduced-condition point estimates.

    The rule ('beat on Brier AND ECE AND AUROC') has no inferential
    criterion; this reports point-estimate comparisons only. Inference is
    in run_repeated_cv.py.
    """
    a = summary[RULE_A][calib]["reduced"]["pooled"]
    out = {}
    for b_name in RULE_BS:
        b = summary[b_name][calib]["reduced"]["pooled"]
        beats = {
            "brier": a["brier"] < b["brier"],
            "ece_sweep": a["ece_sweep"] < b["ece_sweep"],
            "auroc": a["auroc"] > b["auroc"],
        }
        out[b_name] = {"gated_beats_on": beats, "gated_beats_on_all_three": all(beats.values()),
                       "gated": {k: a[k] for k in ("brier", "ece_sweep", "auroc")},
                       "baseline": {k: b[k] for k in ("brier", "ece_sweep", "auroc")}}
    out["rule_fires"] = not all(v["gated_beats_on_all_three"] for k, v in out.items()
                                if k in RULE_BS)
    return out


def print_table(summary: dict, calib: str, scheme: str) -> None:
    print(f"\n=== {scheme} | {calib} | reduced condition, pooled OOF (95% subject-bootstrap CI) ===")
    print(f"  {'model':32s} {'Brier':>22s} {'ECE_sweep':>22s} {'AUROC':>22s} {'wsAUROC':>8s} {'slope':>6s} {'AURC':>22s}")
    for name, r in summary.items():
        e = r[calib]["reduced"]
        p, ci = e["pooled"], e.get("pooled_ci95", {})
        def f(m, d=3):
            lo, hi = ci.get(m, [np.nan, np.nan])
            return f"{p[m]:.{d}f} [{lo:.{d}f},{hi:.{d}f}]"
        print(f"  {name:32s} {f('brier'):>22s} {f('ece_sweep'):>22s} {f('auroc'):>22s} "
              f"{p['within_subject_auroc']:8.3f} {p['calibration_slope']:6.2f} {f('aurc'):>22s}")


def main(schemes=("kfold5", "loso"), n_boot: int = N_BOOT,
         reuse_predictions: bool = False, table: str = "chest") -> dict:
    global OUT_PATH
    df = ec.load_table(table)
    mpath = ec.MANIFEST_PATH if table == "chest" else ec.EXT_MANIFEST_PATH
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    sfx = "" if table == "chest" else f"_{table}"
    OUT_PATH = RESULTS / f"baselines{sfx}.json"
    print(f"[baselines] {len(df)} WESAD windows, {df.subject_id.nunique()} subjects; "
          f"feature table sha256 {manifest['sha256'][:12]}")
    results = {"feature_table_sha256": manifest["sha256"], "table": table,
               "n_boot": n_boot,
               "n_windows": int(len(df)),
               "n_subjects": int(df.subject_id.nunique()),
               "partition_seed": PARTITION_SEED,
               "schemes": {}}
    for scheme in schemes:
        pred_path = RESULTS / f"predictions_{scheme}{sfx}.csv"
        if reuse_predictions and pred_path.exists():
            pred = pd.read_csv(pred_path)
            print(f"  [{scheme}] reusing {pred_path.name} ({len(pred)} rows)")
        else:
            pred = run_predictions(df, scheme)
            pred.to_csv(pred_path, index=False)
        per_subject_table(pred, df).to_csv(RESULTS / f"per_subject_{scheme}{sfx}.csv", index=False)
        summary = summarise(pred, n_boot)
        results["schemes"][scheme] = {
            "folds": [list(map(str, f)) for f in _folds(df, scheme)],
            "models": summary,
            "falsification_rule": {c: falsification_check(summary, c) for c, _ in CALIBS},
        }
        for calib, _ in CALIBS:
            print_table(summary, calib, scheme)
        print("  rule fires: " + "; ".join(
            f"{c}={results['schemes'][scheme]['falsification_rule'][c]['rule_fires']}" for c, _ in CALIBS))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[baselines] wrote {OUT_PATH}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="kfold5,loso")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--reuse-predictions", action="store_true",
                    help="summarise from results/predictions_<scheme>.csv if present")
    ap.add_argument("--table", default="chest", choices=["chest", "wrist"])
    a = ap.parse_args()
    main(tuple(a.schemes.split(",")), a.boot, a.reuse_predictions, a.table)
