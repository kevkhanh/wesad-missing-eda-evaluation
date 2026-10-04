"""
Repeated subject-level 5-fold CV with Bouckaert-Frank corrected inference,
participant-level inference, and post-hoc equivalence analyses.

Re-implemented on eval_core after peer_review_round3.md C1: the previous
version shuffled ROWS before GroupKFold, which yields the same subject
partition in every repeat (verified: 1 distinct partition in 20). Each
repeat now draws its own random permutation of subjects, and the
partitions themselves are recorded in the output. Pre-fix output is
archived in results/archive/repeated_cv_2026-10-01_prefix.json.

Round-4 (peer_reviewe_v2.md) changes:
  M2   the fourth factorial cell (`mlp_nongated_no_dropout`) and the
       complete-trained LightGBM (i) join the default set; factorial
       contrasts (gate x masking interaction, main effects) are estimated
       per fold and per participant.
  M3   post-hoc equivalence: TOST at a smallest effect size of interest
       (SESOI; Brier 0.02, AUROC 0.03, fixed below before running) and a
       Bayesian ROPE probability (correlated Bayesian t-test, Corani &
       Benavoli 2015, for fold-level differences; flat-prior t posterior for
       participant-level differences). Also the smallest equivalence bound
       the data support (max |90% CI bound|), which needs no SESOI.
  M4   Bouckaert-Frank with df = n_subjects - 1 (= 14) as a sensitivity
       analysis, and a participant-level analysis as the inferential anchor:
       each participant's metric is averaged over the repeats, then a t14
       interval and an exact sign-flip test on the n = 15 differences.
  M5 / minor 5   every metric and comparison is computed in BOTH tiers
       (complete and reduced), so complete-condition differences get paired
       CIs too.
  minor 7  the decision rule is evaluated with and without its ECE criterion.
  minor 4 / 14  E-AURC, risk at 60% coverage, SmoothECE per fold.
  m5 / m11 / m13 / E4  extra model sets: masking-rate sweep, HRV subset,
       per-subject normalisation transforms, and the wrist (E4) table.

Inference per comparison (A - B; negative Brier/log-loss/ECE/AURC = A better,
positive AUROC = A better):
  * Bouckaert-Frank corrected t over r*k fold differences,
        se = sqrt((1/(r*k) + n_test/n_train) * var(d)),
    df = r*k - 1 (as proposed) and df = n_subjects - 1 (sensitivity).
  * Participant-level: per-participant mean over repeats; t interval with
    df = n - 1 and an exact sign-flip over 2^n patterns. Participants in the
    same fold share a model, so these are approximate too; both are reported.

CLI:
    python run_repeated_cv.py                    # default set, chest table, 20 repeats
    python run_repeated_cv.py --set tuned        # nested-tuned gated/MLP/(ii)/(iii)
    python run_repeated_cv.py --set capacity     # gated vs MLP at matched sizes
    python run_repeated_cv.py --set hrv_subset   # mean NN + RMSSD only
    python run_repeated_cv.py --set maskrate     # training masking-rate sweep
    python run_repeated_cv.py --set default --transform subjnorm_baseline
    python run_repeated_cv.py --set default --table wrist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

import eval_core as ec
from run_metrics import compute_panel, within_subject_auroc

OUT_PATH = Path(__file__).resolve().parent / "results" / "repeated_cv.json"
N_FOLDS = 5
N_REPEATS_DEFAULT = 20
MODEL_SEED_BASE = 200

# Smallest effect sizes of interest for the post-hoc equivalence analysis
# (review M3). Fixed here, before the round-4 run, from external anchors, not
# from the data: 0.02 Brier is ~9% of the prevalence-only Brier
# (0.35 * 0.65 = 0.2275), i.e. ~0.09 Brier skill; 0.03 AUROC is a commonly
# used minimal clinically relevant AUROC difference. Labelled post hoc
# throughout because the comparisons themselves were already known.
SESOI = {"brier": 0.02, "auroc": 0.03}

G, GN = "gated_dropout_p0.5", "gated_no_dropout"
M, MN = "mlp_nongated_dropout_p0.5", "mlp_nongated_no_dropout"
L1, L2, L3, L4 = ("lgbm_nan_at_test", "lgbm_nan_training_p0.5",
                  "lgbm_ecg_only_per_config", "lgbm_mean_impute_recal")
HR = "hr_only"

MODELS_RCV = [G, GN, M, MN, L1, L2, L3, L4, HR]

COMPARISONS = [
    (G, L2),    # falsification rule (ii)
    (G, L3),    # falsification rule (iii)
    (G, L4),    # (iv)
    (G, HR),
    (G, GN),    # masking within the gated network
    (G, M),     # gating, both masked (M2)
    (M, L2),    # NN vs GBDT, same masking
    (GN, MN),   # gating, neither masked (the missing cell, M2)
    (M, MN),    # masking within the MLP
    (L2, L1),   # masking within LightGBM
    (G, L1),    # gated vs the never-masked comparator / deployment (complete tier)
    (L3, G),    # per-configuration deployment, reduced tier
]

# Factorial contrasts as signed sums of models (per fold, per participant).
CONTRASTS = {
    # (G - GN) - (M - MN): does masking help the gated net more than the MLP?
    # Equivalently (G - M) - (GN - MN): does the gate help more without masking?
    "interaction_gate_x_masking": [(+1, G), (-1, GN), (-1, M), (+1, MN)],
    "main_effect_gate": [(+0.5, G), (-0.5, M), (+0.5, GN), (-0.5, MN)],
    "main_effect_masking": [(+0.5, G), (-0.5, GN), (+0.5, M), (-0.5, MN)],
}

_CAP = [(f"cap_gated_d{d}", f"cap_mlp_h{h}") for d, h in ec.CAPACITY_PAIRS]
_HS = ["hrvsub_gated_dropout_p0.5", "hrvsub_mlp_nongated_dropout_p0.5",
       "hrvsub_lgbm_nan_training_p0.5", "hrvsub_lgbm_ecg_only_per_config"]
_MR = [f"maskrate_{fam}_p{p}" for p in ec.MASK_RATES for fam in ("gated", "mlp", "lgbm")]

# name -> (models, comparisons, contrasts, (rule A, rule Bs) or None, output stem)
SETS = {
    "default": (MODELS_RCV, COMPARISONS, CONTRASTS, (G, (L2, L3)), "repeated_cv"),
    "tuned": ([
        "tuned_gated_dropout_p0.5", "tuned_mlp_nongated_dropout_p0.5",
        "tuned_lgbm_nan_training_p0.5", "tuned_lgbm_ecg_only_per_config"], [
        ("tuned_gated_dropout_p0.5", "tuned_lgbm_nan_training_p0.5"),
        ("tuned_gated_dropout_p0.5", "tuned_lgbm_ecg_only_per_config"),
        ("tuned_gated_dropout_p0.5", "tuned_mlp_nongated_dropout_p0.5"),
        ("tuned_mlp_nongated_dropout_p0.5", "tuned_lgbm_nan_training_p0.5")], {},
        ("tuned_gated_dropout_p0.5",
         ("tuned_lgbm_nan_training_p0.5", "tuned_lgbm_ecg_only_per_config")),
        "repeated_cv_tuned"),
    "capacity": ([m for pair in _CAP for m in pair], _CAP, {}, None, "repeated_cv_capacity"),
    "hrv_subset": (_HS, [(_HS[0], _HS[2]), (_HS[0], _HS[3]), (_HS[0], _HS[1])], {},
                   (_HS[0], (_HS[2], _HS[3])), "repeated_cv_hrv_subset"),
    "maskrate": (_MR, [(f"maskrate_gated_p{p}", f"maskrate_mlp_p{p}") for p in ec.MASK_RATES]
                 + [(f"maskrate_gated_p{p}", f"maskrate_lgbm_p{p}") for p in ec.MASK_RATES],
                 {}, None, "repeated_cv_maskrate"),
}
METRICS = ("brier", "log_loss", "auroc", "ece_sweep", "smooth_ece", "aurc", "e_aurc",
           "risk_at_cov60", "calibration_slope")
PARTICIPANT_METRICS = ("brier", "auroc")
CALIBS = (("raw", "p_raw"), ("platt", "p_platt"), ("platt_pc", "p_platt_pc"))
TIERS = ("reduced", "complete")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _tost_p(mean: float, se: float, df: float, delta: float) -> float:
    """Two one-sided tests: p for H0 |mu| >= delta (equivalence if p < 0.05)."""
    if not (se > 0) or not np.isfinite(mean):
        return float("nan")
    p_lo = tdist.sf((mean + delta) / se, df)    # H0: mu <= -delta
    p_hi = tdist.cdf((mean - delta) / se, df)   # H0: mu >= +delta
    return float(max(p_lo, p_hi))


def _rope_prob(mean: float, se: float, df: float, delta: float) -> float:
    """P(|mu| < delta) under a location-scale t posterior t_df(mean, se)."""
    if not (se > 0) or not np.isfinite(mean):
        return float("nan")
    return float(tdist.cdf((delta - mean) / se, df) - tdist.cdf((-delta - mean) / se, df))


def _interval_block(mean: float, se: float, df: float, delta) -> dict:
    t975, t95 = tdist.ppf(0.975, df), tdist.ppf(0.95, df)
    ci90 = [mean - t95 * se, mean + t95 * se]
    out = {"ci95": [mean - t975 * se, mean + t975 * se], "ci90": ci90,
           "p": float(2 * tdist.sf(abs(mean / se), df)) if se > 0 else float("nan"),
           # Smallest symmetric bound at which TOST (alpha 0.05) would pass.
           "equivalence_bound_attained": float(max(abs(ci90[0]), abs(ci90[1]))),
           # Minimum detectable difference, 80% power, two-sided alpha 0.05.
           "mdd80": float((t975 + tdist.ppf(0.80, df)) * se)}
    if delta is not None:
        out["sesoi"] = delta
        out["tost_p"] = _tost_p(mean, se, df, delta)
        out["rope_prob"] = _rope_prob(mean, se, df, delta)
    return out


def bouckaert_frank(diffs, n_train: float, n_test: float, n_subjects: int | None = None,
                    delta: float | None = None) -> dict:
    """Corrected repeated-CV t (Nadeau-Bengio variance, Bouckaert-Frank df).

    The df = r*k - 1 block is the original proposal; the `df_subjects` block
    repeats the interval with df = n_subjects - 1 (review M4). The ROPE
    probability under df = r*k - 1 is the correlated Bayesian t-test
    (Corani & Benavoli 2015), which uses the same corrected scale.
    """
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    N = len(d)
    if N < 2:
        return {"mean": float("nan"), "t": float("nan"), "p": float("nan"),
                "ci95": [float("nan")] * 2}
    mean = float(d.mean())
    se = float(np.sqrt((1.0 / N + n_test / n_train) * d.var(ddof=1)))
    out = {"mean": mean, "se_corrected": se, "t": mean / se if se > 0 else float("nan"),
           "df": N - 1, **_interval_block(mean, se, N - 1, delta)}
    if n_subjects:
        out["df_subjects"] = {"df": n_subjects - 1,
                              **_interval_block(mean, se, n_subjects - 1, delta)}
    return out


def sign_flip_p(subject_diffs) -> float:
    """Exact two-sided sign-flip p over all 2^n sign patterns (n <= 20)."""
    d = np.asarray([x for x in subject_diffs if np.isfinite(x)], dtype=float)
    n = len(d)
    if n < 2:
        return float("nan")
    masks = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    stats = np.abs((np.where(masks == 1, 1.0, -1.0) * d).mean(axis=1))
    return float(np.mean(stats >= abs(d.mean()) - 1e-12))


def participant_level(diffs: pd.Series, delta: float | None, lower_is_better: bool) -> dict:
    """Inference on n per-participant mean differences (review M4 anchor)."""
    d = diffs.dropna().astype(float)
    n = len(d)
    if n < 3:
        return {"n": n}
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    better = (d < 0) if lower_is_better else (d > 0)
    return {"n": n, "mean": mean, "se": se, "df": n - 1,
            **_interval_block(mean, se, n - 1, delta),
            "sign_flip_p": sign_flip_p(d.values),
            "n_A_better": int(better.sum())}


# ---------------------------------------------------------------------------
# Fold task
# ---------------------------------------------------------------------------

def _fold_task(df: pd.DataFrame, models, rep: int, fold_i: int, test_subj) -> tuple[list, list, list]:
    # One compute thread per worker process; the persistent loky pool
    # provides the parallelism.
    import torch
    torch.set_num_threads(1)
    train_df, test_df = ec.split_by_subjects(df, test_subj)
    seed = MODEL_SEED_BASE + rep * N_FOLDS + fold_i
    fold_rows, subj_rows, chosen = [], [], []
    for name in models:
        pred = ec.fit_predict(name, train_df, test_df, seed=seed)
        if "chosen_config" in pred:
            chosen.append({"repeat": rep, "fold": fold_i, "model": name,
                           "config": pred["chosen_config"].iloc[0]})
        for tier in TIERS:
            sub = pred[pred.tier == tier]
            for calib, pcol in CALIBS:
                pan = compute_panel(sub["y"].values, sub[pcol].values)
                fold_rows.append({"repeat": rep, "fold": fold_i, "model": name, "calib": calib,
                                  "tier": tier, **{m: getattr(pan, m) for m in METRICS}})
                for s_, g in sub.groupby("subject_id"):
                    subj_rows.append({
                        "repeat": rep, "model": name, "calib": calib, "tier": tier,
                        "subject_id": s_,
                        "brier": float(np.mean((g[pcol] - g["y"]) ** 2)),
                        "auroc": within_subject_auroc(g["y"].values, g[pcol].values,
                                                      g["subject_id"].values)})
    return fold_rows, subj_rows, chosen


def _signed_sum(frames: dict, terms, metric: str) -> pd.Series:
    out = None
    for w, name in terms:
        v = w * frames[name][metric]
        out = v if out is None else out + v
    return out


def _prepare_table(table: str, transform: str) -> pd.DataFrame:
    df = ec.load_table(table)
    if transform == "subjnorm_all":
        df = ec.subject_normalise(df, "all")
    elif transform == "subjnorm_baseline":
        df = ec.subject_normalise(df, "baseline")
    elif transform != "none":
        raise ValueError(transform)
    return df


def run_repeated(n_repeats: int = N_REPEATS_DEFAULT, model_set: str = "default",
                 n_jobs: int = 12, table: str = "chest", transform: str = "none") -> dict:
    from joblib import Parallel, delayed
    models, comparisons, contrasts, rule_spec, stem = SETS[model_set]
    if table != "chest":
        stem += f"_{table}"
    if transform != "none":
        stem += f"_{transform}"
    output_path = OUT_PATH.with_name(stem + ".json")
    df = _prepare_table(table, transform)
    if table == "chest":
        sha = json.loads(ec.MANIFEST_PATH.read_text(encoding="utf-8"))["sha256"]
    else:
        sha = json.loads(ec.EXT_MANIFEST_PATH.read_text(encoding="utf-8"))["sha256"]
    n_subj = df["subject_id"].nunique()

    tasks, partitions = [], []
    n_train_list, n_test_list = [], []
    for rep in range(n_repeats):
        folds = ec.subject_kfold(df["subject_id"], N_FOLDS, seed=1000 + rep)
        partitions.append([list(map(str, f)) for f in folds])
        for fold_i, test_subj in enumerate(folds):
            n_train_list.append(n_subj - len(test_subj))
            n_test_list.append(len(test_subj))
            tasks.append((rep, fold_i, test_subj))
    n_distinct = len({tuple(tuple(f) for f in sorted(p)) for p in partitions})
    print(f"[repeated_cv:{model_set}:{table}:{transform}] {len(tasks)} fold tasks x "
          f"{len(models)} models, n_jobs={n_jobs}", flush=True)
    outs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fold_task)(df, models, rep, fold_i, ts) for rep, fold_i, ts in tasks)
    folds_df = pd.DataFrame([r for o in outs for r in o[0]])
    subj_df = pd.DataFrame([r for o in outs for r in o[1]])
    chosen_rows = [r for o in outs for r in o[2]]
    n_train, n_test = float(np.mean(n_train_list)), float(np.mean(n_test_list))

    results = {
        "config": {"model_set": model_set, "table": table, "transform": transform,
                   "models": models, "n_repeats": n_repeats, "n_folds": N_FOLDS,
                   "n_subjects_total": int(n_subj), "n_windows": int(len(df)),
                   "n_distinct_partitions": n_distinct, "partitions": partitions,
                   "n_train_subjects_mean": n_train, "n_test_subjects_mean": n_test,
                   "feature_table_sha256": sha, "sesoi_posthoc": SESOI,
                   "tiers": "paired: every test window scored with and without EDA"},
    }
    lower_better = {m: m != "auroc" for m in METRICS}
    for tier in TIERS:
        sfx = "" if tier == "reduced" else "_complete"
        results[f"per_model{'_reduced' if tier == 'reduced' else '_complete'}"] = {}
        results[f"pairwise{sfx}"] = {}
        results[f"contrasts{sfx}"] = {}
        results[f"falsification_rule{sfx}"] = {}
        for calib, _ in CALIBS:
            fc = folds_df[(folds_df.calib == calib) & (folds_df.tier == tier)]
            sc = subj_df[(subj_df.calib == calib) & (subj_df.tier == tier)]
            pm_key = "per_model_reduced" if tier == "reduced" else "per_model_complete"
            results[pm_key][calib] = {
                name: {f"{m}_mean": float(g[m].mean()) for m in METRICS}
                      | {f"{m}_sd": float(g[m].std(ddof=1)) for m in METRICS}
                      | {"n_folds": int(len(g))}
                for name, g in fc.groupby("model")}
            frames = {name: g.set_index(["repeat", "fold"]) for name, g in fc.groupby("model")}
            subj_means = {name: g.groupby("subject_id")[list(PARTICIPANT_METRICS)].mean()
                          for name, g in sc.groupby("model")}

            def infer(fold_diff: dict, subj_diff: dict) -> dict:
                entry = {m: bouckaert_frank(fold_diff[m].values, n_train, n_test, n_subj,
                                            SESOI.get(m)) for m in METRICS}
                entry["participant"] = {m: participant_level(subj_diff[m], SESOI.get(m),
                                                             lower_better[m])
                                        for m in PARTICIPANT_METRICS}
                # Kept for backward compatibility with earlier outputs.
                pb = entry["participant"]["brier"]
                entry["subject_sign_flip_brier"] = {
                    "n_subjects": pb.get("n"), "mean_subject_diff": pb.get("mean"),
                    "p": pb.get("sign_flip_p"), "n_subjects_A_better": pb.get("n_A_better")}
                return entry

            results[f"pairwise{sfx}"][calib] = {
                f"{a}__vs__{b}": infer({m: frames[a][m] - frames[b][m] for m in METRICS},
                                       {m: subj_means[a][m] - subj_means[b][m]
                                        for m in PARTICIPANT_METRICS})
                for a, b in comparisons}
            results[f"contrasts{sfx}"][calib] = {
                cname: infer({m: _signed_sum(frames, terms, m) for m in METRICS},
                             {m: _signed_sum(subj_means, terms, m) for m in PARTICIPANT_METRICS})
                for cname, terms in contrasts.items()}
            if rule_spec:
                pmod = results[pm_key][calib]
                a_name, b_names = rule_spec
                rule = {}
                for b in b_names:
                    beats = {"brier": pmod[a_name]["brier_mean"] < pmod[b]["brier_mean"],
                             "ece_sweep": pmod[a_name]["ece_sweep_mean"] < pmod[b]["ece_sweep_mean"],
                             "auroc": pmod[a_name]["auroc_mean"] > pmod[b]["auroc_mean"]}
                    rule[b] = {"gated_beats_on": beats,
                               "gated_beats_on_all_three": all(beats.values()),
                               "gated_beats_on_brier_and_auroc": beats["brier"] and beats["auroc"]}
                rule["rule_fires"] = not all(rule[b]["gated_beats_on_all_three"] for b in b_names)
                # Review minor 7: the same rule without its (small-sample-biased)
                # ECE criterion.
                rule["rule_fires_without_ece"] = not all(
                    rule[b]["gated_beats_on_brier_and_auroc"] for b in b_names)
                results[f"falsification_rule{sfx}"][calib] = rule

    if chosen_rows:
        ch = pd.DataFrame(chosen_rows)
        results["chosen_configs"] = {
            m: g["config"].value_counts().to_dict() for m, g in ch.groupby("model")}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    subj_df.to_csv(output_path.with_name(output_path.stem + "_per_subject.csv"), index=False)
    folds_df.to_csv(output_path.with_name(output_path.stem + "_per_fold.csv"), index=False)
    _print(results)
    print(f"\n[repeated_cv] wrote {output_path}")
    return results


def _fmt(e: dict, d: int = 4) -> str:
    return f"{e['mean']:+.{d}f} [{e['ci95'][0]:+.{d}f}, {e['ci95'][1]:+.{d}f}]"


def _print(res: dict) -> None:
    c = res["config"]
    print(f"\n=== Repeated subject-level CV ({c['model_set']}, {c['table']}, {c['transform']}): "
          f"{c['n_repeats']} x {c['n_folds']} folds, {c['n_subjects_total']} subjects, "
          f"{c['n_distinct_partitions']} distinct partitions ===")
    for tier in TIERS:
        sfx = "" if tier == "reduced" else "_complete"
        pm_key = "per_model_reduced" if tier == "reduced" else "per_model_complete"
        for calib, _ in CALIBS:
            if calib == "platt":
                continue
            print(f"\n--- {tier} | {calib}: means (SD across folds) ---")
            for name, r in res[pm_key][calib].items():
                print(f"  {name:34s} Brier {r['brier_mean']:.4f} ({r['brier_sd']:.4f})  "
                      f"AUROC {r['auroc_mean']:.3f}  ECE_sweep {r['ece_sweep_mean']:.3f}  "
                      f"smECE {r['smooth_ece_mean']:.3f}  E-AURC {r['e_aurc_mean']:.3f}")
            print(f"--- {tier} | {calib}: pairwise (A - B) ---")
            for k, e in list(res[f"pairwise{sfx}"][calib].items()) + \
                    list(res[f"contrasts{sfx}"][calib].items()):
                b, au, pb = e["brier"], e["auroc"], e["participant"]["brier"]
                print(f"  {k}\n    Brier BF {_fmt(b)} df14 [{b['df_subjects']['ci95'][0]:+.4f}, "
                      f"{b['df_subjects']['ci95'][1]:+.4f}] TOST p={b['tost_p']:.3f} "
                      f"eqb={b['equivalence_bound_attained']:.3f} ROPE={b['rope_prob']:.2f}"
                      f" | participant {_fmt(pb)} TOST p={pb['tost_p']:.3f} "
                      f"({pb['n_A_better']}/{pb['n']})"
                      f" | AUROC BF {_fmt(au)} TOST p={au['tost_p']:.3f}")
            if calib in res[f"falsification_rule{sfx}"]:
                r = res[f"falsification_rule{sfx}"][calib]
                print(f"  rule fires ({tier}, {calib}): {r['rule_fires']} "
                      f"(without ECE: {r['rule_fires_without_ece']})")
    for m, counts in res.get("chosen_configs", {}).items():
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {m} most-chosen configs: {top}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=N_REPEATS_DEFAULT)
    ap.add_argument("--set", default="default", choices=sorted(SETS))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--table", default="chest", choices=["chest", "wrist"])
    ap.add_argument("--transform", default="none",
                    choices=["none", "subjnorm_all", "subjnorm_baseline"])
    a = ap.parse_args()
    run_repeated(a.repeats, a.set, a.jobs, a.table, a.transform)
