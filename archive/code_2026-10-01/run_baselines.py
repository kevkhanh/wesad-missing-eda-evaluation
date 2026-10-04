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
N_BOOT = 500
POOLED_METRICS = ("brier", "log_loss", "auroc", "auprc", "ece_sweep",
                  "calibration_slope", "calibration_intercept", "aurc", "risk_at_cov80")
CALIBS = (("raw", "p_raw"), ("platt", "p_platt"), ("platt_pc", "p_platt_pc"))
RULE_A = "gated_dropout_p0.5"
RULE_BS = ("lgbm_nan_training_p0.5", "lgbm_ecg_only_per_config")


def _folds(df: pd.DataFrame, scheme: str) -> list[np.ndarray]:
    if scheme == "kfold5":
        return ec.subject_kfold(df["subject_id"], 5, seed=PARTITION_SEED)
    if scheme == "loso":
        return ec.loso(df["subject_id"])
    raise ValueError(scheme)


DEFAULT_MODELS = [m for m in ec.MODELS if not m.startswith(("tuned_", "cap_"))]


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
    """Percentile 95% CI by resampling subjects with replacement."""
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in sub.groupby("subject_id")}
    subjects = np.array(list(groups))
    draws = {m: [] for m in POOLED_METRICS + ("within_subject_auroc",)}
    for _ in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        parts = []
        for j, s in enumerate(pick):
            g = groups[s]
            parts.append(g.assign(boot_subj=f"{s}#{j}"))
        b = pd.concat(parts, ignore_index=True)
        if b["y"].nunique() < 2:
            continue
        pan = panel_to_dict(compute_panel(b["y"].values, b[pcol].values))
        for m in POOLED_METRICS:
            draws[m].append(pan[m])
        draws["within_subject_auroc"].append(
            within_subject_auroc(b["y"].values, b[pcol].values, b["boot_subj"].values))
    return {m: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for m, v in draws.items() if v}


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
            # CIs for the primary (reduced) condition only; complete-condition
            # CIs would double the cost and are not used for any claim.
            if n_boot and tier == "reduced":
                entry["pooled_ci95"] = _cluster_boot(sub, pcol, n_boot, seed=17)
            res[calib][tier] = entry
    return res


def summarise(pred: pd.DataFrame, n_boot: int) -> dict:
    from joblib import Parallel, delayed
    names = list(dict.fromkeys(pred["model"]))
    outs = Parallel(n_jobs=min(len(names), 13))(
        delayed(_summarise_model)(pred[pred.model == n], n_boot) for n in names)
    return dict(zip(names, outs))


def per_subject_table(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    hr = df.groupby(["subject_id", "label"])[ec.HR_FEATURE].mean().unstack()
    hr_resp = (hr[1] - hr[0]).rename("delta_mean_nn_ms_stress_minus_baseline")
    rows = []
    for (name, tier, subj), g in pred.groupby(["model", "tier", "subject_id"]):
        rows.append({
            "model": name, "tier": tier, "subject_id": subj,
            "brier_raw": float(np.mean((g.p_raw - g.y) ** 2)),
            "brier_platt": float(np.mean((g.p_platt - g.y) ** 2)),
            "brier_platt_pc": float(np.mean((g.p_platt_pc - g.y) ** 2)),
            "auroc_raw": within_subject_auroc(g.y.values, g.p_raw.values, g.subject_id.values),
            "n_windows": int(len(g)), "n_stress": int(g.y.sum()),
        })
    out = pd.DataFrame(rows).merge(hr_resp, left_on="subject_id", right_index=True)
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
         reuse_predictions: bool = False) -> dict:
    df = ec.load_wesad_frozen()
    manifest = json.loads(ec.MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"[baselines] {len(df)} WESAD windows, {df.subject_id.nunique()} subjects; "
          f"feature table sha256 {manifest['sha256'][:12]}")
    results = {"feature_table_sha256": manifest["sha256"],
               "n_windows": int(len(df)),
               "n_subjects": int(df.subject_id.nunique()),
               "partition_seed": PARTITION_SEED,
               "schemes": {}}
    for scheme in schemes:
        pred_path = RESULTS / f"predictions_{scheme}.csv"
        if reuse_predictions and pred_path.exists():
            pred = pd.read_csv(pred_path)
            print(f"  [{scheme}] reusing {pred_path.name} ({len(pred)} rows)")
        else:
            pred = run_predictions(df, scheme)
            pred.to_csv(pred_path, index=False)
        per_subject_table(pred, df).to_csv(RESULTS / f"per_subject_{scheme}.csv", index=False)
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
    a = ap.parse_args()
    main(tuple(a.schemes.split(",")), a.boot, a.reuse_predictions)
