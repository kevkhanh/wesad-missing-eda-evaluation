"""
Repeated subject-level 5-fold CV with Bouckaert-Frank corrected inference
and a subject-level sign-flip permutation test.

Re-implemented on eval_core after peer_review_round3.md C1: the previous
version shuffled ROWS before GroupKFold, which yields the same subject
partition in every repeat (verified: 1 distinct partition in 20). Each
repeat now draws its own random permutation of subjects, and the number of
distinct partitions is recorded in the output. Pre-fix output is archived in
results/archive/repeated_cv_2026-10-01_prefix.json.

Per (repeat, fold): every model in MODELS_RCV is trained once, every test
window is scored under both tiers (paired), raw and Platt (inner-CV OOF)
probabilities are kept. Reduced-condition Brier, log-loss, AUROC and
ECE_sweep are computed per fold.

Inference per comparison (A - B; negative Brier/log-loss/ECE = A better,
positive AUROC = A better):
  * Bouckaert-Frank corrected t over r*k fold differences,
        se = sqrt((1/(r*k) + n_test/n_train) * var(d)),  df = r*k - 1,
    with the matching 95% CI for the mean difference.
  * Exact sign-flip over 2^15 on per-subject reduced Brier differences
    (each subject's difference averaged over repeats). Subjects in the
    same fold share a model, so these differences are not independent;
    treat the p-value as descriptive.

CLI:
    python run_repeated_cv.py               # 20 repeats
    python run_repeated_cv.py --repeats 3   # quick check
    python run_repeated_cv.py --set tuned     # nested-tuned gated/MLP/(ii)/(iii)
    python run_repeated_cv.py --set capacity  # gated vs MLP at matched sizes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

import eval_core as ec
from run_metrics import compute_panel

OUT_PATH = Path(__file__).resolve().parent / "results" / "repeated_cv.json"
N_FOLDS = 5
N_REPEATS_DEFAULT = 20
MODEL_SEED_BASE = 200

MODELS_RCV = [
    "gated_dropout_p0.5",
    "gated_no_dropout",
    "mlp_nongated_dropout_p0.5",
    "lgbm_nan_training_p0.5",
    "lgbm_ecg_only_per_config",
    "lgbm_mean_impute_recal",
    "hr_only",
]

COMPARISONS = [
    ("gated_dropout_p0.5", "lgbm_nan_training_p0.5"),     # falsification rule (ii)
    ("gated_dropout_p0.5", "lgbm_ecg_only_per_config"),   # falsification rule (iii)
    ("gated_dropout_p0.5", "lgbm_mean_impute_recal"),     # (iv)
    ("gated_dropout_p0.5", "hr_only"),
    ("gated_dropout_p0.5", "gated_no_dropout"),           # dropout contribution
    ("gated_dropout_p0.5", "mlp_nongated_dropout_p0.5"),  # gating vs no gating (M2)
    ("mlp_nongated_dropout_p0.5", "lgbm_nan_training_p0.5"),  # NN vs GBDT, same masking
]

_CAP = [(f"cap_gated_d{d}", f"cap_mlp_h{h}") for d, h in ec.CAPACITY_PAIRS]

# name -> (models, comparisons, (rule model A, rule baselines) or None, output file)
SETS = {
    "default": (MODELS_RCV, COMPARISONS,
                ("gated_dropout_p0.5", ("lgbm_nan_training_p0.5", "lgbm_ecg_only_per_config")),
                "repeated_cv.json"),
    "tuned": ([
        "tuned_gated_dropout_p0.5", "tuned_mlp_nongated_dropout_p0.5",
        "tuned_lgbm_nan_training_p0.5", "tuned_lgbm_ecg_only_per_config"], [
        ("tuned_gated_dropout_p0.5", "tuned_lgbm_nan_training_p0.5"),
        ("tuned_gated_dropout_p0.5", "tuned_lgbm_ecg_only_per_config"),
        ("tuned_gated_dropout_p0.5", "tuned_mlp_nongated_dropout_p0.5"),
        ("tuned_mlp_nongated_dropout_p0.5", "tuned_lgbm_nan_training_p0.5")],
        ("tuned_gated_dropout_p0.5",
         ("tuned_lgbm_nan_training_p0.5", "tuned_lgbm_ecg_only_per_config")),
        "repeated_cv_tuned.json"),
    "capacity": ([m for pair in _CAP for m in pair], _CAP, None,
                 "repeated_cv_capacity.json"),
}
METRICS = ("brier", "log_loss", "auroc", "ece_sweep", "aurc", "calibration_slope")
CALIBS = (("raw", "p_raw"), ("platt", "p_platt"), ("platt_pc", "p_platt_pc"))


def bouckaert_frank(diffs, n_train: float, n_test: float) -> dict:
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    N = len(d)
    if N < 2:
        return {"mean": float("nan"), "t": float("nan"), "p": float("nan"), "ci95": [float("nan")] * 2}
    mean = float(d.mean())
    se = float(np.sqrt((1.0 / N + n_test / n_train) * d.var(ddof=1)))
    tcrit = float(tdist.ppf(0.975, df=N - 1))
    t = mean / se if se > 0 else float("nan")
    p = float(2 * tdist.sf(abs(t), df=N - 1)) if se > 0 else float("nan")
    return {"mean": mean, "se_corrected": se, "t": t, "p": p, "df": N - 1,
            "ci95": [mean - tcrit * se, mean + tcrit * se]}


def sign_flip_p(subject_diffs) -> float:
    """Exact two-sided sign-flip p over all 2^n sign patterns (n <= 20)."""
    d = np.asarray([x for x in subject_diffs if np.isfinite(x)], dtype=float)
    n = len(d)
    if n < 2:
        return float("nan")
    masks = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    stats = np.abs((np.where(masks == 1, 1.0, -1.0) * d).mean(axis=1))
    return float(np.mean(stats >= abs(d.mean()) - 1e-12))


def _fold_task(df: pd.DataFrame, models, rep: int, fold_i: int, test_subj) -> tuple[list, list, list]:
    # One compute thread per worker process; the persistent loky pool
    # provides the parallelism (workers and their thread pools are reused
    # across tasks, so there is no per-task creation cost).
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
        red = pred[pred.tier == "reduced"]
        for calib, pcol in CALIBS:
            pan = compute_panel(red["y"].values, red[pcol].values)
            fold_rows.append({"repeat": rep, "fold": fold_i, "model": name, "calib": calib,
                              **{m: getattr(pan, m) for m in METRICS}})
            for s_, g in red.groupby("subject_id"):
                subj_rows.append({"repeat": rep, "model": name, "calib": calib,
                                  "subject_id": s_,
                                  "brier": float(np.mean((g[pcol] - g["y"]) ** 2))})
    return fold_rows, subj_rows, chosen


def run_repeated(n_repeats: int = N_REPEATS_DEFAULT, model_set: str = "default",
                 n_jobs: int = 12) -> dict:
    from joblib import Parallel, delayed
    models, comparisons, rule_spec, out_name = SETS[model_set]
    output_path = OUT_PATH.with_name(out_name)
    df = ec.load_wesad_frozen()
    manifest = json.loads(ec.MANIFEST_PATH.read_text(encoding="utf-8"))
    n_subj = df["subject_id"].nunique()

    tasks, partitions = [], set()
    n_train_list, n_test_list = [], []
    for rep in range(n_repeats):
        folds = ec.subject_kfold(df["subject_id"], N_FOLDS, seed=1000 + rep)
        partitions.add(tuple(tuple(f) for f in sorted(folds, key=lambda f: f[0])))
        for fold_i, test_subj in enumerate(folds):
            n_train_list.append(n_subj - len(test_subj))
            n_test_list.append(len(test_subj))
            tasks.append((rep, fold_i, test_subj))
    print(f"[repeated_cv:{model_set}] {len(tasks)} fold tasks x {len(models)} models, n_jobs={n_jobs}",
          flush=True)
    outs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fold_task)(df, models, rep, fold_i, ts) for rep, fold_i, ts in tasks)
    fold_rows = [r for o in outs for r in o[0]]
    subj_rows = [r for o in outs for r in o[1]]
    chosen_rows = [r for o in outs for r in o[2]]

    folds_df = pd.DataFrame(fold_rows)
    subj_df = pd.DataFrame(subj_rows)
    n_train, n_test = float(np.mean(n_train_list)), float(np.mean(n_test_list))

    results = {
        "config": {"model_set": model_set, "models": models,
                   "n_repeats": n_repeats, "n_folds": N_FOLDS, "n_subjects_total": int(n_subj),
                   "n_distinct_partitions": len(partitions),
                   "n_train_subjects_mean": n_train, "n_test_subjects_mean": n_test,
                   "feature_table_sha256": manifest["sha256"],
                   "tiers": "paired: every test window scored with and without EDA"},
        "per_model_reduced": {},
        "pairwise": {},
        "falsification_rule": {},
    }
    for calib, _ in CALIBS:
        fc = folds_df[folds_df.calib == calib]
        results["per_model_reduced"][calib] = {
            name: {f"{m}_mean": float(g[m].mean()) for m in METRICS}
                  | {f"{m}_sd": float(g[m].std(ddof=1)) for m in METRICS}
                  | {"n_folds": int(len(g))}
            for name, g in fc.groupby("model")}
        results["pairwise"][calib] = {}
        for a, b in comparisons:
            ga = fc[fc.model == a].set_index(["repeat", "fold"])
            gb = fc[fc.model == b].set_index(["repeat", "fold"])
            entry = {m: bouckaert_frank((ga[m] - gb[m]).values, n_train, n_test) for m in METRICS}
            sa = subj_df[(subj_df.model == a) & (subj_df.calib == calib)].groupby("subject_id")["brier"].mean()
            sb = subj_df[(subj_df.model == b) & (subj_df.calib == calib)].groupby("subject_id")["brier"].mean()
            sd = (sa - sb).dropna()
            entry["subject_sign_flip_brier"] = {"n_subjects": int(len(sd)),
                                                "mean_subject_diff": float(sd.mean()),
                                                "p": sign_flip_p(sd.values),
                                                "n_subjects_A_better": int((sd < 0).sum())}
            results["pairwise"][calib][f"{a}__vs__{b}"] = entry
        if rule_spec:
            pm = results["per_model_reduced"][calib]
            a_name, b_names = rule_spec
            rule = {}
            for b in b_names:
                beats = {"brier": pm[a_name]["brier_mean"] < pm[b]["brier_mean"],
                         "ece_sweep": pm[a_name]["ece_sweep_mean"] < pm[b]["ece_sweep_mean"],
                         "auroc": pm[a_name]["auroc_mean"] > pm[b]["auroc_mean"]}
                rule[b] = {"gated_beats_on": beats, "gated_beats_on_all_three": all(beats.values())}
            rule["rule_fires"] = not all(v["gated_beats_on_all_three"] for v in rule.values())
            results["falsification_rule"][calib] = rule

    if chosen_rows:
        ch = pd.DataFrame(chosen_rows)
        results["chosen_configs"] = {
            m: g["config"].value_counts().to_dict() for m, g in ch.groupby("model")}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    subj_df.to_csv(output_path.with_name(output_path.stem + "_per_subject.csv"), index=False)
    _print(results)
    print(f"\n[repeated_cv] wrote {output_path}")
    return results


def _print(res: dict) -> None:
    c = res["config"]
    print(f"\n=== Repeated subject-level CV: {c['n_repeats']} x {c['n_folds']} folds, "
          f"{c['n_subjects_total']} subjects, {c['n_distinct_partitions']} distinct partitions ===")
    for calib, _ in CALIBS:
        print(f"\n--- {calib}: reduced-condition means (SD across folds) ---")
        for name, r in res["per_model_reduced"][calib].items():
            print(f"  {name:28s} Brier {r['brier_mean']:.4f} ({r['brier_sd']:.4f})  "
                  f"LL {r['log_loss_mean']:.3f}  AUROC {r['auroc_mean']:.3f} ({r['auroc_sd']:.3f})  "
                  f"ECE_sweep {r['ece_sweep_mean']:.3f}  AURC {r['aurc_mean']:.3f}  slope {r['calibration_slope_mean']:.2f}")
        print(f"--- {calib}: pairwise (A - B), Bouckaert-Frank 95% CI ---")
        for k, e in res["pairwise"][calib].items():
            b, au, ar = e["brier"], e["auroc"], e["aurc"]
            sf = e["subject_sign_flip_brier"]
            print(f"  {k}\n    Brier {b['mean']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] p={b['p']:.3f}"
                  f" | AUROC {au['mean']:+.4f} [{au['ci95'][0]:+.4f}, {au['ci95'][1]:+.4f}] p={au['p']:.3f}"
                  f" | AURC {ar['mean']:+.4f} [{ar['ci95'][0]:+.4f}, {ar['ci95'][1]:+.4f}]"
                  f" | sign-flip p={sf['p']:.3f} ({sf['n_subjects_A_better']}/{sf['n_subjects']} subj A better)")
        if calib in res["falsification_rule"]:
            print(f"  falsification rule fires ({calib}): {res['falsification_rule'][calib]['rule_fires']}")
    for m, counts in res.get("chosen_configs", {}).items():
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {m} most-chosen configs: {top}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=N_REPEATS_DEFAULT)
    ap.add_argument("--set", default="default", choices=sorted(SETS))
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    run_repeated(a.repeats, a.set, a.jobs)
