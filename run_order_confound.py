"""
Order-confound checks (peer_reviewe_v2.md M6), on the round-4 extended table.

In WESAD the baseline always comes first; stress and amusement order was
swapped between the two protocol versions (7 vs 8 participants here). So
baseline-vs-stress is confounded with time on electrode and habituation.
Three checks, all LOSO (every participant scored by a model that never saw
them), complete (EDA present) and reduced (EDA missing) conditions, raw
probabilities:

  A. Stress vs amusement: order-balanced across participants. If the HR
     mechanism is real arousal rather than time-since-start, discrimination
     and its HR dependence should survive here. Within-subject AUROC is also
     split by protocol version (stress before vs after amusement).
  B. Negative control: baseline-vs-stress models applied to the held-out
     participant's amusement and meditation windows. Fraction predicted
     stress (p >= 0.5) and mean p per condition and meditation block. A
     time-since-start detector would call the later, calm conditions stress.
  C. Baseline vs meditation (both low arousal, different times): EDA-only,
     ECG-only, HR-only and all-feature models. High AUROC from EDA alone
     would point to tonic drift. Caveat: guided meditation also changes
     breathing and HRV, so this is not a pure null for ECG.
  D. Descriptive drift: within-baseline Spearman correlation of tonic EDA
     and of mean NN with time, per participant.

Labelled "requested" (review round 4), run after the main results were
known; nothing here was pre-registered.

Output: results/order_confound.json, results/order_confound_predictions.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import eval_core as ec

from run_metrics import within_subject_auroc

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "order_confound.json"
SEED_BASE = 200
N_BOOT = 2000


MODELS_A = ["gated_dropout_p0.5", "mlp_nongated_dropout_p0.5", "lgbm_nan_training_p0.5",
            "lgbm_ecg_only_per_config", "hr_only", "gated_dropout_hr_removed",
            "lgbm_nan_training_hr_removed", "lgbm_ecg_only_hr_removed"]
MODELS_B = ["gated_dropout_p0.5", "mlp_nongated_dropout_p0.5", "lgbm_nan_training_p0.5",
            "lgbm_ecg_only_per_config", "lgbm_nan_at_test", "hr_only"]
MODELS_C = ["lgbm_eda_only", "lgbm_ecg_only_per_config", "hr_only", "lgbm_nan_at_test",
            "gated_no_dropout"]


def _loso_predict(train_pool: pd.DataFrame, score_pool: pd.DataFrame, models) -> pd.DataFrame:
    """For each subject: train on the OTHER subjects' train_pool rows, score
    that subject's score_pool rows under both tiers."""
    from joblib import Parallel, delayed

    def one(i, s, name):
        import torch
        torch.set_num_threads(1)
        tr = train_pool[train_pool.subject_id != s]
        te = score_pool[score_pool.subject_id == s]
        out = ec.fit_predict(name, tr, te, seed=SEED_BASE + i)
        return out.merge(te[["row_id", "condition", "condition_block", "t_start_min",
                             "protocol_stress_before_amusement"]], on="row_id")
    subjects = sorted(score_pool.subject_id.unique())
    outs = Parallel(n_jobs=16)(delayed(one)(i, s, n) for i, s in enumerate(subjects) for n in models)
    return pd.concat(outs, ignore_index=True)


def _boot_ci(sub: pd.DataFrame, fn, n_boot=N_BOOT, seed=17):
    """Participant-cluster percentile 95% CI; fn(y, p, sid) -> statistic."""
    rng = np.random.default_rng(seed)
    arr = {s_: (g["y"].values, g["p_raw"].values) for s_, g in sub.groupby("subject_id")}
    subj = np.array(list(arr))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subj, len(subj), replace=True)
        y = np.concatenate([arr[s_][0] for s_ in pick])
        if len(np.unique(y)) < 2:
            continue
        p = np.concatenate([arr[s_][1] for s_ in pick])
        sid = np.concatenate([np.full(len(arr[s_][0]), j) for j, s_ in enumerate(pick)])
        vals.append(fn(y, p, sid))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def _discrimination_summary(pred: pd.DataFrame) -> dict:
    out = {}
    for (name, tier), g in pred.groupby(["model", "tier"]):
        ws = lambda y, p, sid: within_subject_auroc(y, p, sid)
        pooled = lambda y, p, sid: roc_auc_score(y, p)
        per_subj = {s: within_subject_auroc(d.y.values, d.p_raw.values, d.subject_id.values)
                    for s, d in g.groupby("subject_id")}
        order = g.groupby("subject_id")["protocol_stress_before_amusement"].first()
        out.setdefault(name, {})[tier] = {
            "pooled_auroc": float(roc_auc_score(g.y, g.p_raw)), "pooled_auroc_ci95": _boot_ci(g, pooled),
            "within_subject_auroc": float(within_subject_auroc(g.y.values, g.p_raw.values, g.subject_id.values)),
            "within_subject_auroc_ci95": _boot_ci(g, ws),
            "brier": float(np.mean((g.p_raw - g.y) ** 2)),
            "within_subject_auroc_by_protocol": {
                "stress_before_amusement": float(np.mean([per_subj[s] for s in order.index if order[s]])),
                "stress_after_amusement": float(np.mean([per_subj[s] for s in order.index if not order[s]])),
                "n": {"stress_before_amusement": int(order.sum()),
                      "stress_after_amusement": int((~order).sum())}},
            "per_subject": per_subj,
        }
    return out


def check_a(ext: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    d = ext[ext.condition.isin(["stress", "amusement"])].copy()
    d["label"] = (d.condition == "stress").astype(int)
    pred = _loso_predict(d, d, MODELS_A)
    res = _discrimination_summary(pred)
    hr = d.groupby(["subject_id", "condition"])[ec.HR_FEATURE].mean().unstack()
    res["_hr_delta_mean_nn_stress_minus_amusement_ms"] = {
        "median": float((hr["stress"] - hr["amusement"]).median()),
        "range": [float((hr["stress"] - hr["amusement"]).min()),
                  float((hr["stress"] - hr["amusement"]).max())]}
    res["_n_windows"] = {c: int(n) for c, n in d.condition.value_counts().items()}
    return res, pred.assign(check="A")


def check_b(ext: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    train = ext[ext.condition.isin(["baseline", "stress"])].copy()
    train["label"] = (train.condition == "stress").astype(int)
    score = ext.copy()
    score["label"] = (score.condition == "stress").astype(int)   # placeholder for non-BS rows
    pred = _loso_predict(train, score, MODELS_B)
    pred["cond_block"] = np.where(pred.condition == "meditation",
                                  "meditation_" + (pred.condition_block + 1).astype(str),
                                  pred.condition)
    res = {}
    for (name, tier), g in pred.groupby(["model", "tier"]):
        r = {}
        for cb, h in g.groupby("cond_block"):
            subj_frac = h.groupby("subject_id").apply(lambda x: float((x.p_raw >= 0.5).mean()),
                                                      include_groups=False)
            r[cb] = {"frac_pred_stress": float((h.p_raw >= 0.5).mean()),
                     "frac_pred_stress_platt_pc": float((h.p_platt_pc >= 0.5).mean()),
                     "mean_p": float(h.p_raw.mean()),
                     "subject_frac_median": float(subj_frac.median()),
                     "n_subjects_frac_gt_half": int((subj_frac > 0.5).sum()),
                     "n_windows": int(len(h)), "mean_t_start_min": float(h.t_start_min.mean())}
        res.setdefault(name, {})[tier] = r
    return res, pred.assign(check="B")


def check_c(ext: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    d = ext[ext.condition.isin(["baseline", "meditation"])].copy()
    d["label"] = (d.condition == "meditation").astype(int)
    pred = _loso_predict(d, d, MODELS_C)
    pred = pred[pred.tier == "complete"]
    return _discrimination_summary(pred), pred.assign(check="C")


def check_d(ext: pd.DataFrame) -> dict:
    b = ext[ext.condition == "baseline"]
    out = {}
    for col in ["eda_tonic_mean", ec.HR_FEATURE]:
        rhos = {s: float(spearmanr(g.t_start_min, g[col])[0]) for s, g in b.groupby("subject_id")}
        v = np.array(list(rhos.values()))
        out[col] = {"median_rho_vs_time": float(np.median(v)),
                    "n_positive": int((v > 0).sum()), "n": int(len(v)), "per_subject": rhos}
    return out


def main() -> dict:
    ext = ec.load_extended("chest", ("baseline", "stress", "amusement", "meditation"))
    manifest = json.loads(ec.EXT_MANIFEST_PATH.read_text(encoding="utf-8"))
    res_a, pa = check_a(ext)
    res_b, pb = check_b(ext)
    res_c, pc = check_c(ext)
    res = {"extended_table_sha256": manifest["sha256"],
           "protocol_versions": {
               "stress_before_amusement": int(ext.groupby("subject_id")
                                              .protocol_stress_before_amusement.first().sum()),
               "n_subjects": int(ext.subject_id.nunique())},
           "A_stress_vs_amusement": res_a,
           "B_negative_control": res_b,
           "C_baseline_vs_meditation": res_c,
           "D_within_baseline_drift": check_d(ext)}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.concat([pa, pb, pc]).to_csv(RESULTS / "order_confound_predictions.csv", index=False)
    _print(res)
    return res


def _print(res):
    print("\nA. stress vs amusement (LOSO, raw)")
    for m, t in res["A_stress_vs_amusement"].items():
        if m.startswith("_"):
            print(" ", m, t)
            continue
        for tier, r in t.items():
            print(f"  {m:30s} {tier:8s} pooled {r['pooled_auroc']:.3f} {np.round(r['pooled_auroc_ci95'], 3)}"
                  f" ws {r['within_subject_auroc']:.3f} {np.round(r['within_subject_auroc_ci95'], 3)}"
                  f" by protocol {r['within_subject_auroc_by_protocol']['stress_before_amusement']:.3f}/"
                  f"{r['within_subject_auroc_by_protocol']['stress_after_amusement']:.3f}")
    print("\nB. negative control: fraction of windows predicted stress (p>=0.5)")
    for m, t in res["B_negative_control"].items():
        for tier, r in t.items():
            print(f"  {m:28s} {tier:8s} " + "  ".join(
                f"{cb} {v['frac_pred_stress']:.2f}" for cb, v in sorted(r.items())))
    print("\nC. baseline vs meditation (complete)")
    for m, t in res["C_baseline_vs_meditation"].items():
        r = t["complete"]
        print(f"  {m:28s} pooled {r['pooled_auroc']:.3f} ws {r['within_subject_auroc']:.3f} "
              f"{np.round(r['within_subject_auroc_ci95'], 3)}")
    print("\nD.", {k: (v["median_rho_vs_time"], v["n_positive"]) for k, v in res["D_within_baseline_drift"].items()})


if __name__ == "__main__":
    main()
