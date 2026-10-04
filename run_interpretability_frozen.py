"""
Interpretability regenerated on the frozen feature table (outline Appendix E).

Replaces the 2026-09-20 `run_interpretability.py` numbers, which came from
the deleted 445-window cache and a single random split. Everything here is
out-of-fold under LOSO: each subject's attributions come from a model that
never saw that subject. Descriptive only; no inferential claims.

  * SHAP (TreeExplainer, exact) on (ii) NaN-trained LightGBM, held-out
    windows under both conditions, and on (iii) ECG-only LightGBM.
  * Integrated Gradients on the gated network (dropout p=0.5), logit output,
    256-step midpoint Riemann sum, baseline = 0 in robust-standardised space (= the training
    median of every feature). Completeness is checked per window:
    sum(IG) vs logit(x) - logit(baseline). Under the reduced condition the
    EDA gate is closed, so EDA attributions are zero by construction and only
    ECG attributions are reported.

Grouped attributions (share of total mean |attribution|):
  HR = hrv_mean_nn; other HRV = remaining ECG features; EDA = 4 EDA
  features; coupling = coupling_peak_corr + coupling_lag_sec.

Output: results/interpretability_frozen.json, results/interp_*_frozen.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import eval_core as ec
from feature_extraction import ALL_FEATURE_NAMES, ECG_FEATURE_NAMES, EDA_FEATURE_NAMES

RESULTS = Path(__file__).resolve().parent / "results"
SEED_BASE = 200
IG_STEPS = 256
GROUPS = {
    "HR": ["hrv_mean_nn"],
    "other_HRV": [f for f in ECG_FEATURE_NAMES if f != "hrv_mean_nn"],
    "EDA": EDA_FEATURE_NAMES[:4],
    "coupling": EDA_FEATURE_NAMES[4:],
}


def _shap_fold(df, fold_i, test_subj):
    import shap
    train_df, test_df = ec.split_by_subjects(df, test_subj)
    tiers = ec.paired_tiers(test_df)
    rows = []
    for name, model in (("lgbm_nan_training_p0.5", ec.LGBMModel(ALL_FEATURE_NAMES, train_nan_p=0.5)),
                        ("lgbm_ecg_only_per_config", ec.LGBMModel(ECG_FEATURE_NAMES))):
        model.fit(train_df, seed=SEED_BASE + fold_i)
        expl = shap.TreeExplainer(model.model_)
        for tier, tdf in tiers.items():
            sv = expl.shap_values(tdf[model.features])
            sv = sv[1] if isinstance(sv, list) else sv
            for j, f in enumerate(model.features):
                rows.append({"model": name, "tier": tier, "feature": f,
                             "sum_abs": float(np.abs(sv[:, j]).sum()), "n": len(tdf)})
    return rows


def _ig_fold(df, fold_i, test_subj):
    torch.set_num_threads(1)
    train_df, test_df = ec.split_by_subjects(df, test_subj)
    m = ec.GatedModel(0.5, drop_eda=True).fit(train_df, seed=SEED_BASE + fold_i)
    net = m.net_
    rows, completeness = [], []
    for tier, tdf in ec.paired_tiers(test_df).items():
        ecg, eda, mask, _ = m._tensors(m._prep(tdf))
        b_ecg, b_eda = torch.zeros_like(ecg), torch.zeros_like(eda)
        g_ecg, g_eda = torch.zeros_like(ecg), torch.zeros_like(eda)
        for k in range(IG_STEPS):
            a = (k + 0.5) / IG_STEPS          # midpoint rule
            xe = (b_ecg + a * (ecg - b_ecg)).requires_grad_(True)
            xd = (b_eda + a * (eda - b_eda)).requires_grad_(True)
            out = net(xe, xd, mask, force_no_dropout=True).sum()
            ge, gd = torch.autograd.grad(out, [xe, xd])
            g_ecg += ge
            g_eda += gd
        ig_ecg = ((ecg - b_ecg) * g_ecg / IG_STEPS).numpy()
        ig_eda = ((eda - b_eda) * g_eda / IG_STEPS).numpy()
        with torch.no_grad():
            delta = (net(ecg, eda, mask, force_no_dropout=True)
                     - net(b_ecg, b_eda, mask, force_no_dropout=True)).numpy()
        total = ig_ecg.sum(1) + ig_eda.sum(1)
        completeness.extend(np.abs(total - delta).tolist())
        for j, f in enumerate(ECG_FEATURE_NAMES):
            rows.append({"model": "gated_dropout_p0.5", "tier": tier, "feature": f,
                         "sum_abs": float(np.abs(ig_ecg[:, j]).sum()), "n": len(tdf)})
        for j, f in enumerate(EDA_FEATURE_NAMES):
            rows.append({"model": "gated_dropout_p0.5", "tier": tier, "feature": f,
                         "sum_abs": float(np.abs(ig_eda[:, j]).sum()), "n": len(tdf)})
    return rows, completeness


def _summarise(rows: pd.DataFrame) -> dict:
    out = {}
    for (model, tier), g in rows.groupby(["model", "tier"]):
        per_feat = g.groupby("feature")[["sum_abs", "n"]].sum()
        mean_abs = (per_feat["sum_abs"] / per_feat["n"]).sort_values(ascending=False)
        total = mean_abs.sum()
        groups = {k: float(mean_abs.reindex(v).fillna(0).sum() / total) if total > 0 else float("nan")
                  for k, v in GROUPS.items()}
        out[f"{model}|{tier}"] = {
            "mean_abs_by_feature": {k: float(v) for k, v in mean_abs.items()},
            "rank": list(mean_abs.index),
            "group_share": groups,
        }
    return out


def main(n_jobs: int = 15) -> dict:
    from joblib import Parallel, delayed
    df = ec.load_wesad_frozen()
    folds = ec.loso(df["subject_id"])
    shap_rows = Parallel(n_jobs=n_jobs)(delayed(_shap_fold)(df, i, ts) for i, ts in enumerate(folds))
    ig_out = Parallel(n_jobs=n_jobs)(delayed(_ig_fold)(df, i, ts) for i, ts in enumerate(folds))
    shap_df = pd.DataFrame([r for rs in shap_rows for r in rs])
    ig_df = pd.DataFrame([r for rs, _ in ig_out for r in rs])
    comp = np.concatenate([np.asarray(c) for _, c in ig_out])
    worst_completeness = float(comp.max())
    res = {"scheme": "LOSO, out-of-fold attributions on held-out subjects",
           "ig": {"steps": IG_STEPS, "rule": "midpoint",
                  "baseline": "training median (0 after robust standardisation)",
                  "completeness_error_logit": {"max": worst_completeness,
                                               "p95": float(np.percentile(comp, 95)),
                                               "median": float(np.median(comp))}},
           **_summarise(pd.concat([shap_df, ig_df], ignore_index=True))}
    shap_df.to_csv(RESULTS / "interp_shap_frozen.csv", index=False)
    ig_df.to_csv(RESULTS / "interp_ig_frozen.csv", index=False)
    (RESULTS / "interpretability_frozen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, v in res.items():
        if isinstance(v, dict) and "group_share" in v:
            gs = ", ".join(f"{g} {s:.0%}" for g, s in v["group_share"].items())
            print(f"{k:40s} top3 {v['rank'][:3]} | {gs}")
    print("IG completeness |sum(IG) - (f(x)-f(b))| (logit): max %.4f, p95 %.4f, median %.5f"
          % (worst_completeness, np.percentile(comp, 95), np.median(comp)))
    return res


if __name__ == "__main__":
    main()
