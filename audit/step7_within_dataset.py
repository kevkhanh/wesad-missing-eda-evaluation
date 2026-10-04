"""
Audit Step 7 — within-dataset clinical re-validation.

Runs the Preprint 1 pipeline on the only clinically interpretable
discrimination contrasts the data can support:

  C3  WESAD baseline vs stress   — LOSO (leave-one-subject-out)
  C4  MIT-BIH NSR vs arrhythmic  — StratifiedGroupKFold on subjects

Plus a feature-elimination sweep: iteratively drop the top-|SHAP| identity
feature (as ranked by the Step 4 P2 probe on C1), retrain the clinical
classifier, and plot identity-AUC against clinical-AUC. If clinical AUC
collapses in step with identity AUC, the clinical performance WAS the
shortcut (Branch D in the decision tree).

Run:
    python -m audit.step7_within_dataset
    python -m audit.step7_within_dataset --fast
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import ECG_FEATURE_NAMES
from train_stage3_models import _load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "audit"


def _fit_lgbm(train_X, train_y, n_estimators=50, max_depth=3):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                            random_state=42, verbosity=-1)
    m.fit(train_X, train_y)
    return m


def _cv_auc(df: pd.DataFrame, feature_cols: list,
            positive_label: str, negative_label: str,
            splitter) -> tuple[float, list[float]]:
    """Return (mean_fold_AUC, per_fold_AUCs)."""
    sub = df[df["label"].isin([positive_label, negative_label])].copy()
    if sub.empty:
        return float("nan"), []
    sub["y"] = (sub["label"] == positive_label).astype(int)
    subjects = sub["subject_id"].values
    aucs = []
    for tr, te in splitter.split(sub, groups=subjects):
        tr_df, te_df = sub.iloc[tr], sub.iloc[te]
        if te_df["y"].nunique() < 2 or tr_df["y"].nunique() < 2:
            continue
        m = _fit_lgbm(tr_df[feature_cols], tr_df["y"])
        probs = m.predict_proba(te_df[feature_cols])[:, 1]
        aucs.append(float(roc_auc_score(te_df["y"], probs)))
    return (float(np.mean(aucs)) if aucs else float("nan"), aucs)


def within_wesad_c3(df: pd.DataFrame, feature_cols: list) -> dict:
    """LOSO on WESAD baseline vs stress."""
    wesad = df[df["dataset"] == "WESAD"].copy()
    if wesad.empty:
        return {"auc_mean": None, "per_fold": []}
    n_subj = wesad["subject_id"].nunique()
    if n_subj < 2:
        return {"auc_mean": None, "per_fold": []}
    splitter = LeaveOneGroupOut()
    mean_auc, per_fold = _cv_auc(wesad, feature_cols, "stress", "baseline", splitter)
    return {"auc_mean": mean_auc, "per_fold": per_fold, "n_subjects": int(n_subj)}


def within_mitbih_c4(df: pd.DataFrame, feature_cols: list) -> dict:
    """Subject-grouped 5-fold on MIT-BIH NSR vs arrhythmic."""
    mit = df[df["dataset"] == "MITBIH"].copy()
    if mit.empty:
        return {"auc_mean": None, "per_fold": []}
    n_subj = mit["subject_id"].nunique()
    n_folds = min(5, n_subj)
    if n_folds < 2:
        return {"auc_mean": None, "per_fold": []}
    splitter = GroupKFold(n_splits=n_folds)
    mean_auc, per_fold = _cv_auc(mit, feature_cols, "arrhythmic", "nsr", splitter)
    return {"auc_mean": mean_auc, "per_fold": per_fold, "n_subjects": int(n_subj)}


# ---------------------------------------------------------------------------
# Feature-elimination sweep (Branch D detector)
# ---------------------------------------------------------------------------

def _identity_shap_ranking(df: pd.DataFrame, feature_cols: list) -> list:
    """Fit an identity classifier (dataset ~ features) and rank features
    by mean |SHAP| — the "identity signature" ranking used to drive the
    sweep."""
    try:
        import shap
    except ImportError:
        return list(feature_cols)

    import lightgbm as lgb
    subs = df[df["label"].isin(["baseline", "nsr"])].copy()
    if subs.empty:
        return list(feature_cols)
    subs["y_id"] = (subs["dataset"] == "MITBIH").astype(int)
    m = lgb.LGBMClassifier(n_estimators=100, num_leaves=15,
                            random_state=42, verbosity=-1)
    m.fit(subs[feature_cols], subs["y_id"])
    try:
        explainer = shap.TreeExplainer(m)
        shap_vals = explainer.shap_values(subs[feature_cols])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
    except Exception:
        mean_abs = np.abs(m.feature_importances_)
    order = np.argsort(-mean_abs)
    return [feature_cols[i] for i in order]


def feature_elimination_sweep(df: pd.DataFrame,
                              feature_cols: list,
                              max_drop: int | None = None) -> list:
    """Drop the top-|SHAP| identity feature, re-fit identity and clinical
    classifiers, record both AUCs. Repeat until max_drop features are gone
    or the feature set has <=1 left.
    """
    ranking = _identity_shap_ranking(df, feature_cols)
    if max_drop is None:
        max_drop = min(len(ranking), 5)

    rows = []
    kept = list(feature_cols)
    for step in range(max_drop + 1):
        # Identity AUC (subject-collapsed for consistency with Step 4)
        id_auc = _identity_auc(df, kept)
        # Clinical AUCs
        c3 = within_wesad_c3(df, kept)["auc_mean"]
        c4 = within_mitbih_c4(df, kept)["auc_mean"]
        rows.append({
            "features_kept": list(kept),
            "n_features": len(kept),
            "identity_auc_C1": id_auc,
            "clinical_auc_C3": c3,
            "clinical_auc_C4": c4,
            "dropped_so_far": ranking[:step],
        })
        if step < len(ranking) and len(kept) > 1:
            drop = ranking[step]
            if drop in kept:
                kept.remove(drop)
    return rows


def _identity_auc(df: pd.DataFrame, feature_cols: list) -> float:
    """Mean per-fold identity AUC (dataset ~ features), C1 subset."""
    from sklearn.model_selection import GroupKFold
    subs = df[df["label"].isin(["baseline", "nsr"])].copy()
    if subs.empty:
        return float("nan")
    subs["y_id"] = (subs["dataset"] == "MITBIH").astype(int)
    if subs["y_id"].nunique() < 2:
        return float("nan")
    n = min(5, subs["subject_id"].nunique())
    if n < 2:
        return float("nan")
    aucs = []
    for tr, te in GroupKFold(n_splits=n).split(subs, groups=subs["subject_id"]):
        tr_df, te_df = subs.iloc[tr], subs.iloc[te]
        if te_df["y_id"].nunique() < 2 or tr_df["y_id"].nunique() < 2:
            continue
        m = _fit_lgbm(tr_df[feature_cols], tr_df["y_id"])
        probs = m.predict_proba(te_df[feature_cols])[:, 1]
        aucs.append(roc_auc_score(te_df["y_id"], probs))
    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_step7(fast: bool = False) -> dict:
    df = _load_dataset()
    if df["label"].dtype != object:
        raise RuntimeError("Step 7 needs the string-label dataset.")
    feature_cols = [c for c in ECG_FEATURE_NAMES if c in df.columns]

    c3 = within_wesad_c3(df, feature_cols)
    c4 = within_mitbih_c4(df, feature_cols)
    sweep = feature_elimination_sweep(
        df, feature_cols, max_drop=2 if fast else 5,
    )

    manifest = {
        "C3_wesad_baseline_vs_stress": c3,
        "C4_mitbih_nsr_vs_arrhythmic": c4,
        "feature_elimination_sweep": sweep,
        "verdict": _verdict_step7(c3, c4, sweep),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step7_within_dataset.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Audit Step 7 — within-dataset clinical re-validation ===")
    print(f"  C3 (WESAD, baseline vs stress, LOSO): "
          f"AUC = {_fmt(c3['auc_mean'])}  (n_subj={c3.get('n_subjects','?')})")
    print(f"  C4 (MIT-BIH, NSR vs arrhythmic, GroupKFold): "
          f"AUC = {_fmt(c4['auc_mean'])}  (n_subj={c4.get('n_subjects','?')})")
    print("\n  Feature-elimination sweep (identity | C3 clinical | C4 clinical):")
    for r in sweep:
        print(f"    kept={r['n_features']:>2}  "
              f"identity={_fmt(r['identity_auc_C1'])}  "
              f"C3={_fmt(r['clinical_auc_C3'])}  "
              f"C4={_fmt(r['clinical_auc_C4'])}")
    print(f"\n  Verdict: {manifest['verdict']}")
    return manifest


def _fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  N/A"
    return f"{x:.3f}"


def _verdict_step7(c3: dict, c4: dict, sweep: list) -> str:
    """Detect Branch D pattern: clinical AUC collapses as identity AUC drops."""
    if len(sweep) < 2:
        return "insufficient_sweep"
    identity_first = sweep[0]["identity_auc_C1"]
    identity_last = sweep[-1]["identity_auc_C1"]
    c3_first = sweep[0]["clinical_auc_C3"]
    c3_last = sweep[-1]["clinical_auc_C3"]
    id_drop = (identity_first or 0) - (identity_last or 0)
    c3_drop = (c3_first or 0) - (c3_last or 0)
    if id_drop > 0.15 and c3_drop > 0.15:
        return "clinical_collapses_with_identity_BRANCH_D"
    if c3_drop < 0.05 and id_drop > 0.15:
        return "clinical_survives_identity_removal"
    return "inconclusive"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    run_step7(fast=args.fast)
