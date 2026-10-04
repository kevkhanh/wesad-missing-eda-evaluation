"""
Audit Step 2 — restrict features to the common (ECG-only) modality set.

Rationale (audit doc): every EDA-derived feature is structurally NaN on
MIT-BIH, and LightGBM's native NaN routing turns that into a perfect
splitting rule. Any cross-dataset probe or MMD test must therefore run on
F_ECG (the intersection modality), not the full feature set. Step 0
already proves the collinearity; this step provides the surgical filter.

Two entry points:
  * restrict_to_ecg(df) — return a copy with EDA features dropped
  * eda_ablation_report(model_fit_fn, train_df, test_df) — quantify how much
    of the Preprint 1 headline AUC comes from EDA features specifically, per
    the audit-doc Step 2 recommendation ("re-run the Preprint 1 classifier
    with EDA features removed and report the change in AUC").
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import ALL_FEATURE_NAMES, ECG_FEATURE_NAMES, EDA_FEATURE_NAMES


def restrict_to_ecg(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with EDA-derived features dropped (kept: ECG_FEATURE_NAMES).

    Non-feature columns (subject_id, label, etc.) are preserved. The returned
    frame is a copy; upstream frames are untouched.
    """
    drop = [c for c in EDA_FEATURE_NAMES if c in df.columns]
    return df.drop(columns=drop).copy()


def ecg_only_view_of_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: same as restrict_to_ecg but also zero-fills any EDA
    columns the caller wants to keep in-place (for pipelines that expect
    a fixed schema). Sets EDA columns to NaN so LightGBM treats them as
    missing rather than as a real 0."""
    out = df.copy()
    for c in EDA_FEATURE_NAMES:
        if c in out.columns:
            out[c] = float("nan")
    return out


def eda_ablation_report(train_df: pd.DataFrame,
                        test_df: pd.DataFrame) -> dict:
    """Quantify how much AUC the Preprint 1 classifier loses without EDA.

    Trains two LightGBM baselines on the same folds:
      * full: ALL_FEATURE_NAMES (matches train_stage3_models.train_baseline)
      * ecg_only: only ECG_FEATURE_NAMES

    Reports AUC and delta. A large delta means EDA is doing real work; a
    small delta plus a high identity-probe AUC on EDA-derived features
    means EDA is mostly a dataset shortcut.
    """
    import lightgbm as lgb

    def _fit_eval(feature_cols):
        train_complete = train_df[train_df.get("has_eda", True) == True]  # noqa: E712
        if len(train_complete) < 4:
            train_complete = train_df
        m = lgb.LGBMClassifier(n_estimators=50, max_depth=3,
                                random_state=42, verbosity=-1)
        m.fit(train_complete[feature_cols], train_complete["label"])
        probs = m.predict_proba(test_df[feature_cols])[:, 1]
        if test_df["label"].nunique() < 2:
            return {"auroc": float("nan"), "n": len(test_df)}
        return {"auroc": float(roc_auc_score(test_df["label"], probs)),
                "n": len(test_df)}

    full = _fit_eval(ALL_FEATURE_NAMES)
    ecg = _fit_eval(ECG_FEATURE_NAMES)
    delta = None
    if full["auroc"] == full["auroc"] and ecg["auroc"] == ecg["auroc"]:  # not-NaN
        delta = full["auroc"] - ecg["auroc"]
    return {
        "full_auroc": full["auroc"],
        "ecg_only_auroc": ecg["auroc"],
        "eda_contribution_auroc_delta": delta,
        "n_test": full["n"],
    }


if __name__ == "__main__":
    from train_stage3_models import load_and_split

    train_df, test_df = load_and_split()
    report = eda_ablation_report(train_df, test_df)
    print("Step 2 — EDA ablation report")
    for k, v in report.items():
        print(f"  {k:<32}  {v}")
