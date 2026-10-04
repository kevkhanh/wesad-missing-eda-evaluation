"""
Section 5.5: Interpretability — exploratory, not confirmatory.

Two components in priority order:

1. LightGBM SHAP (TreeExplainer, exact attributions) — global feature importance
   across the full HRV + EDA feature set. Exploratory follow-up to H2a: does
   anything other than coupling_lag_sec carry signal in the panic-proxy-vs-baseline
   classification?

2. Integrated Gradients on the modality-invariant model — conditional feature
   attribution under three conditions: (a) all modalities present, (b) EDA missing,
   (c) ECG inputs zeroed. Shows which features the model reweights onto when a
   channel drops — the mechanistic explanation behind H-B1's calibration result.

Framing enforced throughout:
- Panic-proxy-vs-baseline, NOT panic-vs-cardiac.
- SHAP/IG are descriptive; they do not validate or invalidate H2a statistically.
- No individual-level force plots as headline results (global summary only).
- No claim that high SHAP importance implies clinical relevance.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from feature_extraction import ECG_FEATURE_NAMES, EDA_FEATURE_NAMES, ALL_FEATURE_NAMES
from train_stage3_models import (
    _load_dataset,
    _standardize_pair,
    _subject_of,
    expected_calibration_error,
    load_and_split,
    train_baseline,
    train_invariant_model,
    ModalityInvariantNet,
)

RNG_SEED = 42
MASK_FRACTION = 0.5
N_STEPS_IG = 50  # interpolation steps for Integrated Gradients

torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Component 1: LightGBM SHAP (TreeExplainer, exact)
# ---------------------------------------------------------------------------

def run_lgbm_shap(train_df, test_df):
    """
    Train the LightGBM baseline, compute exact SHAP values via TreeExplainer
    on the complete-modality test partition, and return a ranked importance table.

    Returns:
        shap_df: DataFrame with columns [feature, mean_abs_shap, rank]
        shap_values: raw (n_samples, n_features) SHAP array
    """
    try:
        import shap
    except ImportError:
        print("  [skip] shap not installed — pip install shap")
        return None, None

    print("  Training LightGBM baseline...")
    model, _ = train_baseline(train_df, test_df)

    # Explain on complete-modality test rows (the regime the baseline was trained for)
    test_complete = test_df[test_df["has_eda"]]
    if len(test_complete) < 2:
        print("  [skip] too few complete-modality test rows")
        return None, None

    X_test = test_complete[ALL_FEATURE_NAMES].fillna(0.0)
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_test)
    # shap_values may be list [class0, class1] for binary LightGBM
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class-1 (stress/panic-proxy) SHAP

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": ALL_FEATURE_NAMES,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = shap_df.index + 1

    print("\n  --- LightGBM SHAP: global feature importance (complete-modality test) ---")
    print(f"  {'Rank':>4}  {'Feature':<24}  {'Mean |SHAP|':>12}  Group")
    for _, row in shap_df.iterrows():
        group = "ECG/HRV" if row["feature"] in ECG_FEATURE_NAMES else "EDA"
        print(f"  {row['rank']:>4}  {row['feature']:<24}  {row['mean_abs_shap']:>12.5f}  {group}")

    # Flag H2a's pre-specified feature
    h2a_rank = shap_df[shap_df["feature"] == "coupling_lag_sec"]["rank"].values
    if h2a_rank.size:
        print(f"\n  coupling_lag_sec (H2a's pre-specified feature): rank {h2a_rank[0]} of "
              f"{len(ALL_FEATURE_NAMES)}")
    coupling_corr_rank = shap_df[shap_df["feature"] == "coupling_peak_corr"]["rank"].values
    if coupling_corr_rank.size:
        print(f"  coupling_peak_corr: rank {coupling_corr_rank[0]} of {len(ALL_FEATURE_NAMES)}")

    shap_df.to_csv(OUT_DIR / "interp_lgbm_shap.csv", index=False)
    print("  Saved: interp_lgbm_shap.csv")

    # Attempt summary plot (saves to file if matplotlib available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap as shap_lib

        fig, ax = plt.subplots(figsize=(8, 5))
        shap_lib.summary_plot(
            shap_values, X_test,
            feature_names=ALL_FEATURE_NAMES,
            plot_type="bar", show=False,
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "interp_lgbm_shap_summary.png", dpi=150)
        plt.close()
        print("  Saved: interp_lgbm_shap_summary.png")
    except Exception as e:
        print(f"  [note] plot not saved: {e!r}")

    return shap_df, shap_values


# ---------------------------------------------------------------------------
# Component 2: Integrated Gradients on the modality-invariant model
# ---------------------------------------------------------------------------

def _integrated_gradients(model, ecg_x, eda_x, eda_mask, n_steps=N_STEPS_IG):
    """
    Approximate IG for ModalityInvariantNet.
    Baseline is zeros (= training-distribution mean after standardisation).
    When eda_mask is False, the EDA gate is closed throughout the path, so
    EDA IG is naturally 0 — the model literally cannot use those features.
    Returns (ecg_ig, eda_ig), each (B, n_features).
    """
    baseline_ecg = torch.zeros_like(ecg_x)
    baseline_eda = torch.zeros_like(eda_x)

    ecg_grad_acc = torch.zeros_like(ecg_x)
    eda_grad_acc = torch.zeros_like(eda_x)

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interp_ecg = (baseline_ecg + alpha * (ecg_x - baseline_ecg)).detach().requires_grad_(True)
        interp_eda = (baseline_eda + alpha * (eda_x - baseline_eda)).detach().requires_grad_(True)

        logits = model(interp_ecg, interp_eda, eda_mask, force_no_dropout=True)
        torch.sigmoid(logits).sum().backward()

        ecg_grad_acc = ecg_grad_acc + interp_ecg.grad.detach()
        eda_grad_acc = eda_grad_acc + interp_eda.grad.detach()

    ecg_ig = (ecg_x - baseline_ecg) * ecg_grad_acc / n_steps
    eda_ig = (eda_x - baseline_eda) * eda_grad_acc / n_steps
    return ecg_ig.detach(), eda_ig.detach()


def _apply_simulated_eda_mask_local(df, fraction=MASK_FRACTION, seed=RNG_SEED):
    """Inline EDA masking so this module has no import-order dependency."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    mask = np.zeros(len(df), dtype=bool)
    for lbl in df["label"].unique():
        idx = df.index[df["label"] == lbl]
        n = int(round(len(idx) * fraction))
        mask[df.index.isin(rng.choice(idx, size=n, replace=False))] = True
    df["has_eda"] = ~mask
    df.loc[mask, EDA_FEATURE_NAMES] = float("nan")
    return df


def run_invariant_ig(train_df, test_df):
    """
    Train the modality-invariant model and compute Integrated Gradients under
    three conditions on the test set:
      (a) all_present   — test rows with EDA present, ECG at real values
      (b) eda_missing   — test rows with EDA masked (gate closed), ECG carries all signal
      (c) ecg_zeroed    — test rows with EDA present but ECG inputs set to 0;
                          shows which EDA features the model falls back on

    Returns a DataFrame with mean |IG| per feature per condition.
    """
    print("  Training modality-invariant model...")
    model, _ = train_invariant_model(train_df, test_df)
    model.eval()

    train_s, test_s = _standardize_pair(train_df, test_df)

    # Condition (a): EDA present
    cond_a = test_s[test_s["has_eda"]].copy()
    # Condition (b): EDA missing (already NaN → 0 via fillna)
    cond_b = test_s[~test_s["has_eda"]].copy()
    # Condition (c): EDA present, but ECG inputs zeroed
    cond_c = cond_a.copy()
    cond_c[ECG_FEATURE_NAMES] = 0.0

    def _to_tensors_main(df):
        ecg_x = torch.tensor(df[ECG_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32)
        eda_x = torch.tensor(df[EDA_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32)
        eda_m = torch.tensor(df["has_eda"].values, dtype=torch.bool)
        return ecg_x, eda_x, eda_m

    rows = []
    conditions = [
        ("all_present", cond_a),
        ("eda_missing", cond_b),
        ("ecg_zeroed",  cond_c),
    ]

    for cond_name, cond_df in conditions:
        if len(cond_df) < 2:
            print(f"  [skip] {cond_name}: too few rows")
            continue
        ecg_x, eda_x, eda_m = _to_tensors_main(cond_df)
        ecg_ig, eda_ig = _integrated_gradients(model, ecg_x, eda_x, eda_m)
        all_ig = torch.cat([ecg_ig, eda_ig], dim=1).numpy()  # (B, n_all_features)
        mean_abs = np.abs(all_ig).mean(axis=0)
        for feat, val in zip(ALL_FEATURE_NAMES, mean_abs):
            rows.append({"condition": cond_name, "feature": feat, "mean_abs_ig": float(val)})

    ig_df = pd.DataFrame(rows)

    print("\n  --- Invariant model IG: mean |IG| per condition ---")
    for cond_name in ["all_present", "eda_missing", "ecg_zeroed"]:
        sub = ig_df[ig_df["condition"] == cond_name].sort_values(
            "mean_abs_ig", ascending=False
        ).reset_index(drop=True)
        if sub.empty:
            continue
        print(f"\n  Condition: {cond_name}")
        print(f"  {'Rank':>4}  {'Feature':<24}  {'Mean |IG|':>10}  Group")
        for i, row in sub.iterrows():
            group = "ECG/HRV" if row["feature"] in ECG_FEATURE_NAMES else "EDA"
            print(f"  {i+1:>4}  {row['feature']:<24}  {row['mean_abs_ig']:>10.6f}  {group}")

    ig_df.to_csv(OUT_DIR / "interp_invariant_ig.csv", index=False)
    print("\n  Saved: interp_invariant_ig.csv")

    # Comparison: which ECG features rise in rank from all_present → eda_missing?
    _print_rank_shift(ig_df, from_cond="all_present", to_cond="eda_missing",
                      feature_group=ECG_FEATURE_NAMES,
                      label="ECG feature rank shift: all_present -> eda_missing")

    return ig_df


def _print_rank_shift(ig_df, from_cond, to_cond, feature_group, label):
    """Print how feature ranks change between two conditions, for one feature group."""
    def _ranks(cond):
        sub = ig_df[ig_df["condition"] == cond].sort_values(
            "mean_abs_ig", ascending=False
        ).reset_index(drop=True)
        return {row["feature"]: i + 1 for i, row in sub.iterrows()}

    r_from = _ranks(from_cond)
    r_to   = _ranks(to_cond)
    if not r_from or not r_to:
        return

    print(f"\n  {label}")
    print(f"  {'Feature':<24}  {'Rank in ' + from_cond:>18}  {'Rank in ' + to_cond:>15}  Delta")
    for feat in feature_group:
        rf = r_from.get(feat, "?")
        rt = r_to.get(feat, "?")
        delta = (rf - rt) if isinstance(rf, int) and isinstance(rt, int) else "?"
        arrow = f"+{delta}" if isinstance(delta, int) and delta > 0 else str(delta)
        print(f"  {feat:<24}  {str(rf):>18}  {str(rt):>15}  {arrow}")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_interpretability():
    train_df, test_df = load_and_split(simulated_mask_fraction=MASK_FRACTION)
    print(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows "
          f"({test_df['has_eda'].sum()} complete, {(~test_df['has_eda']).sum()} EDA-masked)")

    print("\n" + "=" * 78)
    print("COMPONENT 1 — LightGBM SHAP (TreeExplainer, exact)")
    print("=" * 78)
    shap_df, _ = run_lgbm_shap(train_df, test_df)

    print("\n" + "=" * 78)
    print("COMPONENT 2 — Integrated Gradients on modality-invariant model")
    print("=" * 78)
    ig_df = run_invariant_ig(train_df, test_df)

    return shap_df, ig_df


if __name__ == "__main__":
    run_interpretability()
