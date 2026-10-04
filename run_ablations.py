"""
Section 5.2b ablations: per-modality dropout isolation and dropout-rate sweep.

Ablation 1 — Per-modality dropout isolation:
    Trains four AblationNet variants differing only in which branch is
    eligible for training-time dropout, then evaluates ECE when the matched
    modality is missing at test time.  The extracted feature set contains ECG
    and EDA only (WESAD does not include PPG or respiration in the current
    extraction pipeline), so the four variants are:
      A. EDA-only droppable  — matches the main H-B1 model; the comparison point
      B. ECG-only droppable  — ECG branch zeroed at train time; ECG simulated-
                                missing at test (symmetric to EDA masking)
      C. Joint               — both branches droppable; evaluated on EDA-missing
                                test tier (consistent with main result)
      D. No dropout          — neither branch droppable; regularization control

    Key question: does calibration improvement come from general dropout
    regularization, or specifically from training for the modality that ends
    up missing at test time?

Ablation 2 — Dropout-rate sweep:
    Retrains the EDA-only dropout model at p in {0.2, 0.4, 0.6} using the
    same 5-fold subject-level GroupKFold.  Reports ECE_complete, ECE_reduced,
    and degradation ratio (ECE_reduced / ECE_complete) per rate — the data
    behind the "benefit scales with dropout rate" claim.

Both ablations use the existing subject-level CV pipeline and WESAD features;
no new data collection.  AblationNet extends ModalityInvariantNet to gate both
branches symmetrically, enabling the ECG-only and joint dropout conditions.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from feature_extraction import ECG_FEATURE_NAMES, EDA_FEATURE_NAMES
from train_stage3_models import (
    _apply_simulated_eda_mask,
    _load_dataset,
    _standardize_pair,
    _subject_of,
    expected_calibration_error,
)

RNG_SEED = 42
MASK_FRACTION = 0.5
N_FOLDS = 5
N_EPOCHS = 100
LR = 1e-2

torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ---------------------------------------------------------------------------
# ECG masking — symmetric to _apply_simulated_eda_mask
# ---------------------------------------------------------------------------

def _apply_simulated_ecg_mask(df, fraction=MASK_FRACTION, seed=RNG_SEED):
    """Zero ECG features for `fraction` of rows, stratified by label."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    mask = np.zeros(len(df), dtype=bool)
    for lbl in df["label"].unique():
        idx = df.index[df["label"] == lbl]
        n = int(round(len(idx) * fraction))
        mask[df.index.isin(rng.choice(idx, size=n, replace=False))] = True
    df["has_ecg"] = ~mask
    df.loc[mask, ECG_FEATURE_NAMES] = float("nan")
    return df


# ---------------------------------------------------------------------------
# Extended model: configurable per-branch training-time dropout
# ---------------------------------------------------------------------------

class AblationNet(nn.Module):
    """
    Two-branch model with per-modality dropout control.
    drop_ecg / drop_eda govern whether each branch is eligible for
    training-time random zeroing.  Both branches are always gated by their
    present_mask at inference time, making ECG gating symmetric with EDA.
    """

    def __init__(self, embed_dim=16, dropout_p=0.5, drop_ecg=False, drop_eda=True):
        super().__init__()
        self.dropout_p = dropout_p
        self.drop_ecg = drop_ecg
        self.drop_eda = drop_eda
        n_ecg, n_eda = len(ECG_FEATURE_NAMES), len(EDA_FEATURE_NAMES)
        self.ecg_branch = nn.Sequential(
            nn.Linear(n_ecg, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.eda_branch = nn.Sequential(
            nn.Linear(n_eda, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, ecg_x, eda_x, ecg_mask, eda_mask, force_no_dropout=False):
        B = ecg_x.shape[0]
        ecg_gate = ecg_mask.float().unsqueeze(1)
        eda_gate = eda_mask.float().unsqueeze(1)
        if self.training and not force_no_dropout:
            if self.drop_ecg:
                ecg_gate = ecg_gate * (torch.rand(B, 1) > self.dropout_p).float()
            if self.drop_eda:
                eda_gate = eda_gate * (torch.rand(B, 1) > self.dropout_p).float()
        combined = ecg_gate * self.ecg_branch(ecg_x) + eda_gate * self.eda_branch(eda_x)
        return self.head(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# Tensor prep and train/eval
# ---------------------------------------------------------------------------

def _to_tensors(df):
    """Convert a DataFrame row to model input tensors.
    has_ecg defaults to all-True if the column is absent (ECG always present
    for EDA-masking variants where no ECG masking was applied)."""
    has_ecg = (
        df["has_ecg"].values if "has_ecg" in df.columns
        else np.ones(len(df), dtype=bool)
    )
    return (
        torch.tensor(df[ECG_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32),
        torch.tensor(df[EDA_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32),
        torch.tensor(has_ecg, dtype=torch.bool),
        torch.tensor(df["has_eda"].values, dtype=torch.bool),
        torch.tensor(df["label"].values, dtype=torch.float32),
    )


def _train_and_eval(train_df, test_df, *, dropout_p, drop_ecg, drop_eda, missing_col):
    """
    Train AblationNet and return per-tier ECE/AUROC.
    missing_col: boolean column in test_df where True = complete-modality window.
    """
    train_df = train_df.copy()
    if "has_ecg" not in train_df.columns:
        train_df["has_ecg"] = True

    train_s, test_s = _standardize_pair(train_df, test_df)

    model = AblationNet(dropout_p=dropout_p, drop_ecg=drop_ecg, drop_eda=drop_eda)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    ecg_x, eda_x, ecg_m, eda_m, y_tr = _to_tensors(train_s)
    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        loss_fn(model(ecg_x, eda_x, ecg_m, eda_m), y_tr).backward()
        opt.step()

    model.eval()
    results = {}
    with torch.no_grad():
        for tier, tier_df in [
            ("complete", test_s[test_s[missing_col]]),
            ("reduced",  test_s[~test_s[missing_col]]),
        ]:
            if tier_df["label"].nunique() < 2 or len(tier_df) < 2:
                results[tier] = None
                continue
            t_ecg, t_eda, t_em, t_dm, t_y = _to_tensors(tier_df)
            probs = torch.sigmoid(
                model(t_ecg, t_eda, t_em, t_dm, force_no_dropout=True)
            ).numpy()
            results[tier] = {
                "auroc": float(roc_auc_score(t_y.numpy(), probs)),
                "ece": expected_calibration_error(t_y.numpy(), probs),
                "n": len(tier_df),
            }
    return results


def _aggregate_folds(fold_list):
    """Mean ± std of ECE and AUROC across folds, per tier."""
    agg = {}
    for tier in ("complete", "reduced"):
        vals = [f[tier] for f in fold_list if f.get(tier)]
        if not vals:
            agg[tier] = None
            continue
        agg[tier] = {
            "ece_mean":   float(np.mean([v["ece"] for v in vals])),
            "ece_std":    float(np.std([v["ece"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
            "auroc_mean": float(np.mean([v["auroc"] for v in vals])),
            "n_folds":    len(vals),
        }
    return agg


# ---------------------------------------------------------------------------
# Ablation 1: per-modality dropout isolation
# ---------------------------------------------------------------------------

def run_ablation1_per_modality(n_folds=N_FOLDS):
    """
    Four AblationNet variants; each evaluated on the modality that was
    made eligible for training-time dropout.

    Interpretation guide for the table:
      eda_only vs no_dropout   — is the ECE benefit from EDA-specific training
                                  or from dropout regularization in general?
      eda_only vs ecg_only     — does it matter that you trained for the *right*
                                  missing modality at test time?
      eda_only vs joint        — cost/benefit of training for both modalities.
    """
    df = _load_dataset()
    subjects = _subject_of(df["subject_id"]).values
    n_unique = len(np.unique(subjects))
    if n_unique < n_folds:
        n_folds = max(2, n_unique)
        print(f"[note] {n_unique} subjects — reducing n_folds to {n_folds}")

    variants = {
        "eda_only_dropout": dict(dropout_p=0.5, drop_ecg=False, drop_eda=True),
        "ecg_only_dropout": dict(dropout_p=0.5, drop_ecg=True,  drop_eda=False),
        "joint_dropout":    dict(dropout_p=0.5, drop_ecg=True,  drop_eda=True),
        "no_dropout":       dict(dropout_p=0.5, drop_ecg=False, drop_eda=False),
    }

    fold_results = {k: [] for k in variants}
    gkf = GroupKFold(n_splits=n_folds)

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=subjects)):
        train_df = df.iloc[train_idx].copy()
        raw_test = df.iloc[test_idx].copy()
        if train_df["label"].nunique() < 2 or raw_test["label"].nunique() < 2:
            print(f"  fold {fold_i}: skipped (single-class)")
            continue

        test_eda = _apply_simulated_eda_mask(raw_test, seed=RNG_SEED + fold_i)
        test_ecg = _apply_simulated_ecg_mask(raw_test, seed=RNG_SEED + fold_i)

        for name, params in variants.items():
            if name == "ecg_only_dropout":
                test_df, missing_col = test_ecg, "has_ecg"
            else:
                test_df, missing_col = test_eda, "has_eda"

            res = _train_and_eval(train_df, test_df, missing_col=missing_col, **params)
            fold_results[name].append(res)

            c_ece = f"{res['complete']['ece']:.4f}" if res.get("complete") else "N/A"
            r_ece = f"{res['reduced']['ece']:.4f}"  if res.get("reduced")  else "N/A"
            print(f"  fold {fold_i}  {name:<22}  complete ECE={c_ece}  reduced ECE={r_ece}")

    print("\n--- Ablation 1: Per-modality dropout isolation ---")
    print(f"  {'Variant':<22}  {'ECE_complete':>14}  {'ECE_reduced':>14}  {'ratio':>7}  folds")
    summary = {}
    for name in variants:
        agg = _aggregate_folds(fold_results[name])
        c, r = agg.get("complete"), agg.get("reduced")
        if c and r:
            ratio = r["ece_mean"] / c["ece_mean"] if c["ece_mean"] > 0 else float("nan")
            print(f"  {name:<22}  {c['ece_mean']:.4f} ± {c['ece_std']:.4f}    "
                  f"{r['ece_mean']:.4f} ± {r['ece_std']:.4f}    {ratio:.2f}x   {c['n_folds']}")
        else:
            print(f"  {name:<22}  insufficient data")
        summary[name] = agg

    return summary


# ---------------------------------------------------------------------------
# Ablation 2: dropout-rate sweep (EDA-only dropout, matching main result)
# ---------------------------------------------------------------------------

def run_ablation2_rate_sweep(rates=(0.2, 0.4, 0.6), n_folds=N_FOLDS):
    """
    Retrains the EDA-only dropout model at each rate and reports the ECE
    degradation ratio — the curve behind "benefit scales with dropout rate."
    The main result uses p=0.5 (ModalityInvariantNet default); that point
    is not re-run here to avoid duplication, but the Ablation 1 eda_only
    result at p=0.5 serves as the anchor for comparison.
    """
    df = _load_dataset()
    subjects = _subject_of(df["subject_id"]).values
    n_unique = len(np.unique(subjects))
    if n_unique < n_folds:
        n_folds = max(2, n_unique)
        print(f"[note] {n_unique} subjects — reducing n_folds to {n_folds}")

    rate_folds = {r: [] for r in rates}
    gkf = GroupKFold(n_splits=n_folds)

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=subjects)):
        train_df = df.iloc[train_idx].copy()
        test_df = _apply_simulated_eda_mask(df.iloc[test_idx].copy(), seed=RNG_SEED + fold_i)
        if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
            print(f"  fold {fold_i}: skipped (single-class)")
            continue

        for rate in rates:
            res = _train_and_eval(
                train_df, test_df,
                dropout_p=rate, drop_ecg=False, drop_eda=True, missing_col="has_eda",
            )
            rate_folds[rate].append(res)
            c_ece = f"{res['complete']['ece']:.4f}" if res.get("complete") else "N/A"
            r_ece = f"{res['reduced']['ece']:.4f}"  if res.get("reduced")  else "N/A"
            print(f"  fold {fold_i}  p={rate:.1f}  complete ECE={c_ece}  reduced ECE={r_ece}")

    print("\n--- Ablation 2: Dropout-rate sweep (EDA-only dropout) ---")
    print(f"  {'Rate':>5}  {'ECE_complete':>14}  {'ECE_reduced':>14}  {'ratio':>7}  folds")
    sweep_results = {}
    for rate in rates:
        agg = _aggregate_folds(rate_folds[rate])
        c, r = agg.get("complete"), agg.get("reduced")
        if c and r:
            ratio = r["ece_mean"] / c["ece_mean"] if c["ece_mean"] > 0 else float("nan")
            print(f"  {rate:.1f}      {c['ece_mean']:.4f} ± {c['ece_std']:.4f}    "
                  f"{r['ece_mean']:.4f} ± {r['ece_std']:.4f}    {ratio:.2f}x   {c['n_folds']}")
        else:
            print(f"  {rate:.1f}      insufficient data")
        sweep_results[rate] = agg

    return sweep_results


if __name__ == "__main__":
    print("=" * 78)
    print("ABLATION 1 — Per-modality dropout isolation")
    print("=" * 78)
    run_ablation1_per_modality()

    print("\n" + "=" * 78)
    print("ABLATION 2 — Dropout-rate sweep")
    print("=" * 78)
    run_ablation2_rate_sweep()
