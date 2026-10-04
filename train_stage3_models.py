"""
Stage 3 models: LightGBM baseline vs modality-invariant PyTorch net.

Interface reconstructed from robustness/__pycache__/train_stage3_models.cpython-313.pyc
to match what run_ablations.py, run_interpretability.py, and
run_locked_comparison.py import. New callers (audit/*.py) should use these
same helpers rather than reimplementing standardization or masking.

Locking / reproducibility posture:
  * RNG_SEED = 42 at module load
  * torch model init is deferred to callers via per-fold model_seed (see
    run_locked_comparison.py) — do not set torch.manual_seed here
  * LightGBM uses random_state=RNG_SEED and verbosity=-1

The invariant net is ModalityInvariantNet — two branches (ECG, EDA), summed
after per-branch embeddings, with training-time random EDA gating (Bernoulli
p=0.5) and inference-time hard gating from the has_eda column. This is the
model whose H-B1 headline ECE ratio is 1.61x in the outline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

from feature_extraction import (
    ECG_FEATURE_NAMES,
    EDA_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
)

RNG_SEED = 42

_HERE = Path(__file__).resolve().parent
DATASET_CSV = _HERE.parent / "stage3_hb1_dataset.csv"

np.random.seed(RNG_SEED)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _subject_of(series: pd.Series) -> pd.Series:
    """Extract a subject group ID from the subject_id column.

    subject_id is a string like "WESAD_S2" or "MITBIH_P201_202"; this is
    already the group identity we want, so we return it as-is. Kept as a
    function (rather than inlined) so we can enforce the 201/202 sharing
    rule here if the raw column ever loses it.
    """
    return series.astype(str)


def _load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the cached feature table, or build one from real/synthetic sources.

    If DATASET_CSV exists we trust it. Otherwise we call the real_data_loaders
    (which fall back to synthetic if the raw datasets aren't present),
    extract features, cache to CSV, and return.

    Path defaults to the module-level DATASET_CSV *at call time* (not at
    def-time), so tests and callers can monkeypatch DATASET_CSV and have it
    take effect.
    """
    path = Path(path) if path is not None else DATASET_CSV
    if path.exists():
        return pd.read_csv(path)

    from real_data_loaders import load_all_windows
    from feature_extraction import extract_batch

    windows = load_all_windows()
    df = extract_batch(windows)
    # Restrict to the labels the H-B1 pipeline expects: WESAD baseline/stress.
    # MIT-BIH windows are kept in the CSV so the audit modules can build C1/C4;
    # H-B1 callers filter with _restrict_to_wesad if they need only WESAD.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except OSError:
        pass
    return df


def _restrict_to_wesad(df: pd.DataFrame) -> pd.DataFrame:
    """The H-B1 pipeline is WESAD-internal (stress vs baseline).

    train_baseline / train_invariant_model expect a numeric label column;
    this helper both filters and re-encodes ('baseline' -> 0, 'stress' -> 1).
    """
    w = df[df["dataset"] == "WESAD"].copy()
    if w["label"].dtype == object:
        w = w[w["label"].isin(["baseline", "stress"])]
        w["label"] = (w["label"] == "stress").astype(int)
    return w.reset_index(drop=True)


def _apply_simulated_eda_mask(df: pd.DataFrame,
                              fraction: float = 0.5,
                              seed: int = RNG_SEED) -> pd.DataFrame:
    """Zero EDA features for a stratified `fraction` of rows.

    Adds has_eda=False to masked rows so downstream code can gate them.
    Stratifying by label keeps class balance intact within each tier.
    """
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    if "has_eda" not in df.columns:
        df["has_eda"] = True
    mask = np.zeros(len(df), dtype=bool)
    for lbl in df["label"].unique():
        idx = df.index[df["label"] == lbl].to_numpy()
        n = int(round(len(idx) * fraction))
        if n and len(idx):
            chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
            mask[chosen] = True
    df.loc[mask, EDA_FEATURE_NAMES] = float("nan")
    df.loc[mask, "has_eda"] = False
    return df


# ---------------------------------------------------------------------------
# Standardization & tensors
# ---------------------------------------------------------------------------

def _standardize_pair(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit robust (median / IQR) stats on train, apply to both.

    NaNs are preserved so the model / gating logic can decide how to handle
    them (LightGBM routes natively; the invariant net receives fillna(0)
    from _to_tensors callers).
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    for feat in ALL_FEATURE_NAMES:
        vals = train_df[feat].astype(float)
        med = vals.median()
        iqr = vals.quantile(0.75) - vals.quantile(0.25)
        scale = iqr if iqr and np.isfinite(iqr) and iqr > 1e-9 else 1.0
        train_df[feat] = (vals - med) / scale
        test_df[feat] = (test_df[feat].astype(float) - med) / scale
    return train_df, test_df


def _prep_tensors(df: pd.DataFrame):
    """Row-oriented DataFrame -> (ecg_x, eda_x, eda_mask, y) tensors."""
    ecg_x = torch.tensor(df[ECG_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32)
    eda_x = torch.tensor(df[EDA_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32)
    eda_m = torch.tensor(df["has_eda"].values, dtype=torch.bool)
    y = torch.tensor(df["label"].values, dtype=torch.float32)
    return ecg_x, eda_x, eda_m, y


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    """Standard equal-width binning ECE.

    Matches the outline's Section 5.2 usage: 10 bins, absolute gap between
    per-bin accuracy and confidence, weighted by bin population.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    if len(y_true) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Baseline: LightGBM
# ---------------------------------------------------------------------------

def train_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """LightGBM baseline trained on complete-modality training rows only.

    Matches run_locked_comparison.run_locked_lgbm's config:
    n_estimators=50, max_depth=3, random_state=42. LightGBM handles NaNs
    natively so we do not fillna the test-side inputs; the ECE / AUROC
    reported below is for information — callers usually recompute per-tier.
    """
    import lightgbm as lgb

    train_complete = train_df[train_df.get("has_eda", True) == True]  # noqa: E712
    if len(train_complete) < 4:
        train_complete = train_df

    model = lgb.LGBMClassifier(
        n_estimators=50, max_depth=3,
        random_state=RNG_SEED, verbosity=-1,
    )
    model.fit(train_complete[ALL_FEATURE_NAMES], train_complete["label"])

    probs = model.predict_proba(test_df[ALL_FEATURE_NAMES])[:, 1]
    metrics = {
        "auroc": float(roc_auc_score(test_df["label"], probs))
                 if test_df["label"].nunique() > 1 else float("nan"),
        "ece": expected_calibration_error(test_df["label"].values, probs),
        "n_test": len(test_df),
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Modality-invariant model
# ---------------------------------------------------------------------------

class ModalityInvariantNet(nn.Module):
    """Two-branch network with training-time EDA dropout + inference-time gating.

    forward signature (kept for backward compat with run_interpretability.py):
        forward(ecg_x, eda_x, eda_mask, force_no_dropout=False)

    ECG branch is always active. EDA branch is:
      * during training and eda_mask=True: kept with prob (1 - dropout_p)
        (random Bernoulli zeroing across the batch)
      * during training and eda_mask=False: gated off
      * at inference (or force_no_dropout=True): gated by eda_mask
    """
    def __init__(self, embed_dim: int = 16, dropout_p: float = 0.5):
        super().__init__()
        self.dropout_p = dropout_p
        self.ecg_branch = nn.Sequential(
            nn.Linear(len(ECG_FEATURE_NAMES), embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.eda_branch = nn.Sequential(
            nn.Linear(len(EDA_FEATURE_NAMES), embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, ecg_x, eda_x, eda_mask, force_no_dropout: bool = False):
        B = ecg_x.shape[0]
        eda_gate = eda_mask.float().unsqueeze(1)
        if self.training and not force_no_dropout:
            eda_gate = eda_gate * (torch.rand(B, 1) > self.dropout_p).float()
        combined = self.ecg_branch(ecg_x) + eda_gate * self.eda_branch(eda_x)
        return self.head(combined).squeeze(-1)


def train_invariant_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    embed_dim: int = 16,
    dropout_p: float = 0.5,
    n_epochs: int = 100,
    lr: float = 1e-2,
    model_seed: int | None = None,
):
    """Fit ModalityInvariantNet on standardized features.

    If model_seed is provided, torch.manual_seed is reset immediately before
    weight initialization — this is what run_locked_comparison relies on to
    make the paired ratio comparison meaningful. Do NOT rely on module-level
    RNG_SEED for parameter init; callers must set it explicitly.
    """
    train_s, test_s = _standardize_pair(train_df, test_df)

    if model_seed is not None:
        torch.manual_seed(model_seed)
    model = ModalityInvariantNet(embed_dim=embed_dim, dropout_p=dropout_p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ecg_x, eda_x, eda_m, y_tr = _prep_tensors(train_s)
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss_fn(model(ecg_x, eda_x, eda_m), y_tr).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        t_ecg, t_eda, t_dm, t_y = _prep_tensors(test_s)
        probs = torch.sigmoid(
            model(t_ecg, t_eda, t_dm, force_no_dropout=True)
        ).numpy()
    metrics = {
        "auroc": float(roc_auc_score(test_s["label"], probs))
                 if test_s["label"].nunique() > 1 else float("nan"),
        "ece": expected_calibration_error(test_s["label"].values, probs),
        "n_test": len(test_s),
    }
    return model, metrics


# ---------------------------------------------------------------------------
# load_and_split (WESAD-internal, H-B1 setup)
# ---------------------------------------------------------------------------

def load_and_split(
    simulated_mask_fraction: float = 0.5,
    test_size: float = 0.3,
    seed: int = RNG_SEED,
):
    """Build the (train_df, test_df) pair used by H-B1-flavored scripts.

    WESAD-only, simulated EDA masking applied ONLY to the test partition
    (matches the "test-time missing modality" evaluation regime — training
    always sees complete modality). Uses subject-level GroupKFold-style
    holdout to preserve subject-level CV semantics; falls back to random
    split when only one subject is available.
    """
    full = _load_dataset()
    wesad = _restrict_to_wesad(full)
    if wesad.empty:
        raise RuntimeError(
            "No WESAD windows in cached dataset; check real_data_loaders "
            "and the synthetic-fallback warning above."
        )
    subjects = _subject_of(wesad["subject_id"]).values
    uniq = np.unique(subjects)

    if len(uniq) < 2:
        train_df, test_df = train_test_split(
            wesad, test_size=test_size, random_state=seed,
            stratify=wesad["label"],
        )
    else:
        rng = np.random.default_rng(seed)
        n_test_subj = max(1, int(round(len(uniq) * test_size)))
        test_subs = set(rng.choice(uniq, size=n_test_subj, replace=False))
        test_df = wesad[np.isin(subjects, list(test_subs))].copy()
        train_df = wesad[~np.isin(subjects, list(test_subs))].copy()

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df["has_eda"] = True
    test_df = _apply_simulated_eda_mask(
        test_df, fraction=simulated_mask_fraction, seed=seed,
    )
    return train_df, test_df
