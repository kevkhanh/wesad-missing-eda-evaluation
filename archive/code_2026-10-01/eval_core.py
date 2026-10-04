"""
Shared evaluation core for the Section 5.2 re-run.

Addresses peer_review_round3.md:
  C1  repeated CV reused one subject partition -> `subject_kfold` draws an
      explicit random permutation of SUBJECTS per repeat (version-independent;
      sklearn GroupKFold ignores row order and the installed sklearn differs
      from the requirements pin).
  C2  Platt protocol described but not implemented -> `fit_predict` fits
      Platt on inner-CV out-of-fold predictions over training subjects only.
  C4  two different feature tables -> `load_wesad_frozen` verifies the
      SHA-256 recorded in results/feature_table_manifest.json.
  M1  complete/reduced scored on disjoint halves -> `paired_tiers` scores
      EVERY test window under both conditions.
  §3  hr-resid kept HR as an input and residualised across pooled subjects
      -> `_hr_resid_transform` drops HR and residualises within subject.
  M2  gating confounded with model family -> `NonGatedMLP`, a capacity-
      matched MLP trained with the same EDA masking.

All models expose fit(train_df, seed) -> self and predict(df) -> p, where
`df` carries `has_eda` and NaN EDA features on reduced rows.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from feature_extraction import ALL_FEATURE_NAMES, ECG_FEATURE_NAMES, EDA_FEATURE_NAMES
from run_locked_comparison import CanonicalNet, LR, N_EPOCHS
from run_metrics import PlattCalibrator
import train_stage3_models as t3

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "results" / "feature_table_manifest.json"
HR_FEATURE = "hrv_mean_nn"
INNER_K = 4


# ---------------------------------------------------------------------------
# Frozen feature table (C4)
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_feature_table() -> dict:
    """Record the current cache's hash. Run once; every analysis then checks it."""
    path = Path(t3.DATASET_CSV)
    df = pd.read_csv(path)
    w = t3._restrict_to_wesad(df)
    manifest = {
        "path": str(path),
        "sha256": _sha256(path),
        "n_rows_total": int(len(df)),
        "n_rows_wesad_baseline_stress": int(len(w)),
        "n_subjects_wesad": int(w["subject_id"].nunique()),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Single source for every Section 5.2 number. Supersedes the "
                 "445-window cache used by the 2026-09-20 locked run, which "
                 "was deleted by run_full_audit.sh on 2026-09-30."),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_wesad_frozen() -> pd.DataFrame:
    """WESAD baseline-vs-stress windows from the frozen table (hash-checked)."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} missing; run `python eval_core.py --freeze` once.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    path = Path(t3.DATASET_CSV)
    digest = _sha256(path)
    if digest != manifest["sha256"]:
        raise RuntimeError(
            f"Feature table {path} has sha256 {digest[:12]}..., manifest expects "
            f"{manifest['sha256'][:12]}.... Refusing to mix feature tables.")
    df = t3._restrict_to_wesad(pd.read_csv(path)).copy()
    df["has_eda"] = True
    df["has_ecg"] = True
    df["row_id"] = np.arange(len(df))
    return df


# ---------------------------------------------------------------------------
# Subject partitions (C1)
# ---------------------------------------------------------------------------

def subject_kfold(subjects, k: int, seed: int) -> list[np.ndarray]:
    """Random partition of the unique subjects into k near-equal test groups."""
    uniq = np.array(sorted(set(subjects)))
    perm = np.random.default_rng(seed).permutation(uniq)
    return [np.sort(g) for g in np.array_split(perm, k)]


def loso(subjects) -> list[np.ndarray]:
    return [np.array([s]) for s in sorted(set(subjects))]


def split_by_subjects(df: pd.DataFrame, test_subjects) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = df["subject_id"].isin(set(test_subjects)).values
    return df[~m].copy(), df[m].copy()


# ---------------------------------------------------------------------------
# Paired tiers (M1)
# ---------------------------------------------------------------------------

def make_reduced(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[EDA_FEATURE_NAMES] = np.nan
    out["has_eda"] = False
    return out


def paired_tiers(test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every test window, once with EDA present and once with EDA missing."""
    complete = test_df.copy()
    complete["has_eda"] = True
    return {"complete": complete, "reduced": make_reduced(test_df)}


# ---------------------------------------------------------------------------
# LightGBM family
# ---------------------------------------------------------------------------

class LGBMModel:
    """LightGBM baselines.

    features: input columns.
    train_nan_p: if > 0, EDA set to NaN on a random fraction of training rows
        (baseline ii).
    test_fill: how missing EDA is delivered at predict time:
        "nan"  -> native LightGBM missing handling,
        "zero" -> fillna(0) (the original straw-man encoding),
        "mean" -> training-mean imputation (baseline iv).
    """

    def __init__(self, features, train_nan_p: float = 0.0, test_fill: str = "nan",
                 n_estimators: int = 50, max_depth: int = 3,
                 learning_rate: float = 0.1, min_child_samples: int = 20):
        self.features = list(features)
        self.train_nan_p = train_nan_p
        self.test_fill = test_fill
        self.hp = dict(n_estimators=n_estimators, max_depth=max_depth,
                       learning_rate=learning_rate, min_child_samples=min_child_samples)

    def fit(self, train_df: pd.DataFrame, seed: int):
        import lightgbm as lgb
        X = train_df[self.features].copy()
        if self.train_nan_p > 0:
            rng = np.random.default_rng(seed)
            m = rng.random(len(X)) < self.train_nan_p
            eda_cols = [c for c in self.features if c in EDA_FEATURE_NAMES]
            X.loc[m, eda_cols] = np.nan
        self.eda_means_ = train_df[EDA_FEATURE_NAMES].mean()
        # n_jobs=1: parallelism lives at the fold-task level (joblib process
        # pool); per-fit OpenMP threads would oversubscribe the cores.
        self.model_ = lgb.LGBMClassifier(random_state=seed, verbosity=-1, n_jobs=1, **self.hp)
        self.model_.fit(X, train_df["label"].values)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].copy()
        eda_cols = [c for c in self.features if c in EDA_FEATURE_NAMES]
        if eda_cols and self.test_fill == "zero":
            X[eda_cols] = X[eda_cols].fillna(0.0)
        elif eda_cols and self.test_fill == "mean":
            X[eda_cols] = X[eda_cols].fillna(self.eda_means_[eda_cols])
        return self.model_.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Neural models: gated (CanonicalNet) and capacity-matched non-gated MLP
# ---------------------------------------------------------------------------

def _hr_resid_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Within-subject residualisation of non-HR ECG features on HR; HR dropped.

    For each subject separately: feature <- residual of OLS(feature ~ HR)
    plus that subject's feature mean, so only the within-subject covariation
    with HR is removed (between-subject levels are kept). HR itself is set
    to 0 so the model cannot use it. Label-free but transductive: a test
    subject's own windows (both conditions) define its regression.
    """
    out = df.copy()
    for s, idx in out.groupby("subject_id").groups.items():
        hr = out.loc[idx, HR_FEATURE].astype(float).values
        hr_c = hr - np.nanmean(hr)
        denom = float(np.nansum(hr_c ** 2))
        for col in ECG_FEATURE_NAMES:
            if col == HR_FEATURE:
                continue
            y = out.loc[idx, col].astype(float).values
            ok = np.isfinite(y) & np.isfinite(hr_c)
            if ok.sum() < 3 or denom <= 0:
                continue
            beta = float(np.sum(hr_c[ok] * (y[ok] - y[ok].mean())) / np.sum(hr_c[ok] ** 2))
            out.loc[idx, col] = y - beta * hr_c
    out[HR_FEATURE] = 0.0
    return out


def _robust_stats(train_df: pd.DataFrame) -> dict:
    stats = {}
    for feat in ALL_FEATURE_NAMES:
        v = train_df[feat].astype(float)
        iqr = v.quantile(0.75) - v.quantile(0.25)
        stats[feat] = (v.median(), iqr if np.isfinite(iqr) and iqr > 1e-9 else 1.0)
    return stats


def _standardize(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    for feat, (med, scale) in stats.items():
        out[feat] = (out[feat].astype(float) - med) / scale
    return out


class _NeuralBase:
    def __init__(self, hr_removed: bool = False, hr_resid: bool = False,
                 epochs: int = N_EPOCHS, lr: float = LR, weight_decay: float = 0.0):
        self.hr_removed = hr_removed
        self.hr_resid = hr_resid
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.hr_resid:
            df = _hr_resid_transform(df)
        out = _standardize(df, self.stats_)
        if self.hr_removed or self.hr_resid:
            out[HR_FEATURE] = 0.0
        return out

    def _tensors(self, df: pd.DataFrame):
        return (
            torch.tensor(df[ECG_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32),
            torch.tensor(df[EDA_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32),
            torch.tensor(df["has_eda"].values.astype(bool)),
            torch.tensor(df["label"].values, dtype=torch.float32),
        )

    def fit(self, train_df: pd.DataFrame, seed: int):
        base = _hr_resid_transform(train_df) if self.hr_resid else train_df
        self.stats_ = _robust_stats(base)
        tr = self._prep(train_df)
        ecg, eda, m, y = self._tensors(tr)
        torch.manual_seed(seed)
        self.net_ = self._build()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        torch.manual_seed(seed + 1)
        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss_fn(self._forward(ecg, eda, m, train=True), y).backward()
            opt.step()
        self.net_.eval()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ecg, eda, m, _ = self._tensors(self._prep(df))
        with torch.no_grad():
            return torch.sigmoid(self._forward(ecg, eda, m, train=False)).numpy()


class GatedModel(_NeuralBase):
    """CanonicalNet (the paper's modality-gated architecture)."""

    def __init__(self, dropout_p: float = 0.5, drop_eda: bool = True,
                 embed_dim: int = 16, **kw):
        super().__init__(**kw)
        self.dropout_p = dropout_p
        self.drop_eda = drop_eda
        self.embed_dim = embed_dim

    def _build(self):
        return CanonicalNet(embed_dim=self.embed_dim, dropout_p=self.dropout_p,
                            drop_eda=self.drop_eda, drop_ecg=False)

    def _forward(self, ecg, eda, m, train: bool):
        return self.net_(ecg, eda, m, force_no_dropout=not train)


class _MLP(nn.Module):
    """~Capacity-matched to CanonicalNet (two 13->16->16 branches + head)."""

    def __init__(self, n_in: int, hidden: int = 32):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, 16), nn.ReLU(),
            nn.Linear(16, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


class NonGatedMLP(_NeuralBase):
    """Plain MLP on [ECG, EDA zero-imputed, EDA-missing indicator].

    Trained with the SAME EDA masking as the gated model's modality dropout
    (each training row's EDA is zeroed and its indicator cleared with prob p),
    so a gated-vs-MLP contrast isolates gating from model family (M2).
    """

    def __init__(self, dropout_p: float = 0.5, hidden: int = 32, **kw):
        super().__init__(**kw)
        self.dropout_p = dropout_p
        self.hidden = hidden

    def _build(self):
        return _MLP(len(ECG_FEATURE_NAMES) + len(EDA_FEATURE_NAMES) + 1, hidden=self.hidden)

    def _forward(self, ecg, eda, m, train: bool):
        gate = m.float().unsqueeze(1)
        if train and self.dropout_p > 0:
            gate = gate * (torch.rand(gate.shape[0], 1) > self.dropout_p).float()
        return self.net_(torch.cat([ecg, eda * gate, gate], dim=1))


# ---------------------------------------------------------------------------
# Nested hyperparameter tuning
# ---------------------------------------------------------------------------

def _inner_oof(factory, train_df: pd.DataFrame, seed: int):
    """Inner subject-level CV over training subjects -> OOF predictions on
    paired tiers. Shared by tuning and by Platt fitting."""
    oof = {"complete": [], "reduced": []}
    ys = []
    for k, inner_test in enumerate(
            subject_kfold(train_df["subject_id"], INNER_K, seed=seed + 7919)):
        itr, ite = split_by_subjects(train_df, inner_test)
        if itr["label"].nunique() < 2:
            continue
        m = factory().fit(itr, seed=seed + 101 * (k + 1))
        tiers = paired_tiers(ite)
        oof["complete"].append(m.predict(tiers["complete"]))
        oof["reduced"].append(m.predict(tiers["reduced"]))
        ys.append(ite["label"].values)
    y = np.concatenate(ys)
    return y, {t: np.concatenate(v) for t, v in oof.items()}


class TunedModel:
    """Select hyperparameters by inner subject-level CV, then refit.

    Criterion: mean of complete- and reduced-condition OOF Brier (the
    deployed model must serve both). The chosen config's OOF predictions
    are kept in `oof_` so fit_predict fits Platt on them without a second
    inner loop. Tuning never sees the outer test subjects.
    """

    def __init__(self, grid: list, labels: list[dict]):
        self.grid = grid
        self.labels = labels

    def fit(self, train_df: pd.DataFrame, seed: int):
        best = None
        for factory, label in zip(self.grid, self.labels):
            y, oof = _inner_oof(factory, train_df, seed)
            score = 0.5 * (np.mean((oof["complete"] - y) ** 2)
                           + np.mean((oof["reduced"] - y) ** 2))
            if best is None or score < best[0]:
                best = (score, factory, label, y, oof)
        _, factory, self.chosen_, y, oof = best
        self.oof_ = (y, oof)
        self.model_ = factory().fit(train_df, seed=seed)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(df)


def _grid(make, **axes):
    import itertools
    keys = list(axes)
    combos = [dict(zip(keys, v)) for v in itertools.product(*axes.values())]
    return [lambda c=c: make(**c) for c in combos], combos


LGBM_GRID = dict(n_estimators=[50, 200], max_depth=[2, 4],
                 learning_rate=[0.05, 0.1], min_child_samples=[10, 30])
NN_GRID = dict(epochs=[100, 300], lr=[3e-3, 1e-2], weight_decay=[0.0, 1e-3])
# Width enters the tuning grid at roughly matched parameter counts:
# gated embed_dim 8/16/32 -> 273/801/2625 params; MLP hidden 8/25/84 -> 281/808/2637.
GATED_WIDTHS = [8, 16, 32]
MLP_WIDTHS = [8, 25, 84]


def _tuned(make, **axes):
    return lambda: TunedModel(*_grid(make, **axes))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ECG_NO_HR = [c for c in ECG_FEATURE_NAMES if c != HR_FEATURE]
ALL_NO_HR = [c for c in ALL_FEATURE_NAMES if c != HR_FEATURE]

# name -> (factory, platt_mode). platt_mode "shared": one Platt fit on
# complete-condition OOF predictions, applied at both tiers. "per_condition":
# separate Platt per tier (baseline iv's defining feature).
MODELS: dict[str, tuple[Callable[[], object], str]] = {
    "lgbm_zero_fill_at_test":     (lambda: LGBMModel(ALL_FEATURE_NAMES, test_fill="zero"), "shared"),
    "lgbm_nan_at_test":           (lambda: LGBMModel(ALL_FEATURE_NAMES, test_fill="nan"), "shared"),
    "lgbm_nan_training_p0.5":     (lambda: LGBMModel(ALL_FEATURE_NAMES, train_nan_p=0.5), "shared"),
    "lgbm_ecg_only_per_config":   (lambda: LGBMModel(ECG_FEATURE_NAMES), "shared"),
    "lgbm_mean_impute_recal":     (lambda: LGBMModel(ALL_FEATURE_NAMES, test_fill="mean"), "per_condition"),
    "hr_only":                    (lambda: LGBMModel([HR_FEATURE]), "shared"),
    "gated_dropout_p0.5":         (lambda: GatedModel(0.5, drop_eda=True), "shared"),
    "gated_no_dropout":           (lambda: GatedModel(0.5, drop_eda=False), "shared"),
    "mlp_nongated_dropout_p0.5":  (lambda: NonGatedMLP(0.5), "shared"),
    "gated_dropout_hr_removed":   (lambda: GatedModel(0.5, hr_removed=True), "shared"),
    "gated_dropout_hr_resid":     (lambda: GatedModel(0.5, hr_resid=True), "shared"),
    "lgbm_nan_training_hr_removed": (lambda: LGBMModel(ALL_NO_HR, train_nan_p=0.5), "shared"),
    "lgbm_ecg_only_hr_removed":   (lambda: LGBMModel(ECG_NO_HR), "shared"),
    # Nested-tuned variants of the four models the conclusion rests on.
    "tuned_gated_dropout_p0.5": (_tuned(lambda **c: GatedModel(0.5, **c),
                                        embed_dim=GATED_WIDTHS, **NN_GRID), "shared"),
    "tuned_mlp_nongated_dropout_p0.5": (_tuned(lambda **c: NonGatedMLP(0.5, **c),
                                               hidden=MLP_WIDTHS, **NN_GRID), "shared"),
    "tuned_lgbm_nan_training_p0.5": (_tuned(lambda **c: LGBMModel(ALL_FEATURE_NAMES, train_nan_p=0.5, **c),
                                            **LGBM_GRID), "shared"),
    "tuned_lgbm_ecg_only_per_config": (_tuned(lambda **c: LGBMModel(ECG_FEATURE_NAMES, **c),
                                              **LGBM_GRID), "shared"),
}

# Capacity sweep: gated vs non-gated MLP at matched parameter budgets
# (gated embed_dim d -> 2d^2+18d+1 params; MLP hidden h -> 31h+33 params).
CAPACITY_PAIRS = [(8, 8), (16, 25), (32, 84), (64, 300)]
for _d, _h in CAPACITY_PAIRS:
    MODELS[f"cap_gated_d{_d}"] = (lambda d=_d: GatedModel(0.5, embed_dim=d), "shared")
    MODELS[f"cap_mlp_h{_h}"] = (lambda h=_h: NonGatedMLP(0.5, hidden=h), "shared")


# ---------------------------------------------------------------------------
# Fit + OOF Platt + predict (C2)
# ---------------------------------------------------------------------------

def fit_predict(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                seed: int) -> pd.DataFrame:
    """Train `name` on train_df and score every test window under both tiers.

    Platt calibrators are fit on inner-CV out-of-fold predictions over the
    training subjects only. Returns one row per (window, tier) with raw,
    shared-Platt (`p_platt`; per-condition for baseline iv) and
    per-condition-Platt (`p_platt_pc`) probabilities.
    """
    factory, platt_mode = MODELS[name]

    model = factory().fit(train_df, seed=seed)
    if hasattr(model, "oof_"):
        # Tuned model: Platt on the chosen config's inner-CV OOF predictions.
        oof_y, oof = model.oof_
    else:
        oof_y, oof = _inner_oof(factory, train_df, seed)
    platt = {"complete": PlattCalibrator().fit(oof_y, oof["complete"])}
    platt_pc = {"complete": platt["complete"],
                "reduced": PlattCalibrator().fit(oof_y, oof["reduced"])}
    platt["reduced"] = platt_pc["reduced"] if platt_mode == "per_condition" else platt["complete"]

    rows = []
    for tier, tdf in paired_tiers(test_df).items():
        p_raw = model.predict(tdf)
        rows.append(pd.DataFrame({
            "model": name,
            "tier": tier,
            "row_id": tdf["row_id"].values,
            "subject_id": tdf["subject_id"].values,
            "y": tdf["label"].values.astype(int),
            "p_raw": p_raw,
            "p_platt": platt[tier].predict(p_raw),
            # Per-condition Platt for every model: a separate calibrator for
            # each sensor configuration, both fit on training-subject OOF.
            "p_platt_pc": platt_pc[tier].predict(p_raw),
        }))
    out = pd.concat(rows, ignore_index=True)
    if hasattr(model, "chosen_"):
        out["chosen_config"] = json.dumps(model.chosen_, sort_keys=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true",
                    help="record the current feature table's SHA-256")
    args = ap.parse_args()
    if args.freeze:
        print(json.dumps(freeze_feature_table(), indent=2))
    else:
        df = load_wesad_frozen()
        print(f"frozen table OK: {len(df)} WESAD windows, {df.subject_id.nunique()} subjects")
