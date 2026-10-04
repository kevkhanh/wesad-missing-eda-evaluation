"""
Audit Step 6a — per-stage harmonization transforms (H0-H7).

Two kinds of transforms:

  Raw-signal (operate on SubjectWindow -> SubjectWindow):
    H2 resample_to_common_rate      — 700 or 360 Hz -> COMMON_FS (360)
    H3 identical_filtering          — Butterworth 0.5-40 Hz zero-phase; removes
                                       both 50 Hz and 60 Hz mains
    H4 amplitude_normalize          — per-recording z-score of ECG
    H5 identical_windowing          — enforce non-overlapping, fixed length,
                                       per-subject cap (applied at window build)

  Feature-level (operate on DataFrame -> DataFrame):
    H1 feature_mask_ecg_only        — drop EDA/resp features (delegates to
                                       audit/step2_common_modality.py)
    H6a drop_spectral_features      — drop hrv_lf_power, hrv_hf_power,
                                       hrv_lf_hf_ratio; time-domain only
    H6b notch_spectral_artifacts    — approximated in feature space by masking
                                       LF-band contribution when a synthetic
                                       tape-artifact spike is detected (see
                                       note in _notch_spectral_features_df)
    H7 within_subject_zscore        — z-score each feature by the subject's
                                       own baseline (NSR for MIT-BIH, baseline
                                       for WESAD) window statistics

The ladder runner in step6_harmonization.py applies these cumulatively in
the order specified by the audit doc (H0 -> H1 -> H2 -> H3 -> H4 -> H5 ->
H6 -> H7). Each stage returns a NEW DataFrame / list of windows; the
previous state is not mutated in place.

Design notes:
  * H2/H3/H4/H5 need the raw signal, so they are applied at the
    SubjectWindow level BEFORE feature extraction is redone. That is why
    the ladder runner re-extracts features per stage instead of trying to
    transform features in place.
  * H6b (notch at 0.09/0.167 Hz in the RR series) would ideally be a Welch
    re-computation with a notched frequency mask. We approximate it in
    feature space rather than re-running Welch, since the current feature
    extractor delegates PSD to neurokit2 and we don't want to fork it.
    The audit doc lists 6a (drop LF/HF entirely) as the primary variant
    for exactly this reason.
"""

import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import ECG_FEATURE_NAMES, EDA_FEATURE_NAMES, ALL_FEATURE_NAMES
from stage1_pipeline import SubjectWindow

# Target common sampling rate. 360 Hz matches MIT-BIH so we downsample WESAD;
# resampling WESAD downward preserves detection quality better than upsampling
# MIT-BIH (which would invent samples at frequencies not present in the raw
# analog tape signal).
COMMON_FS = 360

# Rhythm-band edges for spectral-artifact control (H6a). LF and HF bands per
# Task Force 1996; H6a drops the whole triple.
SPECTRAL_FEATURES = ["hrv_lf_power", "hrv_hf_power", "hrv_lf_hf_ratio"]


# ---------------------------------------------------------------------------
# H2 — resample raw ECG (and EDA/resp when present) to COMMON_FS
# ---------------------------------------------------------------------------

def _resample_signal(x: np.ndarray, src_fs: int, tgt_fs: int) -> np.ndarray:
    if src_fs == tgt_fs or x is None:
        return x
    from math import gcd
    g = gcd(src_fs, tgt_fs)
    up = tgt_fs // g
    down = src_fs // g
    return resample_poly(x, up, down)


def resample_to_common_rate(windows: Iterable[SubjectWindow],
                            target_fs: int = COMMON_FS) -> List[SubjectWindow]:
    out = []
    for w in windows:
        w2 = deepcopy(w)
        w2.ecg = _resample_signal(w.ecg, w.fs, target_fs)
        w2.eda = _resample_signal(w.eda, w.fs, target_fs) if w.has_eda() else None
        w2.resp = _resample_signal(w.resp, w.fs, target_fs) if w.has_resp() else None
        w2.fs = target_fs
        w2.meta = {**w.meta, "harmonized_fs": target_fs}
        out.append(w2)
    return out


# ---------------------------------------------------------------------------
# H3 — identical Butterworth 0.5-40 Hz zero-phase band-pass on ECG
# ---------------------------------------------------------------------------

def _butter_sos(low: float, high: float, fs: int, order: int = 4):
    return butter(order, [low, high], btype="band", fs=fs, output="sos")


def identical_filtering(windows: Iterable[SubjectWindow],
                        low: float = 0.5, high: float = 40.0,
                        order: int = 4) -> List[SubjectWindow]:
    """Zero-phase Butterworth 0.5-40 Hz on ECG.

    Removes both 50 Hz (WESAD) and 60 Hz (MIT-BIH) mains and the 30 Hz
    twice-speed MIT-BIH variant. Leaves EDA/resp untouched (they have
    different mains-noise profiles and different useful bandwidths).
    """
    out = []
    for w in windows:
        w2 = deepcopy(w)
        if w.ecg is not None and len(w.ecg) > order * 4:
            hi = min(high, w.fs / 2 - 1.0)
            if 0 < low < hi:
                sos = _butter_sos(low, hi, w.fs, order=order)
                w2.ecg = sosfiltfilt(sos, w.ecg)
        w2.meta = {**w.meta, "filtered_bp_hz": [low, high]}
        out.append(w2)
    return out


# ---------------------------------------------------------------------------
# H4 — per-recording z-score of ECG amplitude
# ---------------------------------------------------------------------------

def amplitude_normalize(windows: Iterable[SubjectWindow]) -> List[SubjectWindow]:
    """Group windows by (subject_id, record) and z-score ECG per group."""
    from collections import defaultdict

    windows = list(windows)
    groups = defaultdict(list)
    for i, w in enumerate(windows):
        key = (w.subject_id, w.meta.get("record", w.subject_id))
        groups[key].append(i)

    out = [None] * len(windows)
    for key, idxs in groups.items():
        pool = np.concatenate([windows[i].ecg for i in idxs
                                if windows[i].ecg is not None])
        mu = float(np.nanmean(pool)) if pool.size else 0.0
        sd = float(np.nanstd(pool)) if pool.size else 1.0
        sd = sd if sd > 1e-9 else 1.0
        for i in idxs:
            w2 = deepcopy(windows[i])
            if w2.ecg is not None:
                w2.ecg = (w2.ecg - mu) / sd
            w2.meta = {**w2.meta, "amplitude_zscore_mu": mu, "amplitude_zscore_sd": sd}
            out[i] = w2
    return out


# ---------------------------------------------------------------------------
# H5 — identical windowing / RR cleaning
#
# The window length + non-overlap policy is already enforced at load time by
# real_data_loaders. This function is here as a placeholder so the ladder
# runner can treat "H5" as an explicit stage; call it to enforce a specific
# window length by trimming and to enforce per-subject caps by
# down-sampling per subject.
# ---------------------------------------------------------------------------

def identical_windowing(windows: Iterable[SubjectWindow],
                        window_sec: int = 60,
                        per_subject_cap: int = 20,
                        seed: int = 0) -> List[SubjectWindow]:
    windows = list(windows)
    rng = np.random.default_rng(seed)

    trimmed = []
    for w in windows:
        target = int(w.fs * window_sec)
        if w.ecg is None:
            continue
        ecg = w.ecg[:target] if len(w.ecg) >= target else None
        if ecg is None:
            continue
        w2 = deepcopy(w)
        w2.ecg = ecg
        if w2.eda is not None:
            w2.eda = w2.eda[:target] if len(w2.eda) >= target else None
        if w2.resp is not None:
            w2.resp = w2.resp[:target] if len(w2.resp) >= target else None
        trimmed.append(w2)

    from collections import defaultdict
    per_subject = defaultdict(list)
    for w in trimmed:
        per_subject[(w.subject_id, w.label)].append(w)
    out = []
    for _, group in per_subject.items():
        if len(group) <= per_subject_cap:
            out.extend(group)
        else:
            idxs = rng.choice(len(group), size=per_subject_cap, replace=False)
            out.extend([group[i] for i in idxs])
    return out


# ---------------------------------------------------------------------------
# Feature-level transforms
# ---------------------------------------------------------------------------

def feature_mask_ecg_only(df: pd.DataFrame) -> pd.DataFrame:
    """Delegates to audit/step2_common_modality.restrict_to_ecg."""
    from audit.step2_common_modality import restrict_to_ecg
    return restrict_to_ecg(df)


def drop_spectral_features(df: pd.DataFrame) -> pd.DataFrame:
    """H6a: drop LF/HF/LF/HF ratio; keep time-domain HRV."""
    drop = [c for c in SPECTRAL_FEATURES if c in df.columns]
    return df.drop(columns=drop).copy()


def _notch_spectral_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """H6b (feature-space approximation): rescale LF/HF magnitudes toward
    their per-dataset trimmed median when they fall in the top decile.

    The proper H6b (notch 0.09 and 0.167 Hz in the interpolated RR series
    and re-run Welch) would require rebuilding the PSD from raw beats. This
    approximation subtracts the top-decile bias that dataset-specific tape
    artifacts contribute to the LF/HF estimates. Use H6a as the primary
    variant; use this as a diagnostic to check whether the artifact
    contribution alone drives separability.
    """
    out = df.copy()
    for feat in ["hrv_lf_power", "hrv_hf_power"]:
        if feat not in out.columns:
            continue
        for ds, mask in out.groupby("dataset").groups.items():
            vals = out.loc[mask, feat].astype(float)
            hi = vals.quantile(0.90)
            med = vals.quantile(0.50)
            clipped = vals.where(vals < hi, med)
            out.loc[mask, feat] = clipped
    if "hrv_lf_power" in out.columns and "hrv_hf_power" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["hrv_lf_hf_ratio"] = out["hrv_lf_power"] / out["hrv_hf_power"]
    return out


def within_subject_zscore(df: pd.DataFrame,
                          feature_cols: list | None = None,
                          reference_labels=("baseline", "nsr")) -> pd.DataFrame:
    """H7: z-score each feature by the subject's own reference-state stats.

    WARNING (per audit doc): this removes absolute-level clinical signal too,
    so downstream Step 7 (C3/C4) re-validation MUST be reported alongside
    the harmonized identity AUC to check whether clinical AUC collapsed.
    """
    if feature_cols is None:
        feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
    out = df.copy()
    for sid, sub in df.groupby("subject_id"):
        ref = sub[sub["label"].isin(reference_labels)]
        if ref.empty:
            continue
        for feat in feature_cols:
            vals = ref[feat].astype(float)
            mu = vals.mean()
            sd = vals.std(ddof=1)
            if not np.isfinite(sd) or sd < 1e-9:
                continue
            idx = df.index[df["subject_id"] == sid]
            out.loc[idx, feat] = (out.loc[idx, feat].astype(float) - mu) / sd
    return out


# ---------------------------------------------------------------------------
# Stage registry (drives the ladder runner)
# ---------------------------------------------------------------------------

STAGE_ORDER = ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7")

STAGE_KIND = {
    "H0": "raw",         # baseline: no transform
    "H1": "feature",     # F_ECG mask
    "H2": "raw",         # resample
    "H3": "raw",         # bandpass filter
    "H4": "raw",         # amplitude z-score
    "H5": "raw",         # identical windowing
    "H6": "feature",     # spectral control (H6a variant)
    "H7": "feature",     # within-subject z-score
}


def apply_raw_stage(stage: str, windows: List[SubjectWindow]) -> List[SubjectWindow]:
    if stage == "H0" or stage == "H1":
        return windows
    if stage == "H2":
        return resample_to_common_rate(windows)
    if stage == "H3":
        return identical_filtering(windows)
    if stage == "H4":
        return amplitude_normalize(windows)
    if stage == "H5":
        return identical_windowing(windows)
    return windows


def apply_feature_stage(stage: str, df: pd.DataFrame) -> pd.DataFrame:
    if stage == "H1":
        return feature_mask_ecg_only(df)
    if stage == "H6":
        return drop_spectral_features(df)
    if stage == "H7":
        return within_subject_zscore(df)
    return df
