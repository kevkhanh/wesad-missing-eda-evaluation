"""
Feature extraction for the E14 robustness pipeline.

Extends the coupling-lag-only extraction used by H2 to the broader HRV + EDA
feature set that the panic-proxy-vs-baseline classifier (H-B1/H-B2) needs.

Feature name order matches the ranking table in Section 5.5 of the outline
(13 features total). Downstream code — train_stage3_models, run_ablations,
run_interpretability, run_locked_comparison, and the audit modules — imports
these lists by name; do not reorder or rename without updating those.

Provenance note (relevant to audit Step 0 / Driver #1): every feature is
tagged with its raw source channel in FEATURE_PROVENANCE. The audit code uses
this to prove that any EDA-derived feature is structurally NaN on MIT-BIH
(which has no EDA channel), so their missingness pattern is a perfect
dataset label. That fact is by itself sufficient to force EDA features out
of any cross-dataset audit — see audit/step0_freeze.py.
"""

from typing import Iterable

import numpy as np

from stage1_pipeline import SubjectWindow, compute_coupling_lag

ECG_FEATURE_NAMES = [
    "hrv_mean_nn",
    "hrv_sdnn",
    "hrv_rmssd",
    "hrv_pnn50",
    "hrv_lf_power",
    "hrv_hf_power",
    "hrv_lf_hf_ratio",
]

EDA_FEATURE_NAMES = [
    "eda_tonic_mean",
    "eda_tonic_slope",
    "eda_phasic_mean_amp",
    "eda_phasic_rate",
    "coupling_peak_corr",
    "coupling_lag_sec",
]

ALL_FEATURE_NAMES = ECG_FEATURE_NAMES + EDA_FEATURE_NAMES

FEATURE_PROVENANCE = {
    **{f: "ecg" for f in ECG_FEATURE_NAMES},
    "eda_tonic_mean": "eda",
    "eda_tonic_slope": "eda",
    "eda_phasic_mean_amp": "eda",
    "eda_phasic_rate": "eda",
    "coupling_peak_corr": "ecg+eda",
    "coupling_lag_sec": "ecg+eda",
}


def _hrv_time_freq_features(ecg: np.ndarray, fs: int) -> dict:
    """HRV time- and frequency-domain features via neurokit2.

    Uses neurokit2's default R-peak detector and default LF/HF band edges
    (LF 0.04-0.15 Hz, HF 0.15-0.40 Hz — Task Force 1996). Returns all
    ECG_FEATURE_NAMES; unrecoverable segments (too few beats, degenerate
    signal) return NaN so LightGBM's native NaN routing takes over.
    """
    out = {k: float("nan") for k in ECG_FEATURE_NAMES}
    try:
        import neurokit2 as nk
    except ImportError:
        return _hrv_fallback(ecg, fs, out)

    try:
        signals, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 4:
            return out
        time_df = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
        freq_df = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False)
    except Exception:
        return _hrv_fallback(ecg, fs, out)

    def _first(df, key):
        if key in df.columns:
            v = df[key].values[0]
            return float(v) if np.isfinite(v) else float("nan")
        return float("nan")

    out["hrv_mean_nn"] = _first(time_df, "HRV_MeanNN")
    out["hrv_sdnn"] = _first(time_df, "HRV_SDNN")
    out["hrv_rmssd"] = _first(time_df, "HRV_RMSSD")
    out["hrv_pnn50"] = _first(time_df, "HRV_pNN50")
    out["hrv_lf_power"] = _first(freq_df, "HRV_LF")
    out["hrv_hf_power"] = _first(freq_df, "HRV_HF")
    lf, hf = out["hrv_lf_power"], out["hrv_hf_power"]
    out["hrv_lf_hf_ratio"] = float(lf / hf) if hf and hf > 0 else float("nan")
    return out


def _hrv_fallback(ecg: np.ndarray, fs: int, out: dict) -> dict:
    """Pure-numpy HRV when neurokit2 is unavailable.

    Enough to make the smoke test pass without neurokit2 installed. Uses the
    same simple detector as stage1_pipeline._instantaneous_hr, computes
    time-domain HRV directly from RR, and leaves frequency-domain features
    as NaN (Welch-based spectral estimation without windowing controls
    would be misleading in a fallback).
    """
    from scipy.signal import butter, filtfilt, find_peaks

    nyq = fs / 2.0
    b, a = butter(4, [5 / nyq, 15 / nyq], btype="band")
    filt = filtfilt(b, a, ecg)
    thresh = 0.6 * np.nanmax(np.abs(filt)) if np.isfinite(np.nanmax(filt)) else 0
    peaks, _ = find_peaks(filt, height=thresh, distance=int(0.3 * fs))
    if len(peaks) < 4:
        return out
    rr_ms = np.diff(peaks) / fs * 1000.0
    out["hrv_mean_nn"] = float(np.mean(rr_ms))
    out["hrv_sdnn"] = float(np.std(rr_ms, ddof=1))
    dif = np.diff(rr_ms)
    out["hrv_rmssd"] = float(np.sqrt(np.mean(dif ** 2)))
    out["hrv_pnn50"] = float(np.mean(np.abs(dif) > 50) * 100.0)
    return out


def _eda_features(eda: np.ndarray, fs: int) -> dict:
    """EDA tonic + phasic features.

    Tonic: mean and slope of a low-pass-filtered version (SCL proxy).
    Phasic: mean SCR amplitude and event rate per minute from a peak-picker
    on the high-pass residual. Prefer neurokit2's eda_phasic if available.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    out = {k: float("nan") for k in EDA_FEATURE_NAMES[:4]}
    if eda is None or len(eda) < fs * 2 or np.nanstd(eda) < 1e-9:
        return out

    nyq = fs / 2.0
    tonic_cut = min(0.5, nyq * 0.9)
    b, a = butter(2, tonic_cut / nyq, btype="low")
    tonic = filtfilt(b, a, eda)
    out["eda_tonic_mean"] = float(np.mean(tonic))
    t = np.arange(len(tonic)) / fs
    if len(t) > 1:
        slope, _ = np.polyfit(t, tonic, 1)
        out["eda_tonic_slope"] = float(slope)

    phasic = eda - tonic
    if np.nanstd(phasic) > 1e-9:
        peaks, props = find_peaks(
            phasic,
            height=0.05 * np.nanstd(phasic),
            distance=int(0.8 * fs),
        )
        if len(peaks):
            out["eda_phasic_mean_amp"] = float(np.mean(props["peak_heights"]))
            out["eda_phasic_rate"] = float(len(peaks) / (len(eda) / fs) * 60.0)
        else:
            out["eda_phasic_mean_amp"] = 0.0
            out["eda_phasic_rate"] = 0.0
    return out


def extract_full_features(window: SubjectWindow) -> dict:
    """Compute all 13 features for one SubjectWindow.

    Returned dict has exactly ALL_FEATURE_NAMES as keys, plus subject_id,
    dataset, label, and has_eda / has_ecg / has_resp booleans. Callers
    (train_stage3_models._load_dataset, the audit split builders) treat
    these as a pandas row.
    """
    row = {name: float("nan") for name in ALL_FEATURE_NAMES}
    row.update(_hrv_time_freq_features(window.ecg, window.fs))
    if window.has_eda():
        row.update(_eda_features(window.eda, window.fs))
        corr, lag = compute_coupling_lag(window.ecg, window.eda, window.fs)
        row["coupling_peak_corr"] = corr
        row["coupling_lag_sec"] = lag
    row["subject_id"] = window.subject_id
    row["dataset"] = window.dataset
    row["label"] = window.label
    row["has_ecg"] = True
    row["has_eda"] = window.has_eda()
    row["has_resp"] = window.has_resp()
    return row


def extract_batch(windows: Iterable[SubjectWindow]):
    """Convenience: extract features for a list of windows into a DataFrame."""
    import pandas as pd

    rows = [extract_full_features(w) for w in windows]
    return pd.DataFrame(rows)
