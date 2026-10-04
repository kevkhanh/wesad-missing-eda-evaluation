"""
Stage 1 pipeline utilities.

Provides SubjectWindow (a lightweight per-window container carrying the raw
ECG/EDA/label metadata that feature_extraction needs) and compute_coupling_lag
(the EDA-HR coupling-lag feature that Section 5.5's H2a hypothesis is built on
and that ranks 12/13 in the SHAP table).

Kept deliberately small: this module exists so that feature_extraction and the
audit modules have a single, stable place to import per-window primitives
from. If more Stage 1 utilities appear later, add them here rather than in
feature_extraction.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import correlate


@dataclass
class SubjectWindow:
    """One analysis window from one subject.

    ecg / eda / resp arrays are the raw samples at the window's native sample
    rate (may differ across datasets). fs stores that rate so downstream
    feature code can resample as needed without guessing.

    dataset is a short tag ("WESAD" or "MITBIH") the audit code uses as the
    domain label. label is the clinical/state target (baseline/stress for
    WESAD; NSR/arrhythmic for MIT-BIH).
    """
    subject_id: str
    dataset: str
    label: str
    fs: int
    ecg: np.ndarray
    eda: Optional[np.ndarray] = None
    resp: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def has_eda(self) -> bool:
        return self.eda is not None and np.any(np.isfinite(self.eda))

    def has_resp(self) -> bool:
        return self.resp is not None and np.any(np.isfinite(self.resp))


def _instantaneous_hr(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Uniform-time-grid instantaneous HR (bpm) from an ECG segment.

    Uses a simple bandpass + threshold R-peak detector — matches the
    onset-detector step0 wiring rather than a full Pan-Tompkins pipeline.
    Fine for coupling-lag correlation (which only needs a smooth HR curve);
    for HRV features we use neurokit2 in feature_extraction.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    nyq = fs / 2.0
    b, a = butter(4, [5 / nyq, 15 / nyq], btype="band")
    filt = filtfilt(b, a, ecg)
    thresh = 0.6 * np.nanmax(np.abs(filt))
    peaks, _ = find_peaks(filt, height=thresh, distance=int(0.3 * fs))
    if len(peaks) < 3:
        return np.full(len(ecg), np.nan)

    rr = np.diff(peaks) / fs
    hr = 60.0 / rr
    hr_time = peaks[1:] / fs
    grid = np.arange(len(ecg)) / fs
    return np.interp(grid, hr_time, hr, left=hr[0], right=hr[-1])


def compute_coupling_lag(
    ecg: np.ndarray,
    eda: np.ndarray,
    fs: int,
    max_lag_sec: float = 15.0,
) -> tuple[float, float]:
    """Cross-correlation lag (seconds) that maximises corr(EDA, HR).

    Returns (peak_correlation, lag_sec). lag > 0 means EDA follows HR.
    NaN if either channel is degenerate.

    This is the H2a-specification feature; SHAP ranks it 12/13 in the
    complete-modality regime and 3rd when ECG is zeroed (Section 5.5 of the
    outline). Both facts are consistent with a weak-but-nonzero
    physiological signal that becomes load-bearing only when HRV is gone.
    """
    if len(ecg) < fs * 3 or len(eda) < fs * 3:
        return float("nan"), float("nan")

    hr = _instantaneous_hr(ecg, fs)
    if not np.any(np.isfinite(hr)) or np.nanstd(hr) < 1e-6:
        return float("nan"), float("nan")

    n = min(len(hr), len(eda))
    hr = hr[:n]
    eda = np.asarray(eda[:n], dtype=float)
    if np.nanstd(eda) < 1e-6:
        return float("nan"), float("nan")

    hr = (hr - np.nanmean(hr)) / (np.nanstd(hr) + 1e-12)
    eda = (eda - np.nanmean(eda)) / (np.nanstd(eda) + 1e-12)

    max_lag_samples = int(round(max_lag_sec * fs))
    corr = correlate(eda, hr, mode="full") / n
    lags = np.arange(-n + 1, n)
    mask = np.abs(lags) <= max_lag_samples
    corr, lags = corr[mask], lags[mask]

    idx = int(np.nanargmax(np.abs(corr)))
    return float(corr[idx]), float(lags[idx] / fs)
