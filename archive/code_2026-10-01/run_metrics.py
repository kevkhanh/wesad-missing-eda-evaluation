"""
Single-source metric definitions for the Section 5.2 primary panel.

Addresses peer-review C5 (no discrimination metric reported), MA1 (ECE ratio
rewards worse complete-condition calibration), MA2 (per-tier structural
unfairness of the ratio), and MA6 (unspecified calibration-fit protocol).

Primary panel per the updated outline Section 5.2:
    - Reduced-condition Brier score (proper scoring rule, not gameable by base
      rate)
    - Reduced-condition equal-mass ECE_SWEEP (Roelofs et al., AISTATS 2022;
      monotone-sweep bin choice, no bias subtraction, no clipping)
    - Reduced-condition AUROC
    - Reduced-condition AUPRC

Secondary:
    - log loss, calibration slope/intercept
    - absolute delta-ECE = ECE_reduced - ECE_complete
    - ECE ratio (retained for continuity with earlier drafts)

Calibration-fit protocol (MA6; implemented in eval_core.fit_predict):
    Platt scaling is fit on INNER-CV out-of-fold predictions over the
    training subjects only (never on test subjects, never on in-sample
    training predictions). By default one Platt model, fit on complete-
    modality OOF predictions, is applied at both test tiers; baseline (iv)
    uses per-condition Platt by design. Raw and Platt-recalibrated metrics
    are both reported. Isotonic is not run (it would overfit at ~12
    training subjects).

All metrics operate on numpy arrays. Caller is responsible for subject-level
aggregation (see run_repeated_cv.py for the Bouckaert-Frank corrected t
across 20 repeats x 5 folds).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


EPS = 1e-7


# ---------------------------------------------------------------------------
# ECE_SWEEP, equal-mass (Roelofs et al., AISTATS 2022)
# ---------------------------------------------------------------------------
#
# Replaces the earlier `ece_debiased_sweep`, which took the MINIMUM over six
# bin counts of a zero-clipped, ad-hoc bias-corrected ECE. That estimator was
# biased toward 0 (20 of 40 reduced-condition per-fold values were exactly
# 0.000) and is not the Roelofs estimator (peer_review_round3.md C3).

def _equal_mass_bins(y: np.ndarray, p: np.ndarray, n_bins: int):
    """Yield (n_i, mean_y_i, mean_p_i) for equal-mass bins over sorted p."""
    order = np.argsort(p, kind="mergesort")
    y_s, p_s = y[order], p[order]
    edges = np.linspace(0, len(y), n_bins + 1).round().astype(int)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            yield hi - lo, float(np.mean(y_s[lo:hi])), float(np.mean(p_s[lo:hi]))


def _ece_equal_mass(y: np.ndarray, p: np.ndarray, n_bins: int,
                    order: int = 1) -> float:
    """L_order equal-mass ECE with n_bins bins."""
    N = len(y)
    if N == 0:
        return float("nan")
    total = sum(n_i / N * abs(acc - conf) ** order
                for n_i, acc, conf in _equal_mass_bins(y, p, n_bins))
    return float(total ** (1.0 / order))


def ece_sweep_em(y: np.ndarray, p: np.ndarray, order: int = 1) -> float:
    """ECE_SWEEP with equal-mass bins (Roelofs et al., AISTATS 2022).

    Uses the LARGEST bin count b for which the per-bin mean outcome is
    monotonically non-decreasing across the b bins, then reports the
    equal-mass ECE at that b. No bias subtraction and no clipping, so the
    estimator is never forced to 0. With ~40-90 windows per evaluation set
    its sampling SD is large; report its uncertainty, not a single value.
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p).astype(float), EPS, 1 - EPS)
    N = len(y)
    if N == 0:
        return float("nan")
    best_b = 1
    for b in range(2, N + 1):
        accs = [acc for _, acc, _ in _equal_mass_bins(y, p, b)]
        if all(a2 >= a1 for a1, a2 in zip(accs, accs[1:])):
            best_b = b
        else:
            break
    return _ece_equal_mass(y, p, best_b, order=order)


def risk_coverage(y: np.ndarray, p: np.ndarray):
    """Selective-prediction curve: abstain on the least confident windows.

    Confidence = max(p, 1 - p); risk = misclassification rate at the 0.5
    threshold among the retained windows. Returns (coverage, risk) arrays
    ordered from the most confident single window to full coverage.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    conf = np.round(np.maximum(p, 1 - p), 12)
    order = np.argsort(-conf, kind="mergesort")
    err = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    # Tree models emit tied probabilities; within a block of equal confidence
    # the abstention order is arbitrary, so use the block's mean error (the
    # expectation over tie orderings) instead of whatever the row order gives.
    c_sorted = conf[order]
    starts = np.r_[0, np.flatnonzero(np.diff(c_sorted)) + 1]
    ends = np.r_[starts[1:], len(err)]
    for a, b in zip(starts, ends):
        if b - a > 1:
            err[a:b] = err[a:b].mean()
    n = np.arange(1, len(y) + 1)
    return n / len(y), np.cumsum(err) / n


def aurc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better; Geifman et al. 2018)."""
    if len(y) == 0:
        return float("nan")
    _, risk = risk_coverage(y, p)
    return float(np.mean(risk))


def risk_at_coverage(y: np.ndarray, p: np.ndarray, coverage: float = 0.8) -> float:
    """Error rate on the most confident `coverage` fraction of windows."""
    if len(y) == 0:
        return float("nan")
    cov, risk = risk_coverage(y, p)
    return float(risk[max(int(np.ceil(coverage * len(y))) - 1, 0)])


def within_subject_auroc(y: np.ndarray, p: np.ndarray, subj: np.ndarray) -> float:
    """Mean over subjects of the AUROC computed on that subject's windows.

    Pooled AUROC mixes between-subject offsets with within-subject
    discrimination (peer_review_round3.md M3); this isolates the latter.
    Subjects with a single class are skipped.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    subj = np.asarray(subj)
    vals = []
    for s in np.unique(subj):
        m = subj == s
        if len(np.unique(y[m])) == 2:
            vals.append(roc_auc_score(y[m], p[m]))
    return float(np.mean(vals)) if vals else float("nan")


def ece_equal_width(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Legacy equal-width ECE used in train_stage3_models.expected_calibration_error.

    Retained so secondary ratio summaries stay comparable to earlier drafts.
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p).astype(float), EPS, 1 - EPS)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(y)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Calibration slope / intercept
# ---------------------------------------------------------------------------

def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict:
    """Logistic recalibration: fit logit(y) ~ a + b * logit(p); report (a, b)."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p).astype(float), EPS, 1 - EPS)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {"slope": float("nan"), "intercept": float("nan")}
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, max_iter=1000)
    try:
        lr.fit(logit_p, y)
    except Exception:
        return {"slope": float("nan"), "intercept": float("nan")}
    return {"slope": float(lr.coef_[0, 0]),
            "intercept": float(lr.intercept_[0])}


# ---------------------------------------------------------------------------
# Calibration fitting (Platt / isotonic) — MA6 lock
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """Single-parameter-pair logistic recalibration on logit(p).

    Fit on complete-modality training fold only (no exposure to test subjects).
    Apply the SAME fitted model at both complete and reduced test tiers so an
    inter-tier difference reflects the model, not the recalibrator (MA6).
    """

    def __init__(self):
        self._lr: Optional[LogisticRegression] = None

    def fit(self, y_train: np.ndarray, p_train: np.ndarray) -> "PlattCalibrator":
        y = np.asarray(y_train).astype(int)
        p = np.clip(np.asarray(p_train).astype(float), EPS, 1 - EPS)
        logit_p = np.log(p / (1 - p)).reshape(-1, 1)
        self._lr = LogisticRegression(C=1e6, max_iter=1000)
        self._lr.fit(logit_p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError("Fit PlattCalibrator before predict.")
        p = np.clip(np.asarray(p).astype(float), EPS, 1 - EPS)
        logit_p = np.log(p / (1 - p)).reshape(-1, 1)
        return self._lr.predict_proba(logit_p)[:, 1]


class IsotonicCalibrator:
    """Non-parametric isotonic recalibration. Reported as sensitivity only."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_train: np.ndarray, p_train: np.ndarray) -> "IsotonicCalibrator":
        y = np.asarray(y_train).astype(int)
        p = np.asarray(p_train).astype(float)
        self._iso.fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self._iso.predict(np.asarray(p).astype(float))


# ---------------------------------------------------------------------------
# Primary / secondary metric panel
# ---------------------------------------------------------------------------

@dataclass
class MetricPanel:
    n: int
    base_rate: float
    brier: float
    ece_sweep: float
    ece_equal_width: float
    log_loss: float
    auroc: float
    auprc: float
    calibration_slope: float
    calibration_intercept: float
    aurc: float
    risk_at_cov80: float


def compute_panel(y: np.ndarray, p: np.ndarray) -> MetricPanel:
    """Compute the full primary + secondary panel for one (y, p) partition."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p).astype(float), EPS, 1 - EPS)
    n = int(len(y))
    if n == 0:
        return MetricPanel(0, float("nan"), *([float("nan")] * 10))
    base_rate = float(np.mean(y))
    try:
        auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
    try:
        auprc = (float(average_precision_score(y, p))
                 if len(np.unique(y)) > 1 else float("nan"))
    except Exception:
        auprc = float("nan")
    slope_int = calibration_slope_intercept(y, p)
    return MetricPanel(
        n=n,
        base_rate=base_rate,
        brier=float(brier_score_loss(y, p)),
        ece_sweep=ece_sweep_em(y, p),
        ece_equal_width=ece_equal_width(y, p),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        auroc=auroc,
        auprc=auprc,
        calibration_slope=slope_int["slope"],
        calibration_intercept=slope_int["intercept"],
        aurc=aurc(y, p),
        risk_at_cov80=risk_at_coverage(y, p, 0.8),
    )


def panel_to_dict(panel: MetricPanel) -> dict:
    return asdict(panel)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_test = rng.integers(0, 2, size=200)
    p_test = np.clip(rng.beta(2, 5, size=200), 0.01, 0.99)
    p_test[y_test == 1] += 0.1
    p_test = np.clip(p_test, 0.01, 0.99)
    print("Smoke test (random inputs):")
    print(panel_to_dict(compute_panel(y_test, p_test)))
