"""
Regression tests for the peer_review_round3.md fixes (C1, C2, C3, M1, §3).
Synthetic data only; no real feature cache required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_core as ec  # noqa: E402
from feature_extraction import ALL_FEATURE_NAMES, EDA_FEATURE_NAMES  # noqa: E402
from run_metrics import ece_sweep_em, within_subject_auroc  # noqa: E402
from run_repeated_cv import sign_flip_p  # noqa: E402


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(10):
        for w in range(20):
            y = int(w >= 13)
            feats = {f: rng.normal() + (0.8 * y if f in ("hrv_mean_nn", "eda_tonic_mean") else 0)
                     for f in ALL_FEATURE_NAMES}
            rows.append({"subject_id": f"WESAD_S{s}", "label": y, **feats})
    df = pd.DataFrame(rows)
    df["has_eda"] = True
    df["has_ecg"] = True
    df["row_id"] = np.arange(len(df))
    return df


def test_repeated_partitions_differ(toy):
    """C1: each repeat must draw a different subject partition."""
    parts = {tuple(tuple(f) for f in ec.subject_kfold(toy.subject_id, 5, seed=1000 + r))
             for r in range(20)}
    assert len(parts) > 1


def test_partition_covers_each_subject_once(toy):
    folds = ec.subject_kfold(toy.subject_id, 5, seed=3)
    flat = np.concatenate(folds)
    assert sorted(flat) == sorted(toy.subject_id.unique())


def test_paired_tiers_same_windows(toy):
    """M1: complete and reduced tiers contain exactly the same windows."""
    tiers = ec.paired_tiers(toy.iloc[:40])
    assert (tiers["complete"].row_id.values == tiers["reduced"].row_id.values).all()
    assert tiers["reduced"][EDA_FEATURE_NAMES].isna().all().all()
    assert not tiers["reduced"]["has_eda"].any()
    assert tiers["complete"]["has_eda"].all()


def test_fit_predict_never_trains_on_test(toy, monkeypatch):
    """C2: neither the model nor the inner-CV Platt sees test subjects."""
    seen = []
    orig_fit = ec.LGBMModel.fit

    def spy(self, train_df, seed):
        seen.append(set(train_df.subject_id))
        return orig_fit(self, train_df, seed)

    monkeypatch.setattr(ec.LGBMModel, "fit", spy)
    train_df, test_df = ec.split_by_subjects(toy, ["WESAD_S0", "WESAD_S1"])
    out = ec.fit_predict("lgbm_nan_training_p0.5", train_df, test_df, seed=1)
    assert len(seen) == ec.INNER_K + 1
    assert all(not (s & {"WESAD_S0", "WESAD_S1"}) for s in seen)
    assert set(out.tier) == {"complete", "reduced"}
    assert np.isfinite(out.p_platt).all()


def test_hr_resid_removes_hr(toy):
    """§3: the HR-residualised variant must not carry HR, and residuals are
    uncorrelated with HR within each subject."""
    out = ec._hr_resid_transform(toy)
    assert (out[ec.HR_FEATURE] == 0).all()
    for _, g in out.assign(hr=toy[ec.HR_FEATURE]).groupby("subject_id"):
        assert abs(np.corrcoef(g["hrv_rmssd"], g["hr"])[0, 1]) < 1e-8


def test_ece_sweep_not_clipped_and_small_when_calibrated():
    """C3: calibrated predictions give small ECE; miscalibrated give large;
    nothing is forced to zero."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert 0 < ece_sweep_em(y, p) < 0.03
    assert ece_sweep_em(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_within_subject_auroc_ignores_between_subject_offsets():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])   # pooled AUROC 0.5, each subject perfect
    s = np.array(["a", "a", "b", "b"])
    assert within_subject_auroc(y, p, s) == 1.0


def test_sign_flip_exact():
    assert sign_flip_p(np.ones(10)) == pytest.approx(2 / 1024)


def test_aurc_perfect_and_random():
    """Selective risk: a perfect ranker with confident correct predictions has
    AURC 0; abstaining on the least confident windows never raises risk for a
    well-ordered model."""
    from run_metrics import aurc, risk_at_coverage
    y = np.array([0, 0, 1, 1])
    assert aurc(y, np.array([0.01, 0.1, 0.9, 0.99])) == 0.0
    p = np.array([0.4, 0.45, 0.6, 0.99])     # confident window correct, others not
    y2 = np.array([1, 1, 0, 1])
    assert risk_at_coverage(y2, p, 0.25) == 0.0
    assert aurc(y2, p) > 0


def test_risk_coverage_tie_invariant():
    """Tied confidences must not make AURC depend on row order."""
    from run_metrics import aurc
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.8, 0.8, 0.8, 0.2, 0.2, 0.2])
    perm = np.array([5, 4, 3, 2, 1, 0])
    assert aurc(y, p) == pytest.approx(aurc(y[perm], p[perm]))
