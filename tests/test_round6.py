"""
Regression tests for the review-9 additions (post hoc): the intercept-only and
slope-only recalibrators of run_intercept_recal.py, and the no-added-masking
variant of run_mnar.py. Synthetic data only.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_intercept_recal as ir  # noqa: E402
import run_mnar as rm  # noqa: E402


def _sample(n=20000, seed=0, shift=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    p = 1 / (1 + np.exp(-(scale * z + shift)))     # a mis-located / mis-scaled model
    return y, p


def test_intercept_recalibrator_recovers_a_pure_shift():
    y, p = _sample(shift=1.5)
    cal = ir.InterceptRecalibrator().fit(y, p)
    assert abs(cal.b_ + 1.5) < 0.1
    assert abs(cal.predict(p).mean() - y.mean()) < 0.01     # base rate restored


def test_slope_recalibrator_recovers_a_pure_rescale_and_keeps_zero_intercept():
    y, p = _sample(scale=3.0)
    cal = ir.SlopeRecalibrator().fit(y, p)
    assert abs(cal.a_ - 1 / 3.0) < 0.05
    assert abs(cal.predict(np.array([0.5]))[0] - 0.5) < 1e-9   # no intercept term


def test_recalibration_cannot_change_auroc_ordering():
    y, p = _sample(shift=1.0, scale=2.0)
    for cal in (ir.InterceptRecalibrator().fit(y, p), ir.SlopeRecalibrator().fit(y, p)):
        q = cal.predict(p)
        assert np.array_equal(np.argsort(p, kind="stable"), np.argsort(q, kind="stable"))


def test_no_added_masking_variant_removes_every_extra_training_mask():
    s0, s5 = rm.make_systems(0.0), rm.make_systems(0.5)
    assert s0["gated"]().dropout_p == 0.0 and s0["mlp"]().dropout_p == 0.0
    assert s0["lgbm_nan_training"]().train_nan_p == 0.0
    assert s5["gated"]().dropout_p == 0.5 and s5["mlp"]().dropout_p == 0.5
    assert s5["lgbm_nan_training"]().train_nan_p == 0.5   # the pre-specified run is unchanged
