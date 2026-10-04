"""
Regression tests for the round-5 additions (prespec_round5.md: C3 MNAR
missingness, C4 operating points). Synthetic data only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feature_extraction import ALL_FEATURE_NAMES, EDA_FEATURE_NAMES  # noqa: E402
from run_metrics import (confusion_at, net_benefit, net_benefit_contrib,  # noqa: E402
                         selective_by_class, threshold_for_sensitivity,
                         threshold_for_specificity)
import run_mnar as rm  # noqa: E402


def _toy_scores(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(int)
    p = np.clip(0.35 + 0.3 * (y - 0.35) + 0.2 * rng.normal(size=n), 0.01, 0.99)
    return y, p


def test_thresholds_hit_targets_in_sample():
    y, p = _toy_scores()
    t1 = threshold_for_sensitivity(y, p, 0.9)
    t2 = threshold_for_specificity(y, p, 0.9)
    assert abs(confusion_at(y, p, t1)["sens"] - 0.9) < 0.02
    assert abs(confusion_at(y, p, t2, strict=True)["spec"] - 0.9) < 0.02


def test_net_benefit_definition_and_contributions():
    y, p = _toy_scores()
    for pt in (0.2, 0.35, 0.5):
        pred = p >= pt
        expect = (np.sum(pred & (y == 1)) - np.sum(pred & (y == 0)) * pt / (1 - pt)) / len(y)
        assert np.isclose(net_benefit(y, p, pt), expect)
        assert np.isclose(net_benefit_contrib(y, p, pt).mean(), expect)
    # Treat-all at pt equal to prevalence is exactly zero.
    pi = y.mean()
    assert np.isclose(net_benefit(y, np.ones_like(p), pi), 0.0)


def test_selective_by_class_retains_coverage_with_ties():
    y = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.6, 0.4, 0.5, 0.5, 0.1])  # many ties
    out = selective_by_class(y, p, 0.6)
    kept = (1 - out["abstain_stress"]) * y.sum() + (1 - out["abstain_baseline"]) * (1 - y).sum()
    assert np.isclose(kept, 6.0)


def test_label_mechanism_calibration():
    y = np.r_[np.ones(152), np.zeros(282)].astype(int)
    for o in (2, 4, 8):
        p0, p1 = rm.label_probs(y, o)
        assert np.isclose(y.mean() * p1 + (1 - y.mean()) * p0, rm.RATE)
        assert np.isclose(rm._odds(p1) / rm._odds(p0), o)


def test_motion_mechanism_calibration():
    rng = np.random.default_rng(1)
    y = (rng.random(434) < 0.35).astype(int)
    z = rng.normal(size=434) + 1.5 * y
    z = (z - z.mean()) / z.std()
    for o in (2, 4, 8):
        p, info = rm.motion_probs(z, y, o)
        assert np.isclose(p.mean(), rm.RATE)
        assert np.isclose(info["implied_or"], o, rtol=1e-6)


def _toy_df(n_subj=6, per=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        for w in range(per):
            y = int(w >= 13)
            rows.append({"subject_id": f"S{s}", "label": y,
                         **{f: rng.normal() + 0.8 * y for f in ALL_FEATURE_NAMES}})
    df = pd.DataFrame(rows)
    df["has_eda"] = True
    df["row_id"] = np.arange(len(df))
    df["p_test"] = np.where(df.label == 1, 0.9, 0.0)
    return df


def test_apply_missing_follows_probabilities_and_blanks_all_eda():
    df = _toy_df()
    out = rm.apply_missing(df, "test", seed=3)
    assert not out.loc[out.label == 0, "has_eda"].eq(False).any()
    miss = ~out.has_eda
    assert out.loc[miss, EDA_FEATURE_NAMES].isna().all().all()
    assert out.loc[~miss, EDA_FEATURE_NAMES].notna().all().all()


def test_per_config_routes_by_has_eda():
    df = _toy_df()
    df.loc[df.index % 3 == 0, EDA_FEATURE_NAMES] = np.nan
    df["has_eda"] = df[EDA_FEATURE_NAMES[0]].notna()
    m = rm.PerConfig().fit(df, seed=0)
    p = m.predict(df)
    has = df.has_eda.values
    assert np.allclose(p[has], m.full_.predict(df[has]))
    assert np.allclose(p[~has], m.ecg_.predict(df[~has]))


def test_complete_trained_lightgbm_nan_is_zero_fill_for_never_missing_features():
    """Review 8 (2026-10-02): LightGBM maps NaN to 0 at prediction for a feature
    that had no missing values in training, so baseline (i) "NaN at test" is
    zero-fill for those features (the label bug of Appendix A #9, one layer down)."""
    from eval_core import LGBMModel
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(rng.normal(size=(n, len(ALL_FEATURE_NAMES))) + 2.0, columns=ALL_FEATURE_NAMES)
    df["label"] = (df[ALL_FEATURE_NAMES[0]] + df[EDA_FEATURE_NAMES[0]] + rng.normal(size=n) > 4).astype(int)
    m = LGBMModel(ALL_FEATURE_NAMES).fit(df, seed=0)
    a = df.copy()
    a[EDA_FEATURE_NAMES] = np.nan
    b = a.copy()
    b[EDA_FEATURE_NAMES] = 0.0
    pa = m.model_.predict_proba(a[m.features])[:, 1]
    pb = m.model_.predict_proba(b[m.features])[:, 1]
    assert np.allclose(pa, pb)
