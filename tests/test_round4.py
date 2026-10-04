"""
Regression tests for the round-4 (peer_reviewe_v2.md) additions.
Synthetic data only.
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
from run_metrics import aurc, aurc_references, smooth_ece  # noqa: E402
from run_repeated_cv import (_rope_prob, _tost_p, bouckaert_frank,  # noqa: E402
                             participant_level)


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


def test_tost_matches_ci90():
    """TOST at delta passes iff the 90% CI lies inside (-delta, delta)."""
    rng = np.random.default_rng(1)
    d = rng.normal(0.002, 0.01, 100)
    bf = bouckaert_frank(d, 12, 3, n_subjects=15, delta=0.02)
    inside = -0.02 < bf["ci90"][0] and bf["ci90"][1] < 0.02
    assert (bf["tost_p"] < 0.05) == inside
    assert bf["equivalence_bound_attained"] == pytest.approx(max(map(abs, bf["ci90"])))


def test_df14_interval_wider_than_df99():
    d = np.random.default_rng(2).normal(0, 0.03, 100)
    bf = bouckaert_frank(d, 12, 3, n_subjects=15)
    w99 = bf["ci95"][1] - bf["ci95"][0]
    w14 = bf["df_subjects"]["ci95"][1] - bf["df_subjects"]["ci95"][0]
    assert w14 / w99 == pytest.approx(2.145 / 1.984, rel=0.01)


def test_rope_and_tost_limits():
    assert _rope_prob(0.0, 1e-4, 14, 0.02) == pytest.approx(1.0)
    assert _rope_prob(0.1, 1e-4, 14, 0.02) == pytest.approx(0.0, abs=1e-9)
    assert _tost_p(0.0, 1e-4, 14, 0.02) < 1e-6


def test_participant_level_counts_direction():
    d = pd.Series([-0.01] * 10 + [0.01] * 5)
    assert participant_level(d, 0.02, lower_is_better=True)["n_A_better"] == 10
    assert participant_level(d, 0.02, lower_is_better=False)["n_A_better"] == 5


def test_aurc_oracle_and_random_references():
    """A perfect error ranking attains the oracle; E-AURC >= 0."""
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    p = np.array([0.99, 0.95, 0.9, 0.05, 0.1, 0.15, 0.4, 0.6, 0.55, 0.45])  # last 4 = least confident
    ref = aurc_references(y, p)
    r = float(np.mean((p >= 0.5) != y))
    assert ref["aurc_random"] == pytest.approx(r)
    assert ref["e_aurc"] >= -1e-9
    assert aurc(y, p) <= ref["aurc_random"] + 1e-12


def test_smooth_ece_calibrated_vs_miscalibrated():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 600)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert smooth_ece(y, p) < 0.06
    assert smooth_ece(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_mlp_no_mask_never_sees_missing_indicator(toy):
    """M2: the never-masked MLP's indicator is 1 for every training row."""
    m = ec.NonGatedMLP(0.0).fit(toy, seed=0)
    ecg, eda, msk, _ = m._tensors(m._prep(toy))
    import torch
    gate = msk.float().unsqueeze(1)
    x = torch.cat([ecg, eda * gate, gate], dim=1)
    assert (x[:, -1] == 1).all()
    red = ec.make_reduced(toy.iloc[:20])
    assert np.isfinite(m.predict(red)).all()


def test_zero_features_removes_inputs(toy):
    """m11: zeroed HRV features cannot change the network's output."""
    m = ec.GatedModel(0.5, zero_features=ec.HRV_DROPPED).fit(toy, seed=0)
    alt = toy.copy()
    alt[ec.HRV_DROPPED] = alt[ec.HRV_DROPPED] * 100 + 7
    assert np.allclose(m.predict(toy), m.predict(alt))


def test_subject_normalise_baseline_reference(toy):
    out = ec.subject_normalise(toy, "baseline")
    base = out[out.label == 0].groupby("subject_id")[ALL_FEATURE_NAMES].mean()
    assert np.allclose(base.values, 0, atol=1e-9)


def test_maskrate_models_registered():
    for p in ec.MASK_RATES:
        for fam in ("gated", "mlp", "lgbm"):
            assert f"maskrate_{fam}_p{p}" in ec.MODELS
