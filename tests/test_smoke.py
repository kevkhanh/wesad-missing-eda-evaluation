"""
End-to-end smoke test on synthetic data.

Runs the whole pipeline (loaders -> feature extraction -> baseline +
invariant training -> audit Step 3 -> audit Step 4 in --fast mode) against
synthetic WESAD and MIT-BIH windows. Should complete in under 60s on a
laptop with no external downloads and no CUDA.

Not a scientific test — this checks wiring and interfaces. Real numbers
come from running against WESAD_ROOT and MITBIH_ROOT set to actual data.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    """Build a synthetic feature-cache CSV once, hand its path to the tests."""
    from real_data_loaders import (
        _synthesize_wesad_windows,
        _synthesize_mitbih_windows,
    )
    from feature_extraction import extract_batch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wesad = _synthesize_wesad_windows(n_subjects=4, window_sec=30)
        mitbih = _synthesize_mitbih_windows(n_records=6, window_sec=30)
    df = extract_batch(wesad + mitbih)

    out_dir = tmp_path_factory.mktemp("dataset")
    csv = out_dir / "stage3_hb1_dataset.csv"
    df.to_csv(csv, index=False)
    return df, csv


def test_features_have_expected_shape(synthetic_dataset):
    df, _ = synthetic_dataset
    from feature_extraction import ALL_FEATURE_NAMES

    assert len(df) > 0
    assert set(ALL_FEATURE_NAMES).issubset(df.columns)
    assert {"subject_id", "dataset", "label", "has_eda"}.issubset(df.columns)


def test_driver1_pattern_holds_on_synthetic(synthetic_dataset):
    """MIT-BIH has no EDA channel, so EDA features must be all-NaN there."""
    df, _ = synthetic_dataset
    mitbih = df[df["dataset"] == "MITBIH"]
    if mitbih.empty:
        pytest.skip("no MIT-BIH synthetic windows")
    assert mitbih["eda_tonic_mean"].isna().all(), \
        "MIT-BIH windows should never carry EDA features"
    assert not df[df["dataset"] == "WESAD"]["eda_tonic_mean"].isna().all(), \
        "WESAD baseline windows should carry at least some EDA features"


def test_baseline_and_invariant_train(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)

    train_df, test_df = m.load_and_split(simulated_mask_fraction=0.5)
    assert len(train_df) >= 4 and len(test_df) >= 2

    baseline, b_metrics = m.train_baseline(train_df, test_df)
    inv, i_metrics = m.train_invariant_model(train_df, test_df, model_seed=0)
    for metrics in (b_metrics, i_metrics):
        assert np.isfinite(metrics["ece"])


def test_step3_shortcut_runs(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)
    from audit.step3_shortcut import run_step3

    manifest = run_step3()
    assert "conditions" in manifest
    assert "verdict" in manifest
    # PATHOLOGY_LIKELY retired per peer-review M1 (see step3_shortcut.py).
    assert manifest["verdict"]["conclusion"] in {
        "SHORTCUT_CONFIRMED", "inconclusive",
    }


def test_step4_fast_mode(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)
    from audit.step4_identity_probe import run_step4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manifest = run_step4(fast=True)

    assert "C1" in manifest and "P1" in manifest["C1"]
    p1 = manifest["C1"]["P1"]
    assert np.isfinite(p1["subject_auc"])
    assert manifest["verdict"]["band"] in {
        "TRIVIALLY_SEPARABLE", "SUBSTANTIALLY_SEPARABLE",
        "BORDERLINE", "NOT_SEPARABLE", "inconclusive", "no_result",
    }


def test_step0_manifest(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)
    from audit.step0_freeze import run_step0

    manifest = run_step0()
    assert "versions" in manifest and "driver1_verdict" in manifest
    # On synthetic data, MIT-BIH has no EDA -> EDA features must be flagged
    # as dataset_collinear.
    assert manifest["driver1_verdict"].get("eda_tonic_mean") == "dataset_collinear"


# ---------------------------------------------------------------------------
# Steps 2, 5, 6, 7, 8, 9 — smoke coverage on synthetic data.
# Each of these keeps replicate counts small so the whole suite stays <2 min.
# ---------------------------------------------------------------------------

def test_step2_ecg_restriction(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)
    from audit.step2_common_modality import restrict_to_ecg
    from train_stage3_models import load_and_split
    from feature_extraction import EDA_FEATURE_NAMES

    train_df, test_df = load_and_split()
    restricted = restrict_to_ecg(train_df)
    for col in EDA_FEATURE_NAMES:
        assert col not in restricted.columns


def test_step5_mmd_fast(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)
    from audit.step5_mmd import run_step5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manifest = run_step5(fast=True)
    assert "primary" in manifest
    assert "verdict" in manifest
    assert manifest["verdict"] in {
        "not_detected", "significant_but_within_null_heterogeneity",
        "significant_beyond_null_heterogeneity", "insufficient_data",
    }


def test_step6_ladder_short(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)
    from audit.step6_harmonization import run_step6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manifest = run_step6(stages=["H0", "H1", "H6"], leave_one_out=False, fast=True)
    ladder = manifest["cumulative_ladder"]
    assert len(ladder) == 3
    assert all("identity_auc_C1" in r for r in ladder)


def test_step7_within_dataset(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)
    from audit.step7_within_dataset import run_step7

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manifest = run_step7(fast=True)
    assert "C3_wesad_baseline_vs_stress" in manifest
    assert "C4_mitbih_nsr_vs_arrhythmic" in manifest
    assert "verdict" in manifest


def test_step8_alignment(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)
    from audit.step8_alignment import run_step8

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manifest = run_step8()
    assert "verdict" in manifest
    assert "identity_auc_before" in manifest


def test_step9_decision(monkeypatch, synthetic_dataset):
    _, csv = synthetic_dataset
    import train_stage3_models as m
    monkeypatch.setattr(m, "DATASET_CSV", csv)
    from audit.step9_decision import run_step9

    # Step 9 reads Step 3-8 JSON manifests. They exist after the earlier
    # tests run; if this test runs standalone it will emit warnings but
    # still produce a decision using whatever inputs are available.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decision = run_step9()
    assert decision["branch"] in {
        "A_shortcut_confirmed",
        "B_removable_by_harmonization",
        "C_partial_inconclusive",
        "D_clinical_collapses_with_identity",
    }
