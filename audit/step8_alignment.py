"""
Audit Step 8 — conditional alignment diagnostics (CORAL on C1 only).

WARNING (repeated verbatim from the audit doc, because it decides how this
module can be used):

    Do NOT run unconditional DANN on dataset identity; it will erase the
    class signal because domain = label. Acceptable diagnostic: learn a
    feature transform that aligns ONLY the matched normal windows (C1)
    across datasets, e.g. CORAL or an MMD-penalized linear map fitted on
    C1 only, then apply it to all windows and re-test the clinical task.
    If clinical AUC survives alignment of the normal states, that is mild
    evidence of a genuine signal.

That is exactly what this module does. It fits second-order CORAL on the
matched normal windows only (WESAD baseline vs MIT-BIH NSR-pure), applies
the resulting linear map to ALL windows in both datasets, and re-tests:

  * identity AUC on C1  (should drop by design)
  * clinical AUC on C3  (should NOT collapse if signal is genuine)
  * clinical AUC on C4  (should NOT collapse if signal is genuine)

The audit doc explicitly labels this as a diagnostic, not a cure. Do not
report the CORAL-aligned pipeline as the preprint's headline pipeline.

If the `skada` or `adapt` packages are available they can be used for a
richer alignment (KMM, KLIEP, MDD), but this module ships a self-contained
CORAL implementation so it runs without those installs.

Run:
    python -m audit.step8_alignment
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import ECG_FEATURE_NAMES
from train_stage3_models import _load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "audit"


# ---------------------------------------------------------------------------
# CORAL: whiten source, recolor to target
# ---------------------------------------------------------------------------

def _cov_regularized(X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    X = X - X.mean(axis=0, keepdims=True)
    C = np.cov(X, rowvar=False)
    if np.ndim(C) == 0:
        return np.array([[float(C) + eps]])
    return C + eps * np.eye(C.shape[0])


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(M)
    w = np.clip(w, 0, None)
    return v @ np.diag(np.sqrt(w)) @ v.T


def _matrix_inv_sqrt(M: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(M)
    w = np.where(w > 1e-9, 1.0 / np.sqrt(w), 0.0)
    return v @ np.diag(w) @ v.T


def fit_coral_map(src: np.ndarray, tgt: np.ndarray):
    """Return an affine map (A, b) such that A @ (x - src_mean) + tgt_mean
    aligns src to tgt in second-order moments."""
    Cs = _cov_regularized(src)
    Ct = _cov_regularized(tgt)
    A = _matrix_inv_sqrt(Cs) @ _matrix_sqrt(Ct)
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    return A, src_mean, tgt_mean


def apply_coral(X: np.ndarray, A, src_mean, tgt_mean) -> np.ndarray:
    return (X - src_mean) @ A + tgt_mean


# ---------------------------------------------------------------------------
# Runner: fit on C1 only, apply everywhere, re-eval
# ---------------------------------------------------------------------------

def run_step8() -> dict:
    from audit.step1_splits import build_sets
    from audit.step2_common_modality import restrict_to_ecg
    from audit.step7_within_dataset import (
        within_wesad_c3, within_mitbih_c4, _identity_auc,
    )

    df = _load_dataset()
    if df["label"].dtype != object:
        raise RuntimeError("Step 8 needs the string-label dataset.")

    feature_cols = [c for c in ECG_FEATURE_NAMES if c in df.columns]
    if not feature_cols:
        raise RuntimeError("No ECG feature columns to align.")

    # Fit on C1 (matched normal windows) — WESAD baseline as source,
    # MIT-BIH NSR as target. Direction is a modelling choice; we pick
    # source=WESAD because it is smaller and has more variance to align.
    sets = build_sets(df)
    c1 = restrict_to_ecg(sets["C1"])
    wesad_normal = c1[(c1["dataset"] == "WESAD")][feature_cols].astype(float).fillna(0.0).values
    mitbih_normal = c1[(c1["dataset"] == "MITBIH")][feature_cols].astype(float).fillna(0.0).values
    if len(wesad_normal) < 4 or len(mitbih_normal) < 4:
        raise RuntimeError("Too few C1 rows to fit CORAL.")

    A, src_mean, tgt_mean = fit_coral_map(wesad_normal, mitbih_normal)

    # Apply to ALL windows (both datasets, both labels). WESAD windows get
    # actively aligned; MIT-BIH windows pass through unchanged (they are the
    # target). This is standard CORAL directionality — we move source to
    # target's second-order moment.
    aligned = df.copy()
    wesad_mask = aligned["dataset"] == "WESAD"
    if wesad_mask.any():
        Xw = aligned.loc[wesad_mask, feature_cols].astype(float).fillna(0.0).values
        aligned.loc[wesad_mask, feature_cols] = apply_coral(
            Xw, A, src_mean, tgt_mean
        )

    # Re-evaluate.
    id_before = _identity_auc(df, feature_cols)
    id_after = _identity_auc(aligned, feature_cols)
    c3_before = within_wesad_c3(df, feature_cols)["auc_mean"]
    c3_after = within_wesad_c3(aligned, feature_cols)["auc_mean"]
    c4_before = within_mitbih_c4(df, feature_cols)["auc_mean"]
    c4_after = within_mitbih_c4(aligned, feature_cols)["auc_mean"]

    manifest = {
        "notice": "diagnostic only — not a preprint-headline pipeline",
        "coral_map_shape": [int(A.shape[0]), int(A.shape[1])],
        "identity_auc_before": id_before,
        "identity_auc_after": id_after,
        "c3_clinical_before": c3_before,
        "c3_clinical_after": c3_after,
        "c4_clinical_before": c4_before,
        "c4_clinical_after": c4_after,
        "verdict": _verdict(id_before, id_after, c3_before, c3_after, c4_before, c4_after),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step8_alignment.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Audit Step 8 — conditional CORAL alignment (C1 only) ===")
    print(f"  identity C1  {_fmt(id_before)} -> {_fmt(id_after)}")
    print(f"  clinical C3  {_fmt(c3_before)} -> {_fmt(c3_after)}")
    print(f"  clinical C4  {_fmt(c4_before)} -> {_fmt(c4_after)}")
    print(f"  verdict: {manifest['verdict']}")
    return manifest


def _fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  N/A"
    return f"{x:.3f}"


def _verdict(id_before, id_after, c3_b, c3_a, c4_b, c4_a) -> str:
    def _drop(a, b):
        if a is None or b is None:
            return None
        return (a or 0) - (b or 0)
    id_drop = _drop(id_before, id_after)
    c3_drop = _drop(c3_b, c3_a)
    c4_drop = _drop(c4_b, c4_a)
    if id_drop is None:
        return "insufficient_data"
    if id_drop > 0.1 and (c3_drop or 0) < 0.05 and (c4_drop or 0) < 0.05:
        return "identity_reduced_clinical_survived (mild evidence of genuine signal)"
    if id_drop > 0.1 and ((c3_drop or 0) > 0.1 or (c4_drop or 0) > 0.1):
        return "identity_reduced_clinical_collapsed (Branch D signal)"
    return "alignment_ineffective_or_inconclusive"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    run_step8()
