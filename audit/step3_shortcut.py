"""
Audit Step 3 — frozen WESAD-internal baseline, transfer-behaviour diagnostic.

Per peer-review M1 (resolved) and the outline's Section 5.6 Step 3 label
clarification: the "frozen production model" is a LightGBM baseline trained
on WESAD-only data with class 1 = WESAD STRESS vs class 0 = WESAD baseline.
It is NOT a cardiac-detection model. Any variable or verdict that reads as
"P(cardiac)" or "PATHOLOGY_LIKELY" is a legacy modelling convention and is
reported here as P(class 1 = WESAD stress) / "stress-model-fires".

What this step tells us (diagnostic only; not decisive for the Branch
decision — Step 4 is):
  (a) MIT-BIH NSR-pure windows        — model's P(class 1 = stress) output
  (b) MIT-BIH arrhythmic windows      — model's P(class 1 = stress) output
  (c) WESAD baseline windows          — reference; the training class 0
  (d) WESAD stress windows            — reference; the training class 1

Reports:
  * subject-level mean P(class 1 = stress) per condition
  * AUC(b vs a) — whether the stress model ranks arrhythmic MIT-BIH above
                   NSR MIT-BIH. A model attending to high / irregular HR
                   will have AUC > 0.5 here; this does NOT imply the model
                   learned pathology, it reflects that both stress and
                   arrhythmia raise and destabilize heart rate.
  * AUC(a vs c) — dataset-dependence signal: whether the stress model
                   calls MIT-BIH NSR "stress-like" more or less than WESAD
                   baseline. Both directions are evidence that the model's
                   outputs depend on dataset. Report as separability
                   max(AUC, 1-AUC) per the Section 5.6 convention.

Historical note:
  Prior versions of this file used the label "PATHOLOGY_LIKELY" when
  AUC(b vs a) was high and median P on MIT-BIH NSR was low. That label
  was retired per peer-review M1 — a WESAD-stress model has no label
  that would allow it to "learn pathology", so claiming the opposite
  pattern implied anything about pathology was a conceptual conflation.
  The current verdict vocabulary is {SHORTCUT_CONFIRMED, inconclusive}
  only; cross-dataset decisions rest on Step 4's identity probe, not here.

If the real Preprint 1 model is stored elsewhere (a saved pickle from a
previous run), point --frozen-model at it; otherwise we train the
baseline on all WESAD data here and treat that as the frozen artefact.

Run:
    python -m audit.step3_shortcut
    python -m audit.step3_shortcut --frozen-model path/to/lgbm.pkl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import ALL_FEATURE_NAMES
from train_stage3_models import (
    _load_dataset,
    _restrict_to_wesad,
    train_baseline,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "audit"


def _load_or_train_frozen(cached_path: Path | None):
    """Load a serialized LightGBM model from disk, or train fresh on all WESAD."""
    if cached_path and cached_path.exists():
        import pickle
        with open(cached_path, "rb") as f:
            return pickle.load(f)

    df = _restrict_to_wesad(_load_dataset())
    if df.empty:
        raise RuntimeError("No WESAD windows to freeze a Preprint 1 model on.")
    df["has_eda"] = True
    dummy_test = df.iloc[:1].copy()
    model, _ = train_baseline(df, dummy_test)
    return model


def _score(model, df: pd.DataFrame) -> np.ndarray:
    """LightGBM handles NaN natively; do not fillna."""
    return model.predict_proba(df[ALL_FEATURE_NAMES])[:, 1]


def _subject_agg(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Mean probability per subject; used both for AUC and for the summary."""
    t = df.assign(p=probs)
    return t.groupby("subject_id")["p"].mean().reset_index()


def run_step3(frozen_model_path: Path | None = None) -> dict:
    df = _load_dataset()
    if df["label"].dtype != object:
        raise RuntimeError(
            "Step 3 needs the raw string-label cache (WESAD 'baseline'/'stress' "
            "and MIT-BIH 'nsr'/'arrhythmic'). Delete stage3_hb1_dataset.csv "
            "and let real_data_loaders rebuild it."
        )

    conditions = {
        "a_mitbih_nsr":  df[(df["dataset"] == "MITBIH") & (df["label"] == "nsr")],
        "b_mitbih_arr":  df[(df["dataset"] == "MITBIH") & (df["label"] == "arrhythmic")],
        "c_wesad_base":  df[(df["dataset"] == "WESAD")  & (df["label"] == "baseline")],
        "d_wesad_stress":df[(df["dataset"] == "WESAD")  & (df["label"] == "stress")],
    }

    model = _load_or_train_frozen(frozen_model_path)
    summary = {}
    subject_means = {}
    for name, sub in conditions.items():
        if sub.empty:
            summary[name] = None
            continue
        probs = _score(model, sub)
        subj_means = _subject_agg(sub, probs)
        subject_means[name] = subj_means
        summary[name] = {
            "n_windows": int(len(sub)),
            "n_subjects": int(sub["subject_id"].nunique()),
            "median_p": float(np.median(probs)),
            "mean_p": float(np.mean(probs)),
            "subject_mean_p_median": float(subj_means["p"].median()),
        }

    aucs = {}
    if subject_means.get("a_mitbih_nsr") is not None and subject_means.get("b_mitbih_arr") is not None:
        y = np.concatenate([
            np.zeros(len(subject_means["a_mitbih_nsr"])),
            np.ones(len(subject_means["b_mitbih_arr"])),
        ])
        s = np.concatenate([
            subject_means["a_mitbih_nsr"]["p"].values,
            subject_means["b_mitbih_arr"]["p"].values,
        ])
        if len(set(y)) == 2:
            aucs["auc_b_vs_a_pathology"] = float(roc_auc_score(y, s))
    if subject_means.get("a_mitbih_nsr") is not None and subject_means.get("c_wesad_base") is not None:
        y = np.concatenate([
            np.zeros(len(subject_means["c_wesad_base"])),
            np.ones(len(subject_means["a_mitbih_nsr"])),
        ])
        s = np.concatenate([
            subject_means["c_wesad_base"]["p"].values,
            subject_means["a_mitbih_nsr"]["p"].values,
        ])
        if len(set(y)) == 2:
            aucs["auc_a_vs_c_dataset"] = float(roc_auc_score(y, s))

    verdict = _verdict(summary, aucs)
    manifest = {
        "conditions": summary,
        "subject_level_auc": aucs,
        "verdict": verdict,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step3_shortcut.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _print(manifest)
    return manifest


def _verdict(summary: dict, aucs: dict) -> dict:
    """Decide SHORTCUT_CONFIRMED vs inconclusive from the stress-model's
    MIT-BIH outputs.

    Peer-review M1: the historical PATHOLOGY_LIKELY verdict (fired when
    AUC(arr vs NSR) > 0.75 and median P on MIT-BIH NSR < 0.5) was retired
    because the stress model has no label capable of representing
    pathology — a high rank of arrhythmic over NSR reflects heart-rate /
    irregularity sensitivity shared with stress, not learned pathology.
    Verdict vocabulary is now {SHORTCUT_CONFIRMED, inconclusive} only.
    Cross-dataset decisions rest on Step 4's identity probe.
    """
    med_nsr = (summary.get("a_mitbih_nsr") or {}).get("subject_mean_p_median")
    auc_b_vs_a = aucs.get("auc_b_vs_a_pathology")
    conclusion = "inconclusive"
    reasons = []
    if med_nsr is not None and med_nsr > 0.8:
        reasons.append(
            f"P(class 1 = stress | MIT-BIH NSR) subject-median = {med_nsr:.3f} > 0.8"
        )
    if auc_b_vs_a is not None and auc_b_vs_a <= 0.65:
        reasons.append(
            f"AUC(arrhythmic vs NSR) = {auc_b_vs_a:.3f} <= 0.65 "
            "(stress model fails to separate the two MIT-BIH conditions)"
        )

    if len(reasons) == 2:
        conclusion = "SHORTCUT_CONFIRMED"
    return {"conclusion": conclusion, "reasons": reasons}


def _print(manifest):
    print("\n=== Audit Step 3 — production-model shortcut test ===")
    for cond, s in manifest["conditions"].items():
        if s is None:
            print(f"  {cond:<18}  no windows")
            continue
        print(f"  {cond:<18}  n={s['n_windows']:>4}  "
              f"median P(class 1 = stress)={s['median_p']:.3f}  "
              f"subject-median={s['subject_mean_p_median']:.3f}")
    print("\n  Subject-level AUCs:")
    for k, v in manifest["subject_level_auc"].items():
        print(f"  {k:<22}  {v:.3f}")
    v = manifest["verdict"]
    print(f"\n  Verdict: {v['conclusion']}")
    for r in v["reasons"]:
        print(f"    - {r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-model", type=Path, default=None,
                    help="Optional path to a serialized (pickled) frozen LightGBM model.")
    args = ap.parse_args()
    run_step3(args.frozen_model)
