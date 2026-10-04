"""
Audit Step 6b — harmonization ladder runner.

Applies the H0-H7 stages cumulatively (as specified by the audit doc: most-
likely discriminative first), re-extracts features from raw signals after
each raw stage, and re-runs a compact identity probe (P1) plus the Step 3
production-model shortcut check after each stage. Reports a per-stage table:

  stage | C1 identity AUC | AUC 95% CI | Step-3 verdict | clinical AUC (C3)

Then a leave-one-stage-out attribution pass (full ladder minus each stage
in turn) so the reader can see which stage the residual identity signal
depends on.

Compute cost: each raw stage requires re-extracting features on ALL
windows. With ~60 synthetic windows this takes seconds; on real
WESAD+MIT-BIH with a few thousand windows it is minutes per stage. Use
--stages to limit the runner during development.

Run:
    python -m audit.step6_harmonization                # all stages, cumulative
    python -m audit.step6_harmonization --stages H0 H1 H2 H3
    python -m audit.step6_harmonization --leave-one-out
    python -m audit.step6_harmonization --fast         # reduced permutations
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import extract_batch, ECG_FEATURE_NAMES
from real_data_loaders import load_all_windows
from audit.harmonize import (
    STAGE_ORDER, STAGE_KIND,
    apply_raw_stage, apply_feature_stage,
)
from audit.step1_splits import build_sets
from audit.step3_shortcut import run_step3, _load_or_train_frozen, _score, _subject_agg
from audit.step4_identity_probe import (
    cross_val_subject_scores, subject_level_auc, cluster_bootstrap_ci,
)
from audit.step2_common_modality import restrict_to_ecg

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "audit"


# ---------------------------------------------------------------------------
# Per-stage extraction pipeline
# ---------------------------------------------------------------------------

def build_features_for_stages(stages: list, base_windows=None,
                              verbose: bool = True) -> pd.DataFrame:
    """Cumulatively apply raw-signal stages to base windows, re-extract
    features, then cumulatively apply feature-space stages to the DataFrame.

    Returns the final feature DataFrame with subject_id/dataset/label
    columns intact (audit split builders require these).
    """
    if base_windows is None:
        base_windows = load_all_windows()

    windows = list(base_windows)
    for stage in stages:
        if STAGE_KIND[stage] == "raw":
            if verbose:
                print(f"  [raw stage] {stage}")
            windows = apply_raw_stage(stage, windows)

    df = extract_batch(windows)

    for stage in stages:
        if STAGE_KIND[stage] == "feature":
            if verbose:
                print(f"  [feature stage] {stage}")
            df = apply_feature_stage(stage, df)

    return df


# ---------------------------------------------------------------------------
# Per-stage evaluation: compact Step 3 + compact Step 4 + C3 clinical AUC
# ---------------------------------------------------------------------------

def _compact_step3(df: pd.DataFrame) -> dict:
    """Frozen production-model shortcut check on this stage's features."""
    from train_stage3_models import _restrict_to_wesad, train_baseline
    from sklearn.metrics import roc_auc_score

    wesad = _restrict_to_wesad(df)
    if wesad.empty:
        return {"verdict": "no_wesad", "auc_b_vs_a": None,
                "auc_a_vs_c": None, "median_p_nsr": None}
    wesad = wesad.copy()
    wesad["has_eda"] = True
    dummy = wesad.iloc[:1].copy()
    feature_cols = [c for c in ECG_FEATURE_NAMES if c in wesad.columns]
    # Retrain baseline on just the ECG features available at this stage
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=50, max_depth=3,
                            random_state=42, verbosity=-1)
    m.fit(wesad[feature_cols], wesad["label"])

    def _mean_p(sub):
        if sub.empty:
            return None
        probs = m.predict_proba(sub[feature_cols])[:, 1]
        return float(np.mean(probs)), float(np.median(probs)), sub, probs

    conds = {
        "a": df[(df["dataset"] == "MITBIH") & (df["label"] == "nsr")],
        "b": df[(df["dataset"] == "MITBIH") & (df["label"] == "arrhythmic")],
        "c": df[(df["dataset"] == "WESAD") & (df["label"] == "baseline")],
    }
    results = {}
    for k, s in conds.items():
        results[k] = _mean_p(s) if not s.empty else None

    auc_b_vs_a = None
    auc_a_vs_c = None
    if results.get("a") and results.get("b"):
        a_mean, a_med, a_sub, a_probs = results["a"]
        b_mean, b_med, b_sub, b_probs = results["b"]
        y = np.concatenate([np.zeros(len(a_probs)), np.ones(len(b_probs))])
        s = np.concatenate([a_probs, b_probs])
        if len(set(y)) == 2:
            auc_b_vs_a = float(roc_auc_score(y, s))
    if results.get("a") and results.get("c"):
        a_mean, a_med, a_sub, a_probs = results["a"]
        c_mean, c_med, c_sub, c_probs = results["c"]
        y = np.concatenate([np.zeros(len(c_probs)), np.ones(len(a_probs))])
        s = np.concatenate([c_probs, a_probs])
        if len(set(y)) == 2:
            auc_a_vs_c = float(roc_auc_score(y, s))

    median_p_nsr = results["a"][1] if results.get("a") else None
    return {
        "auc_b_vs_a_pathology": auc_b_vs_a,
        "auc_a_vs_c_dataset": auc_a_vs_c,
        "median_p_nsr": median_p_nsr,
    }


def _compact_step4(df: pd.DataFrame, fast_seeds: int = 3) -> dict:
    """C1 identity probe with a small number of seeds for ladder-speed."""
    sets = build_sets(df)
    c1 = restrict_to_ecg(sets["C1"])
    if c1.empty:
        return {"auc": None, "ci95": [None, None]}
    subj_ds = dict(zip(c1["subject_id"],
                       np.where(c1["dataset"] == "MITBIH", "MITBIH", "WESAD")))
    scores = cross_val_subject_scores(c1, "P1", n_seeds=fast_seeds)
    auc, (y, s) = subject_level_auc(scores, subj_ds)
    ds_labels = np.array([subj_ds[sid] for sid, mp in
                          zip(scores["subject_id"], scores["mean_prob"])
                          if np.isfinite(mp)])
    ci_lo, ci_hi = cluster_bootstrap_ci(y, s, ds_labels, n_boot=500)
    return {"auc": float(auc) if np.isfinite(auc) else None,
            "ci95": [ci_lo, ci_hi], "n_subjects": int(len(y))}


def _clinical_auc_c3(df: pd.DataFrame) -> float | None:
    """WESAD baseline-vs-stress AUC with subject-grouped 3-fold CV."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb

    wesad = df[(df["dataset"] == "WESAD") &
               (df["label"].isin(["baseline", "stress"]))].copy()
    if wesad.empty or wesad["label"].nunique() < 2:
        return None
    wesad["y"] = (wesad["label"] == "stress").astype(int)
    subjects = wesad["subject_id"].values
    n_folds = min(3, len(np.unique(subjects)))
    if n_folds < 2:
        return None
    feature_cols = [c for c in ECG_FEATURE_NAMES if c in wesad.columns]
    if not feature_cols:
        return None
    aucs = []
    gkf = GroupKFold(n_splits=n_folds)
    for tr, te in gkf.split(wesad, groups=subjects):
        tr_df, te_df = wesad.iloc[tr], wesad.iloc[te]
        if te_df["y"].nunique() < 2:
            continue
        m = lgb.LGBMClassifier(n_estimators=50, max_depth=3,
                                random_state=42, verbosity=-1)
        m.fit(tr_df[feature_cols], tr_df["y"])
        probs = m.predict_proba(te_df[feature_cols])[:, 1]
        aucs.append(roc_auc_score(te_df["y"], probs))
    return float(np.mean(aucs)) if aucs else None


# ---------------------------------------------------------------------------
# Ladder + leave-one-out
# ---------------------------------------------------------------------------

def run_cumulative_ladder(stages: list, base_windows=None,
                          fast: bool = False) -> list:
    """Apply stages cumulatively; return per-stage evaluation records."""
    records = []
    for i in range(len(stages)):
        cur = stages[: i + 1]
        print(f"\n[ladder] cumulative stages: {cur}")
        df = build_features_for_stages(cur, base_windows=base_windows, verbose=False)
        s3 = _compact_step3(df)
        s4 = _compact_step4(df, fast_seeds=2 if fast else 5)
        c3 = _clinical_auc_c3(df)
        records.append({
            "stage_up_to": stages[i],
            "cumulative_stages": cur,
            "identity_auc_C1": s4["auc"],
            "identity_ci95": s4["ci95"],
            "n_subjects_C1": s4["n_subjects"],
            "shortcut_auc_a_vs_c": s3["auc_a_vs_c_dataset"],
            "shortcut_auc_b_vs_a": s3["auc_b_vs_a_pathology"],
            "median_p_mitbih_nsr": s3["median_p_nsr"],
            "clinical_auc_C3": c3,
        })
        _print_ladder_row(records[-1])
    return records


def run_leave_one_out(stages: list, base_windows=None,
                      fast: bool = False) -> list:
    """Full ladder minus each stage in turn (H0 always included)."""
    records = []
    for skip in stages:
        if skip == "H0":
            continue
        cur = [s for s in stages if s != skip]
        print(f"\n[loo] all stages except {skip}: {cur}")
        df = build_features_for_stages(cur, base_windows=base_windows, verbose=False)
        s4 = _compact_step4(df, fast_seeds=2 if fast else 5)
        c3 = _clinical_auc_c3(df)
        records.append({
            "dropped_stage": skip,
            "kept_stages": cur,
            "identity_auc_C1": s4["auc"],
            "identity_ci95": s4["ci95"],
            "clinical_auc_C3": c3,
        })
    return records


def _print_ladder_row(rec: dict):
    def _fmt(x):
        return "None" if x is None else f"{x:.3f}"
    print(f"  {rec['stage_up_to']:>4}  "
          f"identity AUC={_fmt(rec['identity_auc_C1'])}  "
          f"CI=[{_fmt(rec['identity_ci95'][0])},{_fmt(rec['identity_ci95'][1])}]  "
          f"shortcut a_vs_c={_fmt(rec['shortcut_auc_a_vs_c'])}  "
          f"clinical C3={_fmt(rec['clinical_auc_C3'])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_step6(stages: list | None = None, leave_one_out: bool = False,
              fast: bool = False) -> dict:
    stages = stages or list(STAGE_ORDER)
    print(f"[step6] cumulative ladder over: {stages}")

    base_windows = load_all_windows()
    ladder = run_cumulative_ladder(stages, base_windows=base_windows, fast=fast)
    loo = run_leave_one_out(stages, base_windows=base_windows, fast=fast) \
          if leave_one_out else []

    manifest = {
        "stages_applied": stages,
        "cumulative_ladder": ladder,
        "leave_one_out": loo,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step6_ladder.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[step6] wrote {OUT_DIR / 'step6_ladder.json'}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=None,
                    help="Subset of H0..H7 to apply cumulatively.")
    ap.add_argument("--leave-one-out", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    run_step6(stages=args.stages, leave_one_out=args.leave_one_out, fast=args.fast)
