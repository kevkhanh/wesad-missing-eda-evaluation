"""
Audit Step 1 — build the six comparison sets defined in the protocol.

Comparison sets (labels use dataset+state pairs; column names below):
  C1 normal-vs-normal   — WESAD baseline vs MIT-BIH NSR-pure (primary probe)
  C2 aroused-vs-normal  — WESAD stress vs MIT-BIH NSR-pure
  C3 within-WESAD       — WESAD baseline vs stress (subject-grouped)
  C4 within-MIT-BIH     — MIT-BIH NSR-pure vs arrhythmic (subject-grouped)
  N1 null-MIT-BIH       — NSR subjects randomly split into halves, repeated
  N2 null-WESAD         — baseline subjects randomly split into halves, repeated

Each set is a pandas DataFrame with columns:
  subject_id, dataset, label, has_ecg, has_eda, has_resp,
  <ALL_FEATURE_NAMES>, split_target (the binary target for that probe)

Records 201/202 already share a subject_id at load time (see
real_data_loaders.MITBIH_SHARED_SUBJECT), so subject grouping is enforced
just by grouping on the subject_id column.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import ALL_FEATURE_NAMES
from train_stage3_models import _load_dataset


def _tag(df: pd.DataFrame, positive: dict, negative: dict) -> pd.DataFrame:
    """Filter df to rows matching either the positive or negative filter,
    add a split_target column (1 for positive, 0 for negative)."""
    def _match(row, spec):
        return all(row.get(k) == v for k, v in spec.items())

    pos = df[df.apply(lambda r: _match(r, positive), axis=1)].copy()
    neg = df[df.apply(lambda r: _match(r, negative), axis=1)].copy()
    pos["split_target"] = 1
    neg["split_target"] = 0
    return pd.concat([pos, neg], axis=0, ignore_index=True)


def build_sets(df: pd.DataFrame | None = None) -> dict:
    """Return {"C1": DataFrame, "C2": ..., "N1": ..., "N2": ...}."""
    if df is None:
        df = _load_dataset()

    # Normalize label column: strings expected ("baseline", "stress", "nsr",
    # "arrhythmic"); numeric labels from _restrict_to_wesad are re-mapped only
    # inside the WESAD-internal pipeline, so the raw cached CSV keeps strings.
    if df["label"].dtype != object:
        raise ValueError(
            "Audit split builder needs the raw string-label dataset; the "
            "int-encoded WESAD-only version is for train_stage3_models only."
        )

    sets = {}

    sets["C1"] = _tag(df,
        positive={"dataset": "MITBIH", "label": "nsr"},
        negative={"dataset": "WESAD", "label": "baseline"},
    )
    sets["C2"] = _tag(df,
        positive={"dataset": "MITBIH", "label": "nsr"},
        negative={"dataset": "WESAD", "label": "stress"},
    )
    sets["C3"] = _tag(df,
        positive={"dataset": "WESAD", "label": "stress"},
        negative={"dataset": "WESAD", "label": "baseline"},
    )
    sets["C4"] = _tag(df,
        positive={"dataset": "MITBIH", "label": "arrhythmic"},
        negative={"dataset": "MITBIH", "label": "nsr"},
    )

    sets["N1"] = _null_split(df, dataset="MITBIH", label="nsr", n_repeats=200, seed=1)
    sets["N2"] = _null_split(df, dataset="WESAD", label="baseline", n_repeats=200, seed=2)

    return sets


def _null_split(df, *, dataset, label, n_repeats, seed):
    """Random subject-halving nulls (N1 / N2).

    Returned DataFrame has an extra column 'null_rep' identifying each
    repetition; downstream code (Step 4 / Step 5) computes AUC / MMD per
    repetition and takes the 95th percentile as the "subject-heterogeneity
    floor".
    """
    sub = df[(df["dataset"] == dataset) & (df["label"] == label)].copy()
    if sub.empty:
        return sub.assign(split_target=[], null_rep=[])
    subjects = sorted(sub["subject_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    reps = []
    for r in range(n_repeats):
        perm = rng.permutation(subjects)
        half = len(subjects) // 2
        if half == 0:
            continue
        group_a = set(perm[:half].tolist())
        rep = sub.copy()
        rep["split_target"] = rep["subject_id"].isin(group_a).astype(int)
        rep["null_rep"] = r
        reps.append(rep)
    if not reps:
        return sub.assign(split_target=[], null_rep=[])
    return pd.concat(reps, axis=0, ignore_index=True)


def summarize(sets: dict) -> pd.DataFrame:
    """Print + return a summary of counts per set."""
    rows = []
    for name, s in sets.items():
        if s.empty:
            rows.append({"set": name, "n_rows": 0, "n_subjects": 0,
                         "positive_frac": float("nan")})
            continue
        rows.append({
            "set": name,
            "n_rows": len(s),
            "n_subjects": int(s["subject_id"].nunique()),
            "positive_frac": float(s["split_target"].mean()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sets = build_sets()
    print(summarize(sets).to_string(index=False))
