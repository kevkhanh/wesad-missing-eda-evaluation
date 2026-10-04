"""
Audit Step 0 — freeze the environment and prove Driver #1.

Two outputs, both written to results/audit/step0_manifest.json:

  1. `versions` — the pinned library versions actually in use at run time.
     If any drift from configs/preregistration.yaml, the audit is invalidated
     and must be re-run.

  2. `nan_pattern` — for every feature in ALL_FEATURE_NAMES, the fraction of
     rows in each dataset that are NaN. If any feature is 100% NaN in one
     dataset and 0% NaN in the other, "dataset" is a perfect classifier on
     that column alone — LightGBM's native NaN routing turns the NaN mask
     into a splitting rule. This is Driver #1 in the protocol.

  3. `feature_provenance` — echoes feature_extraction.FEATURE_PROVENANCE so
     the manifest is self-contained.

Run:
    python -m audit.step0_freeze
"""

import json
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import ALL_FEATURE_NAMES, FEATURE_PROVENANCE
from train_stage3_models import _load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "audit"

_PINNED = [
    "numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm",
    "neurokit2", "wfdb", "shap", "statsmodels",
]


def _collect_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for pkg in _PINNED:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _nan_pattern(df: pd.DataFrame) -> dict:
    """Per-dataset NaN fraction for every ALL_FEATURE_NAMES column."""
    if "dataset" not in df.columns:
        return {}
    out = {}
    for feat in ALL_FEATURE_NAMES:
        if feat not in df.columns:
            continue
        row = {}
        for ds, sub in df.groupby("dataset"):
            n = len(sub)
            n_nan = int(sub[feat].isna().sum())
            row[str(ds)] = {"n_total": n, "n_nan": n_nan,
                            "frac_nan": float(n_nan / n) if n else float("nan")}
        out[feat] = row
    return out


def _driver1_verdict(nan_pattern: dict) -> dict:
    """Flag any feature whose NaN pattern is a perfect dataset indicator.

    Verdict is one of:
      * dataset_collinear   — 100% NaN in one dataset, 0% in the other
      * partial_signal      — non-trivial NaN in one but not the other
      * ok                  — same NaN fraction (within 5%) in both
    """
    verdict = {}
    for feat, per_ds in nan_pattern.items():
        fracs = [v["frac_nan"] for v in per_ds.values() if np.isfinite(v["frac_nan"])]
        if len(fracs) < 2:
            verdict[feat] = "single_dataset"
            continue
        hi, lo = max(fracs), min(fracs)
        if hi >= 0.99 and lo <= 0.01:
            verdict[feat] = "dataset_collinear"
        elif hi - lo > 0.05:
            verdict[feat] = "partial_signal"
        else:
            verdict[feat] = "ok"
    return verdict


def run_step0(cache_path=None) -> dict:
    """Build the manifest, print a summary, return the dict."""
    df = _load_dataset(cache_path) if cache_path else _load_dataset()
    nan_pattern = _nan_pattern(df)
    verdict = _driver1_verdict(nan_pattern)

    manifest = {
        "versions": _collect_versions(),
        "feature_provenance": FEATURE_PROVENANCE,
        "nan_pattern": nan_pattern,
        "driver1_verdict": verdict,
        "n_rows": int(len(df)),
        "datasets": sorted(df["dataset"].unique().tolist())
                    if "dataset" in df.columns else [],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "step0_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[step0] wrote {out_path}")
    print(f"[step0] {manifest['n_rows']} rows across datasets {manifest['datasets']}")
    collinear = [f for f, v in verdict.items() if v == "dataset_collinear"]
    if collinear:
        print(f"[step0] DRIVER #1 CONFIRMED — dataset-collinear features: {collinear}")
        print("        Any cross-dataset audit must drop these before Step 4.")
    else:
        print("[step0] No perfectly dataset-collinear NaN patterns found.")
    return manifest


if __name__ == "__main__":
    run_step0()
