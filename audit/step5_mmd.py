"""
Audit Step 5 — MMD two-sample test with block permutation.

Kernel:
  Gaussian RBF k(x, y) = exp(-||x - y||^2 / (2 sigma^2)).
  Bandwidth sigma set by the median heuristic on the pooled sample
  (Gretton et al., JMLR 2012).

Statistic:
  Unbiased MMD^2 (U-statistic), Gretton eq. (3):
    MMD^2_u = 1/(m(m-1)) sum_{i!=j} k(x_i, x_j)
            + 1/(n(n-1)) sum_{i!=j} k(y_i, y_j)
            - 2/(m n)   sum_{i,j}   k(x_i, y_j)

Exchangeability fix:
  Windows within a subject are correlated; a window-level permutation is
  anti-conservative. We use BLOCK PERMUTATION — permute the dataset label
  across SUBJECTS, carrying all of each subject's windows together.

Aggregation ("MMDAgg" fallback):
  Fallback implementation of Schrab et al. 2023's collection-of-bandwidths
  test when the `mmdagg` package isn't installed: run the test at a small
  grid of sigmas relative to the median heuristic (0.5x, 1x, 2x, 4x, 8x),
  Holm-correct across bandwidths, report the minimum-adjusted p.

Effect size (heterogeneity ratio R):
  R = MMD^2(C1) / (95th percentile of MMD^2 from N1/N2 subject-split nulls).
  R <= 1 means the cross-dataset gap is no larger than the within-dataset
  subject-heterogeneity gap.

Run:
    python -m audit.step5_mmd
    python -m audit.step5_mmd --fast
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import ECG_FEATURE_NAMES

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "audit"

N_PERM_DEFAULT = 10000
BANDWIDTH_SCALE_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)


# ---------------------------------------------------------------------------
# Kernel + statistic
# ---------------------------------------------------------------------------

def _pairwise_sq_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    xx = np.sum(X * X, axis=1, keepdims=True)
    yy = np.sum(Y * Y, axis=1, keepdims=True).T
    d2 = xx + yy - 2.0 * X @ Y.T
    return np.maximum(d2, 0.0)


def _median_bandwidth(Z: np.ndarray) -> float:
    """Median pairwise Euclidean distance of the pooled sample."""
    d2 = _pairwise_sq_dist(Z, Z)
    tri = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.sqrt(np.median(tri))) if tri.size else 1.0
    return max(med, 1e-6)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-_pairwise_sq_dist(X, Y) / (2 * sigma ** 2))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD^2 U-statistic, Gaussian RBF."""
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return float("nan")
    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (m * (m - 1))
    term_yy = Kyy.sum() / (n * (n - 1))
    term_xy = Kxy.sum() / (m * n)
    return float(term_xx + term_yy - 2 * term_xy)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _prep_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Robust-scale (pooled median/IQR), then fill NaN with 0."""
    X = df[feature_cols].astype(float).values.copy()
    med = np.nanmedian(X, axis=0)
    q75 = np.nanpercentile(X, 75, axis=0)
    q25 = np.nanpercentile(X, 25, axis=0)
    iqr = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
    X = (X - med) / iqr
    return np.nan_to_num(X, nan=0.0)


def _split_by_target(df: pd.DataFrame, feature_cols: list):
    X = _prep_features(df, feature_cols)
    y = df["split_target"].values.astype(int)
    return X[y == 0], X[y == 1], df.loc[y == 0, "subject_id"].values, \
                                 df.loc[y == 1, "subject_id"].values


# ---------------------------------------------------------------------------
# Block (subject-level) permutation
# ---------------------------------------------------------------------------

def block_permutation_p(
    df: pd.DataFrame,
    feature_cols: list,
    sigma: float,
    n_perm: int = N_PERM_DEFAULT,
    seed: int = 0,
) -> tuple[float, float]:
    """Permute dataset labels across subjects, carrying windows together.

    Returns (observed_mmd2, p_value).
    """
    X0, X1, _, _ = _split_by_target(df, feature_cols)
    if len(X0) < 2 or len(X1) < 2:
        return float("nan"), float("nan")
    observed = mmd2_unbiased(X0, X1, sigma)

    subjects = df.drop_duplicates("subject_id")[["subject_id", "split_target"]].copy()
    subj_ids = subjects["subject_id"].values
    subj_target = subjects["split_target"].values.astype(int)

    if len(np.unique(subj_target)) < 2:
        return observed, float("nan")

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(subj_target)
        subj_to_perm = dict(zip(subj_ids, perm))
        perm_target = df["subject_id"].map(subj_to_perm).values.astype(int)
        Xp = _prep_features(df, feature_cols)
        stat = mmd2_unbiased(Xp[perm_target == 0], Xp[perm_target == 1], sigma)
        if np.isfinite(stat) and stat >= observed:
            hits += 1
    p = (1 + hits) / (1 + n_perm)
    return observed, p


# ---------------------------------------------------------------------------
# MMDAgg fallback (multi-bandwidth aggregation)
# ---------------------------------------------------------------------------

def mmdagg_holm(
    df: pd.DataFrame,
    feature_cols: list,
    scales: tuple = BANDWIDTH_SCALE_GRID,
    n_perm: int = N_PERM_DEFAULT,
    seed: int = 0,
) -> dict:
    """Run MMD at each bandwidth scale, Holm-correct across scales.

    If the `mmdagg` package is importable, prefer its `mmdagg` function; the
    manual grid is a fallback so the pipeline runs without JAX/Torch backends.
    """
    try:
        import mmdagg  # noqa: F401
        # If the user has mmdagg installed, they can call it directly with
        # their own controls. We still return the grid version so the report
        # is self-contained.
    except ImportError:
        pass

    X0, X1, _, _ = _split_by_target(df, feature_cols)
    if len(X0) < 2 or len(X1) < 2:
        return {"per_scale": {}, "min_holm_p": float("nan")}

    sigma_median = _median_bandwidth(np.vstack([X0, X1]))
    per_scale = {}
    p_values = []
    for s in scales:
        sigma = sigma_median * s
        obs, p = block_permutation_p(df, feature_cols, sigma,
                                     n_perm=n_perm, seed=seed)
        per_scale[float(s)] = {"sigma": float(sigma), "mmd2": obs, "p": p}
        if np.isfinite(p):
            p_values.append(p)

    # Holm-Bonferroni across bandwidths
    p_sorted = sorted(p_values)
    holm_p = min([p * (len(p_sorted) - i) for i, p in enumerate(p_sorted)] + [1.0]) \
             if p_sorted else float("nan")
    return {"per_scale": per_scale, "min_holm_p": float(holm_p),
            "sigma_median_heuristic": float(sigma_median)}


# ---------------------------------------------------------------------------
# Null distribution + heterogeneity ratio R
# ---------------------------------------------------------------------------

def null_mmd_distribution(null_df: pd.DataFrame, feature_cols: list,
                           n_reps_cap: int = 30, seed: int = 0) -> np.ndarray:
    """Compute observed MMD^2 (no permutation) for each null repetition."""
    if null_df.empty or "null_rep" not in null_df.columns:
        return np.array([])
    stats = []
    rng = np.random.default_rng(seed)
    reps = list(null_df.groupby("null_rep"))
    rng.shuffle(reps)
    for _, rep in reps[:n_reps_cap]:
        X0, X1, _, _ = _split_by_target(rep, feature_cols)
        if len(X0) < 2 or len(X1) < 2:
            continue
        sigma = _median_bandwidth(np.vstack([X0, X1]))
        s = mmd2_unbiased(X0, X1, sigma)
        if np.isfinite(s):
            stats.append(s)
    return np.asarray(stats)


def heterogeneity_ratio(mmd_obs: float, null_stats: np.ndarray) -> float:
    if not len(null_stats) or not np.isfinite(mmd_obs):
        return float("nan")
    p95 = np.percentile(null_stats, 95)
    if p95 <= 0:
        return float("nan")
    return float(mmd_obs / p95)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_step5(fast: bool = False) -> dict:
    from audit.step1_splits import build_sets
    from audit.step2_common_modality import restrict_to_ecg

    sets = build_sets()
    c1 = restrict_to_ecg(sets["C1"])
    if c1.empty:
        raise RuntimeError("C1 is empty; check Step 1 outputs.")

    feature_cols = [c for c in ECG_FEATURE_NAMES if c in c1.columns]
    n_perm = 200 if fast else N_PERM_DEFAULT
    n_reps = 8 if fast else 30

    # Primary test at median-heuristic sigma
    X0, X1, _, _ = _split_by_target(c1, feature_cols)
    sigma = _median_bandwidth(np.vstack([X0, X1]))
    mmd_obs, p_obs = block_permutation_p(c1, feature_cols, sigma,
                                          n_perm=n_perm, seed=0)

    # Multi-bandwidth (MMDAgg fallback)
    agg = mmdagg_holm(c1, feature_cols,
                      n_perm=max(200, n_perm // 5), seed=0)

    # Null floor
    n1_null = null_mmd_distribution(restrict_to_ecg(sets["N1"]),
                                     feature_cols, n_reps_cap=n_reps, seed=1)
    n2_null = null_mmd_distribution(restrict_to_ecg(sets["N2"]),
                                     feature_cols, n_reps_cap=n_reps, seed=2)
    pooled_null = np.concatenate([n1_null, n2_null]) if len(n1_null) or len(n2_null) \
                   else np.array([])
    R = heterogeneity_ratio(mmd_obs, pooled_null)

    manifest = {
        "primary": {
            "sigma_median_heuristic": float(sigma),
            "mmd2_observed": mmd_obs,
            "p_block_permutation": p_obs,
            "n_permutations": n_perm,
        },
        "mmdagg_fallback": agg,
        "null_floor": {
            "n1_stats": [float(x) for x in n1_null],
            "n2_stats": [float(x) for x in n2_null],
            "p95_pooled": float(np.percentile(pooled_null, 95))
                          if len(pooled_null) else float("nan"),
        },
        "heterogeneity_ratio_R": R,
    }
    manifest["verdict"] = _verdict(mmd_obs, p_obs, R)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step5_mmd.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Audit Step 5 — MMD (F_ECG, C1 normal-vs-normal) ===")
    print(f"  sigma (median heuristic): {sigma:.4f}")
    print(f"  observed MMD^2:           {mmd_obs:.5f}")
    print(f"  block-perm p:             {p_obs:.4f}  (n_perm={n_perm})")
    print(f"  MMDAgg min-Holm p:        {agg['min_holm_p']:.4f}")
    print(f"  null p95 (pooled N1/N2):  {manifest['null_floor']['p95_pooled']:.5f}")
    print(f"  heterogeneity ratio R:    {R:.3f}")
    print(f"  verdict: {manifest['verdict']}")
    return manifest


def _verdict(mmd_obs, p_val, R) -> str:
    if not np.isfinite(mmd_obs) or not np.isfinite(p_val):
        return "insufficient_data"
    if p_val >= 0.05:
        return "not_detected"
    if np.isfinite(R) and R <= 1.0:
        return "significant_but_within_null_heterogeneity"
    return "significant_beyond_null_heterogeneity"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    run_step5(fast=args.fast)
