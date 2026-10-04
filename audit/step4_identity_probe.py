"""
Audit Step 4 — dataset-identity probe on C1 (normal-vs-normal).

Four probes triangulate:
  P1 exact-production mirror   — LightGBM at Preprint 1's hyperparameters
  P2 high-capacity LightGBM    — small grid, upper bound on separability
  P3 L2 logistic regression    — robust-scaled; tests linear separability
  P4 univariate screens        — per-feature subject-level AUC (+ LGBM stump)

Cross-validation:
  StratifiedGroupKFold(n_splits=5) with groups=subject_id, stratified on
  dataset. Repeated with N_SEEDS different shuffle seeds.

Metrics:
  Primary: subject-level ROC AUC (mean out-of-fold predicted probability
           per subject, then AUC across subjects).
  95% CI: stratified cluster bootstrap over subjects (2000 replicates).
  p-value: SUBJECT-LEVEL permutation (hand-rolled — sklearn's
           permutation_test_score permutes within groups, which is useless
           when the label is subject-constant; see the protocol doc).

Null calibration: comparable AUC on N1/N2 (built by audit/step1_splits.py).
The 95th percentile of null AUCs is the "subject-heterogeneity floor" the
observed AUC must clear.

Run:
    python -m audit.step4_identity_probe
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import ALL_FEATURE_NAMES, ECG_FEATURE_NAMES

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "audit"

N_FOLDS = 5
N_SEEDS = 20
N_BOOT = 2000
N_PERM = 1000


# ---------------------------------------------------------------------------
# Feature restriction (Step 2 shortcut: F_ECG only)
# ---------------------------------------------------------------------------

def restrict_to_ecg(df: pd.DataFrame) -> pd.DataFrame:
    """Drop EDA-derived features so identity separability isn't trivially
    driven by MIT-BIH's structural NaN pattern (Step 0's Driver #1)."""
    keep_cols = [c for c in df.columns if c not in
                 set(ALL_FEATURE_NAMES) - set(ECG_FEATURE_NAMES)]
    return df[keep_cols].copy()


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _fit_p1(X_tr, y_tr):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=50, max_depth=3,
                            random_state=0, verbosity=-1, is_unbalance=True)
    m.fit(X_tr, y_tr)
    return m


def _fit_p2(X_tr, y_tr):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=15, min_child_samples=20,
        learning_rate=0.05, feature_fraction=0.8,
        random_state=0, verbosity=-1, is_unbalance=True,
    )
    m.fit(X_tr, y_tr)
    return m


def _fit_p3(X_tr, y_tr):
    scaler = RobustScaler().fit(X_tr)
    Xs = scaler.transform(X_tr)
    m = LogisticRegression(C=0.1, max_iter=1000,
                           class_weight="balanced", solver="lbfgs")
    m.fit(np.nan_to_num(Xs, nan=0.0), y_tr)
    return (scaler, m)


def _predict(model, X):
    if isinstance(model, tuple):
        scaler, m = model
        Xs = np.nan_to_num(scaler.transform(X), nan=0.0)
        return m.predict_proba(Xs)[:, 1]
    return model.predict_proba(X)[:, 1]


PROBES = {"P1": _fit_p1, "P2": _fit_p2, "P3": _fit_p3}


# ---------------------------------------------------------------------------
# Cross-validated subject-level scoring
# ---------------------------------------------------------------------------

def _feature_matrix(df):
    """Use only ECG features (F_ECG); LightGBM handles NaN natively.

    Downstream stages (Step 6 H6/H7) may drop columns, so intersect with
    what's actually in the frame rather than requesting the full list.
    """
    cols = [c for c in ECG_FEATURE_NAMES if c in df.columns]
    return df[cols].astype(float)


def cross_val_subject_scores(
    df: pd.DataFrame,
    probe: str,
    n_folds: int = N_FOLDS,
    n_seeds: int = N_SEEDS,
) -> pd.DataFrame:
    """Return subject-level mean predicted probabilities across all seeds."""
    fit = PROBES[probe]
    y = df["split_target"].values.astype(int)
    groups = df["subject_id"].values
    X = _feature_matrix(df)
    stratify_labels = df["dataset"].astype(str) + "|" + df["split_target"].astype(str)

    subj_probs = {s: [] for s in np.unique(groups)}

    unique_groups = np.unique(groups)
    if len(unique_groups) < n_folds:
        n_folds = max(2, len(unique_groups))

    for seed in range(n_seeds):
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=seed,
            )
        except TypeError:
            splitter = StratifiedGroupKFold(n_splits=n_folds)
        for train_idx, test_idx in splitter.split(X, stratify_labels, groups=groups):
            try:
                model = fit(X.iloc[train_idx], y[train_idx])
            except Exception as e:
                warnings.warn(f"[{probe}] fit failed seed={seed}: {e!r}")
                continue
            probs = _predict(model, X.iloc[test_idx])
            for s, p in zip(groups[test_idx], probs):
                subj_probs[s].append(float(p))

    return pd.DataFrame([
        {"subject_id": s, "n_scores": len(v),
         "mean_prob": float(np.mean(v)) if v else float("nan")}
        for s, v in subj_probs.items()
    ])


# ---------------------------------------------------------------------------
# Subject-level AUC, cluster bootstrap CI, subject-level permutation p
# ---------------------------------------------------------------------------

def subject_level_auc(subject_scores: pd.DataFrame, subject_dataset: dict):
    """AUC across subjects, target = dataset membership (1 for MITBIH)."""
    y, s = [], []
    for _, row in subject_scores.iterrows():
        ds = subject_dataset.get(row["subject_id"])
        if ds is None or not np.isfinite(row["mean_prob"]):
            continue
        y.append(1 if ds == "MITBIH" else 0)
        s.append(row["mean_prob"])
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2:
        return float("nan"), (y, s)
    return float(roc_auc_score(y, s)), (y, s)


def cluster_bootstrap_ci(y, s, dataset_labels, n_boot=N_BOOT, seed=0):
    """Resample subjects with replacement within each dataset; percentile CI."""
    rng = np.random.default_rng(seed)
    aucs = []
    wesad_idx = np.where(dataset_labels == "WESAD")[0]
    mitbih_idx = np.where(dataset_labels == "MITBIH")[0]
    if len(wesad_idx) < 2 or len(mitbih_idx) < 2:
        return (float("nan"), float("nan"))
    for _ in range(n_boot):
        wsel = rng.choice(wesad_idx, size=len(wesad_idx), replace=True)
        msel = rng.choice(mitbih_idx, size=len(mitbih_idx), replace=True)
        sel = np.concatenate([wsel, msel])
        y_b, s_b = y[sel], s[sel]
        if len(set(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, s_b))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def subject_permutation_p(
    df: pd.DataFrame, probe: str, observed_auc: float,
    n_perm: int = N_PERM, n_folds: int = N_FOLDS, seed: int = 0,
) -> float:
    """Permute the dataset label ACROSS SUBJECTS, then re-run CV.

    sklearn's permutation_test_score permutes within groups, which is a
    no-op when the target is subject-constant. Per the audit doc: write the
    permutation loop yourself. This is exactly that loop.
    """
    rng = np.random.default_rng(seed)
    subjects = df.drop_duplicates("subject_id")[["subject_id", "dataset"]].copy()
    subj_to_ds = dict(zip(subjects["subject_id"], subjects["dataset"]))
    hits = 0
    for r in range(n_perm):
        perm = rng.permutation(subjects["dataset"].values)
        subj_to_perm = dict(zip(subjects["subject_id"], perm))
        df_perm = df.copy()
        df_perm["split_target"] = df_perm["subject_id"].map(
            lambda s: 1 if subj_to_perm[s] == "MITBIH" else 0
        )
        scores = cross_val_subject_scores(df_perm, probe, n_folds=n_folds, n_seeds=1)
        auc_perm, _ = subject_level_auc(scores, subj_to_perm)
        if np.isfinite(auc_perm) and auc_perm >= observed_auc:
            hits += 1
    return (1 + hits) / (1 + n_perm)


def null_auc_distribution(null_df: pd.DataFrame, probe: str,
                          n_reps_cap: int = 20, seed: int = 0):
    """AUCs across the null-split repetitions (N1 / N2)."""
    if null_df.empty or "null_rep" not in null_df.columns:
        return np.array([])
    aucs = []
    for rep_id, rep in null_df.groupby("null_rep"):
        if len(aucs) >= n_reps_cap:
            break
        subj_ds = dict(zip(
            rep["subject_id"], rep["split_target"].map({1: "MITBIH", 0: "WESAD"})
        ))
        try:
            scores = cross_val_subject_scores(rep, probe, n_folds=3, n_seeds=1)
            auc, _ = subject_level_auc(scores, subj_ds)
            if np.isfinite(auc):
                aucs.append(auc)
        except Exception:
            continue
    return np.asarray(aucs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_step4(fast: bool = False) -> dict:
    from audit.step1_splits import build_sets

    sets = build_sets()
    c1 = sets["C1"]
    if c1.empty:
        raise RuntimeError("C1 is empty; check real_data_loaders + Step 1 output.")

    subject_ds = dict(zip(c1["subject_id"],
                          np.where(c1["dataset"] == "MITBIH", "MITBIH", "WESAD")))

    n_seeds = 3 if fast else N_SEEDS
    n_perm = 100 if fast else N_PERM
    n_boot = 500 if fast else N_BOOT

    manifest = {"C1": {}, "null_floor": {}}

    for probe in ("P1", "P2", "P3"):
        print(f"\n[step4] Probe {probe} — C1 cross-validated subject-level AUC")
        scores = cross_val_subject_scores(c1, probe, n_seeds=n_seeds)
        auc, (y, s) = subject_level_auc(scores, subject_ds)
        ds_labels = np.array([subject_ds[sid] for sid, mp in
                              zip(scores["subject_id"], scores["mean_prob"])
                              if np.isfinite(mp)])
        ci_lo, ci_hi = cluster_bootstrap_ci(y, s, ds_labels, n_boot=n_boot)
        p_val = subject_permutation_p(c1, probe, auc, n_perm=n_perm)

        manifest["C1"][probe] = {
            "subject_auc": auc,
            "ci95": [ci_lo, ci_hi],
            "p_value_subject_permutation": p_val,
            "n_subjects_scored": int(len(y)),
            "n_seeds": n_seeds,
            "n_boot": n_boot,
            "n_perm": n_perm,
        }
        print(f"  {probe} subject AUC = {auc:.3f}  95% CI [{ci_lo:.2f}, {ci_hi:.2f}]"
              f"  subj-perm p = {p_val:.4f}")

    # Null floors
    for null_name in ("N1", "N2"):
        null_df = sets.get(null_name, pd.DataFrame())
        aucs = null_auc_distribution(null_df, "P1",
                                     n_reps_cap=10 if fast else 30)
        if len(aucs):
            manifest["null_floor"][null_name] = {
                "n_reps": int(len(aucs)),
                "median": float(np.median(aucs)),
                "p95": float(np.percentile(aucs, 95)),
            }
            print(f"  Null {null_name}: median {np.median(aucs):.3f}, "
                  f"95th pct {np.percentile(aucs, 95):.3f} ({len(aucs)} reps)")

    manifest["verdict"] = _verdict_step4(manifest)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step4_identity.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[step4] Verdict: {manifest['verdict']['band']} "
          f"({manifest['verdict']['note']})")
    return manifest


def _verdict_step4(manifest: dict) -> dict:
    """Map primary AUC + CI to the operational bands from the pre-registration.

    Pre-registered rule (configs/preregistration.yaml, auc_bands):
      TRIVIALLY_SEPARABLE   iff cluster-bootstrap lower CI >= 0.80
      SUBSTANTIALLY_SEPARABLE iff lower CI >= 0.70
      BORDERLINE            iff lower CI >= 0.60
      NOT_SEPARABLE         iff upper CI <= 0.60

    Earlier versions compared the point estimate (`auc`) to the SUBSTANTIALLY
    and BORDERLINE thresholds instead of the lower CI bound; that was a
    code-vs-pre-registration mismatch flagged in reviewer MA7 and in the
    outline's Section 5.6 reconciliation box. Fixed here per the audit doc.
    """
    p1 = manifest["C1"].get("P1")
    if not p1:
        return {"band": "no_result", "note": "P1 did not run"}
    auc = p1["subject_auc"]
    lo, hi = p1["ci95"]
    n_floor = manifest.get("null_floor", {}).get("N1", {}).get("p95")

    band = "inconclusive"
    if lo >= 0.80:
        band = "TRIVIALLY_SEPARABLE"
    elif lo >= 0.70:
        band = "SUBSTANTIALLY_SEPARABLE"
    elif lo >= 0.60:
        band = "BORDERLINE"
    elif hi <= 0.60 and (n_floor is None or auc <= n_floor):
        band = "NOT_SEPARABLE"
    return {"band": band,
            "note": f"AUC={auc:.3f} CI=[{lo:.2f},{hi:.2f}] N1-floor={n_floor} "
                    f"(band derived from lower CI, per preregistration)"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="Reduce seeds/permutations for smoke testing.")
    args = ap.parse_args()
    run_step4(fast=args.fast)
