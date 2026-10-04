"""
Locked comparison pipeline — resolves the three-number discrepancy.

Root causes of the prior inconsistency:
  1. Different model classes (ModalityInvariantNet vs AblationNet) consumed the
     global RNG state at different points, giving different random initializations
     for what should be the same configuration.
  2. p=0.5 was never actually included in the rate sweep — the outline
     interpolated between 0.4 and 0.6, which is not a valid comparison.

Fix: single model class (CanonicalNet, functionally equivalent to both predecessors
when ECG is always present), per-fold model seed reset so every condition in a
given fold starts from the same initialization, and p=0.5 explicitly included.

Also includes:
  - LightGBM baseline ECE ratio on the same fold splits (LightGBM is deterministic
    at random_state=42; no stochastic initialization to lock, but using identical
    folds makes the comparison apples-to-apples)
  - Locked H-C1 check using CanonicalNet with seed-reset, to verify the
    interaction coefficient is not sensitive to the same initialization-variance
    issue that affected H-B1
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from feature_extraction import ECG_FEATURE_NAMES, EDA_FEATURE_NAMES
from train_stage3_models import (
    _apply_simulated_eda_mask,
    _load_dataset,
    _standardize_pair,
    _subject_of,
    expected_calibration_error,
)

RNG_SEED = 42
MODEL_SEED_BASE = 200   # separate from training RNG to avoid crosstalk
MASK_FRACTION = 0.5
N_FOLDS = 5
N_EPOCHS = 100
LR = 1e-2


# ---------------------------------------------------------------------------
# Canonical model: mathematically equivalent to ModalityInvariantNet when
# ECG is always present (ecg_gate = 1.0 throughout), extends cleanly to the
# no-dropout and rate-sweep conditions.
# ---------------------------------------------------------------------------

class CanonicalNet(nn.Module):
    def __init__(self, embed_dim=16, dropout_p=0.5, drop_eda=True, drop_ecg=False):
        super().__init__()
        self.dropout_p = dropout_p
        self.drop_eda = drop_eda
        self.drop_ecg = drop_ecg
        n_ecg, n_eda = len(ECG_FEATURE_NAMES), len(EDA_FEATURE_NAMES)
        self.ecg_branch = nn.Sequential(
            nn.Linear(n_ecg, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.eda_branch = nn.Sequential(
            nn.Linear(n_eda, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, ecg_x, eda_x, eda_mask, ecg_mask=None, force_no_dropout=False):
        B = ecg_x.shape[0]
        eda_gate = eda_mask.float().unsqueeze(1)
        ecg_gate = (ecg_mask.float().unsqueeze(1) if ecg_mask is not None
                    else torch.ones(B, 1))
        if self.training and not force_no_dropout:
            if self.drop_eda:
                eda_gate = eda_gate * (torch.rand(B, 1) > self.dropout_p).float()
            if self.drop_ecg:
                ecg_gate = ecg_gate * (torch.rand(B, 1) > self.dropout_p).float()
        combined = ecg_gate * self.ecg_branch(ecg_x) + eda_gate * self.eda_branch(eda_x)
        return self.head(combined).squeeze(-1)


def _to_tensors(df):
    has_ecg = df["has_ecg"].values if "has_ecg" in df.columns else np.ones(len(df), dtype=bool)
    return (
        torch.tensor(df[ECG_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32),
        torch.tensor(df[EDA_FEATURE_NAMES].fillna(0.0).values, dtype=torch.float32),
        torch.tensor(df["has_eda"].values, dtype=torch.bool),
        torch.tensor(has_ecg, dtype=torch.bool),
        torch.tensor(df["label"].values, dtype=torch.float32),
    )


def _apply_simulated_ecg_mask(df, fraction=MASK_FRACTION, seed=RNG_SEED):
    """Simulate ECG-missing windows: zero ECG features for `fraction` of rows."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    mask = np.zeros(len(df), dtype=bool)
    for lbl in df["label"].unique():
        idx = df.index[df["label"] == lbl]
        n = int(round(len(idx) * fraction))
        mask[df.index.isin(rng.choice(idx, size=n, replace=False))] = True
    df["has_ecg"] = ~mask
    df.loc[mask, ECG_FEATURE_NAMES] = float("nan")
    return df


def _train_eval(train_df, test_df, dropout_p, drop_eda, model_seed,
                drop_ecg=False, missing_col="has_eda"):
    """
    Train CanonicalNet with a fixed model seed, evaluate on test tiers split by
    missing_col. The model seed is reset immediately before weight initialization
    so every condition within a fold starts from an identical parameter state.
    """
    train_df = train_df.copy()
    if "has_ecg" not in train_df.columns:
        train_df["has_ecg"] = True

    train_s, test_s = _standardize_pair(train_df, test_df)

    torch.manual_seed(model_seed)
    model = CanonicalNet(dropout_p=dropout_p, drop_eda=drop_eda, drop_ecg=drop_ecg)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    ecg_x, eda_x, eda_m, ecg_m, y_tr = _to_tensors(train_s)
    torch.manual_seed(model_seed + 1)
    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        loss_fn(model(ecg_x, eda_x, eda_m, ecg_m), y_tr).backward()
        opt.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for tier, tier_df in [("complete", test_s[test_s[missing_col]]),
                               ("reduced",  test_s[~test_s[missing_col]])]:
            if tier_df["label"].nunique() < 2 or len(tier_df) < 2:
                out[tier] = None
                continue
            t_ecg, t_eda, t_dm, t_em, t_y = _to_tensors(tier_df)
            probs = torch.sigmoid(
                model(t_ecg, t_eda, t_dm, t_em, force_no_dropout=True)
            ).numpy()
            out[tier] = {
                "auroc": float(roc_auc_score(t_y.numpy(), probs)),
                "ece": expected_calibration_error(t_y.numpy(), probs),
                "n": len(tier_df),
            }
    return out


def _agg_folds(fold_list):
    out = {}
    for tier in ("complete", "reduced"):
        vals = [f[tier] for f in fold_list if f.get(tier)]
        if not vals:
            out[tier] = None
            continue
        eces = [v["ece"] for v in vals]
        out[tier] = {
            "ece_mean": float(np.mean(eces)),
            "ece_std":  float(np.std(eces, ddof=1)) if len(eces) > 1 else 0.0,
            "n_folds":  len(eces),
        }
    return out


def _ci95_ratio(fold_list):
    """95% CI on ECE ratio using t-distribution over per-fold ratios."""
    ratios = []
    for f in fold_list:
        if f.get("complete") and f.get("reduced") and f["complete"]["ece"] > 0:
            ratios.append(f["reduced"]["ece"] / f["complete"]["ece"])
    if len(ratios) < 2:
        return ratios, float("nan"), float("nan"), float("nan")
    mean = float(np.mean(ratios))
    se = float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
    t_crit = float(stats.t.ppf(0.975, df=len(ratios) - 1))
    return ratios, mean, mean - t_crit * se, mean + t_crit * se


def _load_wesad():
    """Frozen, hash-checked WESAD baseline-vs-stress table.

    Previously `_load_dataset()` with no WESAD restriction; on the 2026-09-30
    cache that returns 1,161 rows including 727 MIT-BIH windows with string
    labels (peer_review_round3.md C4). Lazy import: eval_core imports this module.
    """
    from eval_core import load_wesad_frozen
    return load_wesad_frozen()


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_locked_comparison():
    df = _load_wesad()
    subjects = _subject_of(df["subject_id"]).values
    n_unique = len(np.unique(subjects))
    n_folds = min(N_FOLDS, n_unique)
    if n_folds < N_FOLDS:
        print(f"[note] {n_unique} subjects — n_folds reduced to {n_folds}")

    # All conditions run in the same fold loop
    rates = [0.2, 0.4, 0.5, 0.6]
    conditions = {
        "dropout_p0.5": dict(dropout_p=0.5, drop_eda=True),
        "no_dropout":   dict(dropout_p=0.5, drop_eda=False),
        **{f"rate_{r}": dict(dropout_p=r, drop_eda=True) for r in rates},
    }
    cond_keys = list(conditions.keys())
    fold_results = {k: [] for k in cond_keys}

    gkf = GroupKFold(n_splits=n_folds)
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=subjects)):
        train_df = df.iloc[train_idx].copy()
        test_df = _apply_simulated_eda_mask(df.iloc[test_idx].copy(), seed=RNG_SEED + fold_i)
        if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
            print(f"  fold {fold_i}: skipped (single-class)")
            continue

        # Same model_seed for all conditions in this fold — the ONLY thing that
        # varies across conditions is dropout_p and drop_eda.
        model_seed = MODEL_SEED_BASE + fold_i

        for cond_name, params in conditions.items():
            res = _train_eval(train_df, test_df, model_seed=model_seed, **params)
            fold_results[cond_name].append(res)
            c = res.get("complete")
            r = res.get("reduced")
            ratio = f"{r['ece']/c['ece']:.2f}x" if (c and r and c["ece"] > 0) else "N/A"
            c_str = f"{c['ece']:.4f}" if c else "N/A"
            r_str = f"{r['ece']:.4f}" if r else "N/A"
            print(f"  fold {fold_i}  {cond_name:<16}  "
                  f"ECE_complete={c_str}  ECE_reduced={r_str}  ratio={ratio}")

    # --- Summary table ---
    print("\n=== Locked comparison: canonical ECE ratios with 95% CI ===")
    print(f"  {'Condition':<16}  {'ECE_complete':>14}  {'ECE_reduced':>14}  "
          f"{'Ratio (mean)':>13}  {'95% CI':>16}  folds")

    canonical_ratios = {}  # store per-fold ratio lists for paired test
    for cond_name in cond_keys:
        agg = _agg_folds(fold_results[cond_name])
        ratios, ratio_mean, ci_lo, ci_hi = _ci95_ratio(fold_results[cond_name])
        canonical_ratios[cond_name] = ratios
        c, r = agg.get("complete"), agg.get("reduced")
        if c and r:
            print(f"  {cond_name:<16}  {c['ece_mean']:.4f} ± {c['ece_std']:.4f}    "
                  f"{r['ece_mean']:.4f} ± {r['ece_std']:.4f}    "
                  f"{ratio_mean:.3f}x         [{ci_lo:.2f}, {ci_hi:.2f}]    {c['n_folds']}")
        else:
            print(f"  {cond_name:<16}  insufficient data")

    # --- Paired test: dropout_p0.5 vs no_dropout ---
    print("\n=== Paired test: dropout (p=0.5) vs no_dropout ===")
    d_ratios = canonical_ratios.get("dropout_p0.5", [])
    n_ratios = canonical_ratios.get("no_dropout", [])

    if len(d_ratios) == len(n_ratios) and len(d_ratios) >= 2:
        deltas = np.array(d_ratios) - np.array(n_ratios)
        t_stat, p_val = stats.ttest_rel(d_ratios, n_ratios)
        try:
            w_stat, w_pval = stats.wilcoxon(d_ratios, n_ratios)
            w_str = f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_pval:.4f}"
        except Exception:
            w_str = "  Wilcoxon: insufficient data (need at least 1 non-zero difference)"

        print(f"  Per-fold ECE ratios — dropout p=0.5: {[f'{x:.3f}' for x in d_ratios]}")
        print(f"  Per-fold ECE ratios — no_dropout:    {[f'{x:.3f}' for x in n_ratios]}")
        print(f"  Per-fold delta (dropout - no_dropout): {[f'{x:+.3f}' for x in deltas]}")
        print(f"  Mean delta: {np.mean(deltas):+.4f}  SD: {np.std(deltas, ddof=1):.4f}")
        print(f"  Paired t-test (n={len(d_ratios)} folds): t={t_stat:.3f}, p={p_val:.4f}")
        print(w_str)
        print()
        if p_val < 0.05:
            print("  FINDING: Dropout training adds a statistically distinguishable effect")
            print(f"  at alpha=0.05 (p={p_val:.4f}). Architecture + dropout both contribute.")
        else:
            print("  FINDING: No statistically distinguishable difference between dropout")
            print(f"  and no-dropout at alpha=0.05 (p={p_val:.4f}, n={len(d_ratios)} folds).")
            print("  The gated architecture accounts for the bulk of calibration robustness;")
            print("  dropout training adds an increment that cannot be resolved at this sample size.")
            print("  Framing implication: lead with architecture, not dropout, as the mechanism.")
    else:
        print(f"  Cannot pair: dropout has {len(d_ratios)} folds, no_dropout has {len(n_ratios)}")

    # --- Rate sweep summary ---
    print("\n=== Rate sweep (EDA-only dropout, locked seeds) ===")
    print(f"  {'Rate':>6}  {'Ratio':>8}  {'95% CI':>16}")
    for r in rates:
        fold_list = fold_results.get(f"rate_{r}", [])
        ratios, mean, lo, hi = _ci95_ratio(fold_list)
        print(f"  p={r:.1f}    {mean:.3f}x    [{lo:.2f}, {hi:.2f}]")

    return fold_results, canonical_ratios


# ---------------------------------------------------------------------------
# LightGBM baseline on the same locked folds
# ---------------------------------------------------------------------------

def run_locked_lgbm():
    """
    Compute LightGBM baseline ECE ratio on the same GroupKFold splits used by
    run_locked_comparison(). LightGBM is deterministic at random_state=42 given
    identical training data, so no model-seed locking is required — the only
    thing to control is that the folds match.
    """
    import lightgbm as lgb

    df = _load_wesad()
    subjects = _subject_of(df["subject_id"]).values
    n_folds = min(N_FOLDS, len(np.unique(subjects)))

    fold_results = []
    gkf = GroupKFold(n_splits=n_folds)
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=subjects)):
        train_df = df.iloc[train_idx].copy()
        test_df = _apply_simulated_eda_mask(df.iloc[test_idx].copy(), seed=RNG_SEED + fold_i)
        if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
            continue

        from feature_extraction import ALL_FEATURE_NAMES
        train_complete = train_df[train_df["has_eda"]]
        if len(train_complete) < 4:
            continue
        model = lgb.LGBMClassifier(n_estimators=50, max_depth=3,
                                    random_state=RNG_SEED, verbosity=-1)
        model.fit(train_complete[ALL_FEATURE_NAMES], train_complete["label"])

        res = {}
        for tier, tier_df in [("complete", test_df[test_df["has_eda"]]),
                               ("reduced",  test_df[~test_df["has_eda"]])]:
            if tier_df["label"].nunique() < 2 or len(tier_df) < 2:
                res[tier] = None
                continue
            # No fillna — LightGBM handles NaN natively (matching train_baseline behavior)
            probs = model.predict_proba(tier_df[ALL_FEATURE_NAMES])[:, 1]
            res[tier] = {
                "ece": expected_calibration_error(tier_df["label"].values, probs),
                "n": len(tier_df),
            }
        fold_results.append(res)
        c = res.get("complete")
        r = res.get("reduced")
        ratio = f"{r['ece']/c['ece']:.2f}x" if (c and r and c["ece"] > 0) else "N/A"
        c_str = f"{c['ece']:.4f}" if c else "N/A"
        r_str = f"{r['ece']:.4f}" if r else "N/A"
        print(f"  fold {fold_i}  lgbm_baseline  ECE_complete={c_str}  "
              f"ECE_reduced={r_str}  ratio={ratio}")

    ratios, mean, lo, hi = _ci95_ratio(fold_results)
    agg = _agg_folds(fold_results)
    c, r = agg.get("complete"), agg.get("reduced")
    print(f"\n  LightGBM baseline (locked folds):")
    if c and r:
        print(f"  ECE_complete: {c['ece_mean']:.4f} +/- {c['ece_std']:.4f}")
        print(f"  ECE_reduced:  {r['ece_mean']:.4f} +/- {r['ece_std']:.4f}")
        print(f"  Ratio: {mean:.3f}x  95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"  Per-fold ratios: {[f'{x:.3f}' for x in ratios]}")
    return fold_results


# ---------------------------------------------------------------------------
# Locked H-C1: verify interaction coefficient stability
# ---------------------------------------------------------------------------

def run_locked_hc1():
    """
    H-C1 mixed-effects check using CanonicalNet with per-fold seed reset.
    Outcome: whether the tier x model_type interaction coefficient changes
    materially when initialization variance is controlled. Reports both the
    locked and the direction of any shift for the outline.
    """
    try:
        import statsmodels.formula.api as smf
        import pandas as pd
    except ImportError:
        print("  [skip] statsmodels not installed")
        return None

    from feature_extraction import ALL_FEATURE_NAMES

    df = _load_wesad()
    subjects = _subject_of(df["subject_id"]).values
    n_folds = min(N_FOLDS, len(np.unique(subjects)))
    gkf = GroupKFold(n_splits=n_folds)

    import lightgbm as lgb
    rows = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=subjects)):
        train_df = df.iloc[train_idx].copy()
        test_df = _apply_simulated_eda_mask(df.iloc[test_idx].copy(), seed=RNG_SEED + fold_i)
        if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
            continue

        # LightGBM baseline (deterministic)
        train_complete = train_df[train_df["has_eda"]]
        b_model = lgb.LGBMClassifier(n_estimators=50, max_depth=3,
                                      random_state=RNG_SEED, verbosity=-1)
        b_model.fit(train_complete[ALL_FEATURE_NAMES], train_complete["label"])

        # CanonicalNet invariant (seed-locked)
        train_s, test_s = _standardize_pair(train_df, test_df)
        torch.manual_seed(MODEL_SEED_BASE + fold_i)
        i_model = CanonicalNet(dropout_p=0.5, drop_eda=True)
        opt = torch.optim.Adam(i_model.parameters(), lr=LR)
        loss_fn = nn.BCEWithLogitsLoss()
        ecg_x, eda_x, eda_m, y_tr = _to_tensors(train_s)
        torch.manual_seed(MODEL_SEED_BASE + fold_i + 1)
        i_model.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad()
            loss_fn(i_model(ecg_x, eda_x, eda_m), y_tr).backward()
            opt.step()
        i_model.eval()

        for tier_name, tier_df, tier_df_s in [
            ("complete", test_df[test_df["has_eda"]], test_s[test_s["has_eda"]]),
            ("reduced",  test_df[~test_df["has_eda"]], test_s[~test_s["has_eda"]]),
        ]:
            if len(tier_df) < 1:
                continue
            y = tier_df["label"].values
            subj = _subject_of(tier_df["subject_id"]).values
            b_probs = b_model.predict_proba(tier_df[ALL_FEATURE_NAMES].fillna(0.0))[:, 1]
            with torch.no_grad():
                t_ecg, t_eda, t_dm, _ = _to_tensors(tier_df_s)
                i_probs = torch.sigmoid(
                    i_model(t_ecg, t_eda, t_dm, force_no_dropout=True)
                ).numpy()
            eps = 1e-6
            b_ll = -(y * np.log(np.clip(b_probs, eps, 1-eps)) +
                     (1-y) * np.log(np.clip(1-b_probs, eps, 1-eps)))
            i_ll = -(y * np.log(np.clip(i_probs, eps, 1-eps)) +
                     (1-y) * np.log(np.clip(1-i_probs, eps, 1-eps)))
            for k in range(len(y)):
                rows += [
                    {"subject_id": subj[k], "tier": tier_name,
                     "model_type": "baseline",  "log_loss": float(b_ll[k])},
                    {"subject_id": subj[k], "tier": tier_name,
                     "model_type": "invariant", "log_loss": float(i_ll[k])},
                ]

    import pandas as pd
    long_df = pd.DataFrame(rows)
    print(f"  H-C1 locked: {len(long_df)} obs, {long_df.subject_id.nunique()} subjects")
    print(long_df.groupby(["tier", "model_type"])["log_loss"].agg(["mean", "std", "count"]))

    md = smf.mixedlm("log_loss ~ tier * model_type",
                     data=long_df, groups=long_df["subject_id"]).fit()
    key = "tier[T.reduced]:model_type[T.invariant]"
    if key not in md.params.index:
        print(f"  [note] interaction term not found; parameterization: {list(md.params.index)}")
        return None
    coef = float(md.params[key])
    se = float(md.bse[key])
    pval = float(md.pvalues[key])
    print(f"\n  Locked H-C1 interaction (tier[reduced]:model_type[invariant]):")
    print(f"  coef {coef:+.4f}  SE {se:.4f}  p={pval:.4g}")
    print(f"  Direction consistent with H-C1 (coef < 0): {coef < 0}")
    print(f"  Significant at alpha=0.05: {coef < 0 and pval < 0.05}")
    return {"coef": coef, "se": se, "p": pval}


# ---------------------------------------------------------------------------
# Locked per-modality ablation: EDA-only vs ECG-only (replaces unlocked Ablation 1)
# ---------------------------------------------------------------------------

def run_locked_per_modality():
    """
    Locked version of the EDA-only vs ECG-only ablation from 5.2b Ablation 1.
    Both conditions use the same fold splits, same model initialization per fold.
    ECG-only condition: ECG branch randomly zeroed at train time; ECG simulated-
    missing at test time (symmetric to EDA treatment).
    EDA-only condition: standard headline model, repeated here for direct comparison.
    """
    df = _load_wesad()
    subjects = _subject_of(df["subject_id"]).values
    n_folds = min(N_FOLDS, len(np.unique(subjects)))

    fold_results = {"eda_only": [], "ecg_only": []}
    gkf = GroupKFold(n_splits=n_folds)

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=subjects)):
        train_df = df.iloc[train_idx].copy()
        raw_test  = df.iloc[test_idx].copy()
        if train_df["label"].nunique() < 2 or raw_test["label"].nunique() < 2:
            print(f"  fold {fold_i}: skipped (single-class)")
            continue

        test_eda = _apply_simulated_eda_mask(raw_test, seed=RNG_SEED + fold_i)
        test_ecg = _apply_simulated_ecg_mask(raw_test, seed=RNG_SEED + fold_i)
        model_seed = MODEL_SEED_BASE + fold_i

        res_eda = _train_eval(train_df, test_eda, dropout_p=0.5, drop_eda=True,
                              drop_ecg=False, missing_col="has_eda", model_seed=model_seed)
        res_ecg = _train_eval(train_df, test_ecg, dropout_p=0.5, drop_eda=False,
                              drop_ecg=True, missing_col="has_ecg", model_seed=model_seed)
        fold_results["eda_only"].append(res_eda)
        fold_results["ecg_only"].append(res_ecg)

        for name, res in [("eda_only", res_eda), ("ecg_only", res_ecg)]:
            c, r = res.get("complete"), res.get("reduced")
            ratio = f"{r['ece']/c['ece']:.2f}x" if (c and r and c["ece"] > 0) else "N/A"
            c_str = f"{c['ece']:.4f}" if c else "N/A"
            r_str = f"{r['ece']:.4f}" if r else "N/A"
            print(f"  fold {fold_i}  {name:<10}  ECE_complete={c_str}  "
                  f"ECE_reduced={r_str}  ratio={ratio}")

    print("\n=== Locked per-modality ablation (EDA-only vs ECG-only) ===")
    print(f"  {'Condition':<12}  {'Ratio':>8}  {'95% CI':>16}  "
          f"  Interpretation")
    results_out = {}
    for name in ("eda_only", "ecg_only"):
        ratios, mean, lo, hi = _ci95_ratio(fold_results[name])
        results_out[name] = {"ratios": ratios, "mean": mean, "ci": (lo, hi)}
        interp = ("train for right missing modality" if name == "eda_only"
                  else "train for wrong missing modality")
        print(f"  {name:<12}  {mean:.3f}x    [{lo:.2f}, {hi:.2f}]    {interp}")
        print(f"    per-fold: {[f'{x:.3f}' for x in ratios]}")

    # Paired test
    eda_r = results_out["eda_only"]["ratios"]
    ecg_r = results_out["ecg_only"]["ratios"]
    if len(eda_r) == len(ecg_r) >= 2:
        deltas = np.array(eda_r) - np.array(ecg_r)
        t_stat, p_val = stats.ttest_rel(eda_r, ecg_r)
        try:
            w_stat, w_pval = stats.wilcoxon(eda_r, ecg_r)
            w_str = f"  Wilcoxon: W={w_stat:.1f}, p={w_pval:.4f}"
        except Exception:
            w_str = "  Wilcoxon: insufficient non-zero differences"
        print(f"\n  Per-fold delta (eda_only - ecg_only): {[f'{x:+.3f}' for x in deltas]}")
        print(f"  Mean delta: {np.mean(deltas):+.4f}  SD: {np.std(deltas, ddof=1):.4f}")
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(w_str)
        if all(d < 0 for d in deltas):
            print("  EDA-only ratio < ECG-only ratio in every fold (correct-modality advantage "
                  "is directionally consistent).")
    return results_out


if __name__ == "__main__":
    run_locked_comparison()

    print("\n" + "=" * 78)
    print("LightGBM baseline on locked folds")
    print("=" * 78)
    run_locked_lgbm()

    print("\n" + "=" * 78)
    print("Locked H-C1 check")
    print("=" * 78)
    run_locked_hc1()

    print("\n" + "=" * 78)
    print("Locked per-modality ablation: EDA-only vs ECG-only")
    print("=" * 78)
    run_locked_per_modality()
