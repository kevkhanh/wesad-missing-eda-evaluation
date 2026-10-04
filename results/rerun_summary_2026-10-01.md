# Post-fix re-run: Section 5.2 (2026-10-01)

Fixes from `peer_review_round3.md` (C1–C4, M1, M2, §3) implemented in `eval_core.py`, `run_metrics.py`, `run_baselines.py` and `run_repeated_cv.py`. Pre-fix outputs are archived in `results/archive/*_prefix.*`.

**Setup**
- **Feature table:** frozen at SHA-256 `df483a7abae0…` (`results/feature_table_manifest.json`). It holds 434 WESAD windows (60 s, non-overlapping, 27–30 per subject, 35% stress) from 15 subjects.
- **Paired scoring:** every test window is scored both with and without EDA.
- **Calibration:** "raw" means uncalibrated model outputs. "Platt" means one calibrator fit on inner-CV out-of-fold predictions from training subjects and applied to both conditions. Baseline (iv) uses per-condition Platt by design.
- **Model training:** none of the models is tuned. LightGBM uses 50 trees, depth 3. The neural networks use full-batch Adam for 100 epochs at learning rate 1e-2.

Raw files:
- `results/baselines.json`, `results/predictions_{kfold5,loso}.csv`, `results/per_subject_{kfold5,loso}.csv`
- `results/repeated_cv.json`, `results/repeated_cv_per_subject.csv`
- Logs: `results/baselines_run.log`, `results/repeated_cv_full.log`, `results/locked_comparison_rerun.log`, `results/locked_per_modality_rerun.log`

---

## 1. Primary comparison: 20× repeated subject-level 5-fold CV (20 distinct partitions; reduced condition, raw)

All differences are A − B. For Brier, a negative value means A is better; for AUROC, a positive value means A is better. CIs are Bouckaert–Frank corrected 95% CIs.

| Model | Brier | AUROC | ECE_sweep | Log-loss |
|---|---|---|---|---|
| Gated + dropout p=0.5 | 0.148 | 0.892 | 0.126 | 0.555 |
| Non-gated MLP, same masking | 0.146 | 0.892 | 0.125 | 0.580 |
| (ii) LightGBM NaN-training p=0.5 | 0.146 | 0.882 | 0.123 | 0.498 |
| (iii) LightGBM ECG-only per config | 0.146 | 0.882 | 0.123 | 0.523 |
| (iv) Mean-impute + per-condition Platt | 0.161 | 0.886 | 0.141 | 0.567 |
| HR-only | 0.179 | 0.854 | 0.138 | 0.606 |
| Gated, no dropout | 0.186 | 0.877 | 0.170 | 0.980 |

| Comparison | ΔBrier [95% CI] | ΔAUROC [95% CI] | Subjects where A is better (sign-flip p) |
|---|---|---|---|
| Gated vs (ii) | +0.002 [−0.031, +0.035] | +0.010 [−0.058, +0.077] | 7/15 (0.80) |
| Gated vs (iii) | +0.002 [−0.033, +0.037] | +0.010 [−0.062, +0.082] | 6/15 (0.83) |
| Gated vs (iv) | −0.013 [−0.073, +0.048] | +0.006 [−0.065, +0.076] | 6/15 (0.71) |
| Gated vs HR-only | −0.030 [−0.077, +0.016] | +0.038 [−0.039, +0.115] | 8/15 (0.20) |
| Gated vs no-dropout | −0.038 [−0.089, +0.014] | +0.015 [−0.036, +0.065] | 11/15 (0.005)* |
| **Gated vs non-gated MLP** | **+0.002 [−0.016, +0.020]** | **−0.000 [−0.025, +0.025]** | 7/15 (0.49) |
| MLP vs (ii) | +0.000 [−0.031, +0.032] | +0.010 [−0.054, +0.074] | 7/15 (0.96) |

\*Subjects in the same fold share a model, so the sign-flip p-value is descriptive only.

**Falsification rule (as pre-stated: the gated model must beat (ii) and (iii) on Brier AND ECE AND AUROC):** the rule fires in every scheme and under both raw and Platt outputs. But it now fires on **near-ties**:
- The gated model is behind on Brier by 0.002 and on ECE_sweep by 0.003.
- It is ahead on AUROC by 0.010.
- Under Platt in repeated CV, it beats (iii) on all three endpoints and still loses to (ii) on Brier and ECE.

The pre-fix statement "gated is worse on all three" no longer holds.

## 2. Pooled out-of-fold results, reduced condition, raw, with subject-cluster bootstrap 95% CIs

| Model | 5-fold Brier | 5-fold AUROC | LOSO Brier | LOSO AUROC | LOSO within-subject AUROC | LOSO calibration slope |
|---|---|---|---|---|---|---|
| LightGBM zero-fill at test (original straw man) | 0.294 [0.274, 0.311] | 0.850 | 0.300 | 0.861 | 0.916 | 1.16 |
| (i) LightGBM NaN at test | 0.293 | 0.835 | 0.296 | 0.852 | 0.918 | 0.96 |
| (ii) NaN-training | 0.135 [0.078, 0.211] | 0.880 | 0.146 | 0.860 | 0.870 | 0.59 |
| (iii) ECG-only | 0.137 | 0.877 | 0.142 | 0.869 | 0.899 | 0.55 |
| Gated + dropout | 0.131 [0.083, 0.185] | 0.896 | 0.146 | 0.873 | 0.903 | 0.43 |
| Non-gated MLP | 0.123 | 0.914 | 0.145 | 0.878 | 0.899 | 0.42 |
| HR-only | 0.178 | 0.826 | 0.178 | 0.820 | 0.867 | 0.44 |
| Gated, HR removed | 0.224 | 0.707 | 0.233 | 0.692 | 0.662 | 0.23 |
| Gated, HR-residualized (fixed) | 0.235 | 0.594 | 0.238 | 0.600 | 0.616 | 0.22 |
| LightGBM NaN-training, HR removed | 0.213 | 0.681 | 0.217 | 0.681 | 0.668 | 0.42 |
| LightGBM ECG-only, HR removed | 0.216 | 0.687 | 0.224 | 0.674 | 0.674 | 0.36 |

## 3. What changed relative to the pre-fix numbers

1. **"Gated loses to fair baselines" becomes "gated is indistinguishable from fair baselines."** Brier differences are about 0.002, with 95% CIs of about ±0.035.
2. **Gating contributes nothing measurable beyond masked training.** The gated network and a plain MLP trained with the same EDA masking agree to within 0.002 Brier and 0.000 AUROC. This is the tightest CI in the study, ±0.02.
   - What matters is training with missingness: the complete-trained LightGBM (Brier about 0.29) and the gated network without dropout (0.19) are the poor performers.
   - New headline: **"training with simulated missingness is what helps; the gate itself does not."**
3. **Heart-rate dependence holds across model families, and is stronger than the pre-fix data showed.**
   - Removing HR drops reduced-condition AUROC to 0.67–0.71 for both LightGBM and the gated network.
   - The corrected HR-residualized variant drops it further, to about 0.60. Most of the remaining HRV signal is within-subject covariation with HR.
4. **Performance tracks each subject's heart-rate response.** Across the 15 subjects, Spearman rho between the size of the stress−baseline drop in mean NN interval and within-subject reduced AUROC is +0.67 to +0.91 for every model (5-fold).
   - **Correction to review item C6:** within each subject, the neural models still rank S9 and S10 well (0.85 and 0.97). The LightGBM models fail on S10 (0.49–0.66).
   - The pooled per-fold failure was partly a between-subject offset, not purely a ranking failure.
5. **Every model is overconfident under missing EDA.** Calibration slopes are 0.4–0.6 for all the better models.
   - The shared Platt protocol (fit on complete-condition predictions) does not fix this for the neural networks: slope 2.3 in 5-fold, meaning it over-corrects.
   - Per-condition recalibration is the appropriate protocol. This is a methods finding worth one paragraph.
6. **The second reviewer's ECE example partly survives with the correct estimator.**
   - Under LOSO, the HR-removed NaN-training LightGBM has the lowest ECE_sweep (0.042) at AUROC 0.681.
   - Under 5-fold it is mid-pack, and the CIs overlap.
   - Use it only as an illustration, with the calibration slope (0.42) shown alongside.
7. **The legacy ECE-ratio table, regenerated from the frozen table, differs from the outline.**

   | | Outline | Regenerated |
   |---|---|---|
   | LightGBM | 4.23× | **4.51×** [0.05, 8.96] |
   | Gated, no dropout | 2.21× | **2.02×** [0.57, 3.46] |
   | Gated + dropout | 1.61× | **1.15×** [0.67, 1.63] |
   | Dropout vs no-dropout, folds favouring dropout | 5/5 (W=0) | 4/5 (W=1, p=0.125; paired t p=0.095) |
   | EDA-only vs ECG-only droppable ablation | 5/5 folds | 4/5 folds, 1.15× vs 2.48×, p=0.18 |

   The outline's "direction consistent across all 5 folds, d_z = −1.19" no longer holds.
   - Note: `run_locked_lgbm` delivers **NaN**, not zeros, to the complete-trained LightGBM. The outline calls this baseline "zero-fill". Both encodings give Brier ≈ 0.29 under the new pipeline, so the straw-man conclusion stands, but the label must be corrected.
8. **H-C1 code does not run on the current code base.** `run_locked_hc1` crashes because `_to_tensors` returns 5 values and the code unpacks 4. The 890-observation numbers in the outline therefore cannot be reproduced. This is a further reason to cut H-C1.

## 4. Follow-up runs (same day)
- **Capacity sweep** (`results/repeated_cv_capacity.json`): the gated network vs the MLP at matched sizes of 273/281, 801/808, 2,625/2,637 and 9,345/9,333 parameters. Every ΔBrier is between −0.001 and +0.005, with 95% CIs within ±0.025. `cap_gated_d16` reproduces the default gated model exactly (0.1482).
- **Nested tuning** (`results/repeated_cv_tuned.json`; inner subject-level 4-fold CV; 16 LightGBM and 24 network configurations):
  - tuned gated vs tuned (ii): +0.004 [−0.028, +0.036];
  - tuned gated vs tuned (iii): +0.004 [−0.026, +0.034];
  - tuned gated vs tuned MLP: +0.005 [−0.019, +0.029].

  The rule fires again. Tuning moved no model by more than 0.005 Brier.
  - Runtime: 9.2 min on 18 single-threaded workers. A first attempt was stalled by LightGBM thread oversubscription; it now uses `n_jobs=1`, which gives identical predictions.
- **H2a, subject level** (`results/h2a_subject_level.json`): d_z = −0.39 [−0.91, +0.15], p = 0.16. TOST at |d_z| < 0.8 gives p = 0.066, so equivalence is not established. Inconclusive, not a null.
- **Figure 1** (`results/fig_per_subject_forest.png/.pdf`, `make_figures.py`): per-subject LOSO within-subject AUROC vs heart-rate response. Spearman ρ: gated 0.88, NaN-trained LightGBM 0.76, HR-only 0.92.
- **Outline:** `e14-preprint-outline.md` was replaced with the rewritten v3. Earlier versions are in `archive/`.

## 5. Remaining-items pass (same evening)
- **Per-condition Platt for every model** (`p_platt_pc` in `eval_core.fit_predict`).
  - **Networks:** calibration is fixed with EDA missing. LOSO slope 0.43 → 0.92 for the gated network and 0.42 → 1.00 for the MLP; ECE about 0.09–0.10 → about 0.05; Brier unchanged.
  - **Complete-trained LightGBM:** recalibration alone takes it from Brier 0.296 to 0.163 and intercept +3.10 to +0.09.
  - **Decision rule:** it does **not** fire under per-condition Platt in 20× repeated CV or LOSO (the gated network edges both baselines on near-ties). It fires in the single 5-fold scheme. **The verdict depends on calibration and scheme**, which is itself evidence that the models are indistinguishable.
- **Selective prediction** (AURC, tie-aware; `fig_risk_coverage_reduced.png`).
  - Pooled LOSO suggested better abstention for the networks (gated 0.085 vs ECG-only 0.162).
  - 20× repeated CV does not support this: gated vs (iii) ΔAURC −0.026 [−0.113, +0.061]; every comparison's CI includes 0.
  - Abstention lowers error from about 0.20 to about 0.10 at 60% coverage for every missingness-aware model.
- **Reproducibility check.** Raw and shared-Platt repeated-CV numbers match the earlier post-fix run to ≤ 2×10⁻⁵. The earlier outputs are kept as `results/archive/postfix-v1_*`.
- **Interpretability regenerated** on the frozen table, out-of-fold under LOSO (`interpretability_frozen.json`).
  - Heart rate is the top feature everywhere.
  - Integrated Gradients completeness: median 0.004, maximum 0.093 logit (256-step midpoint rule).
- **Feature extraction documented** in outline §4.1. The coupling features were always removed together with EDA, so that "decision" was moot.

## 6. Still open
- Author decisions: the COI statement and the H2a smallest effect of interest.
- H-B2: re-run or report as historical.
- Formal re-run of the data-gap search.
- `git init`.
