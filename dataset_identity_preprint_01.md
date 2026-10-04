# Dataset-Identity Audit Protocol for E14 Preprint 1 (WESAD vs MIT-BIH)

The audit comes down to one question that a physiological match can answer: take windows that are "normal" in both datasets (WESAD baseline vs. MIT-BIH clean normal-sinus-rhythm), harmonize the processing, and ask whether (a) a subject-grouped probe can still tell the datasets apart and (b) the Preprint 1 classifier still calls the MIT-BIH normal windows "cardiac." If the answer to (b) is yes, the shortcut is confirmed, whatever the other metrics say. Below is an ordered 10-step protocol with exact splits, models, tests, thresholds, a decision tree, libraries, and effort estimates.

## TL;DR

- **Most decisive, cheapest test (Step 3, ~half a day):** score the frozen Preprint 1 model on MIT-BIH windows that are pure normal sinus rhythm, then on WESAD baseline windows. If the MIT-BIH normal windows get "cardiac" probabilities close to those of the arrhythmic windows, the model is detecting the dataset, not the pathology. Then run a matched "normal-vs-normal" identity probe (LightGBM with the production hyperparameters plus logistic regression, `StratifiedGroupKFold` by subject, subject-level AUC with a cluster-bootstrap CI, subject-level permutation p-value) and a block-permutation MMD test. Repeat both after each of 7 cumulative harmonization stages.
- **Decision rule (my operational thresholds, not a literature standard):** a harmonized normal-vs-normal identity AUC with a lower 95% CI of at least 0.80, together with a failed production-model test, means "shortcut confirmed." Preprint 1's headline discrimination claim, and the robustness, calibration and fairness results built on it, must then be reframed as dataset discrimination. An AUC ≤0.60 whose CI includes 0.5, an MMD indistinguishable from the within-dataset subject-heterogeneity null, and a passed production-model test mean "not detected." Anything in between is inconclusive or partial and should be reported as a limitation.
- **Biggest likely drivers, in order:** (1) EDA/respiration present vs. absent (missingness alone is a perfect dataset label for LightGBM); (2) 700 vs. 360 Hz R-peak timing quantization; (3) filter, mains (50 vs. 60 Hz) and amplitude-unit differences; (4) window, segmentation and windows-per-subject differences; (5) documented MIT-BIH tape-transport artifacts at 0.090 Hz and 0.167 Hz, which fall inside the LF and HF HRV bands; (6) the age gap (young WESAD cohort vs. MIT-BIH subjects aged 23–89), which no preprocessing can remove. Plan on roughly 3–4 engineer-weeks for one person, or about 2–2.5 calendar weeks with two.

## Key Findings (what the research establishes)

1. **A strict "within-class, cross-dataset" probe cannot be built from Preprint 1's data.** Every panic-proxy window is WESAD and every cardiac-proxy window is MIT-BIH, so class and dataset are perfectly collinear. The only valid substitute is a *physiologically matched* comparison: states that should look alike across datasets (resting/baseline WESAD vs. clean normal-sinus MIT-BIH). This is the core design choice of the audit.
2. **MIT-BIH has documented, dataset-specific artifacts that land directly on Preprint 1's features.** PhysioNet's database directory says the analog tapes were recorded on nine Del Mar Avionics model 445 recorders and replayed for digitization. It lists frequency-domain artifacts from the recorder and playback mechanics, of which "the most significant... by far is the 0.167 Hz artifact on recordings that were played back at real time. The next largest is the 0.090 Hz artifact."\[1\] 0.090 Hz sits in the LF band (0.04–0.15 Hz) and 0.167 Hz in the HF band (0.15–0.40 Hz), so LF/HF is a prime shortcut candidate.\[2\] PhysioNet also warns that "wow and flutter should be studied carefully in the context of heart-rate variability studies, since flutter compensation was not possible in these recordings."\[1\]
3. **Other MIT-BIH acquisition fingerprints:** digitized at 360 Hz\[3\] after an analog 0.1–100 Hz passband, with an 11-bit ADC over ±5 mV. 360 Hz was chosen to ease 60 Hz notch filtering. Some tapes were played back at twice real time, so their mains noise appears at 30 Hz. Four records (102, 104, 107, 217) contain paced beats, and records 201 and 202 come from the same subject. The 23 records in the 100-series were randomly selected. The 25 in the 200-series were deliberately chosen for rare, complex arrhythmias.\[4\] Subjects were 25 men aged 32–89 and 22 women aged 23–89.\[1\]
4. **WESAD fingerprints:** RespiBAN chest ECG/EDA/respiration all at 700 Hz. Label codes are 0 = transient, 1 = baseline, 2 = stress (TSST), 3 = amusement, 4 = meditation, 5/6/7 = ignore. Per MultiWave (arXiv 2306.10164), the baseline session is 20 minutes of neutral reading and the TSST stress session lasts 10 minutes. The 15 subjects are young: TEANet (arXiv 2503.12657) reports 12 males and 3 females "with a mean age of 27.5 ± 2.4 years."
5. **HRV methodology sets the harmonization target.** The 1996 Task Force standard says the optimal ECG sampling range is "250 to 500 Hz or perhaps even higher." Lower rates (in any case ≥100 Hz) "may behave satisfactorily only if an algorithm of interpolation (parabolic) is used to refine the R-wave fiducial point."\[5\] Resampling to a common rate plus parabolic R-peak refinement is therefore standard practice, not an ad hoc fix.
6. **Precedent: models readily learn site identity.** Zech et al. (PLOS Medicine, 2018) found that pneumonia CNNs "robustly identified hospital system and department within a hospital," and warned that internal test performance overstates real-world performance.\[6\]\[7\] Rabanser et al. (NeurIPS 2019) found two-sample tests on classifier-reduced representations performed best for detecting shift, and that domain classifiers are most useful for *characterizing* it.\[8\]\[9\] That supports running both an MMD test and a probe.
7. **Small-N error bars dominate interpretation.** Varoquaux (NeuroImage 180:68–77, 2018) shows that cross-validation error bars are about ±10% at 100 samples and that the standard error across folds "strongly underestimates them." With 15 WESAD subjects, report subject-level cluster-bootstrap CIs, never across-fold standard deviations.
8. **DANN-style adversarial training is contraindicated as a fix here.** Domain equals label, so a gradient-reversal adversary on dataset identity would strip out the clinical signal too. Adversarial or MMD alignment is only meaningful when conditioned on matched "normal" windows, and even then it is a diagnostic rather than a cure.

## Details: The Ordered Protocol

### Step 0 — Freeze and reproduce Preprint 1 (0.5–1 day)
- Pin library versions (LightGBM, NeuroKit2, wfdb, scikit-learn), random seeds, the feature list, and the exact LightGBM hyperparameters. Re-generate the Preprint 1 headline metric bit-for-bit. Every later number is compared against this frozen artifact.
- Write down, for every feature, which raw channel it comes from. Specifically, check whether MIT-BIH EDA and respiration features are NaN, zero-imputed, or ECG-derived (EDR). LightGBM routes missing values natively, so NaN-vs-value on any EDA or respiration feature is by itself a perfect dataset indicator. Treat this as "Driver #1" before doing anything else.

### Step 1 — Construct the audit data splits (2–3 days)

**WESAD (15 subjects, chest RespiBAN, 700 Hz)**
- Keep label 1 (baseline) and label 2 (stress/TSST). Drop 0, 5, 6 and 7, and keep 3 (amusement) and 4 (meditation) only as optional secondary contrasts.
- Window rule: accept a window only if 100% of its samples carry one label. Use one window length for everything (see Step 6). Preprint 1 probably used heavily overlapping windows; Schmidt et al. used 60 s windows with a 0.25 s shift.\[10\] For the audit, use **non-overlapping** windows so that window counts don't inflate apparent N.
- Cap windows per subject per condition (e.g., at most 20 baseline and at most 10 stress windows of 60 s) so that no subject dominates.

**MIT-BIH (read with the `wfdb` Python package: `wfdb.rdrecord`, `wfdb.rdann(..., 'atr')`)**
- Exclude paced records **102, 104, 107, 217** (AAMI convention). Assign **one group ID to records 201 and 202** (same patient).\[1\]\[11\] That leaves 44 records from 43 subjects.
- Use lead MLII (channel 0 in almost all records; 102 and 104 used modified V5 and are excluded anyway; in record 114 the signals are reversed, so select by the `sig_name == 'MLII'` field rather than the channel index).\[1\]
- **NSR-pure windows (the "normal" MIT-BIH set):** the rhythm annotation (`aux_note`) in force across the whole window must be `(N`. *Every* beat symbol in the window must be `N` (so none of the ectopic or bundle-branch symbols `V, A, a, J, S, F, E, j, e, L, R, /, f, Q`). There must be no noise or quality annotations (`~` quality changes to noisy, `U`, tape-slippage comments) in the window. Records with persistent bundle branch block (e.g., those dominated by `L` or `R` beats) contribute no NSR-pure windows by this definition. That is intended.
- **Arrhythmic windows:** any window where ≥50% of the duration is under `(AFIB`, `(AFL`, `(SVTA`, `(VT`, `(VFL`, `(B` (ventricular bigeminy) or `(T` (ventricular trigeminy), or that contains ≥3 `V`/`A`-class ectopic beats. Report AF-only as the main subset, because AF is the most HRV-relevant arrhythmia. AF episodes are confined to the 200-series;\[4\] records 201, 202, 203, 210, 219, 221 and 222 are the ones usually listed.\[4\] **Compute the actual per-record rhythm durations from the `.atr` files by summing the time between rhythm-change annotations. Do not trust hard-coded lists**, since PhysioNet relabelled many record-222 AF episodes as atrial flutter in 2005.\[1\]
- Record which series (100 random vs. 200 selected) each window comes from, and the subject's age and sex from the record headers. You will need both as covariates.

**Comparison sets derived from these:**

| Set | Contents | Purpose |
|---|---|---|
| **C1 "normal-vs-normal"** | WESAD baseline (label 1) vs. MIT-BIH NSR-pure | Primary dataset-identity probe; both should be physiologically "normal resting" |
| C2 "aroused-vs-normal" | WESAD stress (label 2) vs. MIT-BIH NSR-pure | Shows whether stress features push WESAD *toward* or *away from* MIT-BIH |
| C3 within-WESAD | Baseline vs. stress, subject-grouped | Clinical-proxy signal without dataset confound |
| C4 within-MIT-BIH | NSR-pure vs. arrhythmic (esp. AF), subject-grouped | Clinical-proxy signal without dataset confound |
| N1 negative control | MIT-BIH NSR-pure, subjects randomly split into two halves (repeat 200×) | Empirical null for probe AUC and MMD under pure subject heterogeneity |
| N2 negative control | WESAD baseline, subjects split into halves (repeat 200×) | Same, for WESAD |

### Step 2 — Restrict to a common-modality feature set (0.5 day)
- Build feature set **F_ECG**: only features computable from the ECG channel in *both* datasets, e.g. mean NN, SDNN, RMSSD, pNN50, HR, LF, HF, LF/HF, LFnu, HFnu, plus ECG-derived respiration rate *only if* computed identically for both datasets.
- EDA features cannot be audited cross-dataset because MIT-BIH has no EDA channel. Any EDA contribution to the Preprint 1 decision is, by construction, a dataset shortcut unless it is only ever used within WESAD. Quantify it: re-run the Preprint 1 classifier with EDA features removed and report the change in AUC. Do the same for respiration features that are missing in MIT-BIH.

### Step 3 — Production-model shortcut test (0.5 day; do this first)
- Using the **frozen Preprint 1 model** (no retraining), predict P(cardiac) for: (a) MIT-BIH NSR-pure windows, (b) MIT-BIH arrhythmic windows, (c) WESAD baseline, (d) WESAD stress.
- Compute subject-level means and these AUCs: AUC(a vs c) and AUC(b vs a).
- **Shortcut signature:** P(cardiac | MIT-BIH NSR) close to P(cardiac | MIT-BIH arrhythmic) (e.g., median >0.8), and AUC(a vs c) close to the headline AUC. A model that had learned pathology should rank arrhythmic MIT-BIH windows well above NSR-pure MIT-BIH windows (AUC(b vs a) well above 0.5) and should *not* confidently call healthy-looking MIT-BIH ECG "cardiac."
- Caveat: MIT-BIH NSR windows come from patients referred for Holter monitoring, so some real "cardiac-ness" in them is plausible. That is why this test is paired with C4. If the within-MIT-BIH NSR-vs-arrhythmia ranking is near chance under the production model, the model is not using arrhythmia physiology.

### Step 4 — Dataset-identity probe (2–3 days)

**Models (triangulate; don't rely on one):**
1. **P1: exact production mirror.** Same LightGBM hyperparameters and feature pipeline, with target = dataset. This measures what *your* model could exploit.
2. **P2: high-capacity LightGBM.** A small grid over `num_leaves ∈ {7, 15, 31}`, `min_child_samples ∈ {20, 50}`, `learning_rate = 0.05`, `n_estimators` up to 500 with early stopping on an inner grouped split, `feature_fraction = 0.8`, `is_unbalance = True`. This gives an upper bound on separability.
3. **P3: L2 logistic regression** on robust-scaled features (`sklearn.preprocessing.RobustScaler` fitted inside each training fold; `C ∈ {0.01, 0.1, 1}`, `class_weight='balanced'`). This tests whether separability is essentially linear, i.e. mean or scale offsets.
4. **P4: univariate screens.** Per-feature subject-level AUC plus a depth-1 LightGBM stump. This identifies single-feature shortcuts (e.g., RMSSD quantization, LF/HF).

**Cross-validation:**
- `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True)` with `groups = subject_id`. Stratify on dataset so each test fold holds about 3 WESAD and about 8–9 MIT-BIH subjects. Repeat with 20 different seeds (100 fits per model per stage).
- All scaling, imputation and feature selection happen inside folds. Hyperparameter tuning for P2 and P3 uses a nested inner `StratifiedGroupKFold(n_splits=3)`.

**Metrics:**
- **Primary: subject-level ROC AUC.** Average out-of-fold predicted probabilities per subject, then compute AUC across the roughly 15 + 43 subjects. The subject is the unit of inference. Also report window-level AUC, balanced accuracy, and log-loss.
- **95% CI:** a stratified cluster bootstrap. Resample subjects with replacement within each dataset, 2,000 replicates, percentile interval. Do not use DeLong or across-fold SD, because windows are clustered and Varoquaux shows fold SD underestimates error.\[12\]
- **p-value: a subject-level label permutation**, 1,000 permutations (minimum attainable p ≈ 0.001). Shuffle the dataset label *across subjects* (keeping the 15/43 split), re-run the full grouped CV, and take p = (1 + #{AUC_perm ≥ AUC_obs}) / (1 + 1000). Note that `sklearn.model_selection.permutation_test_score(groups=...)` permutes labels *within* groups.\[13\] Because dataset is constant within a subject, that permutation does nothing, so **write the permutation loop yourself**.
- **Null calibration:** run the same probe on N1 and N2. The 95th percentile of those AUCs is the "subject-heterogeneity floor." A cross-dataset AUC must clear that floor, not just 0.5.

**Interpretation bands** (my proposed operational thresholds; no published consensus exists for this setting):

| Subject-level AUC (C1, F_ECG) | Label |
|---|---|
| ≥0.90, or lower CI ≥0.80 | **Trivially separable**: strong dataset signature |
| 0.70–0.90 | **Substantially separable** |
| 0.60–0.70 | **Borderline** |
| ≤0.60 and CI includes 0.5 and ≤ N1/N2 95th percentile | **Not separable** (not detected) |

- A 95% CI that straddles two bands is **inconclusive**. With 15 WESAD subjects, expect CIs of roughly ±0.08–0.12.

**Attribution:** fit P2 on all C1 data and compute SHAP values (`lightgbm` `predict(..., pred_contrib=True)` or the `shap` package's `TreeExplainer`). Rank features by mean |SHAP|. This ranked list drives Step 7.

### Step 5 — MMD two-sample test (1–2 days)
- **Input:** F_ECG features for C1 (and separately for C2, C4-normal), each feature robust-scaled using pooled median/IQR.
- **Kernel:** Gaussian RBF, k(x,y) = exp(−‖x−y‖² / (2σ²)), with σ = median pairwise Euclidean distance of the pooled sample (Gretton et al., JMLR 2012, median heuristic).\[14\] Also run **MMDAgg** (Schrab et al., JMLR 2023; `mmdagg` package, NumPy or JAX), which aggregates over a collection of bandwidths with level control and removes the arbitrariness of the bandwidth choice.\[15\]\[16\]
- **Statistic:** unbiased MMD² (U-statistic).
- **Exchangeability fix (critical):** windows within a subject are dependent, so a window-level permutation test is anti-conservative. Use one of:
  - **(A) Block permutation (preferred):** permute the dataset label across *subjects*, carrying all of each subject's windows together.
  - **(B) Subject aggregation:** one vector per subject (median of the subject's windows), then a standard permutation test on n = 15 vs. about 43.
- **Permutations:** 10,000 (minimum p ≈ 1e-4). Note that alibi-detect `MMDDrift` defaults to `n_permutations=100`, which is too few here, and hyppo's `MMD().test()` defaults to `reps=1000` but can switch to a fast chi-square approximation (`auto=True`); set `auto=False`.\[17\]\[18\]
- **Significance:** α = 0.05, Holm-corrected across all (comparison × harmonization-stage) tests you run.
- **Effect size (what you decide on):** with hundreds of windows, MMD will reject even for trivial shifts, so the p-value is necessary but not sufficient. Compute the **heterogeneity ratio R = MMD²(C1) / 95th percentile of MMD²(N1 ∪ N2 subject-split nulls)**. R ≤1 means the cross-dataset gap is no larger than the gap between random groups of subjects within one dataset. Track R across harmonization stages.
- **Cross-check (C2ST):** the Step 4 probe *is* a classifier two-sample test (Lopez-Paz & Oquab, ICLR 2017).\[19\] Report MMD and probe AUC side by side. Disagreement (e.g., significant MMD but AUC ≈ 0.55) usually means a small, diffuse shift. A near-perfect AUC with a modest R usually means a few features carry a sharp shortcut.

### Step 6 — Harmonization ladder (4–6 days; the bulk of the work)
Apply the stages **cumulatively** in this order (most likely discriminative first). After each stage, re-extract features from raw signals and re-run Step 3 (retrained clinical model), Step 4 (P1–P4) and Step 5. Then run a **leave-one-stage-out** pass (the full ladder minus one stage at a time) to attribute separability to individual differences.

| Stage | Change | Why it's suspect | Implementation |
|---|---|---|---|
| H0 | Preprint 1 as-is | Baseline | — |
| H1 | ECG-only features (F_ECG); drop all EDA/resp features or compute EDR identically | Missingness = dataset label | Feature mask |
| H2 | Resample raw ECG to a common rate *before* R-peak detection | 700 vs. 360 Hz RR quantization (≈1.43 ms vs. ≈2.78 ms) fingerprints RMSSD/pNN50/successive-difference distributions | `scipy.signal.resample_poly(ecg_wesad, up=18, down=35)` (700→360 Hz exactly); same detector for both (e.g., `neurokit2.ecg_peaks`), plus parabolic R-peak refinement per Task Force |
| H3 | Identical filtering + mains handling | MIT-BIH 0.1–100 Hz analog passband, 60 Hz (and 30 Hz on twice-speed records) mains; WESAD recorded on 50 Hz mains\[1\] | Zero-phase Butterworth band-pass 0.5–40 Hz (order 4, `sosfiltfilt`) on both; this removes both mains families |
| H4 | Amplitude normalization | mV (MIT-BIH, 11-bit ±5 mV) vs. RespiBAN raw units; lead differences (MLII vs. chest) | Per-recording z-score of ECG before detection; drop any amplitude/morphology features |
| H5 | Identical windowing and RR cleaning | Different window lengths/overlap, windows per subject, ectopic-correction rules | Same window (60 s for time-domain; ≥120 s, ideally 300 s, for LF per Task Force guidance), non-overlapping, same cap per subject, same RR artifact correction (e.g., `neurokit2.signal_fixpeaks`)\[5\] |
| H6 | Spectral-artifact control | MIT-BIH tape artifacts at 0.090 Hz (LF) and 0.167 Hz (HF); uncompensated wow/flutter\[1\] | Variant 6a: drop LF, HF, LF/HF, LFnu, HFnu (time-domain only). Variant 6b: keep spectral features but notch 0.08–0.10 and 0.16–0.18 Hz in the 4 Hz-interpolated RR series and recompute. If 6a removes separability that 6b doesn't, the tape artifacts are a driver |
| H7 | Within-subject normalization | Between-subject offsets (age, fitness, medication) | Z-score each feature by the subject's own NSR/baseline windows. Warning: this also removes absolute-level clinical signal, so evaluate the clinical task (C3/C4) at this stage too |

**Irreducible covariate: age.** WESAD's young cohort barely overlaps MIT-BIH's 23–89 range, and HRV declines strongly with age. No preprocessing fixes a positivity violation. As a sensitivity analysis, restrict MIT-BIH to the youngest available subjects (read ages from the `.hea` headers) and check whether C1 AUC falls. Also report C1 AUC separately for 100-series (random) and 200-series (selected) MIT-BIH subjects.

**Feature-elimination sweep (after H7):** iteratively drop the top-|SHAP| identity feature, retrain both the identity probe (C1) and the clinical classifier (the Preprint 1 task and C3/C4), and plot identity-AUC against clinical-AUC. If clinical AUC collapses in step with identity AUC, the clinical performance *was* the shortcut.

### Step 7 — Within-dataset clinical re-validation (1–2 days)
- Run the Preprint 1 pipeline (harmonized, F_ECG) on **C3** (WESAD baseline vs. stress, LOSO) and **C4** (MIT-BIH NSR vs. AF/arrhythmia, `StratifiedGroupKFold`). These are the only clinically interpretable discrimination results the data can support.
- **Cross-contrast direction check:** compute the standardized mean shift of each feature for stress-minus-baseline (WESAD) and for arrhythmia-minus-NSR (MIT-BIH). Features the Preprint 1 model relies on should show a direction or magnitude difference *within* datasets that matches what the cross-dataset model uses. A feature that separates the datasets at "normal" (C1) but doesn't move within either dataset is a shortcut feature.

### Step 8 — Optional: conditional alignment diagnostics (1–2 days)
- Do **not** run unconditional DANN on dataset identity; it will erase class signal because domain = label.
- Acceptable diagnostic: learn a feature transform that aligns **only the matched normal windows** (C1) across datasets, e.g. CORAL or an MMD-penalized linear map fitted on C1 only, or a conditional adversary trained only on normal windows. Then apply it to all windows and re-test the clinical task. If clinical AUC survives alignment of the normal states, that is mild evidence of a genuine signal.
- Libraries: **SKADA** (scikit-learn/PyTorch-compatible; `make_da_pipeline`, CORAL and DANN-family estimators) or **ADAPT** (`pip install adapt`; `adapt.feature_based.DANN(lambda_=...)`, `CORAL`, `mdd`, instance-reweighting methods like `KMM`, `KLIEP`).\[20\]\[21\] Treat these as secondary; with N=15 WESAD subjects, deep adversarial methods will be unstable.

### Step 9 — Decide and act (decision tree)

**Inputs:** (i) Step 3 production-model test (pass/fail); (ii) C1 subject-level identity AUC with 95% CI at the final harmonization stage; (iii) MMD heterogeneity ratio R and Holm-adjusted p; (iv) whether clinical AUC survives the Step 6 sweep and holds within datasets (C3/C4).

**Branch A — Shortcut confirmed**
- *Criteria:* the Step 3 test fails (MIT-BIH NSR windows scored as cardiac, AUC(b vs a) ≤0.65) **and/or** final-stage C1 AUC lower CI ≥0.80 with R >2 and adjusted p <0.05.
- *Conclusion:* a genuine, hard-to-remove dataset signature exists, and Preprint 1's cross-dataset discrimination cannot be interpreted as panic-vs-cardiac.
- *Actions:*
  1. Withdraw or rewrite the headline claim as "separates WESAD-lab-stress recordings from 1975–1979 Beth Israel Hospital Holter recordings."
  2. Mark the robustness, calibration and equity-across-device-tier results as properties of a dataset classifier, not a clinical one.
  3. Promote C3 and C4 within-dataset results as the only clinical-proxy findings.
  4. Publish the audit as a methods contribution (it is valuable as such).
  5. Prioritize acquiring data where both arousal and cardiac states are recorded on the same device and population (e.g., your own wearable pilot with ED chest-pain presentations, or a single cohort with both).

**Branch B — Removable by harmonization**
- *Criteria:* the Step 3 test passes after harmonization; final-stage C1 AUC ≤0.60 with the CI including 0.5 and at or below the N1/N2 95th percentile; R ≤1 (or adjusted p ≥0.05); clinical AUC retained at ≥80% of its H0 value and replicated within datasets.
- *Conclusion:* the shortcut was preprocessing-induced, and the harmonized pipeline shows no detectable dataset identity.
- *Actions:*
  1. Re-run *all* Preprint 1 analyses on the harmonized pipeline and replace the reported numbers.
  2. Add the ladder table (identity AUC and clinical AUC per stage) to the paper.
  3. State plainly that "not detected" is not "absent." The positivity problem (all panic-proxy windows from one young lab cohort) still limits clinical claims.

**Branch C — Partial / inconclusive**
- *Criteria:* final C1 AUC 0.60–0.80, or a CI spanning bands, or MMD and probe disagreeing.
- *Actions:*
  1. Identify the residual driver via the leave-one-stage-out and SHAP results (likely age or tape spectral artifacts).
  2. Drop the implicated features if clinical AUC survives; otherwise report them as a named limitation.
  3. Restrict the preprint's claims to within-dataset contrasts plus a hedged cross-dataset result.
  4. Do not report the device-tier equity findings as clinical fairness.

**Branch D — Separable only before harmonization, but clinical AUC collapses with it**
- *Criteria:* identity AUC drops to chance, but so does the Preprint 1 clinical AUC.
- *Conclusion:* the model's performance was the shortcut. Treat as Branch A for claims, even though the "shortcut" is now removed.

## Toolkits and Reference Implementations

| Need | Package / function | Notes |
|---|---|---|
| Reading MIT-BIH | `wfdb` (`rdrecord`, `rdann`; `ann.symbol`, `ann.aux_note`) | Rhythm strings like `(N`, `(AFIB`, `(AFL`, `(VT`, `(SVTA`, `(B`, `(T`, `(VFL`\[1\] |
| ECG/HRV/EDA features | `neurokit2` (`ecg_peaks`, `signal_fixpeaks`, `hrv_time`, `hrv_frequency`, `eda_phasic` incl. `cvxeda`)\[22\] | Use identical calls and parameters for both datasets |
| Grouped CV | `sklearn.model_selection.StratifiedGroupKFold`, `LeaveOneGroupOut` | Groups = subject; 201/202 share a group |
| Classifier permutation test | `sklearn.model_selection.permutation_test_score` (Ojala & Garriga, JMLR 2010)\[23\] | Its `groups` argument permutes *within* groups, which is useless for subject-constant labels; write a subject-level permutation loop |
| MMD test | `mmdagg` (Schrab et al.; NumPy/JAX), `hyppo.ksample.MMD` (`reps`, `auto=False`), `alibi_detect.cd.MMDDrift` (`n_permutations`, default 100)\[16\]\[17\]\[18\] | None natively does block/subject permutation. Aggregate to subject level or wrap your own permutation around an MMD² function |
| Classifier two-sample test | `alibi_detect.cd.ClassifierDrift`, or the Step 4 probe itself | Lopez-Paz & Oquab C2ST framing |
| Shortcut toolkit | **ShortKit-ML** (`from shortcut_detect import ShortcutDetector`; methods `'hbac', 'probe', 'statistical', 'geometric', 'equalized_odds'`; `group_labels=dataset_id`)\[24\] | Cajas et al. (MIT Critical Data) medRxiv preprint of 30 Apr 2026, not peer-reviewed, with over 20 detection methods benchmarked on CheXpert, MIMIC-CXR and CelebA. It accepts any embedding matrix, so your hand-crafted feature matrix works. The authors report that requiring at least 7 of 12 methods to agree cut the false-positive rate on null data from 1.0 to 0.0. Use as a secondary, multi-method cross-check, not the primary test |
| Domain adaptation (diagnostic only) | `skada` (scikit-learn/PyTorch), `adapt` (`adapt.feature_based.DANN`, `CORAL`; `adapt.instance_based.KMM`, `KLIEP`) | See the Step 8 warning about domain = label |
| Attribution | `shap.TreeExplainer`, or LightGBM `pred_contrib=True` | Rank identity-driving features |

## Effort Estimate (1–2 ML engineers)

| Step | 1 engineer | Notes |
|---|---|---|
| 0 Freeze/reproduce | 0.5–1 d | Longer if Preprint 1 isn't scripted end-to-end |
| 1 Splits (WESAD labels, MIT-BIH rhythm/beat parsing, NSR-purity filter, grouping) | 2–3 d | The annotation-parsing edge cases take most of the time |
| 2 Common-modality set | 0.5 d | |
| 3 Production-model test | 0.5 d | Do first; may decide the outcome alone |
| 4 Identity probe (P1–P4, grouped CV, cluster bootstrap, subject permutation) | 2–3 d | Compute: 1,000 permutations × 5 folds of LightGBM on a few thousand windows takes minutes to low hours on a laptop |
| 5 MMD (median heuristic + MMDAgg, block permutation, nulls) | 1–2 d | 10,000 permutations on ~2–5k windows is fine in JAX/NumPy |
| 6 Harmonization ladder (7 stages + leave-one-out + SHAP sweep) | 4–6 d | Largest item: re-extracting features from raw signals per stage |
| 7 Within-dataset re-validation | 1–2 d | |
| 8 Conditional alignment (optional) | 1–2 d | Skip if Branch A is already clear |
| 9 Decision, re-run Preprint 1 on the chosen pipeline, write-up | 3–5 d | |
| **Total** | **~16–25 working days (≈3–5 weeks)** | **With 2 engineers: ≈2–2.5 calendar weeks** (one on data/harmonization, one on probe/MMD/stats) |

## Recommendations

1. **Run Step 3 this week.** It costs half a day and, if it fails, settles the headline claim without waiting for the full ladder.
2. **Pre-register the thresholds in this document** (AUC bands, R ≤1, Holm α = 0.05, 1,000/10,000 permutations) in the repo *before* seeing results, so the preprint revision can't be accused of threshold-shopping.
3. **Treat EDA as unauditable cross-dataset.** Any EDA-based discriminative claim in Preprint 1 should be limited to within-WESAD analyses.
4. **Expect Branch A or C.** Given the documented MIT-BIH spectral tape artifacts in exactly the LF/HF bands, the 700/360 Hz quantization, the missing modalities, and the non-overlapping age distributions, I would bet on a C1 AUC well above 0.8 at H0. Whether harmonization brings it below 0.6 is the real open question; the age gap makes full removal unlikely.
5. **Rewrite the robustness, calibration and fairness sections conditionally.** Their validity inherits entirely from the audit outcome; in Branch A they must be reframed.

## Caveats

- The AUC bands, the R ≤1 criterion and the Step 3 cut-offs are **my proposed operational thresholds**, calibrated against within-dataset null splits. They are not established literature standards.
- A passed audit (Branch B) shows only that these probes failed to detect dataset identity. It cannot show that the model learned panic-vs-cardiac physiology. WESAD TSST stress is not panic, and MIT-BIH arrhythmias are not acute coronary events. The proxy-validity gap stays.
- The MIT-BIH record lists for AF and other rhythms quoted here reflect common usage. Compute the actual rhythm durations from the `.atr` annotations, because PhysioNet revised labels (e.g., record 222 AF→AFL in 2005; a record-203 rhythm label corrected in October 1998).
- The ShortKit-ML evidence comes from a non-peer-reviewed medRxiv preprint (Cajas et al., 30 Apr 2026) evaluated on imaging embeddings; its behaviour on low-dimensional hand-crafted biosignal features is untested.
- With 15 WESAD subjects, every subject-level estimate has wide uncertainty. Expect some results to land "inconclusive," and report them as such rather than forcing a branch.

## Sources

1. [MIT-BIH Arrhythmia Database Directory (Introduction)](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm)
2. [Frontiers | An Overview of Heart Rate Variability Metrics and Norms](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2017.00258/full)
3. [Training neural networks with synthetic electrocardiograms](https://arxiv.org/pdf/2111.06175)
4. [Detection of cardiac arrhythmia using mathematical representation of standard ΔRR probability density histograms](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7146206)
5. [Heart Rate Variability | Circulation](https://www.ahajournals.org/doi/10.1161/01.cir.93.5.1043)
6. [Confounding variables can degrade generalization performance of radiological deep learning models — AI for Science](https://awesomepapers.io/ai-for-science/papers/1807.00431)
7. [Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6219764/)
8. [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/pdf/1810.11953)
9. [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift | ML Anthology](https://mlanthology.org/neurips/2019/rabanser2019neurips-failing/)
10. [PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning](https://arxiv.org/pdf/2510.24058)
11. [A systematic review of ECG arrhythmia classification: embedded feasibility, adherence to standards, and fair evaluation | Research on Biomedical Engineering | Springer Nature Link](https://link.springer.com/article/10.1007/s42600-026-00485-z)
12. [Cross-validation failure: Small sample sizes lead to large error bars - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1053811917305311)
13. [permutation\_test\_score — scikit-learn 1.6.dev0 documentation](https://dsyemen.github.io/scikit-learn/modules/generated/sklearn.model_selection.permutation_test_score.html)
14. [A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
15. [MMD Aggregated Two-Sample Test](https://www.jmlr.org/papers/volume24/21-1289/21-1289.pdf)
16. [GitHub - antoninschrab/mmdagg: MMDAgg package implementing the MMDAgg test proposed in MMD Aggregated Two-Sample Test by Schrab, Kim, Albert, Laurent, Guedj and Gretton: https://arxiv.org/abs/2110.15073](https://github.com/antoninschrab/mmdagg)
17. [alibi\_detect.cd package — alibi-detect 0.11.1 documentation](https://docs.seldon.io/projects/alibi-detect/en/stable/api/alibi_detect.cd.html)
18. [MMD — hyppo alpha documentation - NeuroData](https://mgc.neurodata.io/api/generated/hyppo.ksample.mmd)
19. [\[1610.06545\] Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545)
20. [pypi.org](https://pypi.org/project/skada)
21. [Awesome Domain Adaptation Python Toolbox for Tensorflow and Scikit-learn](https://pypi.org/project/adapt/0.3.1)
22. [EDA — NeuroKit2 0.2.13.dev214 documentation](https://neuropsychology.github.io/NeuroKit/functions/eda.html)
23. [permutation\_test\_score — scikit-learn 1.9.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html)
24. [ShortcutDetector - ShortKit-ML](https://criticaldata.github.io/ShortKit-ML/api/shortcut-detector/)
