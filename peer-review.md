# Peer Review: "Calibration-Robust Modality-Gated Architecture with Dropout Training for Clinical Wearable Physiological Monitoring" (E14 Health preprint outline, target: ML4H Findings)

**Bottom line: in its current form I would score this weak reject for ML4H Findings.** The problem is not the scientific attitude, which is unusually self-critical. The problem is that three of the load-bearing claims rest on evidence that is invalid or misread. First, the headline significance (p=0.001, BH p=0.00257) comes from a superseded, unlocked pipeline. Second, the audit's "decisive" Step 3 appears to run a WESAD stress detector whose output is labelled P(cardiac), and then reads an AUC of 0.22 as the absence of a shortcut. Third, the audit is called "pre-registered, confirmatory" although the run used `--fast` mode and a fraction of the pre-registered permutations. All of this can be fixed. Two scoping facts matter even more, though. The ML4H 2026 paper deadline was **September 10, 2026**, not Dec 15, and reviews are already underway.\[1\]\[2\] And a Findings paper is capped at **4 pages**, which cannot hold three loosely connected contributions.\[1\]

## TL;DR

- **Recommendation: weak reject as it stands.** It could reach weak accept at ML4H Findings, or a later venue, if it is cut down to one contribution (the calibration-under-missingness result or the dataset-identity audit) and the fatal issues below are corrected. As written, the numbers contradict each other in at least eight places. The headline p-value belongs to a different pipeline. No reported interval separates the gated model from the baseline. And the audit's key step appears to use a model whose "cardiac" output is really a WESAD stress probability.
- **The novelty claims are overstated compared with the literature.** Ma et al. (ICML 2023, "Calibrating Multimodal Learning") already frame calibration under modality removal as the central problem. Recent wearable work already applies modality dropout to WESAD ECG/EDA (CognifySSL v2.0, *Sensors* 2026; PULSE, arXiv 2510.24058). Device heterogeneity is already named as a fairness axis in ubiquitous health sensing. What remains novel is narrower: calibration robustness specifically for wearable physiological proxies, plus a structured dataset-identity audit for WESAD × MIT-BIH.
- **Statistics and reporting need re-analysis, not rewording.** The CIs are naive t-intervals over 5 dependent fold values. A Nadeau–Bengio correction turns the dropout effect from p=0.056 into roughly p≈0.15. The Wilcoxon test cannot get below p=0.0625 at n=5, so it could never have rejected. The "890 observations" and "n=445, well-powered" analyses almost certainly pseudoreplicate 15 subjects. The ECE ratio is unstable and ignores discrimination. WESAD *does* contain PPG/BVP and respiration, so skipping the pre-registered ablations was a pipeline choice, not a property of the dataset.

---

## 1. Summary of the submission

The outline presents a methods paper motivated by the clinical problem of telling panic attacks apart from cardiac events on wearables. It explicitly does not claim to solve that problem. There are three parts:

1. **H-B1 (headline).** Calibration robustness under missing modalities, measured as the ECE ratio ECE_reduced/ECE_complete, within WESAD (baseline vs TSST stress, 15 subjects, 5-fold subject-level CV). The three-layer comparison is LightGBM 4.23× → gated architecture without dropout 2.21× → gated + modality dropout (p=0.5) 1.61×. Dropout vs no-dropout gives p=0.056 (paired t, n=5 folds). Post-hoc ablations cover which modality is droppable and the dropout rate.
2. **H-C1.** A "device-tier equity" claim under simulated tiers: a mixed-effects model with a tier×model interaction (coef −0.484, p=0.000253, "890 observations from 15 subjects"). Accuracy drops 7.5% for the gated model vs 10.6% for the baseline.
3. **Dataset-identity audit (Section 5.6).** A 10-step protocol described as pre-registered, testing whether WESAD (panic proxy) and MIT-BIH (cardiac proxy) can be told apart by shortcut features on ECG-only features. Tools: frozen-model shortcut test, identity probes, MMD with block permutation, a harmonization ladder, CORAL, and a decision tree. The result is "Branch C (partial/inconclusive)", which is used to explain the null for contrastive pretraining (H-B2).

The outline also reports a null mechanistic test (H2a, EDA–HR coupling lag) and exploratory interpretability (SHAP, Integrated Gradients).

## 2. Strengths

- **Epistemic honesty and scoping.** The outline says outright that no public dataset pairs confirmed panic attacks with confirmed cardiac events. It forbids "this model distinguishes panic from cardiac" language and wants proxy caveats inline. Few wearable-panic papers show this discipline.
- **Calibration is the right target.** A triage-adjacent wearable has to be able to say "I don't know" safely, so framing robustness as calibration rather than accuracy is clinically well motivated.
- **Nulls reported as findings.** H2a and H-B2 are reported rather than buried. ML4H's Findings call explicitly solicits "insightful negative results".\[1\]
- **Transparency in the working notes.** Per-fold values are disclosed for the dropout comparison, arithmetic and geometric means are distinguished, the ratio of means (1.24×) is contrasted with the mean of ratios (1.61×), and the Wilcoxon floor is acknowledged. That level of disclosure is what makes this review possible.
- **Taking cross-dataset confounding seriously.** Designing an audit before claiming anything cross-dataset is the correct instinct. It sits squarely in the lineage of Zech et al. (2018), DeGrave et al. (2021), and an npj Digital Medicine (2024) study of 207,487 patients across 13 datasets, including ECGs, which found that "model performance is frequently overestimated by up to 20% on average due to shortcut learning of hidden data acquisition biases (DAB)".
- **Subject-level CV, plus disclosure of a caught leak.** Saeb et al. (GigaScience 2017) found that "record-wise CV often massively overestimates the prediction accuracy of the algorithms," so disclosing a caught leak has real educational value. The MIT-BIH count reconciliation (48 → 40) is also careful.

## 3. Major concerns

Severity labels: **[FATAL]** means the claim as stated cannot stand without new analysis. **[MAJOR-FIXABLE]** means it is fixable before the deadline. **[MINOR]** and **[COSMETIC]** follow in Section 4.

### M1 [FATAL for the audit's Step 3] — The "frozen production-model shortcut test" appears to use a stress detector labelled as P(cardiac); the results are misread

**Evidence in the text.** Subject-level median P(cardiac) is 0.008 on MIT-BIH NSR and 0.024 on MIT-BIH *arrhythmic*, but 0.97 on *WESAD stress*. The outline's own gloss on AUC(NSR vs WESAD baseline)=0.22 reads: "the frozen model calls MIT-BIH NSR *less stress-like* than WESAD baseline."

**Why it matters.** A model that detects cardiac events should give real arrhythmias high P(cardiac). This one gives them roughly 2%, and gives TSST stress windows 97%. Only three readings are possible:
- (a) the output is P(stress), mislabelled as P(cardiac);
- (b) the class encoding is inverted;
- (c) the "production model" was trained as the WESAD stress model.

Under any of these, the test does not probe the cross-dataset shortcut it was designed to probe. A shortcut test needs a model trained on the cross-dataset contrast.

Separately, **an AUC of 0.22 is not "the opposite of the shortcut signature."** It is a separability of 0.78 with the sign flipped: the frozen model's output clearly depends on which dataset a window came from. The same applies to the harmonization table's "Shortcut AUC (a vs c)" of 0.36–0.39, which corresponds to 0.61–0.64 separability. For a two-sample or shortcut question, the statistic is |AUC − 0.5| (or max(AUC, 1−AUC)), as in classifier two-sample tests (Lopez-Paz & Oquab, ICLR 2017).\[3\]

AUC(arrhythmic vs NSR)=0.82 on probabilities with medians of 0.024 vs 0.008 is a ranking of near-zero, out-of-distribution outputs. The simplest explanation is that a stress/heart-rate detector responds to the higher, more irregular rates in arrhythmic records. That is not evidence of "PATHOLOGY_LIKELY", and "This alone rules out Branch A" does not follow.

**Required fix.**
- State exactly which model was frozen, what it was trained on, and what its positive class is. Include the label-encoding line of code in the appendix.
- If it is the WESAD stress model, relabel Step 3 as "transfer behaviour of the stress model on MIT-BIH." Then either re-run Step 3 with a model actually trained on the cross-dataset task, or drop Step 3 from the decision tree.
- Report all AUCs as separability, i.e. max(AUC, 1−AUC), with subject-level cluster-bootstrap CIs.
- Re-derive the Branch verdict without Step 3.

### M2 [FATAL for the headline inference] — p=0.001 / BH p=0.00257 belongs to a superseded pipeline and cannot be attached to the locked numbers

**Evidence.** The outline says the prior "~3.6×" came "from an unlocked code path on a different split." Yet it keeps "p=0.001, BH-adjusted p=0.00257 (original H-B1 test statistic, still applies to the headline invariant-vs-baseline comparison)." The title note also still says "3.6× → 2.21×."

**Why it matters.** A p-value is a property of a specific statistic computed on specific data. Change the code path, the split, and the effect estimate (3.6× → 4.23×) and the old p-value no longer describes anything in the paper. It is also mechanically implausible on the locked design. At n=5 folds the exact Wilcoxon floor is 0.0625. A paired t-test reaching p=0.001 would need tightly clustered per-fold differences, but the LightGBM CI [0.18, 8.27] implies a per-fold SD of about 3.3. The original p=0.001 was therefore most likely computed at the window level, which is pseudoreplication (see M5).

**Required fix.** Recompute the locked LightGBM vs gated comparison on the same 5 folds: a paired test with the Nadeau–Bengio correction plus a subject-level cluster bootstrap. Report whatever comes out. Remove "3.6×" everywhere, including the title note.

### M3 [MAJOR-FIXABLE, currently overclaimed] — No reported interval supports "substantial" robustness

**Evidence.**
- LightGBM 4.23× [0.18, 8.27]; no-dropout gated 2.21× [0.52, 3.90]; dropout 1.61× [0.32, 2.90].
- All three CIs include 1, which means no degradation, and all overlap one another.
- The LightGBM CI includes values below 1, i.e. missing EDA *improving* calibration.
- The LightGBM per-fold ratios are "not preserved in outline." Yet the CI [0.18, 8.27] can only have been computed from per-fold values. Either they exist or the CI cannot be reproduced.

**Why it matters.** The interval arithmetic is easy to reconstruct. For dropout, the per-fold ratios [1.54, 1.56, 3.32, 1.02, 0.60] have mean 1.608 and SD 1.037. Then 1.608 ± 2.776 × 1.037/√5 gives [0.32, 2.90] exactly. So the CIs are naive t-intervals over 5 fold values. They treat folds as independent, although folds share about 75% of their training subjects. Varoquaux (NeuroImage 2018) shows that "the standard error across folds strongly underestimates" true CV error bars.\[4\] **Even the naive intervals do not separate the conditions**, and properly corrected intervals would be wider still. "Substantial calibration robustness (2.21× vs 4.23×)" is a point-estimate claim with no inferential support.

**Required fix.**
- Compute the paired difference in the robustness metric between models on each fold, and then:
  - a Nadeau–Bengio corrected t-interval, and preferably repeated (e.g. 10×) 5-fold subject-level CV, which Bouckaert & Frank recommend for replicability;\[5\]
  - a subject-level cluster bootstrap. Resample the 15 subjects, recompute pooled out-of-fold ECE_complete and ECE_reduced, and form BCa intervals.
- Replace "substantial" with the interval-supported wording, e.g. "a point-estimate reduction from X to Y; the 95% CI for the difference is [a, b]."

### M4 [MAJOR-FIXABLE] — The ECE ratio is a fragile primary metric, and ECE itself is poorly estimated at this sample size

**Evidence and reasoning.**

- **Ratio instability.** ECE_complete is around 0.09 with a fold SD of about 0.05, so some folds must have ECE_complete near 0.03–0.04, where the ratio blows up. The disagreement between summaries shows this directly:
  - The mean of ratios (1.61×) and the ratio of means (1.24×) differ by 30% for dropout.
  - LightGBM's ratio of means is 0.255/0.087 = 2.93×, against a reported mean of ratios of 4.23×. The gated no-dropout model is 0.160/0.095 = 1.68× against 2.21×.
  - Fold 5 of the dropout model has a ratio of 0.60, so removing EDA "improved" calibration. That is almost certainly estimator noise.
- **Absolute differences tell a different story.** ΔECE is 0.168 for LightGBM, 0.065 for gated no-dropout, and 0.023 for gated + dropout. That is arguably a clearer and more stable result than the ratios, and it does not depend on a noisy denominator.
- **ECE estimator bias.** There are roughly 445 windows across 15 subjects, about 30 per subject, consistent with ~60-s windows over WESAD's protocol of "about 20 min baseline, 10 min of Trier Social Stress Test" (Schmidt et al., ICMI 2018). So each test fold has only about 90 windows (3 subjects), and ECE_bin is biased upward in a way that depends on sample size and binning. Roelofs et al. (AISTATS 2022) found that "binning-based estimators with bins of equal mass (number of instances) have lower bias than estimators with bins of equal width" and recommend the debiased estimator or ECE_sweep.\[6\] The outline gives neither bin count nor binning scheme. Removing a modality also changes the shape of the confidence distribution, so the bias does not cancel in the ratio.
- **Calibration without discrimination is meaningless.** A model that loses EDA and falls back to predicting the stress prevalence (about 1/3) is almost perfectly calibrated and clinically useless. Any calibration-robustness claim has to be reported alongside discrimination under missingness.

**Required fix.**
- Pre-specify, as a documented deviation, the absolute ΔECE computed with a debiased or equal-mass estimator as primary, with the ratio as secondary.
- Report per condition:
  - Brier score and log loss (proper scoring rules);
  - calibration slope and intercept;
  - reliability diagrams on pooled out-of-fold predictions;
  - AUROC/AUPRC under complete and reduced inputs;
  - bin count and scheme (e.g. 10 equal-mass bins, plus ECE_sweep).
- State how isotonic/Platt calibration was fit: on which data, and whether per condition. If the gated model was recalibrated on reduced-modality validation data and LightGBM was not, the comparison is unfair. If recalibration used test-fold subjects, it leaks. Isotonic regression fit on about 12 subjects' windows will overfit, so prefer Platt, or isotonic with nested subject-level CV.

### M5 [MAJOR-FIXABLE] — Pseudoreplication in H-C1 and H2a; "well-powered" is false at the subject level

**Evidence.** H-C1 has "890 observations from 15 subjects, giving far more power than the 5-fold ECE-ratio comparison." H2a has "n=445, d=0.06, p=0.94, well-powered." Note that 890 = 2 × 445, so H-C1 very likely takes the same 445 windows under two tier conditions.

**Why it matters.** Windows within a subject are strongly correlated. The unit of generalization is the subject: n=15. Extra "power" from 890 correlated rows is mostly illusory unless the random-effects structure absorbs within-subject correlation *of the effect of interest*. Barr et al. (2013) show that models without the random slopes the design justifies inflate Type I error.\[7\] Here tier (and model, if evaluated on the same windows) varies within subject, so the model needs at least (1 + tier + model + tier:model | subject). If that does not converge, report a subject-level aggregate analysis instead: one tier×model contrast per subject, then a test across the 15 subjects.

For H2a, window-level power is irrelevant. With 15 subjects, a paired design has 80% power at α=0.05 only for effects of about d_z ≥ 0.78. d=0.06, p=0.94 is a *non-significant* result, not evidence of absence. To claim a null, run an equivalence test (TOST) against a pre-stated smallest effect size of interest at the subject level, or report a Bayes factor.

**Further issues in H-C1.**
- The metric switches from ECE (H-B1) to accuracy (H-C1). Accuracy at a 0.5 threshold does not reflect calibration, so H-C1 does not test "equitable calibration."
- The coefficient scale (logit? proportion?) and the model family (LMM on binary correctness? GLMM?) are not stated.
- The effect is 7.5% vs 10.6%, a 3.1-point difference. Whether that is clinically meaningful is never argued.

**Required fix.** Refit with a maximal-justified random structure, or aggregate to subject level. Report H-C1 on both ECE/Brier and AUROC. Change "well-powered" to a subject-level sensitivity statement plus TOST.

### M6 [MAJOR-FIXABLE] — "Pre-registered, confirmatory" labelling of the audit is not supported by the run that was reported

**Evidence.**
- **Mode.** The results come from "--fast mode, seed=42, run 2026-09-30." H3/H4/H5/H7 were not run, and leave-one-stage-out is "pending."
- **MMD.** "block-permutation p = 0.005 (n=200)" against a pre-registered 10,000 permutations. With 200 permutations, the smallest valid Monte Carlo p is 1/(200+1) ≈ 0.005. Phipson & Smyth show that naive permutation p-values are "understated by about 1/m", which is why the (b+1)/(m+1) form is required.\[8\]\[9\] The reported p is therefore exactly the floor.
- **Identity probes.** "Subject-level permutation p = 0.010 for all three" probes. That is the floor for about 99–100 permutations, not the pre-registered 1,000, whose floor would be about 0.001. Identical p-values across three different probes are the signature of a floor.
- **Holm correction.** Holm–Bonferroni "across all (comparison × stage) tests" is pre-registered but never shown. With p-values sitting at 0.005–0.010 floors and a family of about 10 or more tests, several could fail Holm.
- **Pre-registered bands not followed.**
  - Step 4 declares SUBSTANTIALLY_SEPARABLE, yet reports a cluster-bootstrap 95% CI of about [0.68, 0.95]. The pre-registered rule for "substantially" is *lower CI ≥ 0.70*. A lower bound of 0.68 falls in the **borderline** band (≥0.60).
  - The H6 CI [0.640, 0.904] is also borderline.
- **Null calibration contradicts the thresholds.** The within-dataset null split N2 has a 95th percentile of 0.71. Random subject splits *inside one dataset* thus reach an AUC above the pre-registered "substantially separable" cut-off of 0.70. If the thresholds had been "calibrated against within-dataset null splits," they could not sit below the null's own 95th percentile.
- **Pre-registration timing.** If the thresholds were calibrated against null splits computed on the same data, they were set after seeing (some of) the data.

**Required fix.**
- Run the full protocol with the pre-registered counts: 10,000 MMD permutations, 1,000 probe permutations, 2,000 bootstrap replicates, all stages.
- Apply Holm and present the full table.
- Timestamp the pre-registration, e.g. via an OSF or a git commit hash of `configs/preregistration.yaml`, and disclose any data exposure before that point.
- If anything deviates, label it "pre-registered protocol with documented deviations" and keep "confirmatory" only for tests executed as specified.
- Consider setting thresholds relative to the null, e.g. identity AUC exceeds the N2 p95 by a pre-set margin, rather than fixed bands.

### M7 [MAJOR-FIXABLE] — Numerical inconsistencies across the audit and results

Each of these must be reconciled before submission.

| Quantity | Value A | Value B | Location |
|---|---|---|---|
| C1 identity AUC at "final stage" | 0.833 (decision-tree text) | 0.783 (H6, harmonization table) | 5.6 |
| C1 identity AUC at baseline | 0.820 (H0 table) | 0.833 (Step 4, P1) / 0.824 (CORAL pre-alignment) | 5.6 |
| C3 WESAD baseline vs stress AUC | 0.899 (Step 7, LOSO) | 0.807 (H0 table, LOSO) | 5.6 / 5.2 |
| H0 vs H1 | identical on all four metrics | H1 is described as "F_ECG only" | Step 2 already restricts to ECG, so H1 is a no-op, or H0 is mislabelled |
| LightGBM robustness | 4.23× (locked) | 3.6× (title note) | Title vs 5.2 |
| LightGBM per-fold ratios | "not preserved" | CI [0.18, 8.27] requires them | 5.2 |
| Separability verdict | "SUBSTANTIALLY_SEPARABLE" | CI lower bound 0.68 < 0.70 threshold | Step 4 |
| Residual signal | "identity AUC 0.82 → 0.78" (Limitations) | "0.833 ... borderline-to-substantially" (decision tree) | 7 vs 5.6 |

The C3 discrepancy (0.899 vs 0.807) probably reflects different feature sets or classifiers (e.g. ECG+EDA vs ECG-only). If so, say so. If not, it points to nondeterminism.

### M8 [MAJOR-FIXABLE] — The H-B2 explanation is a causal overreach and may conflate two different pretraining setups

**Evidence.** Section 4.2 describes "contrastive *cross-modal* pretraining (InfoNCE)." The only paired modalities are ECG and EDA, and by the outline's own systematic search no dataset pairs EDA with confirmed cardiac events, so MIT-BIH has no EDA. Cross-modal ECG↔EDA alignment can therefore only be trained *within WESAD*. Section 5.4 nonetheless explains the null by cross-dataset separability: "you cannot align two datasets whose class labels are the datasets."

**Why it matters.** If H-B2 was intra-WESAD cross-modal pretraining, WESAD–MIT-BIH separability is irrelevant to it. If H-B2 had a cross-dataset component (e.g. ECG-only pretraining across both datasets), the paper must describe it. Either way, "the audit explains why H-B2 was foreseeably null" is a post-hoc causal story, and the simpler competing explanation (about 15 subjects is far too few for contrastive pretraining) is not ruled out. Branch C is also "partial/inconclusive" with identity AUC in the borderline band. That is weak ground for saying the shortcut "would overwhelm" an alignment signal.

**Required fix.** Specify the pretraining data, the positive/negative pairs, and the encoders. Downgrade the language to "consistent with" and name the sample-size alternative. If you want the causal claim, run the discriminating experiment: pretrain on a larger external ECG/PPG corpus and test whether the null persists.

### M9 [MAJOR-FIXABLE] — The baseline is weak and may be handicapped by how missingness is encoded

**Evidence.** The comparison baseline is "LightGBM (no missingness handling)." The mean/zero-imputation baseline is still listed as "to add." The outline never says how missing EDA is presented to LightGBM at test time.

**Why it matters.** The official LightGBM docs say: "LightGBM enables the missing value handle by default," and "LightGBM uses NA (NaN) to represent missing values by default." Zeros are *not* treated as missing unless `zero_as_missing=true`.\[10\] If missing EDA is zero-filled, LightGBM sees physiologically impossible values (tonic EDA = 0 µS, phasic rate = 0) and routes them down whichever branch those values fall into. That is a straw man. Native NaN handling only learns a useful default direction if missingness occurs during training. The fair tree-model counterpart of "modality dropout" is therefore LightGBM trained with random EDA→NaN masking.

**Required fix.** At minimum add:
- (i) LightGBM with NaN at test time;
- (ii) LightGBM trained with random EDA-NaN masking at the same rate as the dropout;
- (iii) separate per-tier models (an ECG-only LightGBM for the reduced condition). This is the obvious deployment baseline, and it bounds what any single "robust" model can achieve;
- (iv) mean imputation plus per-condition recalibration;
- optionally, privileged-information distillation, since PULSE (arXiv 2510.24058) reports 0.994 AUROC on WESAD LOSO *without EDA at inference* using EDA as a privileged teacher.\[11\] That is directly relevant.

If the gated model does not beat (ii) and (iii) on Brier/ECE and AUROC, the architectural claim does not hold.

### M10 [MAJOR-FIXABLE] — The pre-registered PPG-only and respiration-only ablations were substituted on the basis of a false claim

**Evidence.** "WESAD has no PPG or respiration channels in the current extraction pipeline." The pre-registered PPG-only and respiration-only conditions "were substituted with ECG-only and no-dropout." The Limitations section calls them "not executable 'on this dataset'."

**Why it matters.** WESAD's chest RespiBAN records ECG, EDA, EMG, respiration, temperature, and ACC at 700 Hz.\[12\] The wrist Empatica E4 records BVP/PPG at 64 Hz, EDA at 4 Hz, TEMP at 4 Hz, and ACC at 32 Hz.\[11\]\[13\]\[14\] Both channels exist. The substitution is a pipeline choice. Calling them "not executable on this dataset" is factually wrong, and swapping in pre-registered conditions after the fact is a protocol deviation that carries HARKing risk.

The PPG omission also cuts against the device-tier story. The most realistic "lower tier" is a wrist-only device (E4: BVP + EDA + TEMP), not chest ECG without EDA. That is exactly the tier the pipeline cannot represent.

**Required fix.** Either extract E4 BVP (PPG-derived HRV) and RespiBAN respiration and run the pre-registered ablations, or state honestly that they were not run for engineering/time reasons and record this as a deviation. Better still, define device tiers realistically: Tier 1 = chest ECG + EDA; Tier 2 = wrist PPG + EDA; Tier 3 = wrist PPG only.

### M11 [MAJOR-FIXABLE] — The headline "ECG fallback hierarchy" amounts largely to heart rate; clinical and shortcut implications are under-discussed

**Evidence.**
- hrv_mean_nn has mean |SHAP| 2.498, versus 0.697 for the next feature.
- Dropping it collapses C3 from 0.899 to 0.674.
- It is "also WESAD's dominant clinical signal for stress" and the top identity-driving feature.
- Under IG with the EDA gate closed, hrv_mean_nn is #1 (0.537).

**Why it matters.** Mean NN interval is simply 60,000/HR. So the headline robustness claim, "calibration stability under EDA dropout is explained by a stable ECG fallback hierarchy," comes down to: *when EDA is missing, the model falls back to heart rate.* That TSST raises heart rate relative to seated reading is well established. It is not a mechanistic insight.

Worse for the eventual clinical goal, heart rate rises in *both* panic and many cardiac events, and it differs systematically between WESAD (young and healthy; Schmidt et al. 2018 report "15 subjects (12 males, 3 females) with a mean age of 27.5 ± 2.4 years") and MIT-BIH (PhysioNet: Holter recordings from 1975–1979 of "25 men aged 32 to 89 years, and 22 women aged 23 to 89 years," with "approximately 60%" from inpatients). The feature carrying H-B1 is therefore the same one most likely to fail at the panic-vs-cardiac boundary, and the same one that drives dataset identity. This does not invalidate H-B1 as a WESAD-internal result, but it has to be stated plainly.

**Required fix.** Report H-B1 with hrv_mean_nn removed, or with HR-residualized features, to see whether the robustness pattern survives. Add an HR-only baseline, which makes it clear how much the multimodal architecture adds beyond heart rate. Add a sentence to the Discussion on why this matters for the target application.

### M12 [MAJOR-FIXABLE] — The novelty claims overstate the gap in the literature

See Section 6. In brief: calibration under modality removal is directly addressed by Ma et al. (ICML 2023). Modality dropout for WESAD ECG/EDA robustness already exists (CognifySSL v2.0; PULSE ablations).\[15\] Device type is already named as a fairness heterogeneity axis. "Not found addressed anywhere" and "has received little attention" need to be rewritten with specific citations and a precise statement of the delta.

### M13 [MAJOR — venue and scope] — The deadline, page limit, and a three-contribution structure are incompatible with ML4H Findings

**Evidence.**
- The ML4H 2026 call lists "Sep 10: Paper submission deadline." As of today (Sep 30, 2026) the ML4H site says "Paper and demonstration submissions are now closed."\[1\]\[2\]
- "Findings papers can be up to 4 pages at submission (excluding references and appendices)... reviewers will review the work based predominantly on the main text."\[1\]
- The Findings track is non-archival: "Findings papers will not appear in the ML4H proceedings".\[1\]

**Why it matters.** Dec 15 does not match ML4H 2026. The author needs to confirm which venue and cycle the date refers to, e.g. ML4H 2027 or another venue, and check that cycle's rules. Even at the right venue, four pages cannot hold H-B1 with ablations, H-C1, H2a, H-B2, a 10-step audit, and interpretability. The paper currently reads as three loosely connected contributions without a unifying thesis. ML4H Findings does explicitly welcome "insightful negative results" and "exciting preliminary results," so a focused paper would fit well.\[1\]

**Required fix.** Choose one of two papers:
- **Paper A (recommended for Findings):** "Calibration under missing wearable modalities: a gated architecture vs strong baselines on WESAD." H-B1 plus corrected statistics, strong baselines, and PPG/respiration tiers; everything else goes to the appendix.
- **Paper B:** "A dataset-identity audit protocol for proxy-vs-proxy wearable comparisons (WESAD × MIT-BIH)." This is a methods/negative-result paper, which fits Findings' "non-traditional artifacts" remit well, but only once M1 and M6 are fixed.

Submit the other paper separately.

## 4. Minor concerns

- **[MINOR] Wording error in the geometric-mean check.** "The invariant/dropout gap *widens* from 0.73 (arith) to 0.72 (geo)." A ratio moving from 0.73 to 0.72 is essentially unchanged, and if anything the gap is marginally *larger* (a lower ratio means a bigger relative difference). Either way, "widens" and "essentially unchanged" in the same sentence contradict each other. Pick one ("essentially unchanged"). Report geometric means for the dropout-rate sweep too, which currently uses arithmetic means only.
- **[MINOR] Pre-specify the aggregation of ratio metrics.** Ratios are multiplicative. The geometric mean (equivalently, the arithmetic mean of log ratios, with CIs on the log scale) is the natural summary and is less dominated by fold 3 (3.32/4.43). Making the arithmetic mean the headline, when the geometric mean gives a smaller effect, looks like selection. Make log-ratio the primary summary, or state why not.
- **[MINOR] Wilcoxon at n=5.** The "Wilcoxon-floor argument" is correct: the exact two-sided minimum is 2/2⁵ = 0.0625. That makes the test uninformative by design. Pre-registering a test that cannot reject at α=0.05 is a design flaw. Use repeated CV or subject-level resampling instead.
- **[MINOR] The dropout-rate sweep.** p=0.4 gives 1.78× [0.02, 3.54]. A lower bound near zero comes from the symmetric t-interval on a positive-only quantity, which is another reason to use log scale. "Benefit present across tested range" is not supported: every CI includes the no-dropout value (2.21) and the value 1.
- **[MINOR] Nadeau–Bengio corrected dropout test.** With 3 test subjects and 12 training subjects per fold, the correction factor is (1/5 + 3/12) = 0.45 instead of 0.2.\[16\]\[17\] The SE grows by √(0.45/0.2) = 1.5, so t = −2.66 becomes about −1.77 (df=4), p ≈ 0.15. "Currently underpowered, p=0.056" should read "no evidence of a benefit after correcting for CV dependence (p≈0.15)." The point estimate (d_z = −1.19) can still be reported as a direction for future work.
- **[MINOR] Ablation 1 substitution.** "ECG-only droppable (wrong modality)" gives 2.37×, essentially the no-dropout value (2.21×). That is expected, since ECG is never removed at test time. It is a sanity check, not an ablation of the pre-registered hypothesis.
- **[MINOR] Frequency-domain HRV from short windows.** Window length is never stated. If it is 60 s (implied by the MIT-BIH "60-s windows" and the window counts), LF power is not interpretable. The Task Force guidance says "recording of approximately 1 minute is needed to assess the HF components of HRV, while approximately 2 minutes are needed to address the LF component," and prefers 5-minute recordings.\[18\] LF power's SHAP of 0.000 is consistent with this. The IG claim that LF moves from rank 8 to 5 when the EDA gate closes, called "a plausible autonomic-regulation mechanism," should be removed or tagged as unreliable. Also report window overlap: overlapping windows further inflate the effective n.
- **[MINOR] Integrated Gradients with a gate.** With a zero baseline and a closed gate that zeroes EDA inputs, EDA IG = 0 by construction, which is uninformative. State the IG baseline (zero, mean, or blurred), the number of steps, and check completeness. IG rank shifts are sensitive to baseline choice. SHAP on highly correlated HRV features (SDNN, RMSSD, pNN50, HF) spreads attribution unpredictably. Use grouped/modality-level attributions, or report correlation-aware variants.
- **[MINOR] Terminology.** "Modality-invariant model," "gated architecture," and "CanonicalNet" are used interchangeably. Define one name and one architecture diagram. The model is not demonstrably "modality-invariant," since its outputs change with the gate state, so "invariant" is misleading.
- **[MINOR] "Favors, rather than penalizes, lower-tier users."** The data show lower-tier users still lose accuracy (7.5%). The gap is only *smaller* than with the baseline (10.6%). "Penalizes lower-tier users less than a standard model does, under simulated tiers" is accurate. "Favors" is not. The intro's "generalizable claim sentence" also contradicts the H-C1 downgrade in 5.3 and must be aligned with it.
- **[MINOR] Limitation text: "this reflects real uncertainty, not measurement error."** Much of the fold-to-fold variance *is* estimator noise: ECE bias and variance at about 90 windows per fold, ratio instability, and a fold ratio below 1. The honest statement is that the width reflects both between-subject heterogeneity and ECE estimation noise at small n, and that the naive fold-based interval probably still *understates* the true uncertainty.
- **[MINOR] Leaky split disclosure.** If any leaked numbers appeared publicly (an earlier preprint, website, pitch deck, grant, or investor material), the paper should say so and show before/after. The ML4H Findings dual-submission guidance also asks authors to disclose overlapping submitted work in a confidential comment.\[1\]
- **[MINOR] CORAL threshold.** "Drop 0.087, just under the module's 0.10 threshold" invites the question of where 0.10 came from. It is not in the pre-registered threshold list shown. Either pre-register it with a rationale (e.g. relative to N-split variability) or report the drop with a bootstrap CI and no dichotomization.
- **[MINOR] MMD details.** State the kernel, the bandwidth selection (σ=2.55 via the median heuristic?), whether the bandwidth was chosen on the same data used for the test (which inflates Type I error unless MMDAgg-style aggregation is used), and the block structure (subject blocks?). R=1.67 "below R > 2 bar" needs a definition and a pre-registration pointer.
- **[COSMETIC] Section numbering.** 5.5 appears after 5.6, and "5.2b" is non-standard. Remove internal notes from the manuscript ("don't skip this," "use near-verbatim," "Lock the title once..."). Use a consistent × notation. Report p-values to consistent precision.

## 5. Statistical and methodological checklist

| Item | Status | Comment |
|---|---|---|
| Unit of analysis = subject | ⚠️ Partial | CV splits by subject, but CIs, H-C1, and H2a treat windows or folds as independent |
| CIs account for CV dependence | ❌ | Naive t over 5 folds. Use Nadeau–Bengio / repeated CV / subject cluster bootstrap |
| Paired test appropriate for n=5 folds | ❌ | Uncorrected t is anticonservative; Wilcoxon cannot reach p<0.0625 |
| p-values correspond to reported numbers | ❌ | p=0.001 from superseded pipeline |
| Primary metric robust | ❌ | ECE ratio unstable; switch to ΔECE (debiased) + Brier + slope/intercept + AUROC |
| ECE estimator specified | ❌ | Bins, scheme, debiasing missing |
| Calibration procedure leakage-free | ❓ | Not described (which data, per-condition?) |
| Aggregation of ratios | ⚠️ | Arithmetic headline; geometric/log should be primary |
| Mixed-effects random structure | ❓ | Random slopes for tier/model by subject not stated |
| Null results tested for equivalence | ❌ | H2a needs TOST / Bayes factor at subject level |
| Multiple comparisons | ⚠️ | BH on H-B1 from old pipeline; Holm for audit not shown; ablations and sweeps uncorrected (fine if labelled exploratory) |
| Permutation p-values valid | ⚠️ | (b+1)/(m+1) floors match reported values; counts below pre-registration |
| Pre-registration integrity | ⚠️ | Deviations (ablations, permutations, --fast, thresholds vs null) must be tabulated |
| Strong baselines | ❌ | Missing NaN-aware and masked-trained LightGBM, per-tier models, distillation |
| Class balance and prevalence | ❓ | Baseline:stress ≈ 2:1 by protocol duration. Report per-subject counts; ECE depends on prevalence |

## 6. Novelty assessment against the literature

| Claimed novelty | Closest prior work found | Assessment |
|---|---|---|
| Modality dropout for missing-sensor robustness | ModDrop (Neverova et al., TPAMI 2015); CognifySSL v2.0 (*Sensors* 2026): "H1 (Dropout Simulation): Stochastic modality dropout during training enhances the robustness under missing input," evaluated on WESAD with IMU/ECG/EDA/TEMP;\[15\] PULSE (arXiv 2510.24058) includes modality-dropout ablations on WESAD and achieves 0.994 AUROC without EDA at inference\[11\]\[19\]\[20\] | Correctly *not* claimed as novel. Cite these specifically instead of "several wearable missing-sensor robustness papers." PULSE is also a necessary baseline |
| (1) Calibration (ECE) under missing modalities | Ma et al., "Calibrating Multimodal Learning" (ICML 2023): finds "the confidence estimated by current models could even increase when some modalities are corrupted" and proposes the principle that "the confidence should not increase when one modality is removed"; SimMLM (arXiv 2507.19264) reports ECE under missing modalities; Lechuga López et al. (arXiv 2603.02719) on calibration and selective prediction in multimodal clinical models\[21\]\[22\]\[23\] | **Not novel as a framing.** What remains is the empirical study of calibration under missing *wearable physiological* modalities with strong baselines. That is a legitimate but modest Findings contribution. Cite Ma et al. prominently and consider the CML regularizer as a baseline |
| (2) Device-tier equity | Work on fairness in ubiquitous health sensing notes that parity methods "can miss other important subgroup failures arising from heterogeneity factors such as device type" (arXiv 2603.13373, citing Adler et al. 2024 and Yfantidou et al. 2023); Zinzuwadia & Singh, "Wearable devices—addressing bias and inequity," Lancet Digit Health 2022;4(12):e856–e857, noting that wearables "are not used as widely in low-income and minority populations" | "Not found anywhere" is too strong. The narrower gap, *calibration* equity across sensor-availability tiers, may still be open, but it rests on simulated tiers and an accuracy metric. Frame it as a measurement proposal, not a finding |
| (3) Dataset-identity / shortcut audit | Zech et al. 2018; DeGrave et al. 2021; Rabanser et al. 2019; npj Digital Medicine 2024 (shortcut learning from data-acquisition bias across 13 datasets including ECGs, performance "overestimated by up to 20% on average");\[24\] Chaibub Neto et al. (arXiv 1712.03120) permutation test for identity confounding; C2ST (Lopez-Paz & Oquab 2017)\[3\]\[25\] | The components are standard. The *integration* into a staged protocol for proxy-vs-proxy wearable comparisons is a reasonable contribution, and "no wearable panic-prediction paper has published a shortcut audit of this scope" is probably true, since the panic literature is small. But its value depends on fixing M1 and M6 |
| Wearable panic literature | Tsai et al., JMIR Med Inform 2022 (7-day PA prediction, Garmin, accuracy 67.4%–81.3%); JMIR 2025 e69045 (wearable ECG plus psychological assessment, accuracy 71.43%); scoping review (PubMed 41181552) that identified seven studies; McGinnis et al. 2024 (Apple Watch)\[26\]\[27\]\[28\] | The outline's contrast ("next-day risk, single-channel HR") is roughly fair but should cite these specific papers. Note that none use WESAD-style proxies, so "onset-level" in this paper is TSST onset, not panic onset |

**Overall:** the defensible novelty is the *empirical*, wearable-specific evidence on calibration under missing modalities against strong baselines, plus a reusable audit protocol. It is not a new concept.

## 7. Reporting and reproducibility gaps

The work is a development and internal-validation study of a prediction model on proxy data, so **TRIPOD+AI (BMJ 2024;385:e078378; 27-item checklist)** is the relevant reporting guideline. It explicitly supersedes TRIPOD 2015.\[29\] CONSORT-AI and SPIRIT-AI cover randomized trials and trial protocols, so they do not apply. DECIDE-AI covers early-stage live clinical evaluation, so it does not apply either.

Missing TRIPOD+AI-relevant items:
- **Data source and participants:** per-subject window counts, class balance per subject, age/sex (WESAD: "15 subjects (12 males, 3 females) with a mean age of 27.5 ± 2.4 years" per Schmidt et al. 2018), MIT-BIH exclusions (already good), the E4-vs-RespiBAN channel choice.
- **Outcome definition:** "stress" = TSST segment labels. Say whether amusement/meditation were excluded, and how transitions were handled.
- **Predictors:** the full feature list with window length, overlap, filters, R-peak detector, EDA decomposition method (cvxEDA? NeuroKit?), and SQI status.
- **Missing data:** exactly how "missing EDA" is simulated at test time (zeros, NaN, gate closure) *for each model*.
- **Sample size justification:** acknowledge that 15 subjects precludes precise estimation. TRIPOD+AI expects a sample-size rationale.
- **Model development:** architecture, hyperparameters and how they were tuned (nested CV?), seeds, compute, training time.
- **Performance measures:** discrimination plus calibration (slope/intercept, plots), not ECE alone.
- **Open science:** the Findings call says "If you are not sharing code, you must explicitly state this in the paper."\[1\] Provide an anonymized repository (audit/, configs/preregistration.yaml with commit hash), a model card, and a datasheet-style description of the derived feature tables. WESAD and MIT-BIH are public, so full reproducibility is achievable and should be expected.
- **Ethics:** use of public de-identified datasets. The ML4H call requires an IRB statement or an explicit justification of why none is needed.\[1\]
- **Conflict of interest:** the author is a co-founder of E14 Health. Disclose this. It is particularly relevant because the paper motivates a commercial application.

## 8. Questions for the authors

1. What exactly is the "frozen production model" in Step 3: its training data, positive class, and label encoding? Why does it give 0.97 to WESAD stress and 0.024 to MIT-BIH arrhythmias under a label called P(cardiac)?
2. How is missing EDA delivered to LightGBM at test time: 0, NaN, or column removal? Was LightGBM ever trained with missing EDA?
3. Was isotonic/Platt calibration fit per condition? On which subjects? Was it the same for all models?
4. What window length and overlap are used? How many windows per subject and per class?
5. What ECE bin count and binning scheme were used? Did you try a debiased estimator?
6. Where do p=0.001 and BH p=0.00257 come from (which test, which unit, which pipeline)? What is the locked equivalent?
7. Is the "890 observations" = 445 windows × 2 tiers? What random-effects structure was fitted?
8. Was H-B2's pretraining intra-WESAD ECG↔EDA or cross-dataset? What were the pretraining data and the pairs?
9. When was `configs/preregistration.yaml` committed relative to the first audit run? Were the N1/N2 null splits run before the thresholds were fixed?
10. Why do C3 (0.899 vs 0.807) and C1 (0.820 / 0.824 / 0.833 / 0.783) differ across sections?

## 9. Recommendation

**For ML4H Findings, as submitted: Weak Reject (score about 4/10). Confidence: high on statistics and internal consistency, medium on venue fit.**

Justification. The Findings bar favours insight over polish, and this work has real insight potential: calibration-first robustness, honest nulls, and a structured shortcut audit. But a Findings reviewer will stop at these problems:
- (i) a headline p-value from a different pipeline (M2);
- (ii) "substantial" robustness with every CI including 1 and overlapping (M3);
- (iii) an audit whose decisive step appears to read a stress detector as a cardiac detector and misreads AUC=0.22 (M1);
- (iv) "pre-registered, confirmatory" results from a `--fast` run with 50× fewer MMD permutations, reported at the permutation floor, and a verdict band the pre-registered rule does not support (M6);
- (v) at least eight numerical inconsistencies (M7).

None of these requires new data. If the paper is cut down to one contribution (M13), with corrected inference, the strong baselines from M9, and an honest framing, I would expect **weak accept** at a Findings-type venue. The most likely honest headline is: "a gated architecture reduces calibration degradation under EDA loss by point estimate, but on 15 subjects this is not statistically distinguishable from strong baselines. We release a protocol and an audit showing that WESAD-vs-MIT-BIH comparisons are shortcut-prone." That is a publishable, useful Findings paper.

**Fatal (to the specific claims, not the project):** M1, M2, and the "confirmatory" labelling in M6. **Major-fixable:** M3–M5 and M7–M13. **Minor/cosmetic:** Section 4.

## 10. Prioritized, time-boxed fix plan (today Sep 30 → Dec 15; about 10.5 weeks)

**Week 0 (Days 1–2) — Venue triage [blocking].**
- Confirm which venue and cycle "Dec 15" refers to. ML4H 2026 closed Sep 10.\[1\]\[2\]
- Read that venue's page limit and anonymity rules.
- Decide between Paper A (calibration) and Paper B (audit), and draft a one-sentence thesis.

**Week 1 — Fix the fatal issues (M1, M2).**
- Audit Step 3: print the model's training config and label map. Re-run Step 3 with a model actually trained on the cross-dataset contrast, or remove it. Recompute all AUCs as max(AUC, 1−AUC) with subject-cluster CIs.
- Delete p=0.001 / BH 0.00257 and "3.6×" everywhere. Recompute the locked LightGBM-vs-gated test.
- Recover the LightGBM per-fold ratios (the CI requires them).

**Week 2 — Correct the inference (M3, M5, minor stats).**
- Implement the Nadeau–Bengio corrected t-test, 10× repeated 5-fold subject-level CV, and a subject-level cluster bootstrap (2,000 replicates, BCa) for all H-B1 contrasts.
- Switch the primary summary to log-ratio or ΔECE and report both means.
- Refit H-C1 with random slopes, or aggregate to subject level. Add TOST for H2a.

**Week 3 — Metrics (M4).**
- Implement debiased and equal-mass ECE (plus ECE_sweep), Brier, log loss, calibration slope/intercept, AUROC/AUPRC, and reliability diagrams for complete and reduced inputs.
- Document the calibration procedure and move it inside nested subject-level CV.

**Weeks 4–5 — Strong baselines (M9) and pre-registered ablations (M10).**
- Add the LightGBM variants (NaN at test; EDA-NaN masked training), per-tier models, mean imputation + recalibration, an HR-only baseline, and optionally PULSE-style distillation or the Ma et al. CML regularizer.
- Extract E4 BVP and RespiBAN respiration. Run PPG-only and respiration-only tiers.
- Redefine the device tiers realistically (chest ECG+EDA / wrist PPG+EDA / wrist PPG only).

**Week 6 — Full audit run, if Paper B, or as appendix support (M6, M7).**
- Run the full protocol: 10,000 MMD permutations, 1,000 probe permutations, all H-stages, leave-one-stage-out, and the Holm table.
- Produce one canonical results table and reconcile every number in the M7 table.
- Tabulate every deviation from the pre-registration and give the timestamp or commit hash.

**Week 7 — Sensitivity and interpretation (M8, M11, minor).**
- H-B1 without hrv_mean_nn or with HR-residualized features.
- State the window length and drop LF claims if windows are shorter than 2 min.
- Grouped attributions and a stated IG baseline.
- Rewrite the H-B2 explanation as "consistent with," naming the sample-size alternative.

**Week 8 — Writing.**
- Cut to the page limit. Move the secondary contributions to the appendix or a companion paper.
- Rewrite the novelty section with the citations in Section 6.
- Align the intro claim sentence with the H-C1 downgrade. Replace "favors" and "substantial" with interval-supported language.
- Fix the "widens" wording, the terminology, and the section numbering.

**Week 9 — Reporting and reproducibility.**
- TRIPOD+AI checklist in the appendix; model card; data statement; code availability (anonymized repository); ethics/IRB statement; COI disclosure; leaky-split disclosure.

**Week 10 (Dec 1–15) — Internal red-team and buffer.**
- Have a colleague recompute three headline numbers from the repository alone.
- Freeze. Submit.

**If time runs short, the minimum viable submission is:** M1 + M2 + M3 (corrected CIs) + M4 (ΔECE, Brier, AUROC) + M9(ii)/(iii) + M7 reconciliation + scope cut (M13). Everything else can be listed as a limitation or future work.

## Caveats on this review

- I reviewed an outline, not a full manuscript or code. Some concerns (e.g. how missingness is encoded, window length) may be resolved in the implementation. They are flagged as questions where the text is silent.
- My inferences that the CIs are fold-level t-intervals, that 890 = 2 × 445, and that the probe p=0.010 reflects about 100 permutations come from reverse-engineering the reported numbers. They should be confirmed against the code. The Nadeau–Bengio figure (p ≈ 0.15) assumes about 3 test and 12 training subjects per fold, and venue rules must be re-checked for any later cycle.

## Sources

1. [Call for Participation · ML4H 2026](https://ml4h.ahli.cc/submit/call-for-papers/)
2. [ML4H 2026 - AHLI – Association for Health Learning and ...](https://ml4h.ahli.cc/)
3. [Revisiting Classifier Two-Sample Tests - ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv161006545L/abstract)
4. [\[1706.07581\] Cross-validation failure: small sample sizes lead to large error bars](https://arxiv.org/abs/1706.07581)
5. [The Variance Brain Foundation Models Forgot: Third-Order Statistics Predict Cognition Where Billion-Parameter Models Fail](https://arxiv.org/pdf/2606.04010)
6. [\[2012.08668\] Mitigating Bias in Calibration Error Estimation](https://arxiv.org/abs/2012.08668)
7. [Random effects structure for confirmatory hypothesis testing: Keep it maximal - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0749596X12001180)
8. [Exact testing with random permutations](https://arxiv.org/pdf/1411.7565)
9. [www.arxiv.org](https://www.arxiv.org/abs/1603.05766)
10. [Advanced Topics — LightGBM 4.7.0 documentation](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html)
11. [PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensorsfor Embodied Multi-Sensory Learning](https://arxiv.org/html/2510.24058)
12. [\[2502.18733\] Cross-Modality Investigation on WESAD Stress Classification](https://arxiv.org/abs/2502.18733)
13. [GitHub - WJMatthew/WESAD: E4 data, EDA stress detection · GitHub](https://github.com/WJMatthew/WESAD)
14. [Semi-Supervised Generative Adversarial Network for Stress Detection Using Partially Labeled Physiological Data](https://arxiv.org/pdf/2206.14976)
15. [Detect and Repair: Robust Self-Supervised Wearable Sensing Under Missing Modalities - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13119957/)
16. [Nadeau-Bengio Corrected Resampled t-Test Calculator](https://metricgate.com/docs/nadeau-bengio-corrected-resampled-t-test/)
17. [Bayesian Inference of Psychometric Variables From Brain and Behavior in Implicit Association Tests](https://arxiv.org/pdf/2603.16741)
18. [Heart Rate Variability | Circulation](https://www.ahajournals.org/doi/10.1161/01.cir.93.5.1043)
19. [Detect and Repair: Robust Self-Supervised Wearable Sensing Under Missing Modalities - PubMed](https://pubmed.ncbi.nlm.nih.gov/42076526/)
20. [\[1501.00102v1\] ModDrop: adaptive multi-modal gesture recognition](https://arxiv.org/abs/1501.00102v1)
21. [SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](https://arxiv.org/pdf/2507.19264)
22. [\[2603.02719\] An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification](https://arxiv.org/abs/2603.02719)
23. [\[2306.01265\] Calibrating Multimodal Learning](https://arxiv.org/abs/2306.01265)
24. [Shortcut learning in medical AI hinders generalization: method for estimating AI model generalization without external data | npj Digital Medicine](https://www.nature.com/articles/s41746-024-01118-4)
25. [Learning Disease vs Participant Signatures: a permutation test approach to detect identity confounding in machine learning diagnostic applications](https://arxiv.org/pdf/1712.03120)
26. [Journal of Medical Internet Research - Panic Attack Prediction for Patients With Panic Disorder via Machine Learning and Wearable Electrocardiography Monitoring: Model Development and Validation Study](https://www.jmir.org/2025/1/e69045)
27. [Panic Attack Prediction Using Wearable Devices and Machine Learning: Development and Cohort Study - PubMed](https://pubmed.ncbi.nlm.nih.gov/35166679/)
28. [Utility of wearable technology in predicting panic attacks: A scoping review - PubMed](https://pubmed.ncbi.nlm.nih.gov/41181552)
29. [TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods - UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10191161/)
