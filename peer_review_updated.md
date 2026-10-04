# Peer Review of the Revised E14 Health Preprint Outline: "Calibration-Robust Modality-Gated Architecture with Dropout Training for Clinical Wearable Physiological Monitoring"

The revision is not ready for arXiv: it is much more honest than the previous version, but it now openly admits that its headline effect is not statistically separated, its baselines have not been run, and its "pre-registration" cannot be verified, while several sections still make the old, stronger claims.

## TL;DR

- **Verdict: do not post yet.** Of the 13 prior concerns, about 1 is resolved (M1, at the interpretation level), 10 are partially resolved (acknowledged in text, analysis pending), and 2 are unresolved (M9 baselines, M13 scope). The revision also introduces at least 14 internal contradictions where updated and un-updated sections disagree. As a hypothetical ML4H Findings / CHIL submission today I would score it 4/10 (weak reject, confidence 4/5). If it is completed as planned, and the gated model survives the per-tier and heart-rate-only baselines, I would expect 6–7/10.
- **The core scientific risk is that the architectural claim is mostly a heart-rate claim.** Removing `hrv_mean_nn` cuts within-WESAD AUC from 0.899 to 0.674. Heart rate alone carries about 50% of total LightGBM SHAP mass. Published HRV-only, leave-one-subject-out WESAD classifiers already reach F1 0.813–0.859 (Prajod & André, ICMLA 2022: RFC 0.813, SVM 0.832, MLP 0.859). A per-tier ECG-only model has an ECE ratio of about 1 *by construction*. The pre-stated falsification rule (gated must beat baselines ii and iii on Brier, ECE and AUROC) is the outline's best feature. In my judgement it has a substantial chance of being triggered, so the authors should write the paper so that it is publishable either way.
- **Minimum before posting (about 2–3 weeks for 1–2 engineers):** put the project under git and freeze numbers to one commit. Remove or substantiate every "pre-registered" claim. Run the cheap LightGBM baselines (NaN-aware, EDA-masked, per-tier ECG-only, HR-only). Report absolute ECE_reduced, ΔECE with a debiased estimator, Brier and AUROC, using repeated subject-level CV with a corrected test or a subject-level paired test. Rerun the full audit. Cut or re-instantiate H-C1 (WESAD's own Empatica E4 wrist BVP+EDA makes realistic tiers possible without new data). Fix every contradiction listed in Section 5. Disclose the E14 Health conflict of interest on page 1.

---

## 1. Executive summary

The revision is a real improvement in scientific integrity. It retracts the superseded p=0.001 / BH p=0.00257, relabels the Step 3 "frozen production model" as a WESAD stress detector, reports per-fold numbers, admits that every naive CI includes 1 and that the intervals overlap, computes a Nadeau–Bengio-corrected p≈0.15 for dropout, documents the `--fast` audit deviations, and adds a falsification rule for the architectural claim.

The revision also changes what the paper *is*. With the inflated statistics removed, what remains is a set of point estimates on 15 subjects, a list of roughly 20 pending analyses, and a framing that the authors' own corrections no longer support: "Calibration-Robust" in the title, "degrades gracefully and equitably… with pre-registered evaluation discipline" in the Discussion, and "has received little attention" in the Introduction. The revision acknowledged most problems but fixed few. In an outline, an acknowledgement is a promise, not a result.

Trajectory: integrity is up sharply and evidentiary strength is down, which is the correct direction. The paper should not go to arXiv until (i) the contradictions are removed, (ii) the cheap decisive baselines are run, and (iii) the pre-registration language matches what can be documented. If the gated model does not beat a per-tier ECG/HR-only model on reduced-modality absolute calibration and discrimination, the most defensible preprint is a candid evaluation paper, with the dataset-identity audit as a separate short note.

---

## 2. Scorecard: prior concerns M1–M13

| ID | Concern | Status | One-line evidence |
|---|---|---|---|
| M1 | Step 3 "frozen production model" was a WESAD stress detector; AUC 0.22 misread | **Resolved (interpretation); residual cleanup** | Step 3 relabelled "Frozen WESAD-internal baseline… (diagnostic only)"; separability max(AUC, 1−AUC)=0.78 adopted; Branch A exclusion moved to Step 4. Residual: purge `PATHOLOGY_LIKELY` labels from code and artifacts; Step 3 adds little and could be cut. |
| M2 | Headline p from superseded unlocked pipeline | **Resolved in text; replacement test pending** | p=0.001 / BH 0.00257 removed and attributed to likely window-level pseudoreplication; the locked LightGBM-vs-gated paired test is "pending", so the headline has no inferential support yet. |
| M3 | No CI supports "substantial" robustness | **Partially resolved** | 5.2 now says CIs "all include 1 and overlap", but the title, Discussion and Conclusion still assert robustness. |
| M4 | ECE ratio fragile; need ΔECE, Brier, slope, AUROC, debiased ECE | **Partially resolved** | ΔECE computed (0.168/0.065/0.023) and ratio-of-means vs mean-of-ratios disclosed. Debiased ECE, Brier, log loss, AUROC/AUPRC, slope/intercept and reliability diagrams are all "to add". No discrimination metric is reported anywhere. |
| M5 | Pseudoreplication in H-C1 and H2a | **Partially resolved** | "890 = 2 × 445" and the "well-powered" error acknowledged; maximal random-effects refit and subject-level TOST pending. 5.3 still says "significant tier×model interaction". |
| M6 | "Pre-registered, confirmatory" audit run in `--fast` | **Partially resolved; integrity issue sharpened** | Good deviations table; "confirmatory" reserved for tests run as pre-registered ("currently none"). The admission that `configs/preregistration.yaml` is not in version control makes *every* pre-registration claim unverifiable, yet 4.3, 7 and 8 still use the term. |
| M7 | Eight numerical inconsistencies | **Partially resolved; new ones introduced** | Reconciliation box explains C1 0.833/0.820/0.824 and C3 0.807/0.899; MIT-BIH 48→40 documented. New: "LightGBM CI below 1" (it is [0.18, 8.27]); intro says Nadeau–Bengio is pending while 5.2b reports it; Limitations p=0.056 vs 5.2b p≈0.15. |
| M8 | H-B2 causal overreach | **Partially resolved** | Retracted in 5.4 but still present in Intro Angle 3 ("the audit that explains why") and the 5.6 Motivation ("why H-B2 was null"). |
| M9 | Weak LightGBM baseline | **Unresolved (plan only)** | Six baselines (i)–(vi) listed with an explicit decision rule; none run. This is the decisive experiment. |
| M10 | False claim that WESAD lacks PPG/respiration | **Partially resolved** | Corrected and labelled a protocol deviation with HARKing risk; neither extraction nor the deviations-table entry is done. |
| M11 | "ECG fallback" is essentially heart rate | **Partially resolved** | Honest reading added ("the model falls back to heart rate"); HR-only baseline and HR-removed H-B1 pending. 5.5 still says "Signal is distributed across both modalities, which motivates the multi-modal architecture." |
| M12 | Overstated novelty | **Largely resolved in Novelty note; contradicted elsewhere** | Intro still says "has received little attention"; Lechuga López et al. mis-described as missing-modality work; SimMLM's learnable gating not acknowledged as prior art for *gating*. |
| M13 | Venue/scope; three loosely connected contributions | **Unresolved** | "three-contribution structure retained"; no unifying thesis; H2a, H-B2 and interpretability still included. |
| New | No version control; leaky-split history; ratio rewards worse complete-condition calibration; suspicious coupling-lag attribution; H-C1 tiers simulated only from ECG+EDA | **Newly identified** | See Section 4. |

---

## 3. Strengths

1. **A pre-stated falsification rule for the main claim.** "If the gated architecture does not beat baselines (ii) and (iii) on Brier and ECE and AUROC under the reduced-modality condition, the architectural claim does not hold and must be downgraded." Few wearable-ML papers commit to this in advance. Timestamp it in git *before* running the baselines and report the result whichever way it goes.

2. **Transparent per-fold reporting.** The per-fold ratios (dropout [1.54, 1.56, 3.32, 1.02, 0.60]; no-dropout [2.01, 1.69, 4.43, 2.19, 0.74]) let a reader recompute everything. I reproduced the means, t-statistic, d_z and CIs from them (Section 6).

3. **Correct self-diagnosis of design-level statistical flaws.** "Pre-registering a test that cannot reject at α=0.05 is a design flaw" (Wilcoxon floor 0.0625 at n=5) is correct and rarely acknowledged. So is the recognition that LightGBM's per-fold SD (~3.3) makes the old p=0.001 implausible.

4. **Honest mechanistic reading of interpretability.** The outline recognizes that mean NN interval = 60,000/HR, that HR rises in both panic and cardiac events, and that HR drives dataset identity. That is the most clinically important observation in the document.

5. **A good audit deviations table.** Permutation counts (200/10,000; ~100/1,000), skipped stages, missing Holm, the code-vs-preregistration mismatch (Step 4 compares the point estimate 0.833 to 0.70 instead of the lower CI), and N2 p95 (0.71) above the cut-off are all disclosed. With Phipson–Smyth (b+1)/(m+1) p-values, 200 permutations give a floor of 1/201 ≈ 0.005 and ~100 give 1/101 ≈ 0.010, which explains the "identical p=0.010 across probes".\[1\]

6. **Within-dataset sanity checks.** C3 (WESAD LOSO 0.899) and C4 (MIT-BIH NSR vs arrhythmic 0.868) separate within-dataset signal from cross-dataset identity.

7. **Correct retraction of LF over-interpretation.** This matches the Task Force guidance that a recording should cover at least ten cycles of the lowest frequency of interest (≈1 min HF, ≈2 min LF, 5 min short-term standard).\[2\]\[3\]

8. **Appropriate clinical guardrails and reporting plan.** "No claim should read as 'this model distinguishes panic from cardiac events'"; a TRIPOD+AI-aligned reporting section; a leaky-split disclosure obligation that extends to investor materials; and numerical versioning with a changelog.

---

## 4. Weaknesses

### 4.1 Critical (block posting)

**C1. The headline claim contradicts the paper's own evidence in at least five places.**
- *Evidence:* The title says "Calibration-Robust". The Discussion says "What's established: a model architecture that degrades gracefully and equitably under missing sensors… with pre-registered evaluation discipline". The Conclusion says "with pre-registered subject-level evaluation". Against this, 5.2 says CIs "all include 1 and overlap" and 5.2b gives a corrected p≈0.15.
- *Why it matters:* arXiv versions are permanent, and readers quote the title and Discussion, not the 5.2b caveats. A co-founder-authored paper whose title asserts an unsupported property invites justified criticism and may later appear in commercial materials.
- *Fix:* Rewrite the title, abstract, Discussion and Conclusion to match the strongest *supported* statement (Sections 5 and 10).

**C2. The pre-registration claims cannot be verified and are partly contradicted.**
- *Evidence:* 4.3 says "decided and pre-registered before results were seen", and Limitations says "pre-registered before results". But 5.6 says "configs/preregistration.yaml is not currently in git", and N2 p95 = 0.71 > 0.70 "is prima facie evidence thresholds set after seeing null data". The "leaky-split case study (before/after)" also implies that leaky-split results were seen before the subject-level protocol was finalized.
- *Why it matters:* An unverifiable pre-registration claim is worse than none.
- *Fix:* Remove "pre-registered" everywhere unless there is a dated external artifact (email, OSF, cloud-drive revision history). Replace it with "analysis plan specified in a configuration file prior to the reported runs; not independently timestamped; see deviations table". Register the baseline plan on OSF or in a signed git tag before running baselines.

**C3. Without version control there is no reproducible artifact.**
- *Evidence:* "the project is not under version control"; numbers come from multiple runs (locked pipeline, `--fast` audit 2026-09-30, H-C1 reruns).
- *Fix:* Run `git init` today. Commit configs, seeds and an environment lockfile, tag `preprint-v1`, regenerate *every* number from one scripted run, and archive a snapshot (e.g., a Zenodo DOI).

**C4. The decisive baselines are not run, and at least two are cheap.**
- *Evidence:* Baselines (i)–(vi) are "required before architectural claim lands"; the HR-only baseline is "to add".
- *Why it matters:* The per-tier ECG-only LightGBM (iii) and the HR-only model take hours of compute on ~445 windows and can decide the paper's framing.
- *Fix:* Run (i), (ii), (iii), (iv) and HR-only before posting. (v) PULSE-style distillation and (vi) CML can wait for the venue version if labelled as future work.

**C5. No discrimination metric is reported for the headline comparison.**
- *Evidence:* The H-B1 table contains only ECE; the outline itself notes "Calibration without discrimination meaningless (prevalence fallback ~1/3)."
- *Why it matters:* A model that degrades to predicting the base rate has low ECE and so looks "calibration-robust". PULSE reports 0.994 AUROC without EDA at inference on WESAD under LOSO, which is the reference point readers will use.\[4\]
- *Fix:* Report AUROC, AUPRC, Brier and log loss for complete vs reduced conditions, per model, with subject-level uncertainty.

### 4.2 Major (must fix before venue submission; ideally before posting)

**MA1. The ECE ratio rewards worse complete-condition calibration.**
- *Evidence:* ECE_complete is 0.087 (LightGBM), 0.095 (gated no-dropout) and 0.098 (gated + dropout). The gated models are *worse* calibrated on complete data, which inflates the denominator. Absolute ECE_reduced (0.255 / 0.160 / 0.121) is the clinically relevant quantity.
- *Fix:* Make reduced-condition Brier plus debiased ECE primary, with ΔECE secondary and any log-ratio tertiary.

**MA2. The per-tier comparison is structurally unfair to the ratio.**
- *Evidence:* A per-tier ECG-only model never sees EDA, so its ratio is ≈1 by construction.
- *Fix:* Compare on *absolute* reduced-condition Brier, ECE and AUROC. That is the deployment question: one gated model, or one model per device configuration?

**MA3. Heart-rate dominance undermines the multimodal story.**
- *Evidence:* LightGBM SHAP gives `hrv_mean_nn` 2.498 of 4.969 total (≈50%); the EDA features sum to 2.168 (≈44%); the other HRV features to 0.253 (≈5%); coupling to 0.050 (≈1%). Removing `hrv_mean_nn` drops C3 from 0.899 to 0.674. In Schmidt et al.'s WESAD paper, ECG-only HR/HRV features with LDA reach 85.44% accuracy (F1 81.31%) for stress vs non-stress under LOSO, and µHR is the second most important feature overall.\[5\] Prajod & André (ICMLA 2022) report HRV-only LOSO F1 of 0.813 (RFC), 0.832 (SVM) and 0.859 (MLP) on 60-s windows with 50-s overlap.
- *Fix:* Run an HR-only model, H-B1 with `hrv_mean_nn` removed, and H-B1 with HR-residualized HRV. Replace the "distributed across both modalities" sentence with the quantified split.

**MA4. H-C1 measures simulated tiers that correspond to no real device.**
- *Evidence:* The tiers are simulated from ECG+EDA features; "current H-C1 tiers instantiate none of Tiers 2/3." A pseudoreplicated p=0.000253 on 890 observations from 15 subjects is not evidence.
- *Fix (preferred):* Use WESAD's hardware. The chest RespiBAN and wrist Empatica E4 (BVP 64 Hz, EDA 4 Hz) were recorded simultaneously on the same 15 subjects, so Tier 1 = chest ECG+EDA, Tier 2 = wrist BVP+EDA and Tier 3 = wrist BVP only can be built without new data. Schmidt et al. report BVP-only performance comparable to ECG-only.\[5\]
- *Fix (alternative):* Demote H-C1 to an appendix "measurement protocol" without "equity".

**MA5. Inference rests on 5 folds (df=4; Wilcoxon floor 0.0625; corrected p≈0.15).**
- *Fix:* Use 20× repeated subject-level 5-fold CV with the Bouckaert–Frank corrected test.\[6\]\[7\] Add a subject-level paired test on per-subject Brier or log-loss differences: n=15, exact sign-flip permutation over 2^15 = 32,768 sign patterns, giving a floor near 6×10⁻⁵ instead of 0.0625. Per-subject ECE (~30 windows) is too noisy, but Brier and log loss are well defined per subject.

**MA6. Calibration fitting is unspecified.**
- *Evidence:* "isotonic on ~12 subjects overfits; prefer Platt". It is not stated whether recalibration is per condition, nested, or shared. Per-condition recalibration would mean the ratio measures the recalibrator; calibration fit on test folds would leak.
- *Fix:* Specify the protocol and report pre- and post-recalibration numbers.

**MA7. The audit's Branch C decision is fragile.**
- *Evidence:* The Step 4 lower CI is 0.68 against the 0.70 boundary, and N2 p95 is 0.71.
- *Fix:* After the full run, report the band with its sensitivity to seeds and bootstrap replicates. Adopt null-relative thresholds via a dated amendment.

**MA8. The Holm correction is infeasible at `--fast` counts.**
- *Evidence:* With ~100 probe permutations the floor is ≈0.0099. For a family of 6 or more tests, Holm's first threshold (0.05/6 ≈ 0.0083) is below the floor, so nothing can reject.
- *Fix:* Rerun with the 1,000/10,000 permutations specified in the plan.

**MA9. The PPG/respiration substitution (M10) is open.**
- *Fix:* Extract E4 BVP and RespiBAN respiration and run the ablations specified in the plan (this also serves MA4), or log the substitution with its decision date.

**MA10. The novelty positioning still has errors.**
- *Evidence:* Lechuga López et al. (arXiv 2603.02719) studies calibration and selective prediction in multimodal ICU classification; it is not a missing-modality paper.\[8\] SimMLM (arXiv 2507.19264) uses "a dynamic, learnable gating mechanism that automatically adjusts each modality's contribution in both full and partial modality settings", which is prior art for gating.\[9\]
- *Fix:* Recharacterize Lechuga López et al., and add learnable modality gating to "Explicitly not claimed as new".

**MA11. The conflict of interest is not yet on the manuscript.**
- *Fix:* State the E14 Health co-founder COI on page 1, and say whether results have appeared in investor or marketing materials.

### 4.3 Minor

- **m1.** "LightGBM CI below 1" is false; the CI is [0.18, 8.27]. Write "includes 1".
- **m2.** The fold-5 dropout ratio of 0.60 means reduced was *better* than complete. Flag it as estimator noise or fold composition rather than averaging it silently.
- **m3.** `coupling_lag_sec` (an ECG–EDA cross-correlation feature) rises to IG rank 3 (0.065) when ECG is zeroed. A coupling feature should be undefined without ECG. This suggests a sentinel or default value acting as a missingness indicator, or coupling computed upstream of zeroing. Audit the pipeline.
- **m4.** H-C1 design is unclear. 890 = 2 × 445 implies two observations per window, but a tier×model interaction needs four cells. State whether the accuracy drop is absolute or relative, and whether 3.1 points is clinically meaningful.
- **m5.** The H2a TOST needs an *a priori* SESOI. Report the minimum detectable effect (d_z ≈ 0.78).
- **m6.** The dropout-rate sweep is non-monotonic with every CI including 1 and 2.21; report "no detectable dose–response" with log-scale CIs (the 0.02 lower bound is a symmetric-interval artifact).
- **m7.** The H-B2 "structural gap" claim needs a documented, dated systematic dataset search.
- **m8.** For MMD, specify the kernel, the bandwidth-selection data (a median heuristic on the same data affects Type I error), subject-block permutation, and the R>2 rationale.
- **m9.** BCa cluster bootstrap with 15 clusters is unstable. Use permutation or sign-flip tests for inference.

### 4.4 Cosmetic

- "CanonicalNet" (5.3) and "invariant/dropout gap" (5.2) violate the 4.2 terminology lock.
- Section order is 5.3 H-C1, 5.4 H-B2, 5.6 audit, 5.5 interpretability, despite the renumbering note.
- "Priority order if time is tight before Dec 15" refers to an obsolete deadline.
- Strip reviewer-response annotations ("per M11", "M9 to-do", "(corrected)", "(unchanged)") and title notes.
- Standardize × and p precision (2.21× vs 2.21; p=0.063 vs 0.0625).

---

## 5. Internal-consistency audit

| # | Location A (current wording) | Location B (conflicting wording) | Correct replacement |
|---|---|---|---|
| 1 | Novelty note: "Prior 'has received little attention' was an overclaim." | Intro claim sentence: "its effect on calibration… has received little attention" | "Calibration under missing modalities has been framed and addressed in general multimodal learning (Ma et al., 2023; SimMLM, 2025); empirical evidence for wearable physiological signals against strong reduced-sensor baselines remains limited." |
| 2 | 5.4: H-B2 pretraining is intra-WESAD; causal story "retracted" | 5.6 Motivation: "which is why H-B2 tried contrastive pretraining as a workaround, and why H-B2 was null" | "Class and dataset are perfectly collinear in any cross-dataset regime; this motivates the audit. H-B2 (intra-WESAD) is unrelated to this collinearity, and its null is most plausibly explained by small N." |
| 3 | 5.3: H-C1 downgraded to "consistent with H-C1 under simulated tiers" | 5.6 Motivation: "the equity/device-tier finding (H-C1)" | "the simulated device-tier measurement protocol (H-C1)" |
| 4 | Intro Angle 3: "the audit that explains why [H-B2 failed]" | 5.4: "Earlier 'audit explains H-B2'… retracted" | "Angle 3: audit the cross-dataset proxy pairing before any cross-dataset claim is made." |
| 5 | 5.2: "CIs all include 1 and overlap"; 5.2b corrected p≈0.15 | Discussion: "What's established: a model architecture that degrades gracefully and equitably under missing sensors… with pre-registered evaluation discipline" | "What the data show: point estimates in which the gated model's reduced-modality calibration degrades less than a LightGBM baseline's, without statistical separation at n=15; equity and pre-registration claims are not supported." |
| 6 | 5.2b: "After correcting for CV dependence (Nadeau-Bengio), p ≈ 0.15" | Limitations: "Dropout effect… lands at p=0.056"; title note "paired t p=0.056" | "uncorrected paired t p=0.056; Nadeau–Bengio-corrected p≈0.15; exact Wilcoxon floor 0.0625 at n=5." |
| 7 | Intro: "the Nadeau-Bengio-corrected paired test… pending" | 5.2b: Nadeau–Bengio already computed for dropout | "pending for LightGBM vs gated; computed for dropout vs no-dropout (p≈0.15)." |
| 8 | 5.6: preregistration file "not currently in git"; N2 p95 0.71 > 0.70 | Limitations: thresholds "calibrated against within-dataset null splits and pre-registered before results" | "Thresholds were specified in a configuration file before the reported audit run; the file is not independently timestamped, and the N2 null p95 exceeding the cut-off suggests the thresholds may have been informed by earlier null data." |
| 9 | 5.6 (as above); leaky-split case study implies earlier results seen | 4.3: "Subject-level cross-validation, decided and pre-registered before results were seen" | "Subject-level 5-fold cross-validation was adopted after a record-level split was found to leak (Appendix X)." |
| 10 | 5.2: no statistically separated effect | Working title: "Calibration-Robust…"; Conclusion "with pre-registered subject-level evaluation" | See Section 10 title; delete "pre-registered". |
| 11 | 5.6 Actions: "do not report device-tier equity as clinical fairness" | Discussion: "degrades… equitably" | Delete "equitably". |
| 12 | 4.2 terminology lock: "modality-gated architecture" | 5.3 "using CanonicalNet"; 5.2 "invariant/dropout gap" | "modality-gated model"; "no-dropout/dropout gap" |
| 13 | 5.2 table: LightGBM CI [0.18, 8.27] | 5.2 headline: "LightGBM CI below 1" | "LightGBM CI includes 1 (lower bound 0.18)" |
| 14 | 5.5 SHAP: `hrv_mean_nn` ≈50% of SHAP; "model falls back to heart rate" | 5.5: "Signal is distributed across both modalities, which motivates the multi-modal architecture." | "Signal is concentrated in heart rate (~50% of SHAP mass) and EDA (~44%); HRV beyond heart rate contributes ~5%." |
| 15 | Venue deferred; ML4H 2026 closed | Notes: "Priority order if time is tight before Dec 15" | "pre-posting priority order" |
| 16 | 5.5/5.6: `hrv_mean_nn` drives both WESAD signal and dataset identity | 5.6 and Notes: "--fast audit… confirms H-B1/H-C1 unaffected" | "H-B1/H-C1 are within-WESAD and outside the audit's scope; the audit does not test them, though their shared reliance on heart rate bears on interpretation." |
| 17 | 5.2: headline paired test pending | Related Work: "reported with CV-dependence-corrected inference" | Keep only once the corrected tests are done. |
| 18 | 5.3 downgrade; Limitations (no real tiers) | Novelty note: "Defensibly new… (2) device-tier calibration equity" | "(2) a protocol for measuring calibration across sensor configurations, demonstrated on [real WESAD chest vs wrist / simulated] tiers." |

---

## 6. Statistical and methodological assessment

### 6.1 Verification of reported computations

| Quantity | Reported | My recomputation | Verdict |
|---|---|---|---|
| Dropout mean of per-fold ratios | 1.61× | 8.04/5 = 1.608 | ✓ |
| No-dropout mean ratio | 2.21× | 11.06/5 = 2.212 | ✓ |
| Dropout 95% CI (t, df=4) | [0.32, 2.90] | SD 1.037, SE 0.464, ×2.776 | ✓ |
| No-dropout 95% CI | [0.52, 3.90] | SD 1.36, SE 0.608 | ✓ |
| LightGBM CI | [0.18, 8.27] | Implies SD ≈ 3.26 | ✓ |
| Geometric means | 1.37 / 1.89 | exp(mean ln) = 1.373 / 1.894 | ✓ |
| Gap ratios | 0.73 / 0.72 | 0.728 / 0.725 | ✓ |
| Per-fold deltas | [−0.47, −0.13, −1.10, −1.17, −0.14] | Third is −1.11 (3.32−4.43) | ✓ (rounding) |
| Mean delta (SD) | −0.60 (0.51) | −0.602 (0.506) | ✓ |
| Paired t | −2.662, p=0.056 | −0.602/(0.506/√5) = −2.66; crit(df=4) 2.776 | ✓ |
| d_z | −1.19 | −0.602/0.506; approximate 95% CI about −2.3 to −0.05 | ✓ (very imprecise) |
| Wilcoxon | W=0, p=0.0625 | All 5 negative → 2/2⁵ = 0.0625 (the floor) | ✓ |
| Ratio of means | 2.93 / 1.68 / 1.24 | 0.255/0.087; 0.160/0.095; 0.121/0.098 | ✓ |
| ΔECE | 0.168 / 0.065 / 0.023 | Differences of fold means | ✓ (see 6.3) |
| Nadeau–Bengio | t≈−1.77, p≈0.15 | (1/5 + 3/12) = 0.45 vs 0.2; √2.25 = 1.5; −2.662/1.5 = −1.775; df=4 → p≈0.15 | ✓ |
| H2a power | d_z ≥ ~0.78 | n=15 paired, α=0.05, 80% | ✓ |
| Windows | ~90/fold | 445/5 ≈ 89; ≈30/subject, consistent with non-overlapping 60-s windows and stress prevalence ≈1/3 | ✓ (confirm) |

The arithmetic is sound. The problems are with inference and design.

### 6.2 Is the Nadeau–Bengio correction the right tool here?

1. **Origin.** Nadeau & Bengio (Machine Learning, 2003) derived the corrected resampled t-test for J *random* train/test splits, replacing 1/J with (1/J + n₂/n₁) as a heuristic for training-set overlap.\[10\] They suggested n₁ be 5–10× n₂; here n₁/n₂ = 4, slightly outside that range.\[7\]\[11\]
2. **Single k-fold CV.** Bouckaert & Frank (PAKDD 2004) applied the same statistic to *repeated* k-fold CV (factor 1/(r·k) + n₂/n₁) and found 10×10-fold most replicable.\[6\]\[7\] Applying it to a *single* 5-fold run is common practice but heuristic, and it leaves df=4, which is the dominant problem. Treat it as a conservative sanity check, not as the primary test.
3. **Paired comparison on identical folds.** This is the case the correction was designed for. Nadeau & Bengio explicitly treat the difference between two algorithms, and per-fold *differences* stay correlated across folds through overlapping training sets.\[11\] Pairing removes shared fold-difficulty variance but not that dependence, so the correction is appropriate in kind.
4. **Better tools.**
   - (a) **Repeated subject-level 5-fold CV** with the Bouckaert–Frank correction. At r=20 the factor is 1/100 + 0.25 = 0.26 with df=99. The n₂/n₁ = 0.25 term is an irreducible floor at 15 subjects.
   - (b) **Subject-level sign-flip tests** on per-subject proper-scoring-rule differences.
   - (c) **Nested-CV intervals** (Bates, Hastie & Tibshirani, JASA 2024), which show that naive CV intervals can badly under-cover.\[12\]

   I would not predict that the dropout effect becomes significant under (a).

### 6.3 ECE: summary statistics, bias and metric choice

- **ΔECE is linear, so the averaging order does not change the point estimate.** 0.168 / 0.065 / 0.023 are valid, but report the per-fold Δs for intervals.
- **The ratio is non-linear.** The mean of ratios vs ratio of means differs by 44% for LightGBM (4.23 vs 2.93) and 30% for dropout (1.61 vs 1.24). The ordering LightGBM > no-dropout > dropout holds under every summary. That is the one robust qualitative statement.
- **Estimator bias.** Roelofs et al. (AISTATS 2022, PMLR 151:4036–4054) report that "ECE_bin is more strongly biased for perfectly calibrated models", and recommend two estimators: the debiased estimator (Bröcker, 2012; Ferro & Fricker, 2012) and ECE_SWEEP, which uses equal-mass bins. The complete-condition ECEs are closer to perfect, so the bias inflates denominators more than numerators and does not cancel in the ratio.
- **The dropout ΔECE (0.023) is probably within noise.** It is below the fold-level ECE SDs (0.035–0.048).
- **Recommended primary endpoint:** reduced-condition Brier or log loss (proper, and not gameable by prevalence fallback), with debiased ECE and AUROC co-primary, at the subject level.

### 6.4 Mixed effects, power and the audit

- **H-C1.** A maximal `(1 + tier + model + tier:model | subject)` model (Barr et al., 2013) will likely be singular with 15 subjects. Pre-specify a simplification sequence, or test the per-subject difference-of-differences directly. State the model family.
- **H2a.** At n=15 only large effects can be excluded. Use a TOST with a pre-justified SESOI or a Bayes factor with a stated prior.
- **Audit.** The components are standard: C2ST (Lopez-Paz & Oquab), the identity-confounding permutation test (Chaibub Neto et al., arXiv 1712.03120, originally for *subject* identity and here adapted to *dataset* identity; say so), MMD and CORAL.\[13\] The finding that WESAD and MIT-BIH are separable is expected given different decades, hardware, sampling rates (700 vs 360 Hz), ages and settings. The audit's most informative result is that dropping `hrv_mean_nn` reduces within-WESAD C3 from 0.899 to 0.674, which ties it to H-B1 more tightly than the outline admits. Report p-values as (b+1)/(m+1) with their floors.

---

## 7. Novelty and literature positioning

### 7.1 Verification of cited works

| Citation | Verified? | What it actually says / note |
|---|---|---|
| Ma et al., ICML 2023, arXiv 2306.01265 | ✓ (ICML oral) | Confidence can rely on partial modalities; the principle that "the confidence should not increase when one modality is removed", implemented as the CML regularizer.\[14\]\[15\] Targets confidence monotonicity rather than ECE. Correct baseline. |
| SimMLM, arXiv 2507.19264 | ✓ | DMoME with learnable gating plus the MoFe ranking loss; calibration error reported across 15 missing/full settings.\[9\]\[16\] **Prior art for gating.** |
| Lechuga López, Shamout, Rudner, arXiv 2603.02719 | ✓ (v4, May 2026) | Calibration and selective prediction in multimodal ICU classification; class-dependent miscalibration.\[8\] **Not missing-modality work.** |
| CognifySSL v2.0, Sensors 2026 (26(8):2419) | ✓ | Modality dropout plus masked pretraining on WESAD, with presence prediction and reconstruction.\[17\]\[18\] Verify the exact modality list before quoting. |
| PULSE, arXiv 2510.24058 | ✓ | EDA-teacher privileged transfer; "0.994 AUROC and 0.988 AUPRC… without EDA at inference" on WESAD LOSO; CVPR 2026 workshop.\[4\] Correctly quoted. |
| ModDrop, arXiv 1501.00102 | Not re-verified here | Confirm the TPAMI year before citing. |
| arXiv 2603.13373 (Roy, Sharma, Salekin) | ✓ (v1) | Flare framework; names "device usage" heterogeneity and evaluates "Sensor Band (Empatica, Basis)" in OhioT1DM; cites Adler et al. 2024.\[19\] A later version is retitled.\[20\] Cite a specific version; the device axis is evaluated on glucose data, not stress wearables. |
| Zinzuwadia & Singh, Lancet Digit Health 2022;4:e856–e857 | ✓ | Comment on bias and inequity in wearables. Supports the general equity motivation only.\[21\] |
| Chaibub Neto et al., arXiv 1712.03120 | ✓ | Subject-identity confounding permutation test.\[13\] |
| Nadeau & Bengio 2003; Bouckaert & Frank 2004 | ✓ | See 6.2. |
| Roelofs et al., AISTATS 2022 (PMLR 151:4036–4054) | ✓ | "ECE_bin is more strongly biased for perfectly calibrated models"; recommends the debiased estimator (Bröcker 2012; Ferro & Fricker 2012) and equal-mass ECE_SWEEP. |
| Varoquaux, NeuroImage 180(A):68–77, 2018 | ✓ | "sample sizes of many neuroimaging studies inherently lead to large error bars, eg ±10% for 100 samples… The standard error across folds strongly underestimates them." |
| Phipson & Smyth 2010 | ✓ | (b+1)/(m+1).\[1\] |
| Task Force 1996 | ✓ (secondary sources) | ≥10 cycles of the lowest frequency; 5-min standard.\[3\]\[22\] |
| TRIPOD+AI, BMJ 2024;385:e078378 | ✓ | 27-item checklist plus a 13-item abstract checklist.\[23\] |
| Schmidt et al., ICMI 2018 (WESAD) | ✓ | 15 subjects; RespiBAN plus E4; LOSO; binary accuracy up to 93%.\[5\]\[24\]\[25\] |
| MIT-BIH Arrhythmia DB | ✓ | 48 records from 47 subjects, drawn from over 4,000 Beth Israel Holter recordings made in 1975–1979, of which "approximately 60%… were obtained from inpatients" (PhysioNet); 23 records chosen at random and 25 selected for rare but clinically important phenomena; 25 men aged 32–89, 22 women aged 23–89; 360 Hz. |

### 7.2 Missing references to add

- Bates, Hastie & Tibshirani, JASA 2024 (nested-CV intervals).\[26\]
- Prajod & André, ICMLA 2022 (arXiv 2210.06225): HRV-only WESAD LOSO F1 0.813–0.859 on 60-s windows with 50-s overlap, and poor transfer to SWELL-KW (RFC F1 0.467, MLP 0.478, SVM 0.535). Bears on both HR dominance and cross-dataset transfer.
- Schmidt et al.'s per-modality ECG-only and BVP-only LOSO results.\[5\]
- Pre-registration in ML: the NeurIPS 2020/2021 pre-registration workshops (PMLR 148, 181) and "Pre-registration for Predictive Modeling" (arXiv 2311.18807).\[27\]\[28\]\[29\]
- Saeb et al. 2017 on record-wise vs subject-wise splits, for the leaky-split case study.\[30\]

### 7.3 Are the narrowed claims accurate?

Mostly, in the Novelty note. The defensible delta is *a subject-level evaluation of calibration under EDA removal on WESAD against reduced-sensor and imputation baselines, with an explicit falsification rule*. "Device-tier calibration equity" is not a contribution until real tiers are instantiated, and the audit is an integration contribution conditional on a full run.

---

## 8. Clinical and translational validity

1. **The proxies have weak construct validity.** WESAD's stressor is the Trier Social Stress Test (public speaking and mental arithmetic, about 10 minutes), administered in a laboratory to 15 young healthy adults (27.5 ± 2.4 years). Panic attacks are abrupt surges, often with dyspnea and chest pain. MIT-BIH draws on Holter recordings made at Beth Israel Hospital in 1975–1979, approximately 60% of them from inpatients (PhysioNet), with 25 of its 48 records deliberately selected for rare arrhythmias. It contains no ACS, MINOCA or symptom-labelled SVT episodes.
2. **The motivation and the experiments are disconnected.** No experiment measures panic-vs-cardiac differentiation. Link what is measured (calibration of a stress-vs-baseline detector under sensor dropout) to a plausible use (abstention when sensors fail), and imply nothing more.
3. **HR dominance is a clinical red flag.** Tachycardia is shared by panic, SVT, AF with rapid ventricular response and ischaemia. A heart-rate fallback is least informative exactly at the decision boundary that motivates the company.
4. **Calibration matters for safety only through abstention.** Evaluate risk–coverage curves under the reduced condition. Lechuga López et al. found that aggregate calibration did not guarantee useful selective prediction.\[8\]\[31\]
5. **COI.** Disclose it on page 1, and add to the abstract: "These results do not evaluate, and should not be interpreted as evidence for, differentiation of panic attacks from cardiac events."

---

## 9. Reproducibility and reporting

| Item | Status | Required action |
|---|---|---|
| Version control | **Absent** | git init; tag the preprint commit; one script regenerates all numbers. |
| Pre-registration integrity | **Unverifiable** | Remove the claim or produce a dated artifact; OSF-register the baseline plan before running it. |
| Seeds | Partial (seed 42; "per-fold seed reset") | Report every seed; ≥5 seeds for neural models. |
| Code/data | Planned | Release code or explain why not; both datasets are public. |
| TRIPOD+AI | Planned | Complete the 27-item and 13-item abstract checklists. Gaps: window length/overlap, R-peak detector, EDA decomposition, SQI, missing-data handling per model, sample size, calibration procedure, tuning. |
| Model card / datasheet | Planned | Intended use "research only"; proxy limitations; HR dependence. |
| Leaky-split disclosure | Planned | Before/after numbers in an appendix; state any public use. |
| Deviations table | Good (audit only) | Extend to the H-B1 ablations, the H-C1 tiers, and the CV decision timeline. |

---

## 10. Recommended framing and scope

### 10.1 Scope decision

The contributions do not share a thesis. H-B1 and H-C1 are within-WESAD, the audit is cross-dataset, and H2a and H-B2 are unrelated nulls. I recommend **splitting**:

- **Preprint 1 (post first): the calibration evaluation.** When EDA drops out, is one gated model better calibrated and more discriminative than a model built for the reduced sensor set? It has a pre-stated falsification rule and is publishable either way. Include H-C1 only with real WESAD chest vs wrist tiers. Move H2a, H-B2 and interpretability to appendices or drop them.
- **Note 2 (after the full audit run): the proxy-pairing audit.** "Proxy-vs-proxy pairings for panic-vs-cardiac research are dataset-identifiable, and heart rate drives both identity and within-dataset signal." A short cautionary methods note.

The audit is not currently the stronger standalone paper: its central result is expected and it rests on a `--fast` run. The calibration paper is stronger *if* the baselines are run, because it has a decisive pre-stated test.

### 10.2 Proposed title

**"Does Modality Gating Preserve Calibration When Wearable Sensors Drop Out? A Subject-Level Evaluation on WESAD Against Reduced-Sensor and Heart-Rate Baselines"**

### 10.3 Abstract skeleton

> *Background.* Wearable systems for distinguishing acute psychological from cardiac events will face missing sensors; calibrated uncertainty under missingness is necessary for safe abstention. Modality dropout and calibration under missing modalities are established in general multimodal learning, but evidence on wearable physiological signals against strong reduced-sensor baselines is limited.
> *Methods.* On WESAD (15 subjects; stress vs baseline; 60-s windows), we compare a modality-gated network (with and without modality dropout) against LightGBM variants (NaN-aware, EDA-masked training, per-configuration ECG-only), mean-imputation with recalibration, and a heart-rate-only model, when EDA is removed at test time. Primary endpoints are reduced-condition Brier score, debiased ECE and AUROC, evaluated with repeated subject-level cross-validation and subject-level paired tests. A decision rule specified before running the baselines declared the architectural claim falsified if the gated model did not beat the EDA-masked and per-configuration baselines on all three endpoints.
> *Results.* [Per-endpoint estimates with subject-level CIs; whether the decision rule was met; HR-only and HR-removed analyses.]
> *Interpretation.* [If met: gating provides X; if not: deploying a reduced-sensor model is as good or better, and apparent robustness reflects reliance on heart rate.] With 15 young, healthy laboratory participants, these results do not evaluate differentiation of panic attacks from cardiac events.
> *Disclosure.* The author is a co-founder of E14 Health.

---

## 11. Checklists

### 11.1 Minimum-viable preprint (~2–3 weeks, 1–2 engineers)

| # | Task | Effort |
|---|---|---|
| 1 | git init; commit configs, environment, seeds; tag; one script regenerates all numbers | 1–2 days |
| 2 | Remove or substantiate all "pre-registered" language; add a dated decision-timeline appendix including the leaky split | 0.5 day |
| 3 | Timestamp the baseline plan and decision rule (OSF or signed git tag) **before** running baselines | 0.5 day |
| 4 | Run baselines (i)–(iv), an HR-only model, and H-B1 without `hrv_mean_nn` | 2–4 days |
| 5 | Reduced-condition Brier, debiased/equal-mass ECE, ΔECE, AUROC/AUPRC and reliability diagrams; document the calibration protocol | 2–3 days |
| 6 | 20× repeated subject-level CV with the Bouckaert–Frank correction, plus subject-level sign-flip tests | 2–3 days |
| 7 | Full audit rerun (10,000/1,000 permutations, all 8 stages, Holm) with seed sensitivity, or remove the audit | 1–2 days |
| 8 | H-C1 with real chest vs wrist tiers, or demote it without "equity" | 3–5 days or 0.5 day |
| 9 | Apply every row of Section 5; strip annotations; enforce terminology | 1 day |
| 10 | Fix the Lechuga López characterization, acknowledge SimMLM gating, cite PULSE 0.994, add Prajod & André and Bates et al. | 0.5 day |
| 11 | COI on page 1; non-clinical disclaimer in the abstract | 0.5 hour |
| 12 | Retitle per 10.2 | — |

### 11.2 Full venue version (additional ~4–8 weeks)

- PULSE-style distillation (v) and the CML regularizer (vi).
- E4 BVP and RespiBAN respiration ablations run as specified in the plan.
- H-C1 with a justified random-effects structure, reporting Brier and ECE per tier.
- H2a TOST, or move it to the appendix.
- Selective prediction (risk–coverage) under reduced inputs.
- An external stress dataset for transfer.
- TRIPOD+AI checklist, model card, code release.
- Neural seed variance (≥5) and nested tuning.
- Grouped SHAP, a documented IG baseline, and the coupling-feature audit.

### 11.3 Hypothetical scores (ML4H Findings / CHIL-style, 1–10)

| State | Score | Rationale |
|---|---|---|
| Current outline | **4/10** (weak reject) | Honest, but no supported headline, missing baselines, contradictions, unverifiable pre-registration. |
| Minimum-viable done, gated **beats** baselines | 6/10 | Modest, clean empirical contribution; small N. |
| Minimum-viable done, gated **does not beat**, reframed honestly | 5–6/10 | Useful negative result with a pre-stated falsification rule. |
| Full venue checklist with real tiers and an external dataset | 6–7/10 | Competitive for ML4H Findings / CHIL. |

### 11.4 How likely is the headline to disappear?

My subjective estimate is above even odds that the gated model does not clearly beat a per-configuration ECG-only model on reduced-condition absolute Brier and AUROC. The reasons:
- (1) Within-WESAD signal is HR-heavy (C3 0.899 → 0.674).
- (2) HRV-only LOSO baselines already reach F1 0.813–0.859 (Prajod & André, ICMLA 2022).
- (3) The per-configuration model is trained on exactly the reduced input distribution.
- (4) Part of the gated model's advantage over LightGBM may reflect zero-fill handling (M9) rather than architecture.

This is a judgement, not a measurement. Its implication is to draft the paper so that either outcome is a publishable, honest result.

---

## 12. Questions for the author

1. Is there *any* dated external record of `configs/preregistration.yaml` or of the subject-level CV decision before results were seen? If not, will you drop the term?
2. When did the leaky split occur relative to the CV decision and the audit thresholds? Were leaky-split numbers shared externally (website, pitch deck, grant, investor update)?
3. Are windows non-overlapping 60 s, and is amusement excluded?
4. How is calibration fitted (subjects, nesting, per condition, Platt or isotonic)? Are the reported ECEs pre- or post-recalibration?
5. How is missing EDA encoded per model at test time, and did LightGBM see missingness during training?
6. What are the 890 H-C1 observations, and is the 7.5% vs 10.6% drop absolute or relative?
7. Why does `coupling_lag_sec` carry attribution when ECG is zeroed?
8. What are the AUROC/AUPRC values per model in the complete and reduced conditions?
9. Can you instantiate real tiers from WESAD's wrist E4?
10. Which ECE binning scheme and how many bins?
11. Which result would change E14 Health's product decisions? Stating it would sharpen the contribution and strengthen the COI handling.

## Sources

1. [Chapter 4 Resampling: permutation testing and bootstrap](https://www.cs.helsinki.fi/u/ahonkela/teaching/compstats1/book/resampling-permutation-testing-and-bootstrap.html)
2. [Shortening heart rate variability measurement time to 1 minute using deep learning: Implication of real-time measurement of heart rate variability](https://www.medrxiv.org/content/10.1101/2023.10.23.23297390.full.pdf)
3. [Thirty Years On: What the 1996 HRV Task Force Report Still Teaches HRV Biofeedback Trainers](https://www.biosourcesoftware.com/post/thirty-years-on-what-the-1996-hrv-task-force-report-still-teaches-hrv-biofeedback-trainers)
4. [\[2510.24058\] PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning](https://arxiv.org/abs/2510.24058)
5. <https://ubi29.informatik.uni-siegen.de/usi/pdf/ubi_icmi2018.pdf>
6. [Evaluating the Replicability of Signiﬁcance Tests for](https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf)
7. [Evaluating the Replicability of Significance Tests for ...](https://sci2s.ugr.es/keel/pdf/specific/congreso/evaluating-the-replicability-of.pdf)
8. [\[2603.02719\] An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification](https://arxiv.org/abs/2603.02719)
9. [\[2507.19264\] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](https://arxiv.org/abs/2507.19264)
10. [Constructing Confidence Intervals for 'the' Generalization Error -- a Comprehensive Benchmark Study](https://arxiv.org/pdf/2409.18836)
11. [Inference for the Generalization Error - Springer Nature](https://link.springer.com/content/pdf/10.1023/a:1024068626366.pdf)
12. [Cross-Validation: What Does It Estimate and How Well Does It Do It?: Journal of the American Statistical Association: Vol 119, No 546](https://www.tandfonline.com/doi/abs/10.1080/01621459.2023.2197686)
13. [\[1712.03120\] Learning Disease vs Participant Signatures: a permutation test approach to detect identity confounding in machine learning diagnostic applications](https://arxiv.org/abs/1712.03120)
14. [\[2306.01265\] Calibrating Multimodal Learning](https://arxiv.org/abs/2306.01265)
15. [ICML 2023 Calibrating Multimodal Learning Oral](https://icml.cc/virtual/2023/oral/25541)
16. [SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality · Pith](https://pith.science/paper/2507.19264)
17. [Detect and Repair: Robust Self-Supervised Wearable Sensing Under Missing Modalities](https://doi.org/10.3390/s26082419)
18. [Detect and Repair: Robust Self-Supervised Wearable Sensing Under Missing Modalities - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13119957/)
19. <https://arxiv.org/html/2603.13373v1>
20. <https://arxiv.org/pdf/2603.13373>
21. [Wearable devices: underrepresentation in the ageing society - The Lancet Digital Health](<https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00069-9/fulltext>)
22. [5-Min Science: The Task Force Report Remains Relevant](https://www.biosourcesoftware.com/post/5-min-science-the-task-force-report-remains-relevant?cid=a891c265-08a2-4757-9d50-ba09594b6bc0)
23. [TRIPOD+AI statement: updated guidance for reporting ...](https://epub.ub.uni-muenchen.de/116670/1/bmj-2023-078378.full.pdf)
24. [Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection](https://www.researchgate.net/publication/328034332_Introducing_WESAD_a_Multimodal_Dataset_for_Wearable_Stress_and_Affect_Detection)
25. [(PDF) Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection](https://www.academia.edu/91329794/Introducing_WESAD_a_Multimodal_Dataset_for_Wearable_Stress_and_Affect_Detection)
26. [Confidence intervals for the Cox model test error from cross-validation - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10734684/)
27. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v148/)
28. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v181/)
29. [Pre-registration for Predictive Modeling](https://arxiv.org/pdf/2311.18807)
30. [How You Split Matters: Data Leakage and Subject Characteristics Studies in Longitudinal Brain MRI Analysis](https://arxiv.org/html/2309.00350)
31. [An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification](https://arxiv.org/html/2603.02719)
