# Peer Review: "Missingness-Aware Training, Not Modality Gating, Determines Performance When EDA Drops Out: A Subject-Level Evaluation on WESAD" (revision 5)

**Bottom line: the paper is not ready to post in its current form, but it is close.** The negative result is real and well reported. The problems are a causal title, an "indistinguishable" conclusion that has no equivalence analysis behind it, a factorial design with one cell missing, a few factual and citation errors, and unfinished provenance items, chiefly an empty disclosure placeholder and no version control. All of these can be fixed in days, not months.

## TL;DR

- **The core finding holds up; the framing does not.** Against fair baselines, the gated network shows no detectable advantage when EDA is missing. But the CIs (about ±0.035 Brier, roughly 15% of the 0.2275 prevalence-only Brier) do not rule out differences that matter in practice. "Determines" and "Not Modality Gating" also go beyond what the design can show, because the "MLP without masking" cell is missing, so the gate × masking interaction cannot be estimated. A defensible title would be: *"No evidence that modality gating adds to missingness-aware training…"*.
- **Novelty is modest, and the broad literature claim is not supported as written.** Ma et al. (CVPR 2022, pp. 18177–18186) already reported that multi-task training with missing-modal inputs "contributes more than the fusion strategy". Training with modality dropout is a standard baseline in the computer-vision and NLP missing-modality literature, and PULSE compared against a model built for the reduced sensor set. The paper's most valuable and most original contribution is the documented evaluation-pitfalls case study together with the mechanistic heart-rate finding, and the paper should lead with these.
- **Minimum before posting:** fill or remove the leaked-number placeholder; initialise git and release code plus the frozen table; retitle; add a clearly labelled post-hoc TOST/ROPE analysis; correct the "±0.037" statement, which does not cover baseline (iv); fix the Roy et al. citation and the Korea dataset version; and report complete-condition paired CIs. This is about 2–4 working days. Adding the missing factorial cell and an order-balanced stress-vs-amusement contrast would add about 1 week and would raise the paper's value substantially.

---

## 1. Executive verdict

This revision is scientifically honest in a way that is unusual for the field. The authors ran fair baselines, the headline claim collapsed, and both evaluations of the decision rule, the implementation-error history, the superseded numbers and the analysis classes (pre-specified, corrected, reviewer-requested, exploratory) are all reported. The central empirical statement survives inspection. With EDA removed at test time, the gated network, a size-matched MLP with the same masking, NaN-trained LightGBM and ECG-only LightGBM all sit within 0.002–0.005 Brier of each other, and every paired 95% CI includes zero.

Four things still stand between the draft and a defensible arXiv posting:

- **(a) Overclaiming.** "Determines" and "not modality gating" are causal and general claims. The study supports a conditional statement: once missingness is seen in training, there is no evidence of a gating benefit. That is narrower than either phrase.
- **(b) No equivalence analysis.** The "indistinguishable" conclusion rests on non-significance. The minimum detectable difference is about 0.047 Brier, so a non-significant result is weak evidence of equivalence.
- **(c) A missing factorial cell.** No MLP was trained without masking. That cell is exactly what separates "gate" from "masking".
- **(d) Unresolved integrity items.** These are the unfilled "[List any leaked number…]" placeholder, git not yet initialised, and code and table not released.

**Verdict:** ready for arXiv after a short blocking checklist (Section 11). The best fit as it stands is a negative-results or evaluation workshop (ICBINB-type) or ML4H Findings. It is not ready for a main-track clinical-AI venue.

## 2. Claim-by-claim assessment

| # | Claim | Evidence in paper | Supported? | Required change |
|---|---|---|---|---|
| T | Title: missingness-aware training, *not gating*, *determines* performance | Table 2 CIs include 0; gated vs MLP ±0.02; capacity sweep; nested tuning | **Partially.** Absence of a detectable gate effect, conditional on masking, is supported. "Determines" is causal and the 2×2 design is incomplete. | Retitle (Section 11). Add the MLP-no-masking cell. |
| 1a | Gated network ≈ reduced-sensor LightGBM | ΔBrier +0.002 [−0.031, +0.035] vs (ii); +0.002 [−0.033, +0.037] vs (iii) | **No evidence of difference.** Equivalence is not shown: the 90% CI is about [−0.026, +0.030], so equivalence holds only at margins of about ±0.03. | Post-hoc TOST with a justified SESOI, or a Bayesian ROPE analysis. |
| 1b | Gated ≈ plain MLP with the same masking | +0.002 [−0.016, +0.020]; capacity sweep at 4 sizes all within ±0.024 | **Strongest claim in the paper.** The 90% CI is about ±0.015, so TOST at ±0.02 Brier would likely pass (approximate, given Bouckaert–Frank caveats). | Lead with this comparison. It is the cleanest "gate adds nothing" evidence. |
| 1c | The earlier gating advantage came "entirely" from a never-trained-with-missingness comparator | Complete-trained LightGBM reduced Brier 0.296 vs 0.142–0.155 | **Supported for the comparator artefact.** However, the gated no-dropout network (0.196) still beats complete-trained LightGBM (0.296) by about 0.10, which suggests the gate's structural "drop-branch" behaviour *does* help when there is no masking. | Say "the earlier advantage was a comparator artefact". Acknowledge the plausible gate × masking interaction. |
| 1d | "Training with missingness is the active ingredient" | LightGBM: 0.296 → 0.146 (large). Gated: dropout vs no dropout −0.038 [−0.089, +0.014], 11/15 | **Strong for LightGBM; not significant within the gated network.** The category also mixes masking, per-configuration models and per-condition recalibration (recalibrated (i) reaches 0.163, outside the 0.142–0.155 band). | Rephrase as "exposure to missingness somewhere in the pipeline". Report the gated within-family contrast as non-significant. |
| 1e | Holds untuned, nested-tuned, and across 4 sizes | Tuned diffs ≤0.005; the modal config was chosen in only 25/100 folds | **Supported**, with the caveat that the tuning surface is flat and the networks may all be underfit (Section 5, M5). | Report tuned complete-condition Brier for the networks. |
| 2a | Without EDA, discrimination collapses toward chance when HR is removed | Pooled AUROC 0.873 → 0.692 (gated), 0.860 → 0.681 (ii), 0.869 → 0.674 (iii) | **Supported** ("about half the above-chance AUROC": retained 51%, 50%, 47%). Within-subject AUROC loses about 60% (0.903 → 0.662). "Toward chance" is too strong at 0.67–0.69. | State the numbers. Run an order-confound check (Section 9). |
| 2b | Per-subject discrimination tracks HR response (ρ 0.76–0.92) | Spearman, n=15, three non-independent correlations | **Descriptive and near-tautological for HR-only.** | Normalise the response by within-condition HR SD. Report CIs. Label as exploratory. |
| 2c | "The failure mode that matters at the panic-vs-cardiac boundary" | No cardiac or panic data | **Hypothesis, not finding.** | Rephrase as a hypothesis for future work. |
| 3 | Five evaluation errors, each of which changed or could change the conclusion | Appendix A plus regression tests | **Supported and valuable.** This is the most transferable contribution. | Consider making it the primary framing. |
| D | Per-configuration system "matches or beats" the single gated network in both conditions | Complete 0.070 vs 0.122; reduced 0.142 vs 0.146 | **Point estimates support it; no paired CI.** It also shows the networks are much worse than trees when EDA is present. | Add paired CIs. Discuss the network deficit (M5). |
| I | "Architectural claims are usually tested against weak missing-data baselines" | No systematic citation | **Not supported as stated.** | Narrow it to specific wearable or physiological papers, or delete it (Section 8). |
| C | Lessmeier 1997 numbers | Verified | **Accurate** (with a selection-population caveat). | Add context (Section 9). |

## 3. Strengths (substantive)

1. **Self-correction done properly.** The authors identified the weak-baseline artefact, built fair comparators (NaN-masked LightGBM, ECG-only LightGBM, mean-imputation with per-condition Platt, a size-matched masked MLP) and report the collapse of their own earlier claim. Both decision-rule evaluations are reported, including the pre-fix one that was invalidated by the single-partition "repeated" CV.
2. **Paired, subject-level design.** Every held-out window is scored in both the complete and reduced condition, which removes window-sampling noise from between-condition contrasts. All splits are at subject level. There are 20 genuinely distinct partitions, recorded.
3. **Capacity matching.** Gated and non-gated networks are matched within about 1% in parameters at four sizes (273/281 through 9,345/9,333). The capacity confound is addressed directly, as Grinsztajn et al. would suggest.
4. **Calibration treated as a first-class endpoint.** Calibration intercept and slope are reported alongside ECE_SWEEP. The paper separates two failure modes: calibration-in-the-large for never-masked models (intercept +3.10) and over-confidence in all models without EDA (slopes 0.42–0.59). It shows that per-configuration recalibration fixes the slope (gated 0.43 → 0.92) without changing Brier.
5. **The ECE estimator lesson is concrete and verifiable.** A minimum-over-bins, zero-clipped "debiased" ECE produced 18/40 exact zeros and ranked the worst model best. That is a reproducible pitfall worth publishing.
6. **Mechanistic decomposition.** The analysis separates HR-only, HR-removed, and within-subject HR-residualised models. The fixed residualisation (0.600 vs the erroneous 0.878) is especially informative: most of the non-EDA signal is heart rate expressed in other units.
7. **Transparent analysis classes.** Each analysis is labelled pre-specified, corrected, reviewer-requested or exploratory, and the paper explicitly declines to draw a conclusion from the decision rule because the rule flips under calibration mode.
8. **H2a handled correctly.** The pseudoreplicated n=445 test was replaced by a subject-level test, with the equivalence bounds the data can and cannot support. I checked the numbers (Section 7).

## 4. Fatal issues

**None in the current revision.** Earlier fatal issues (record-level leakage, single-partition "repeated" CV, the broken ECE estimator, the weak-baseline headline) have been fixed. Two items are *posting-blocking* even though they are not scientifically fatal:

- **F-block-1: the unfilled placeholder "[List any leaked number that appeared publicly]".** This is a research-integrity disclosure, not a formatting matter. Given the COI (co-founder of E14 Health), any superseded number that appeared in a pitch, talk, social post or grant must be listed, or the paper must state "none". Posting with the placeholder would be a disclosure failure.
- **F-block-2: no version control and no released artefacts.** Every §5 number depends on a frozen feature table that is identified only by a SHA-256 hash in the author's own outline. Until the code, the table and the hash are public under a tagged commit, the 337-pass cross-check cannot be verified by anyone else.

## 5. Major issues

**M1. The title and main claim are causal and over-general.**
- "Determines" implies a demonstrated causal decomposition. What was shown is that once masking or reduced-set training is present, adding a gate gives no detectable benefit on one dataset, one dropped modality (EDA, all six features removed, MCAR), 60-s hand-crafted features and networks of at most 9,345 parameters.
- The "missingness-aware" category is defined after the fact. It groups three different mechanisms:
  - training-time masking ((ii), the gated network with dropout, the masked MLP);
  - per-configuration models ((iii));
  - post-hoc per-condition recalibration ((iv), and (i) recalibrated).
- Recalibrated (i) reaches 0.163, which is *between* the 0.142–0.155 band and 0.296. It recovers (0.296 − 0.163)/(0.296 − 0.142) ≈ 86% of the gap, so recalibration "mostly" rescues, not fully. These mechanisms have different deployment costs and failure modes, and lumping them hides that.

**M2. The factorial is incomplete: the MLP-without-masking cell is missing.**
- The design has gate+mask, gate−mask and MLP+mask, but no MLP−mask (zero-fill plus indicator, trained on complete data).
- Without that cell, "masking, not gating" is a contrast between two cells, not a decomposition. The interaction cannot be estimated.
- The existing data hint at a real interaction. With no masking, the gated network (0.196 pooled) is far better than the never-masked tree (0.296). That is roughly the original "gating advantage", and it may be genuine *in the no-masking regime*: closing a gate is a principled missing-mode, while a never-trained missingness indicator or NaN branch is out-of-distribution.
- The honest conclusion may be "gating substitutes for missingness training but does not add to it", which is a more interesting result. This takes about a day of compute.

**M3. "Indistinguishable" rests on absence of evidence.**
- From Table 2, the standard error of Gated − (ii) is about 0.033/1.984 ≈ 0.0166. The minimum detectable difference at 80% power and α = 0.05 is about 2.8 × 0.0166 ≈ **0.047 Brier**, roughly 21% of the prevalence-only Brier (0.2275) and about 32% of the observed Brier (0.146).
- In Brier-skill terms, BSS ≈ 1 − 0.146/0.2275 = 0.36, and a ±0.035 Brier band corresponds to BSS from about 0.20 to 0.51. Under any reasonable clinical reading, that range is practically meaningful.
- **Recommendation:** add an explicitly post-hoc, clearly labelled equivalence analysis.
  - TOST at a SESOI justified independently of the data. Candidates are 0.02 Brier (about 9% of the uninformative Brier, about 0.09 BSS) and 0.03 AUROC.
  - Or a Bayesian ROPE analysis on the participant-level differences.
- Expected outcome from the reported CIs:
  - Gated vs MLP passes at ±0.02 (90% CI about [−0.013, +0.017]).
  - Gated vs (ii)/(iii) passes only at about ±0.03 or wider.
- That split is the true shape of the evidence and should be stated.

**M4. The Bouckaert–Frank corrected test is heuristic here and probably anti-conservative.**
- The formula is implemented correctly: SE² = (1/100 + 0.25)·var(d), since n_test/n_train = 3/12 = 0.25 for 5-fold on 15 participants, with df = 99 as in the original Bouckaert–Frank proposal.\[1\]
- Three concerns remain:
  - **(i)** The Nadeau–Bengio correction assumes a particular correlation between fold estimates. With only 15 clusters and 80% training-set overlap, there is no simulation evidence that it gives nominal coverage.
  - **(ii)** df = 99 treats 100 fold differences as nearly independent pieces of information. The effective number of independent units is about 15 participants. Using t₁₄ (2.145) instead of t₉₉ (1.984) widens every CI by about 8%. That is cheap and more honest.
  - **(iii)** Bates, Hastie & Tibshirani (JASA 2024) show that naive CV intervals undercover, and their nested-CV estimator of the CV MSE gives intervals with approximately correct coverage, typically wider.\[2\]\[3\]
- **Recommendation:** report B–F with df = 14 as primary, or as a sensitivity analysis. Add a participant-level analysis (per-participant mean difference across the 20 repeats, then a t-interval and exact sign-flip on n = 15) as the inferential anchor. Keep B–F as secondary.

**M5. The networks underperform trees badly when EDA is present, and this goes unremarked.**
- Complete-condition Brier is 0.070 for complete-trained LightGBM, 0.090 for NaN-trained LightGBM, 0.122 for the gated network, 0.128 for gated without dropout, and 0.131 for the MLP.
- EDA improves the networks by only about 0.024 Brier (0.146 → 0.122), but improves LightGBM by about 0.072 (0.142 → 0.070).
- Two consequences:
  - **(a)** The gate is meant to arbitrate between an informative and an absent modality. If the networks barely exploit EDA, the gate's job is nearly trivial, and "gate adds nothing" may be a statement about an underfit regime: 100 full-batch epochs, no early stopping, at most 9k parameters, on 7 + 6 features.
  - **(b)** The deployment conclusion ("per-configuration matches or beats single gated") is driven by the complete-condition gap, which has no paired CI.
- Nested tuning does not settle this because tuned complete-condition Brier is not reported.
- **Required:** report complete-condition Brier and AUROC with paired CIs for all models, untuned and tuned. Discuss whether LightGBM's EDA advantage reflects tonic-level drift tied to time-on-electrode (M6).

**M6. The fixed condition order is an unaddressed threat to both the EDA and HR results.**
- In WESAD, baseline always comes first. Stress and amusement order was swapped between the two protocol versions and alternated across subjects "in order to avoid effects of order" (Schmidt et al., ICMI 2018), and each was followed by meditation.
- So baseline-vs-stress is confounded with time on electrode (tonic EDA drift, skin hydration) and with habituation.
- Within-subject AUROC, which is the per-participant analysis, could partly be a time-order classifier.
- An unreviewed public code-repository note (low evidential weight, but directly on point) reports that baseline-vs-stress models on baseline-referenced WESAD features labelled 51–85% of post-stressor rest/meditation/amusement windows as stress.\[4\]
- **Recommended checks**, all possible with data already available:
  - **(i)** Stress vs amusement, which is order-balanced across participants.
  - **(ii)** Apply the baseline-vs-stress model to amusement and meditation windows, and report the fraction predicted as stress. This is a negative control for "time-since-start".
  - **(iii)** Baseline vs meditation (both low arousal, different times). High AUROC here would indicate drift.
- If the HR mechanism survives (i), claim 2 becomes much stronger.

**M7. The HR-response correlation is close to tautological and statistically fragile.**
- For the HR-only model, within-subject AUROC is literally the AUC of mean NN separating that participant's stress and baseline windows. Under a binormal approximation, AUC ≈ Φ(Δ/(σ√2)). It is a monotone function of the *standardised* separation, so a high Spearman ρ with the raw drop is close to mechanical.
- With n = 15, the approximate Fisher-z 95% CIs are:
  - ρ = 0.88 → about [0.67, 0.96];
  - ρ = 0.76 → about [0.40, 0.92];
  - ρ = 0.92 → about [0.77, 0.97].
- The three correlations share participants and are not independent replications.
- **Recommendation:**
  - Normalise the response (Δ mean NN / pooled within-condition SD).
  - Report the CIs.
  - Frame the result descriptively, for example: "an HR-reliant detector works to the extent that HR separates the conditions; in 4/15 participants (27%) HR barely moved (12–51 ms)".
  - The interesting quantity is the *non-HR* models' dependence on HR response, which is shown via the HR-removed collapse. Lead with that.

**M8. The broad literature claim is unsupported (details in Section 8).**
- "Architectural claims are usually tested against weak missing-data baselines" needs systematic evidence or must be narrowed.
- Several of the closest papers *do* use missingness-trained or reduced-set baselines:
  - PULSE includes a no-teacher model trained on deployment sensors only;\[5\]
  - SimMLM itself trains with random modality dropout and compares against dedicated missing-modality methods;\[6\]
  - modality dropout is the standard baseline in Ma et al. (CVPR 2022)-style work, the ACL 2024 prompt-learning work, MMP, and the CVPR 2025 workshop egocentric study.\[7\]\[8\]\[9\]\[10\]
- The claim *is* supported for CognifySSL v2.0, which compares only against SSL methods (CroSSL, MU-MAE, COCOA). It has no zero-fill, mean-imputation or explicitly masked supervised baseline, and its only "no-dropout" comparison is an ablation of its own model.\[11\]

**M9. Analytic flexibility and residual error risk.**
- Nine implementation errors were found across revisions. The decision rule was written the same day as the rewrite, after the gated model's estimates were known. The code was finalised after the first rule outcome. A large share of the analyses were requested by reviewers.
- The labelling is honest, but the posterior probability that an undetected error remains is not small. The cross-check script also reports **95 untraced tokens**.
- **Recommendation:**
  - Trace or remove all 95.
  - Have an independent minimal re-implementation of the primary comparison (gated vs (ii) vs MLP, reduced-condition Brier, LOSO, from raw WESAD .pkl) written by someone who has not seen the main code, and report agreement within a stated tolerance.
  - Release everything under a tagged commit.

## 6. Minor issues

1. **The "±0.037" statement is inaccurate.** "Compatible with differences of up to about ±0.037 Brier" matches Gated vs (iii) [−0.033, +0.037] and vs (ii) [−0.031, +0.035]. But Gated vs (iv), which the paper also calls a baseline "built for missing EDA", has [−0.073, +0.047]. Either restrict the sentence to (ii)/(iii) or widen it.
2. **Within-subject AUROC precision.** With about 10 stress and 19 baseline windows, the Hanley–McNeil SE is about 0.07 at AUC 0.90 and about 0.11 at AUC 0.70, before accounting for autocorrelation between consecutive 60-s windows. S10's 0.16 is about 3 null-SEs below 0.5, so it is probably a real inversion and not noise. Investigate it: posture, movement, or an HR decrease under stress.
3. **Participant bootstrap.** Percentile intervals with 15 clusters are known to undercover, and Cameron & Miller's practitioner's guide to cluster-robust inference (J Hum Resour 2015) notes that "few" may range from fewer than 20 to fewer than 50 clusters. With B = 500, the 2.5%/97.5% endpoints carry Monte Carlo error. Use B ≥ 2,000, BCa or studentised intervals, and label the CIs as approximate.
4. **Selective prediction baselines.** "Abstention halves error (~0.20 → ~0.10 at 60% coverage)" should be benchmarked against random abstention (error stays at about 0.20) and the oracle (error 0 at 60% coverage when full-coverage error is 0.20). Report E-AURC (excess over oracle) as defined by Geifman et al. The pooled finding that ECG-only's most confident 10% is wrong 23% of the time, at or above its average error, is a striking inverted-confidence result. It echoes the Lechuga López et al. failure mode,\[12\] but it does not replicate under repeated CV ([−0.113, +0.061]). Say so.
5. **The cost of masking in the complete condition** (LightGBM 0.070 → 0.090, about 9% of the prevalence Brier) is a real deployment trade-off. Report it in the main text with a paired CI, along with sensitivity to masking rate (50% here vs the expected deployment missingness).
6. **The MLP encoding is unclear.** "Zeroed EDA plus indicator": if features are standardised, zero equals the training mean, which makes this mean imputation plus indicator. State which it is.
7. **ECE_SWEEP per fold on about 87 windows** is upward-biased, as noted. Since the decision rule uses it, report the rule both with and without the ECE criterion.
8. **The H2a lag reliability argument can be made stronger.** A uniform distribution on ±15 s has SD 30/√12 ≈ 8.66 s. The observed within-rest SD of 8.3 s (range 5.9–10.6) is essentially that of a random lag in the search window. Saying so makes the "uninformative" verdict quantitative.
9. **Legacy Wilcoxon.** With n = 5 folds, the smallest attainable two-sided Wilcoxon p is 0.0625 (W = 0). The historical "5/5, W = 0" could never have been significant at 0.05 two-sided. State this in Appendix B, since it shows the original evidence was inadequate by construction.
10. **The legacy ECE ratio** has a Wald-type CI of [0.05, 8.96] for 4.50×. Ratio estimators with noisy denominators need Fieller or log-scale intervals. The paper is right to retire the ratio metric.
11. **HRV at 60 s.** The paper correctly restricts interpretation to mean NN and RMSSD. Consider dropping SDNN, pNN50, LF and LF/HF from the model altogether, or showing that results are unchanged without them, since they add noise dimensions that may disadvantage the networks.
12. **No signal-quality index or artefact gating.** Report R-peak detection plausibility (the fraction of windows with physiologically implausible NN intervals). TSST speech and movement may contaminate chest ECG.
13. **PULSE comparison.** Describe PULSE accurately (Section 8): it is the same binary baseline-vs-stress task under LOSO, but with wrist BVP/ACC/TEMP plus ECG, per-subject z-scoring from the test subject's baseline windows (transductive and label-informed), and per-subject mean AUROC. Its *no-teacher* baseline already reaches 0.963.\[5\] The 0.87 vs 0.99 gap is therefore mostly protocol (modalities, normalisation, metric aggregation), not a sign that the present features are uniquely weak. Still, run a sensitivity analysis with per-subject baseline normalisation to show that the negative result is not specific to a weak-feature regime.
14. **Roelofs ECE_SWEEP, SmoothECE and AURC references.** Consider reporting SmoothECE (Błasiok & Nakkiran) as a robustness check, since it avoids binning choices entirely.

## 7. Statistical verification

| Check | Computation | Result |
|---|---|---|
| Prevalence-only Brier | 0.35 × 0.65 | **0.2275 ✓** (434 windows × 35% ≈ 152 stress, consistent with 9–11 stress windows × 15) |
| "About half the above-chance AUROC" (pooled) | Gated (0.692 − 0.5)/(0.873 − 0.5) = 0.192/0.373; (ii) 0.181/0.360; (iii) 0.174/0.369 | Retains **51% / 50% / 47%** ✓ "about half" is correct for pooled AUROC |
| Same, within-subject | (0.662 − 0.5)/(0.903 − 0.5) = 0.162/0.403 | Retains **40%**. Within-subject loss is about 60%, more than half. |
| HR-only share of gated above-chance AUROC | 0.320/0.373 (pooled); 0.367/0.403 (within) | **86% / 91%**. HR alone carries most of the signal. |
| Table 1 → Table 2 consistency (ΔBrier) | 0.148 − 0.146; 0.148 − 0.161; 0.148 − 0.179; 0.148 − 0.186; 0.148 − 0.146 | +0.002, −0.013, −0.031 (reported −0.030, rounding), −0.038, +0.002 ✓ |
| Table 1 → Table 2 (ΔAUROC) | 0.892 − 0.882; −0.886; −0.854; −0.877; −0.892 | +0.010, +0.006, +0.038, +0.015, 0.000 ✓ |
| CI symmetry | e.g. 0.002 ± 0.033 → [−0.031, +0.035]; −0.013 ± 0.060 → [−0.073, +0.047] | All Table 2 CIs internally consistent ✓ |
| "±0.037" | Max \|bound\| over (ii)/(iii) = 0.037; over (iv) = 0.073 | ✓ only if restricted to (ii)/(iii) (Minor 1) |
| Practical size of ±0.035 | 0.035/0.2275; BSS range | **≈15% of the uninformative Brier**; BSS spans about 0.20–0.51 → practically meaningful |
| B–F SE formula | SE² = (1/(k·r) + n₂/n₁)·σ̂², k·r = 100, n₂/n₁ = 3/12 | **0.26·σ̂² ✓**, df = k·r − 1 = 99 matches Bouckaert–Frank; df = 14 recommended |
| Implied SD of fold differences | Half-width/(1.984·√0.26): Gated−(ii) 0.033 → 0.033; Gated−MLP 0.018 → 0.018 | Plausible given fold Brier SD about 0.075–0.084 |
| Minimum detectable ΔBrier (80% power) | 2.8 × SE: Gated−(ii) SE ≈ 0.0166; Gated−MLP SE ≈ 0.0091 | **≈0.047** and **≈0.025** |
| Post-hoc TOST feasibility | 90% half-width ≈ 1.66 × SE | Gated−MLP ≈ ±0.015 → passes at ±0.02. Gated−(ii) ≈ ±0.028 → fails at ±0.02, passes at ±0.03. |
| Exact sign-flip | 2¹⁵ | 32,768 ✓ |
| H2a | d_z = t/√n = −1.49/√15 | −0.385 ✓. SE = 1.625/2.145 = 0.758. 90% CI = −1.13 ± 1.761 × 0.758 = [−2.47, +0.20] ✓. SD_diff = 0.758√15 = 2.94 → 2.47/2.94 = **0.84** ✓ |
| Lag noise floor | SD of U(−15, 15) = 30/√12 | 8.66 s vs observed 8.3 s → lag ≈ noise |
| Legacy Wilcoxon, n = 5 | Minimum two-sided p | 0.0625 → "W = 1, p = 0.125" ✓; "W = 0" could never reach p < 0.05 |
| Spearman CIs (n = 15, Fisher-z approx.) | z ± 1.96/√12 | 0.88 → [0.67, 0.96]; 0.76 → [0.40, 0.92]; 0.92 → [0.77, 0.97] |
| Within-subject AUROC SE (Hanley–McNeil, 10 vs 19) | AUC 0.9 → √(0.929/190); AUC 0.7 → √(2.19/190) | ≈0.070; ≈0.107 (before autocorrelation) |
| Recalibration rescue fraction | (0.296 − 0.163)/(0.296 − 0.142) | **86%** |
| Complete-condition network deficit | 0.122 − 0.070 | 0.052 Brier (≈23% of uninformative Brier), larger than any reduced-condition difference, and untested |

## 8. Novelty assessment with verified citations

**Verified citations:**

- **Lessmeier et al., Arch Intern Med 1997;157(5):537–543.** All four numbers check out:
  - a retrospective survey of 107 consecutive patients with documented re-entrant PSVT referred for electrophysiological study;\[13\]
  - PSVT unrecognised after initial evaluation in 59 (55%), including 13 of 32 (41%) with pre-excitation;\[14\]
  - unrecognised for a median of 3.3 years;\[14\]
  - 67% met DSM-IV panic disorder criteria.\[13\]
- **CognifySSL v2.0 (Cisse & Ishimaru, Sensors 2026;26(8):2419).** Verified points:
  - It is a self-supervised detect-and-repair framework with random modality dropout during training, evaluated on WESAD under LOSO.\[11\]
  - It reports accuracy, macro-F1 and reconstruction RMSE. It reports **no AUROC, ECE, Brier or quantitative calibration**; calibration is mentioned only qualitatively.\[11\]
  - Baselines are CroSSL, MU-MAE and COCOA. **None** is zero-fill, mean imputation or a supervised masked model, and the paper does not say whether the baselines were retrained with dropout.\[11\]
  - Its dropout evidence is a self-ablation: 85.5% vs 74.1% F1 with vs without dropout under partial input.\[11\]
  - Full-modality WESAD F1 is 92.2% vs MU-MAE 89.6% and CroSSL 88.7%.\[11\]
  - It uses a single seed with no variance reported.\[11\]
  - Its WESAD modality list is internally inconsistent: full modality is defined as IMU, ECG, EDA and TEMP, while RESP appears only in reconstruction.\[11\]
  - *Correction for the manuscript:* it reports accuracy as well as F1, and it does not clearly use respiration for classification.
- **PULSE (Zhao et al., arXiv 2510.24058; CVPR 2026 Workshop).** Verified points:
  - 0.994 AUROC / 0.988 AUPRC on binary baseline-vs-stress, LOSO, without EDA at inference.\[5\]
  - Student modalities are ECG, BVP, ACC and TEMP; windows are 60 s with a 0.25-s stride; per-subject z-scoring uses baseline windows.\[5\]
  - S12 is a fixed validation fold, leaving 13 LOSO test subjects, even though the table caption says 15.\[5\]
  - Baselines include a no-teacher model on deployment sensors only (AUROC 0.963), symmetric alignment (0.972) and a full-sensor model with EDA (0.983).\[5\]
  - So PULSE *did* compare against a reduced-sensor-trained model, and the task matches but the protocol does not.
- **SimMLM (Li, Chen & Han, ICCV 2025, arXiv 2507.19264).** Verified points:
  - DMoME with a learnable gating network, plus the MoFe ranking loss.\[6\]
  - Evaluated on BraTS 2018, UPMC Food-101 and avMNIST.\[6\]
  - ECE and SCE (20 bins) averaged over all 15 full and missing settings, on the BraTS *validation* set.\[6\]
  - Baselines are RobustSeg, mmFormer, ShaSpec and MoMKE (dedicated missing-modality methods). Calibration comparisons are against MoMKE, DMoME without MoFe, and DMoME with a confidence-ranking loss.\[6\]
  - SimMLM itself uses random modality dropout ("a common practice").\[6\]
  - It does **not** include a no-gate-with-same-dropout ablation in the main text. The present paper's gate-vs-masked-MLP control is therefore a genuine complement.
- **Ma et al., "Calibrating Multimodal Learning," ICML 2023.** Verified that confidence can *increase* when modalities are removed or corrupted, and that the CML regularizer was proposed.\[15\]\[16\]\[17\] PMLR volume and page numbers were not re-verified.
- **Lechuga López, Shamout & Rudner, arXiv 2603.02719.** Verified: selective prediction in multimodal ICU multilabel classification can degrade despite strong aggregate metrics, driven by class-dependent miscalibration.\[12\] Appropriately cited.
- **Roy et al., arXiv 2603.13373.** This is "Ethical Fairness without Demographics in Human-Centered AI" (Flare), a demographic-agnostic fairness framework evaluated on EDA, IHS and OhioT1DM.\[18\] **It is not about device variation or unequal device access.** The citation for that point is mismatched: replace it or re-describe it.
- **Zinzuwadia & Singh, Lancet Digit Health 2022;4(12):e856–e857**, "Wearable devices—addressing bias and inequity." Verified and appropriate.\[19\]\[20\]
- **Tsai et al., JMIR Med Inform 2022;10(2):e33063.** Verified: 7-day panic-attack prediction, best random forest accuracy 81.3%.\[21\]
- **Oh et al., J Med Internet Res 2025;27:e69045.** Verified: wearable ECG, a ConvNetQuake adaptation pretrained on PTB-XL, accuracy 71.43%, F1 76.60%.\[22\]\[23\]
- **Zahed et al., Digital Health 2025.** Verified: a scoping review of 7 studies with accuracy 67.4–94.8%.\[24\]
- **Korea University dataset.** Zenodo 10.5281/zenodo.12194136 is v1, published 21 June 2024, and Zenodo notes that a **newer version exists**.\[25\] The Scientific Data descriptor (Jang et al., 2024) cites 10.5281/zenodo.14190138. It uses Fitbit Inspire 1/2.\[26\] **Cite the descriptor and the current version.**
- **Bouckaert & Frank, PAKDD 2004.** Verified: the corrected repeated k-fold CV test, with 10 × 10 CV giving the best replicability.\[27\]\[28\]\[29\]
- **Bates, Hastie & Tibshirani, JASA 2024;119(546):1434–1445.** Verified: naive CV intervals undercover, and nested CV gives approximately correct coverage.\[3\]\[30\]
- *Not re-verified in this pass because of the search budget, but consistent with my knowledge:* ModDrop (Neverova et al., TPAMI 38(8):1692–1706, 2016), Grinsztajn et al. (NeurIPS 2022 D&B), Roelofs et al. (AISTATS 2022), Błasiok & Nakkiran (ICLR 2024), Geifman et al. (ICLR 2019), Nadeau & Bengio (Mach Learn 2003), TRIPOD+AI (Collins et al., BMJ 2024;385:e078378).

**Is the negative result novel?** Only partly.
- **Ma et al., CVPR 2022 ("Are Multimodal Transformers Robust to Missing Modality?", pp. 18177–18186)** found that the optimal fusion strategy is dataset-dependent and that "multi-task learning contributes more than the fusion strategy", with the largest gap when only 10% of text is available at test time. That is essentially the same "training regime beats architecture" message, in vision-language transformers.
- **Ramazanova et al. (CVPR 2025 Workshop, egocentric datasets)** report that the modality-dropping baseline "consistently outperforms the training-free approaches by a significant margin", which they call expected.\[10\]
- **The SimBaMM benchmark (arXiv 2512.22991)** re-implemented 19 multimodal methods on 9 datasets under standardised tuning, CV and statistical testing. It found that complex multimodal architectures do not reliably beat unimodal baselines or a simple baseline, and it documents methodological shortcomings even in top-tier papers.\[31\] This is the most important **missing reference**: it supports the general thesis, but it also lowers the novelty of "architecture doesn't matter under fair evaluation".
- **Woo et al. (AAAI 2023, missing-modality robust action recognition)**, **MMP (2024)** and the **ACL 2024 multimodal prompt-learning** work all use modality dropout as a named baseline.\[7\]\[8\]\[9\]\[32\] Masked-training baselines are therefore *standard* outside wearables.

**What is genuinely new:**
- **(i)** A matched-capacity gate-vs-masked-MLP control, with *calibration* and selective-prediction endpoints, in wearable physiology. SimMLM, CognifySSL and PULSE all lack a gate-ablation-at-matched-dropout.
- **(ii)** Subject-level evidence that, without EDA, every model's performance reduces to heart-rate response.
- **(iii)** A documented, regression-tested catalogue of evaluation errors in a small-n wearable study, with the errors' effect on the conclusion shown.

**Recommendation:** make (iii) plus (ii) the headline, frame (i) as a confirmatory negative result consistent with Ma et al. (2022) and SimBaMM, and drop or narrow the "usually tested against weak baselines" sentence.

## 9. Construct and clinical validity

- **TSST is not panic.** The paper says this clearly, which is good. Additional points to add:
  - TSST HR reactivity is heterogeneous. A meta-analysis comparing traditional and virtual TSST (Helminen et al., Psychosom Med 2021) reports significant between-study heterogeneity in HR reactivity, with a pre-post heart-rate effect size of 0.85 for the traditional TSST (0.98 for the virtual TSST).
  - Reactivity is lower in endurance- and resistance-trained men (a study of 12 endurance-trained, 10 resistance-trained and 12 untrained men found that trained men "exhibited lower heart rate responses to psychosocial stress"), and lower with higher neuroticism.
  - So the 4/15 weak responders here (27%) are plausible rather than anomalous. In panic disorder, HR responses during attacks are also variable, and some attacks occur without marked tachycardia.
- **Fixed order** (M6). Baseline always comes first; stress and amusement order was swapped between the two protocol versions and alternated across subjects "in order to avoid effects of order" (Schmidt et al., ICMI 2018). Use this for an order-balanced contrast.
- **Clinical extrapolation.** "HR reliance is the failure mode at the panic-vs-cardiac boundary" is a reasonable *hypothesis*, but it does not follow from these data:
  - A detector that collapses to HR would flag PSVT, AF-RVR and sinus tachycardia from any cause as "stress-like". That supports caution.
  - But the true discriminating information at that boundary (abrupt onset/offset, rate above about 150 bpm, RR regularity, P-wave morphology) is *rhythm* information that a 60-s mean-NN/RMSSD feature set discards by design.
  - The finding is therefore about this feature set and this proxy task, not about wearables' ability to separate panic from cardiac events.
- **Lessmeier use.** The citation is accurate, but add context:
  - it is a selected electrophysiology-referral population, studied retrospectively;\[13\]
  - the 67% figure reflects *symptom-criteria overlap*, not co-diagnosed panic disorder;
  - the direction is cardiac misattributed as psychiatric.\[33\]
  - This supports the motivation that a wrong "it's panic" call is the dangerous error, which argues for abstention or escalation and for reporting sensitivity to tachyarrhythmia-like inputs. That is a useful framing for the discussion.
- **Population.** 15 young, healthy, mostly male participants (12M/3F, 27.5 ± 2.4 y). Generalisation to the target population (older adults and people with cardiac comorbidity) is untested. The device-access equity concern (Zinzuwadia & Singh) is relevant,\[19\] but needs a correct citation in place of Roy et al.

## 10. Research integrity and reproducibility

**Credit:**
- Superseded numbers are disclosed (Appendix A/B).
- Both rule evaluations are reported.
- "Pre-registered" has been removed.
- The COI is on page 1.
- There are 22 regression tests, the frozen table has a hash, and Python and package pins are specified.

**Outstanding:**
1. **Placeholder.** "[List any leaked number that appeared publicly]" must be resolved, and it is posting-blocking. Given the COI, state explicitly whether any superseded number (e.g. 4.23×, 1.61×, "5/5 folds") appeared in investor decks, accelerator applications, talks or social media.
2. **Provenance.** The decision rule, the rewrite and the rerun are all dated 2026-10-01 with no independent timestamp. Tag the current state as "post-rerun". If further confirmatory analyses are planned (for example the stress-vs-amusement contrast, an external dataset, or the wrist-only E4), **pre-register those on OSF before running them**. That would turn the next revision into genuinely confirmatory evidence.
3. **Residual error risk.** Nine errors have been found so far, and there are 95 untraced tokens. Commission an independent re-implementation (Section 5, M9) and report the agreement.
4. **Release.** Release the code, the frozen feature table (WESAD licence permitting derived features), seeds, the 20 partitions and per-participant results.
5. **COI management.** Ideally, a co-author or acknowledged statistician without equity should check the primary analysis.

## 11. Recommended framing and checklist

**Title:**
> *"No Evidence That Modality Gating Adds to Missingness-Aware Training When Wearable EDA Drops Out: A 15-Participant WESAD Case Study in Evaluation Pitfalls"*

**Alternative:**
> *"A Modality-Gating Advantage Under Missing EDA Was a Baseline Artefact: Fair Comparators, Heart-Rate Dependence, and Five Evaluation Errors"*

**Two-sentence claim:**
> With EDA removed at test time, a modality-gated network trained with modality dropout showed no detectable advantage over a capacity-matched MLP trained with the same masking (ΔBrier +0.002, 95% CI −0.016 to +0.020), nor over reduced-sensor LightGBM baselines (CIs about ±0.035, which cannot exclude practically relevant differences). The earlier apparent advantage came from comparing against a model never exposed to missing EDA, and without EDA every model's discrimination depended mainly on each participant's heart-rate response to the stressor.

**Minimum pre-posting checklist (blocking):**

| # | Item | Effort |
|---|---|---|
| 1 | Resolve the leaked-number placeholder; confirm the COI statement | 1 h |
| 2 | `git init`; commit full history as available; tag "post-rerun"; release code, frozen table, partitions, per-participant results | 0.5–1 day |
| 3 | Retitle; replace "determines" and "not gating" with conditional, evidence-matched language; narrow or remove "usually tested against weak baselines" | 2–3 h |
| 4 | Post-hoc TOST (justified SESOI, e.g. 0.02 Brier / 0.03 AUROC) or ROPE, clearly labelled post hoc | 0.5 day |
| 5 | Fix the "±0.037" sentence to cover (iv) or restrict its scope | 15 min |
| 6 | Report complete-condition paired CIs for all models, untuned and tuned; discuss the network deficit | 0.5 day |
| 7 | Re-run B–F with df = 14 as a sensitivity analysis; add the participant-level paired analysis | 0.5 day |
| 8 | Fix citations: Roy et al. (mismatch), the Korea dataset (current version plus the Sci Data descriptor), the CognifySSL description (accuracy + F1; RESP ambiguity), the PULSE protocol details (13 test subjects, baseline-referenced normalisation, no-teacher 0.963); add Ma et al. CVPR 2022 and SimBaMM | 2–3 h |
| 9 | Trace or remove the 95 untraced tokens | 0.5 day |

**Optional analyses, ranked by value added:**
1. **The MLP-without-masking cell** (completes the 2×2; estimates the gate × masking interaction). About 1 day. This could turn the paper into "gating substitutes for, but does not add to, missingness training".
2. **Order-balanced stress-vs-amusement contrast, plus the negative control** (the baseline-vs-stress model applied to amusement and meditation windows). About 1–2 days. This directly tests M6 and strengthens the HR mechanism.
3. **An independent minimal re-implementation** of the primary comparison. About 2–3 days for a second person.
4. **Wrist-only E4 replication** (BVP-derived HR, wrist EDA dropout). The data are already available, it is the realistic deployment hardware, and it can be pre-registered on OSF first. About 3–5 days.
5. **A second dataset** with the same pre-registered protocol (e.g. a public E4 stress dataset with 36 participants, as used by PULSE). About 1 week.
6. **Realistic missingness** (partial, intermittent, motion-correlated EDA loss) instead of complete MCAR removal. About 2–3 days.
7. **Normalised HR-response analysis, SmoothECE, E-AURC with random and oracle references, BCa bootstrap.** About 1 day in total.

## 12. Venue recommendations and hypothetical scores

| Venue | Fit | Current score (est.) | After blocking + top-2 optional |
|---|---|---|---|
| NeurIPS/ICLR "I Can't Believe It's Not Better"-type negative-results workshop (check 2026–27 availability) | **Excellent.** A negative result plus a mechanism plus pitfalls | 6/10 (weak accept) | 7/10 (accept) |
| ML4H Findings / symposium | Good | 5/10 (borderline) | 6–7/10 |
| TMLR (correctness and interest criteria; no novelty bar) | Good if the claims are scoped | Major revision | Likely accept (with the independent re-implementation) |
| CHIL / MLHC main track | Weak. Single small dataset, no clinical cohort | 4/10 (reject) | 5/10 (borderline) |
| DMLR (data-centric ML) | Moderate. The evaluation-pitfalls angle fits | Borderline | Weak accept |
| npj Digital Medicine / Lancet Digital Health / Nature Machine Intelligence | **Poor.** No clinical data; proxy task; n = 15 | Desk reject | Desk reject unless a clinically labelled paired cohort is added |
| ReScience C | Not applicable (not a replication of a published paper) | — | — |

**Overall:** the scientific substance is honest and largely correct. With the blocking items done, the paper is a useful, citable negative result and pitfalls case study. Its main-track value depends on completing the 2×2 factorial and breaking the condition-order confound.

## Sources

1. [Introduction to correctR](https://cran.r-project.org/web/packages/correctR/vignettes/correctR.html)
2. [(PDF) Cross-validation: what does it estimate and how well does it do it?](https://www.researchgate.net/publication/350578420_Cross-validation_what_does_it_estimate_and_how_well_does_it_do_it)
3. [\[2104.00673\] Cross-validation: what does it estimate and how well does it do it?](https://arxiv.org/abs/2104.00673)
4. [Rebuild feature extraction and evaluation for the JBHI revision by ALVAROI12 · Pull Request #3 · ALVAROI12/smartwatch-stress-detection](https://github.com/ALVAROI12/smartwatch-stress-detection/pull/3)
5. <https://arxiv.org/html/2510.24058>
6. <https://arxiv.org/html/2507.19264>
7. [Multimodal Prompt Learning with Missing Modalities for ...](https://aclanthology.org/2024.acl-long.94.pdf)
8. [www.arxiv.org](https://www.arxiv.org/pdf/2410.03010)
9. [MMP: Towards Robust Multi-Modal Learning with Masked Modality Projection](https://arxiv.org/html/2410.03010v1)
10. [Exploring Missing Modality in Multimodal Egocentric Datasets Merey Ramazanova](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/papers/Ramazanova_Exploring_Missing_Modality_in_Multimodal_Egocentric_Datasets_CVPRW_2025_paper.pdf)
11. [Detect and Repair: Robust Self-Supervised Wearable Sensing Under Missing Modalities](https://doi.org/10.3390/s26082419)
12. <https://arxiv.org/abs/2603.02719>
13. [Is It Panic Disorder or PSVT?](https://www.clinician.com/articles/37321-is-it-panic-disorder-or-psvt)
14. [Unrecognized Paroxysmal Supraventricular Tachycardia: Potential for Misdiagnosis as Panic Disorder](https://jamanetwork.com/journals/INTEMED/articlepdf/623013/archinte_157_5_013.pdf)
15. [ICML Poster Calibrating Multimodal Learning](https://icml.cc/virtual/2023/poster/24749)
16. [ICML 2023 Calibrating Multimodal Learning Oral](https://icml.cc/virtual/2023/oral/25541)
17. [\[2306.01265\] Calibrating Multimodal Learning](https://arxiv.org/abs/2306.01265)
18. <https://arxiv.org/abs/2603.13373>
19. [Wearable devices: underrepresentation in the ageing society - The Lancet Digital Health](<https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00069-9/fulltext>)
20. [Short Paper Patterns of Ownership and Usage of Wearable Devices in the](https://www.jmir.org/2024/1/e56504/PDF)
21. [JMIR Medical Informatics - Panic Attack Prediction Using Wearable Devices and Machine Learning: Development and Cohort Study](https://medinform.jmir.org/2022/2/e33063/)
22. [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/41092077/)
23. [Journal of Medical Internet Research - Panic Attack Prediction for Patients With Panic Disorder via Machine Learning and Wearable Electrocardiography Monitoring: Model Development and Validation Study](https://www.jmir.org/2025/1/e69045)
24. [Utility of wearable technology in predicting panic attacks: A scoping review - PubMed](https://pubmed.ncbi.nlm.nih.gov/41181552)
25. [Digital Phenotyping Dataset for Impending Panic Symptoms: A Prospective Longitudinal Study](https://zenodo.org/records/12194136)
26. [A digital phenotyping dataset for impending panic symptoms: a prospective longitudinal study](https://www.nature.com/articles/s41597-024-04147-6)
27. [Evaluating the Replicability of Signiﬁcance Tests for](https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf)
28. [Statistical Significance Tests for Comparing Machine Learning Algorithms - MachineLearningMastery.com](https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/)
29. [Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms](https://link.springer.com/chapter/10.1007/978-3-540-24775-3_3)
30. [A Honest Cross-Validation Estimator for Prediction Performance](https://arxiv.org/pdf/2510.07649)
31. <https://arxiv.org/abs/2512.22991>
32. [Towards Good Practices for Missing Modality Robust Action Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/25378/25150)
33. [Unrecognized paroxysmal supraventricular tachycardia. Potential for misdiagnosis as panic disorder - PubMed](https://pubmed.ncbi.nlm.nih.gov/9066458/?dopt=Abstract)
