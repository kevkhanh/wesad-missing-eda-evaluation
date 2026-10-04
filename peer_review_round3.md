# Peer review, round 3, merged: `e14-preprint-outline.md` (state as of 2026-10-01 19:14)

**Sources:**
- **[R1]** Code-and-results review: claims checked against `run_baselines.py`, `run_repeated_cv.py`, `run_metrics.py`, `run_locked_comparison.py`, `train_stage3_models.py` and `results/*.json`.
- **[R2]** External reviewer: a document-only reading.
- **[R1+R2]** Both reviewers raised the item independently.
- **[verified]** The item was reproduced from code or result files. It is not taken from the outline's own description.

---

## 0. Consolidated verdict

Both reviewers agree on what matters most. **The falsification rule fired and the paper reported it.** That is the strongest thing in the project. The paper you now have, a negative result showing that the gated model's advantage was an artefact of how the baseline encoded missing data, is better than the one you planned.

The reviewers disagree on how close it is to posting:

| | R2 (document only) | R1 (document + code) |
|---|---|---|
| Readiness | "Close to postable" | Not yet |
| Remaining work | About 1 week of writing, no new experiments | The same writing, **plus** code fixes and a re-run of the existing pipeline |
| Score | 5–6/10 at a Findings-type venue | Agree, *after* the re-run |

The difference is that R2 could not see the code. Four implementation defects (§1, items C1–C4) affect the numbers the negative result is built on. None of them is a new experiment: each is a bug fix, followed by re-running scripts that take minutes on 434 rows. The direction of the conclusion will probably survive. The specific numbers, p-values and one of R2's three proposed headline findings will not (see §3).

**Merged plan:** fix the code (1–2 days) → re-run from one frozen feature table → write the R2 rewrite around the re-run numbers (about 1 week). Do not polish the text before the re-run, or it will need rewriting again.

---

## 1. CRITICAL: code defects behind the result [R1, verified]

### C1. The "20× repeated subject-level CV" uses one subject partition 20 times
`run_repeated_cv.py:294-299` shuffles the rows and then calls `GroupKFold(5)`. `GroupKFold` assigns groups by size and does not depend on row order. A re-run of the partition logic gives **1 distinct partition across all 20 repeats**: `{S5,S7,S8} {S4,S6,S17} {S3,S15,S16} {S2,S11,S14} {S9,S10,S13}`. Only the model seeds and test-time masks vary.

Consequences:
- The Bouckaert–Frank t and p values are not valid as described.
- Every "20× repeated subject-level CV" statement is false: lines 74, 149, 151, 171, 176, 186, 521 and 536.

**Fix:** use `GroupKFold(5, shuffle=True, random_state=rep)` (supported in the installed sklearn 1.9.0) and add a test asserting more than 1 distinct partition. At n=15, also report LOSO.

### C2. The calibration protocol in Methods is not implemented
Lines 75 and 178 describe one Platt model fit on the training fold and applied to both tiers, an isotonic sensitivity check, and reporting before and after recalibration. In the code:
- Baselines (i), (ii), (iii), (hr) and every gated variant are **not recalibrated at all**.
- Only (iv) is recalibrated, using per-condition Platt fit on **in-sample** LightGBM training predictions (`run_baselines.py:245-261`). This overfits.
- Isotonic is not run anywhere.

**Fix:** implement Platt on inner-CV out-of-fold predictions for every model, or rewrite Methods to describe what was actually done.

### C3. The "debiased ECE" estimator is biased toward zero, and this undermines one of R2's headline findings
`run_metrics.py:109-127` clips each bin-count estimate at 0 and takes the **minimum** over six bin counts. That is not Roelofs' ECE_SWEEP. **20 of the 40** reduced-condition per-fold values are exactly 0.000, each computed on about 43 windows.

The effect on R2's proposed finding #3 (§3) is decisive:

| Reduced condition, fold mean | "debiased ECE" (current) | equal-width 10-bin ECE | Calibration slope per fold |
|---|---|---|---|
| Gated, HR removed | **0.042 (ranked best)** | **0.229 (ranked worst)** | −0.06, 0.56, 0.66, 1.24, −0.01 |
| Gated + dropout | 0.056 | 0.189 | 0.33, 1.10, 0.89, 0.67, 0.31 |
| LGBM NaN-training | 0.047 | 0.177 | 1.26, 2.53, 1.71, 1.02, 0.21 |
| LGBM ECG-only | 0.052 | 0.199 | 0.87, 2.12, 1.74, 0.97, 0.18 |
| HR-only | 0.056 | 0.187 | 1.52, 2.42, 0.81, 1.08, 0.07 |

**Fix:**
- Make the primary calibration endpoints calibration intercept and slope, plus Brier and log-loss, on **pooled out-of-fold** predictions with subject-level uncertainty.
- If an ECE is kept, implement Roelofs' monotone sweep or smooth ECE, with no clipping.
- One of the falsification rule's three endpoints was this estimator. Say that in the text, and re-evaluate the rule after the fix.

### C4. The headline table cannot be reproduced from the current data
The ratio table, the ablations, H-C1 and H2a cite **445** WESAD windows. The current cache has **434** (27–30 per subject; 35% stress). `run_locked_comparison.py` calls `_load_dataset()` without `_restrict_to_wesad()`, and on the current cache that loads **1,161 rows, including 727 MIT-BIH windows**.

**Fix:** freeze one feature table, record its SHA-256 in §9, and re-run every number from it.

---

## 2. CRITICAL: provenance of the falsification rule [R1+R2, verified]

Line 111 says the rule is "registered on OSF (DOI to add) and tagged in git (`baselines-plan-v1`)." No such artefacts exist:
- The project is not a git repository.
- `decision_timeline.md` rows 9–10 show that the registration was only *planned*, and that the baselines ran the same day.
- Line 545 plans to tag "before `run_baselines.py` executes," which can no longer happen. **Tagging now would create a timestamp that implies something false.** [R2]
- Line 12's "committed as a static YAML document" has the same problem: without git, nothing was committed. [R2]

**Replacement wording [R2]:** *"The decision rule was written in the analysis outline before the baselines were run; it was not independently timestamped."*

Also disclose, from row 9 of the timeline, that the gated model's prior-draft point estimates had already been seen when the rule was written.

This weakens the claim less than it might seem, because the rule went against the author and was kept. Say exactly that. Apply the same correction at lines 12, 111, 521 and 545. When git is initialised, tag the current state as what it is (e.g., `post-baselines-2026-10-01`), not as a pre-registration.

---

## 3. What the paper should claim

R2 proposed three headline findings. Here is how each holds up against the code and results.

| R2 finding | Status | Corrected form |
|---|---|---|
| **1. The gain was a missing-data artefact.** The advantage disappears once missing EDA is handled fairly. | **Keep, with a CI.** The direction holds in every analysis. But the gated-vs-(ii) Brier difference is +0.014, approximate 95% CI [−0.033, +0.060] (vs (iii): [−0.029, +0.059]). That interval is compatible with ±30% relative differences. [R1] | "No advantage remained against fair baselines; point estimates favoured the baselines; equivalence was not tested." Do **not** write "as good or better" (lines 113 and 165). |
| **2. Discrimination is mostly heart rate.** Removing `hrv_mean_nn` drops AUROC from 0.881 to 0.706. | **Keep, reworded.** 0.706 still retains about 54% of the above-chance AUROC, so "nearly all … comes from `hrv_mean_nn`" (lines 167 and 472) is too strong. Also, the hr-resid variant keeps `hrv_mean_nn` as an input and residualises across pooled subjects, not within subject (`run_baselines.py:343`). Its AUROC of 0.878 therefore cannot test "HRV beyond HR." [R1] | "HR alone matches the gated model's discrimination (0.858 vs 0.881); removing HR roughly halves above-chance AUROC." Fix hr-resid to remove HR and residualise within subject. Run hr-removed for (ii) and (iii) too. |
| **3. Calibration can look good when discrimination is gone.** The HR-removed model has the best ECE (0.042) at AUROC 0.706. | **Do not use as stated.** It ranks best only on the broken estimator (C3). On the standard ECE it is the worst-calibrated model, with calibration slope ≈ 0 in two folds. | Either (a) re-test after C3 and keep it only if it survives, or (b) turn it into the opposite cautionary example: *a small-sample, clipped ECE estimator ranked the worst-calibrated model as the best.* That is a real methods point at n≈43 windows per fold. |

**A fourth finding that neither draft reports [R1, verified].** The fold containing **S9, S10 and S13** is where every model fails:

| | that fold | the other folds |
|---|---|---|
| Reduced AUROC, every model | 0.58–0.75 | 0.78–0.998 |
| Reduced Brier, every model | 0.32–0.44 | 0.06–0.22 |

S9 and S10 are the two weakest heart-rate responders: their mean NN interval changes by −25 and −39 ms from baseline to stress, compared with −100 to −450 ms for 11 of the 15 subjects. This fold produces most of the ±0.13 Brier SDs. Model rankings also change by fold: the gated model is best on this fold and worst on fold 0.

*Every model fails on subjects whose heart rate does not respond to the stressor.* That is the panic-vs-cardiac risk observed directly. Report it with a per-subject forest plot.

**Title.** R2's suggestion ("…Disappear Against Fair Baselines…") is the right shape: a title that answers its own question. "Disappear" slightly overstates the result given the CIs above. Suggested alternatives:
> *"No Calibration Advantage for Modality Gating Under Missing Wearable Sensors Against Fair Baselines: A Subject-Level Evaluation on WESAD"*
> *"Modality Gating's Calibration Advantage Under Missing EDA Was a Baseline Artefact: A Subject-Level Evaluation on WESAD"*

---

## 4. MAJOR: the text has not caught up with the result [R1+R2]

Rewrite the following around the falsification outcome. Do this **after** the re-run so the numbers are final.

| Location | Problem | Action |
|---|---|---|
| Introduction (line 33) | "The main empirical question … is whether it beats a per-configuration ECG-only LightGBM…", which is now answered. The sentence still quotes 7.5% vs 10.6%. | Rewrite: the question was asked, the rule fired, and here is what remains. |
| Novelty item 1 (line 20) | Still conditional ("if the rule fires…"). | State the outcome. |
| Discussion (line 452) | "…is the venue-version's main question." | Rewrite around the negative result, HR dependence and the low-HR-responder failure. |
| Conclusion (line 472) | Leads with the ratio improvement over zero-fill. [R2] | Lead with the negative result. Mention the ratio only as the artefact being explained. |
| Abstract skeleton (lines 522–523) | Placeholders ("If met… if not…"). [R2] | Fill in with the re-run numbers and CIs. |
| Ratio block (lines 117–135, 171) | Still labels 1.61× the "headline", keeps a "Headline claim to use" paragraph, and says "the architecture's built-in test-time robustness." [R2] | Move to an appendix captioned "comparison against the zero-fill baseline; superseded as evidence of architectural benefit." |
| Line 145 | "(prior-version headline)" label in the primary table. | Remove. |
| Line 185 | LightGBM vs gated still "pending." | Remove. |
| Lines 442–444, 496 (3), §11 table | Scripts that have run are still listed as to-dos. [R1+R2] | Mark them done, with dates. |
| Line 127 | Open TODO in body text. | Resolve or move to an appendix. |
| Lines 307 vs 326 | Step 4 described as both "fixed" and "will be fixed"; dead reference to "line 168 above". | Reconcile. |
| Line 454 | "…decision boundary that motivates the company". | Remove from the paper. |
| Line 4 | "If any pre-correction number did [appear in investor material]…" | State the answer definitively. |
| Throughout | About 40% of the text is revision bookkeeping ("per peer-review C1…"). | Produce a clean manuscript and make sure none of it leaks into the arXiv version. |

---

## 5. MAJOR: cut H-C1 [R1+R2, verified]

R2 suspected that H-C1 inherits the zero-fill artefact. **Confirmed.** `run_locked_hc1` scores LightGBM with `fillna(0.0)` (`run_locked_comparison.py:420`), so the −0.484 interaction measures the straw-man baseline. In addition:
- Tier B is identical to the H-B1 reduced condition (line 249).
- The outcome is per-window log-loss with only a subject random intercept, not accuracy.
- "890 obs" is 445 windows × 2 model types, not the explanation given at line 252.
- Nothing in the code produces the 7.5% / 10.6% figures.

**Action:** remove H-C1 from the Introduction, Results, Discussion, Conclusion and abstract. Keep the Tier 1/2/3 protocol as future work, pending the raw WESAD data.

---

## 6. Number and reporting fixes

| Location | Problem | Fix | Source |
|---|---|---|---|
| Line 162 | "*worse* on AUROC (0.869 vs 0.865)". These are repeated-CV numbers, unlabelled, and 0.869 is *higher*. | Label the source, and say "indistinguishable." | R1+R2 |
| Line 167 | "better than it on debiased ECE (0.056 vs 0.056)". | "Equal," or drop it after the C3 fix. | R1+R2 |
| Line 167 | "within noise on AUROC (0.858 vs 0.881)" mixes the 5-fold and repeated-CV numbers. | Quote one source per sentence. | R1+R2 |
| Lines 160–161 | Gated vs (iii) AUROC 0.881 vs 0.883 called "worse". | "Not better / indistinguishable." | R2 |
| Primary table | Debiased ECE has no SD. | Report uncertainty for all four metrics, and label each table as single 5-fold or repeated CV. | R1+R2 |
| Line 464 | "~90 windows per fold". | Reduced metrics use about 43 windows (about 15 stress). See M1. | R1 |
| Conclusion (line 472) | Cites (iv) mean-imputation, which is not in the repeated CV. | Include (iv) in the repeated CV or label it 5-fold-only. | R1 |
| Global | p-value precision is inconsistent. | Pick one and apply it everywhere. | R1 |

---

## 7. MAJOR: design and validity [R1]

R2 did not raise these. They are needed for the paper to be credible to a medical audience, but they do not block posting.

- **M1. Paired masking.** The complete and reduced conditions are scored on disjoint random halves of each test fold (`_apply_simulated_eda_mask(fraction=0.5)`). Score every test window under both conditions. This is a code change, not a new experiment.
- **M2. Gating vs model family.** "Gated NN loses to LightGBM" mixes gating with NN-vs-gradient-boosted trees on about 350 rows (Grinsztajn et al., NeurIPS 2022). Either add a matched non-gated MLP with the same masking and dropout, or limit the conclusion to *this* network. Also state that no model was tuned (LightGBM: 50 trees, depth 3; NN: fixed epochs, no early stopping).
- **M3. WESAD construct validity.**
  - TSST involves speech and arithmetic, so part of the HR rise is not anxiety, and panic onset is uncued.
  - Baseline always precedes stress, so tonic EDA (the top EDA feature) is confounded with drift.
  - Pooled-fold AUROC mixes between-subject offsets; add within-subject AUROC.
- **M4. Ultra-short HRV.** Windows are 60 s and appear non-overlapping (434 windows ≈ 15 × 30 min); confirm this explicitly. SDNN, pNN50, LF and LF/HF are unreliable at that length; only RMSSD has reasonable support. Shrink the IG rank-shift table.
- **M5. H2a.**
  - The subject-level TOST has not been run.
  - SESOI = 0.80 was derived from power, which reverses the logic.
  - "Equivalence … could not be rejected" is inverted.
  - Move H2a to an appendix (R2 agrees).
- **M6. Audit.**
  - The C1 pool of 38 (Step 4 scores 35) out of 55 subjects is unexplained.
  - Every p-value is at the `--fast` Monte Carlo floor.
  - Release the audit separately after the full re-run (R2 agrees).

---

## 8. Scope of the preprint (R1+R2 agree)

- **Main text:** H-B1 evaluated against fair baselines; the falsification outcome; the HR dependence; the per-subject finding on low-HR responders; and the calibration-metric lesson (only if it survives the C3 re-test).
- **Appendices:** H2a, H-B2, interpretability, and the zero-fill ratio block.
- **Separate note:** the dataset-identity audit, after the full non-`--fast` run.
- **Future work:** H-C1 real tiers and the BVP/respiration ablations, both blocked on the raw WESAD data.

---

## 9. Merged order of work

1. **Code (1–2 days):**
   - Shuffled group partitions plus a test (C1).
   - Paired masking (M1).
   - Out-of-fold Platt for every model, or honest Methods text (C2).
   - Replace the ECE estimator; add calibration slope and intercept (C3).
   - Fix hr-resid; add hr-removed for (ii) and (iii) (§3).
   - Freeze one feature table with its hash (C4).
2. **Re-run (hours):**
   - `run_baselines.py`, `run_repeated_cv.py` (with (iv) included), plus LOSO.
   - Per-subject forest plot and HR-responder stratification.
   - Re-evaluate the falsification rule on the fixed endpoints, and report it whichever way it goes.
3. **Write (about 1 week):**
   - Correct the provenance wording (§2).
   - Do the R2 rewrite of the Introduction, Discussion, Conclusion and abstract around the re-run numbers, with CIs (§3–4).
   - Demote the ratio block, cut H-C1, fix the numbers (§4–6), and settle the title.
4. **Optional, before a venue:** matched non-gated MLP (M2); the M3–M4 validity paragraphs.
5. **Git:** initialise and tag the current state truthfully (`post-baselines-2026-10-01`). Use OSF only for *future* analysis plans.
