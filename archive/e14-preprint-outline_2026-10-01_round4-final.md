# Preprint Outline: Missingness-Aware Training, Not Modality Gating, Determines Performance When EDA Drops Out: A Subject-Level Evaluation on WESAD
*(working title, see §10.1 for alternatives; target: arXiv preprint; venue selection deferred)*

**Status of this outline (2026-10-01, rewritten).**
- This version was rewritten after the round-3 review (`peer_review_round3.md`) and the post-fix re-run (`results/rerun_summary_2026-10-01.md`).
- Earlier versions are archived at `archive/e14-preprint-outline_2026-10-01_pre-rerun.md`; a version with in-place post-fix edits (21:49) is at `archive/e14-preprint-outline_2026-10-01_2149-inplace-edits.md`.
- Per-item revision tags have moved to Appendix H (change log), so the body reads as a manuscript draft.
- References were checked against the source papers on 2026-10-01 (`results/citation_check_2026-10-01.md`).
- Numbers are cross-checked by script: `python check_numbers.py` ties each table cell and claim to a specific results-file key and writes `results/number_check_report.md` (2026-10-01: 337 pass, 0 fail; 95 untraced tokens listed for review by eye).
- **Every number in §5 comes from the frozen feature table** (SHA-256 `df483a7abae0…`, `results/feature_table_manifest.json`). Numbers taken from older pipelines are labelled as such and confined to the appendices.

**Conflict-of-interest statement (page 1).** The author is a co-founder of E14 Health, a company whose product thesis depends on wearable physiological monitoring for acute-event differentiation. No result from this project has appeared in E14 Health investor, marketing, or recruitment materials.

**Non-clinical disclaimer (page 1, repeated in the abstract).** This paper studies a stress-vs-baseline detector trained on 15 WESAD participants (young, healthy, laboratory setting). It does not evaluate, and should not be read as evidence for, differentiating panic attacks from cardiac events in patients.

**Provenance of the decision rule (page 1).**
- **When the rule was written.** The falsification rule in §4.6 was written in this analysis outline on 2026-10-01, before the reduced-sensor baselines were run. It was not independently timestamped: the project was not under version control and no OSF registration exists.
- **What had already been seen.** When the rule was written, the gated model's earlier point estimates were known; the baselines' results were not.
- **How many times it was evaluated.** The rule was evaluated twice. The first evaluation used a pipeline later found to contain implementation errors (Appendix A). The second used the corrected pipeline. Both outcomes are reported.
- **Other configuration files.** `configs/preregistration.yaml` (audit thresholds) was likewise written as a file without version control. The word "pre-registered" is not used anywhere in this paper.

**Framing.** This is a methods paper motivated by a clinical differentiation problem. It does not claim to solve that problem. Each Results subsection carries its own proxy-data caveat.

---

## Contributions (stated against prior work)

**Not claimed as new:**
- modality dropout as a technique (ModDrop, Neverova et al., *IEEE TPAMI* 2016; CognifySSL v2.0, Cisse & Ishimaru, *Sensors* 2026; PULSE, Zhao et al., arXiv 2510.24058);
- learnable modality gating for missing modalities (SimMLM, Li, Chen & Han, ICCV 2025, arXiv 2507.19264);
- calibration under modality removal as a problem framing (Ma et al., "Calibrating Multimodal Learning," ICML 2023);
- calibration and selective prediction in multimodal clinical classifiers (Lechuga López, Shamout & Rudner, arXiv 2603.02719; a calibration/selective-prediction paper, not a missing-modality paper);
- device-level variation and unequal wearable access as fairness concerns in health sensing (Roy et al., arXiv 2603.13373; Zinzuwadia & Singh, *Lancet Digit Health* 2022).

**Claimed:**
1. **A negative result against fair baselines, with a decomposition.**
   - On WESAD with EDA removed at test time, a modality-gated network is statistically indistinguishable from reduced-sensor LightGBM baselines and from a plain MLP trained with the same simulated sensor loss.
   - That holds untuned and nested-tuned, and across four matched model sizes.
   - The earlier apparent gating advantage came entirely from comparing against a model never trained with missing EDA.
   - What matters is training with missingness, not the gate.
2. **Heart-rate dependence as the dominant mechanism, shown at subject level.**
   - Without EDA, every model's discrimination collapses toward chance when heart rate is removed.
   - Per-subject discrimination tracks each participant's heart-rate response to the stressor (Spearman ρ = 0.76–0.92 under LOSO).
   - This is the failure mode that matters at the panic-vs-cardiac boundary, where tachycardia is shared.
3. **An evaluation-pitfalls case study for small-n wearable studies.** Five concrete errors, each of which changed or could have changed the conclusion:
   - a record-level leak;
   - "repeated" CV that reused one subject partition;
   - a zero-clipped minimum-over-bins ECE estimator that ranked the worst-calibrated model as best;
   - complete and reduced conditions scored on disjoint windows;
   - a feature table silently replaced between runs.

   Each is paired with the check that catches it, and the decision rule's two evaluations (before and after the fixes) are both reported.

---

## 1. Introduction
- **Clinical motivation.** Panic attacks and acute cardiac events overlap in symptoms (palpitations, chest discomfort, dyspnoea), and the confusion has documented costs.
  - In a cohort of 107 patients with paroxysmal supraventricular tachycardia, 55% were unrecognised after initial evaluation.
  - 67% met DSM-IV criteria for panic disorder.
  - The median delay to diagnosis was 3.3 years (Lessmeier et al., *Arch Intern Med* 1997;157:537–543).

  Wearables that could flag which is more likely would need to stay *calibrated* when a sensor drops out, because the safe use of such a system is abstention or escalation when uncertain (Lechuga López et al.).
- **The data ceiling, stated immediately.** In a dated rapid search (2026-10-01; `results/hb2_search.md`, Appendix D) we found no public dataset that pairs confirmed panic attacks and confirmed cardiac events in the same participants with the same wearable sensors.
  - Panic-only wearable datasets exist, with self-reported panic and consumer heart rate (e.g. the Korea University digital-phenotyping dataset, Zenodo 10.5281/zenodo.12194136).
  - Cardiac-only datasets exist (12-lead or Holter).
  - This paper therefore studies a proxy question that the data can answer.
- **The question.** When the EDA sensor is missing at test time, does a modality-gated network trained with modality dropout give better-calibrated and more discriminative stress predictions than models built for the reduced sensor set?
- **What we found** (written as a candid result):
  - The gated network's advantage in our earlier drafts was an artefact of the baseline: a LightGBM trained only on complete data and given missing EDA at test time.
  - Against baselines that handle missingness fairly, the gated network is indistinguishable from them, and from a plain MLP trained with the same simulated sensor loss.
  - Training with missingness is what matters; the gate does not.
  - Without EDA, all models rely on heart rate, and their per-subject performance tracks how much each participant's heart rate responded to the stressor.
- **Why report it.** This is a decisive negative result in a literature where architectural claims are usually tested against weak missing-data baselines, and in which small-n evaluation errors are common. We report the implementation errors we found in our own pipeline and how each moved the result.

## 2. Related Work
- **Wearable stress and panic detection.** WESAD (Schmidt et al., ICMI 2018) is the standard laboratory stress benchmark. Wearable panic-attack work forecasts attacks over days to a week from consumer-wearable heart rate, sleep and activity plus questionnaires, with self-reported attack labels (Tsai et al., *JMIR Med Inform* 2022: 7-day horizon), and more recently from wearable-ECG HRV (Oh et al., *J Med Internet Res* 2025). See the scoping review by Zahed et al. (*Digital Health* 2025). This paper differs from both by working at the window level and with multiple modalities, but makes no claim beyond the proxy task.
- **Missing-modality robustness.** Three techniques are directly relevant:
  - ModDrop (Neverova et al., *IEEE TPAMI* 38(8):1692–1706, 2016; arXiv 1501.00102) is the canonical modality-dropout technique.
  - CognifySSL v2.0 (Cisse & Ishimaru, "Detect and Repair: Robust Self-Supervised Wearable Sensing Under Missing Modalities," *Sensors* 26:2419, 2026) applies random modality dropout on WESAD (ECG, EDA, respiration, temperature, ACC). It reports F1 under missing modalities, but not AUROC or calibration.
  - PULSE (Zhao et al., arXiv 2510.24058) reports 0.994 AUROC and 0.988 AUPRC on WESAD stress monitoring without EDA at inference, via privileged-information distillation with modality dropout. **Its task definition and CV protocol are not verified to match ours; do not use the number as a reference bar without checking.**
- **Learnable gating.** SimMLM (Li, Chen & Han, ICCV 2025; arXiv 2507.19264) uses "a dynamic, learnable gating mechanism" to weight modalities under full and partial availability, and reports ECE averaged across 15 full and missing-modality settings (BraTS 2018 brain MRI, not physiological signals). This is prior art for the architecture studied here.
- **Calibration under missing or corrupted modalities.** Ma et al. (ICML 2023, PMLR 202:23429–23450; arXiv 2306.01265) show that confidence can *increase* when modalities are removed, and propose the CML regularizer. Our complete-trained LightGBM shows the opposite failure: it confidently under-predicts stress when EDA is missing (§5.5).
- **Tabular model families.** Gradient-boosted trees routinely match or beat neural networks on small tabular data (Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 Datasets & Benchmarks). This is why a similar-size non-gated MLP (1,025 vs 801 parameters), plus a capacity sweep at four exactly matched sizes, is needed to isolate gating (§5.2).
- **Evaluation methodology.**
  - Subject-level CV and pseudoreplication.
  - Corrected repeated-CV inference: Nadeau & Bengio 2003; Bouckaert & Frank, PAKDD 2004.
  - Interval estimates for CV: Varoquaux 2018.
  - ECE estimator bias: Roelofs et al., AISTATS 2022; Błasiok & Nakkiran, ICLR 2024 (SmoothECE).
  - TRIPOD+AI (BMJ 2024;385:e078378).
- **Shortcut learning and dataset identity:**
  - Zech et al., *PLOS Med* 2018.
  - Rabanser et al., NeurIPS 2019.
  - DeGrave et al., *Nat Mach Intell* 2021.
  - These matter for the companion audit note (Appendix F), not for this paper's within-WESAD results.

## 3. Data
- **WESAD** (Schmidt et al., 2018):
  - 15 participants (12M/3F, 27.5 ± 2.4 years).
  - Chest RespiBAN (ECG, EDA, respiration, EMG, temperature, ACC at 700 Hz) and wrist Empatica E4 (BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz).
  - **Used here:** chest ECG and chest EDA. Baseline vs stress (TSST) segments; amusement and meditation excluded.
- **Windows** (`real_data_loaders._window_wesad_subject`):
  - Non-overlapping 60-s windows on the 700 Hz chest signals.
  - A window is kept only if its WESAD label is constant across all 60 s, so windows spanning a condition transition are dropped.
  - At most the first 20 windows per condition per participant are kept. After the label-constancy filter the cap is never reached: baseline yields 18–19 windows and stress 9–11.
  - Result: 434 windows, 27–30 per participant, 35% stress.
  - The table is frozen by hash. It replaces a 445-window table used by the 2026-09-20 runs that no longer exists (Appendix A).
- **Construct validity (stated here, not only in Limitations):**
  - **TSST is not panic.** It is a social-evaluative task with public speaking and mental arithmetic, so part of the heart-rate rise reflects speech, respiration and motor activity rather than anxiety. Panic onset is typically uncued and often accompanied by hyperventilation.
  - **Fixed condition order.** In WESAD the baseline always precedes the stressor. Tonic EDA level and slope are therefore partly confounded with time-on-electrode drift and habituation.
  - **Population.** The sample is young, healthy and mostly male.
- **MIT-BIH Arrhythmia** is used only in the companion dataset-identity audit (Appendix F) and plays no role in the results below.

## 4. Methods

### 4.1 Features
- **ECG (7 features).**
  - Features: mean NN interval (`hrv_mean_nn`, equivalent to heart rate), SDNN, RMSSD, pNN50, LF power, HF power, LF/HF.
  - **Reliability at 60 s.** Only mean NN and RMSSD are reasonably supported at ultra-short windows. SDNN, pNN50, LF and LF/HF are not: the Task Force 1996 guidance (*Circulation* 93:1043) asks for about 2 minutes for LF and prefers 5 minutes. LF-based results are therefore not interpreted.
- **EDA (4 features):** tonic mean, tonic slope, phasic mean amplitude, phasic rate.
- **Coupling (2 features):** ECG–EDA peak cross-correlation and its lag (`coupling_lag_sec`, the H2a feature; Appendix C).
- **Extraction specifics** (`feature_extraction.py`, `stage1_pipeline.py`; neurokit2 0.2.13):
  - **ECG/HRV.**
    - R-peaks: `neurokit2.ecg_peaks` with its default detector, on the raw 700 Hz chest ECG. No separate cleaning step, no mains notch.
    - HRV: `nk.hrv_time` / `nk.hrv_frequency`, with LF 0.04–0.15 Hz and HF 0.15–0.40 Hz (Task Force bands).
    - Fallback: if neurokit2 fails, a 5–15 Hz band-pass peak detector computes time-domain HRV only. It was not triggered on this table (0% missing HRV).
  - **EDA.**
    - Tonic: 2nd-order Butterworth low-pass at 0.5 Hz on the chest EDA. This is not cvxEDA or another model-based decomposition. Tonic features are the mean and the linear slope of the tonic signal.
    - Phasic: the residual after removing the tonic component. Skin-conductance responses are found by peak detection (height ≥ 5% of the phasic SD, minimum spacing 0.8 s) and summarised as mean SCR amplitude and SCRs per minute.
  - **Coupling.**
    - Instantaneous heart rate from a 5–15 Hz band-pass R-peak detector.
    - Cross-correlated with z-scored EDA within ±15 s. Outputs: the peak |correlation| and its lag.
    - Undefined in 0.2% of windows.
  - **Quality control.** No SQI or artifact gating is implemented: every label-homogeneous window is used. **This is a limitation for any ambulatory claim.**

### 4.2 Missing-EDA simulation (paired)
- Every held-out window is scored twice: once with all features (complete condition) and once with all **six** EDA-dependent features (four EDA plus two coupling) set to missing (reduced condition).
- The coupling features need the EDA channel, so they are treated as part of the EDA modality throughout:
  - removed in the reduced condition;
  - included in the EDA branch of the gated network;
  - masked together with EDA in all missingness training.
- How each model receives missing EDA:

  | Model | Missing-EDA encoding |
  |---|---|
  | LightGBM | native NaN, zero-fill, or training-mean imputation, as named |
  | Gated network | closed EDA gate |
  | MLP | zeroed EDA plus a missingness indicator |

### 4.3 Models
| Name | Description |
|---|---|
| Complete-trained LightGBM, NaN at test (i) | trained on complete windows; missing EDA delivered as NaN. *This is the baseline behind earlier drafts' headline* (Appendix A) |
| Complete-trained LightGBM, zero-fill at test | as above with EDA = 0 |
| (ii) NaN-trained LightGBM | EDA set to NaN on a random 50% of training windows |
| (iii) ECG-only LightGBM | trained on ECG features only. Together with the complete-trained model this forms the obvious per-configuration deployment: full model when EDA is present, ECG-only model when absent |
| (iv) Mean-imputation + per-condition Platt | |
| HR-only | LightGBM on `hrv_mean_nn` alone |
| **Gated network** | two branches (ECG 7→16→16, EDA 6→16→16), summed, ReLU, linear head; EDA gate closed when EDA is missing; modality dropout p = 0.5 on the EDA gate during training; 801 parameters |
| Gated network, no dropout | trained on complete data only |
| **Non-gated MLP** | [ECG, EDA zero-imputed, missingness indicator] → 32 → 16 → 1, with the same 50% EDA masking during training; 1,025 parameters (28% more than the gated network; the capacity sweep below uses size-matched pairs) |
| HR-removed variants | gated network, (ii) and (iii) with `hrv_mean_nn` removed |
| HR-residualized gated network | `hrv_mean_nn` removed; every other ECG feature residualized on heart rate *within each participant* (subject mean retained). Label-free but transductive: a test participant's own windows define its regression |

- **Training (untuned defaults):** LightGBM with 50 trees, depth 3, learning rate 0.1; networks with full-batch Adam, learning rate 1e-2, 100 epochs, no early stopping.
- **Nested tuning (§5.2):** each outer training fold selects hyperparameters by inner subject-level 4-fold CV, minimising the mean of complete- and reduced-condition out-of-fold Brier.

  | Family | Grid | Configurations |
  |---|---|---|
  | LightGBM | trees {50, 200} × depth {2, 4} × learning rate {0.05, 0.1} × min child samples {10, 30} | 16 |
  | Networks | width {3 matched sizes} × epochs {100, 300} × learning rate {3e-3, 1e-2} × weight decay {0, 1e-3} | 24 |

- **Capacity sweep:** gated embedding width 8/16/32/64 (273/801/2,625/9,345 parameters) vs MLP hidden width 8/25/84/300 (281/808/2,637/9,333 parameters).

### 4.4 Calibration
Three outputs are reported for every model. All calibrators are fit on inner-CV out-of-fold predictions from the training participants only: never on test participants, never on in-sample predictions.
- **Raw:** the model's own probabilities.
- **Shared Platt:** one calibrator fit on complete-condition predictions and applied to both conditions. Baseline (iv) uses per-condition calibrators by design.
- **Per-condition Platt:** a separate calibrator for each sensor configuration. The reduced-condition calibrator is fit on the same out-of-fold windows with EDA removed.

Isotonic regression is not used, because about 12 training participants is too few.

### 4.5 Evaluation design
- **Primary inference:** 20× repeated subject-level 5-fold CV. Each repeat draws its own random partition of participants (20 distinct partitions, recorded in the output). There are 100 (repeat, fold) pairs, with about 12 training and 3 test participants each.
- **Pooled estimates and per-participant analysis:** leave-one-subject-out (LOSO; 15 folds), and a single random 5-fold partition as a check.
- **Metrics:**
  - Brier score (primary);
  - log-loss;
  - AUROC pooled across held-out participants;
  - **within-subject AUROC** (mean of per-participant AUROCs, which removes between-participant offsets);
  - equal-mass ECE_SWEEP (Roelofs et al. 2022: largest bin count with monotone bin frequencies; no bias subtraction, no clipping);
  - calibration intercept and slope;
  - **selective prediction:** area under the risk–coverage curve (AURC; Geifman, Uziel & El-Yaniv, ICLR 2019) and error at 80% coverage. Windows are retained in order of confidence max(p, 1 − p), and risk is the error rate at the 0.5 threshold. Tied confidences, which tree models produce, are averaged over orderings.

  Pooled estimates carry 95% CIs from resampling participants with replacement (500 resamples).
- **Inference:**
  - Bouckaert–Frank corrected t over the 100 fold-level differences: SE² = (1/100 + n_test/n_train)·var(d), df = 99. Report the mean difference with its corrected 95% CI.
  - An exact sign-flip test over 2^15 patterns on per-participant Brier differences, reported as descriptive only, because participants in the same fold share a model.
- **Software and reproducibility:**
  - The feature table is hash-checked by every analysis script.
  - Package versions are recorded in `requirements.txt` (Python 3.13.13, CPU).
  - Fixed seeds per (repeat, fold).
  - 22 automated tests, including a regression test for every implementation error in Appendix A.

### 4.6 Decision rule (provenance on page 1)
> If the gated network (with dropout) does not beat (ii) NaN-trained LightGBM and (iii) ECG-only LightGBM on reduced-condition Brier **and** ECE **and** AUROC, the architectural claim is not supported, and the paper is reported as an evaluation paper.

- **Inferential gap.** The rule as written compares point estimates and has no inferential criterion. We therefore report the CIs alongside it and do not claim equivalence, which was not pre-specified.
- **Endpoint substitution.** The rule's ECE endpoint was originally a "debiased ECE" whose implementation was wrong (Appendix A). It is evaluated here with ECE_SWEEP.

### 4.7 Which analyses were pre-specified
Every analysis in §5 falls into one of four classes:
- **Pre-specified:** written in the analysis outline before the reduced-sensor baselines were run on 2026-10-01. Not independently timestamped (see page 1).
- **Correction:** a fix to an implementation error found after the first evaluation of the decision rule (Appendix A).
- **Requested:** asked for by a reviewer before it was run, but specified after the first results were known.
- **Exploratory:** added by us after seeing results.

Only pre-specified analyses can confirm anything. Everything else is reported as supporting or exploratory evidence, and no claim rests on an exploratory analysis alone.

| Analysis | Status | Note |
|---|---|---|
| Baselines (i)–(iv), HR-only, HR-removed and HR-residualized gated network | Pre-specified | HR variants requested in review round 2 |
| Decision rule (gated vs (ii), (iii) on Brier, ECE, AUROC) | Pre-specified | Point-estimate rule with no inferential margin |
| 20× repeated subject-level CV with Bouckaert–Frank inference; sign-flip test | Pre-specified | Partitioning implementation corrected (Appendix A #2) |
| Shared Platt calibration, fit on training participants only | Pre-specified | Implementation corrected (Appendix A #5) |
| ECE_SWEEP in place of the mis-implemented "debiased ECE" | Correction | Appendix A #3 |
| Paired scoring of complete and reduced conditions | Correction | Appendix A #4 |
| Fixed HR-residualization (HR removed, within participant) | Correction | Appendix A #6 |
| Frozen feature table and re-run of everything | Correction | Appendix A #7 |
| Non-gated MLP with the same masking | Requested | Round-3 review M2 |
| Capacity sweep (4 matched sizes) | Exploratory | Robustness check of the MLP comparison |
| Nested hyperparameter tuning | Requested | Round-3 review M2 |
| HR-removed LightGBM (ii), (iii) | Requested | Round-3 review C7 |
| LOSO, within-subject AUROC, per-participant / HR-response analysis (Figure 1) | Requested / exploratory | Per-participant analysis requested (C6); the HR-response correlation is exploratory |
| Per-condition Platt for every model | Exploratory | Added after the shared-Platt results showed over-correction |
| Risk–coverage / AURC (Figure 2) | Exploratory | Planned in earlier drafts' Discussion; computed after main results |
| H2a subject-level reanalysis and TOST | Requested | No equivalence bound; feature unreliable at 60 s (Appendix C) |
| Interpretability (Appendix E) | Exploratory | Descriptive only |

**Consequence for the decision rule.** The rule's verdict changes across calibration modes and CV schemes (§5.1). The per-condition calibration mode under which it does not fire is *exploratory*. We therefore make no claim in either direction from the rule. The paper's conclusion ("indistinguishable") rests on the confidence intervals of the pre-specified comparisons in Table 2, which include zero under every calibration mode.

## 5. Results

*Every result is within WESAD (stress vs baseline in healthy young adults). None of it concerns panic or cardiac events.*

### 5.1 Decision-rule outcome and primary comparison

**Table 1. Reduced condition (EDA missing), 20× repeated subject-level 5-fold CV, raw probabilities: mean across 100 folds (SD).**

| Model | Brier | AUROC | ECE_SWEEP* | Log-loss |
|---|---|---|---|---|
| Gated network + dropout | 0.148 (0.075) | 0.892 (0.075) | 0.126 | 0.555 |
| Non-gated MLP, same masking | 0.146 (0.075) | 0.892 (0.075) | 0.125 | 0.580 |
| (ii) NaN-trained LightGBM | 0.146 (0.084) | 0.882 (0.100) | 0.123 | 0.498 |
| (iii) ECG-only LightGBM | 0.146 (0.083) | 0.882 (0.103) | 0.123 | 0.523 |
| (iv) Mean-imputation + per-condition Platt | 0.161 (0.104) | 0.886 (0.101) | 0.141 | 0.567 |
| HR-only | 0.179 (0.087) | 0.854 (0.100) | 0.138 | 0.606 |
| Gated network, no dropout | 0.186 (0.094) | 0.877 (0.093) | 0.170 | 0.980 |

\*Per-fold ECE on about 87 windows is biased upward relative to the pooled estimates in Table 3. It is used here only for the rule's comparison.

**Table 2. Paired differences (A − B), reduced condition, Bouckaert–Frank 95% CI.** Negative ΔBrier means A is better; positive ΔAUROC means A is better.

| Comparison | ΔBrier [95% CI] | ΔAUROC [95% CI] | Participants where A is better |
|---|---|---|---|
| Gated vs (ii) | +0.002 [−0.031, +0.035] | +0.010 [−0.058, +0.077] | 7/15 |
| Gated vs (iii) | +0.002 [−0.033, +0.037] | +0.010 [−0.062, +0.082] | 6/15 |
| Gated vs (iv) | −0.013 [−0.073, +0.047] | +0.006 [−0.065, +0.076] | 6/15 |
| Gated vs HR-only | −0.030 [−0.077, +0.016] | +0.038 [−0.039, +0.115] | 8/15 |
| Gated vs no dropout | −0.038 [−0.089, +0.014] | +0.015 [−0.036, +0.065] | 11/15 |
| **Gated vs non-gated MLP** | **+0.002 [−0.016, +0.020]** | **−0.000 [−0.025, +0.024]** | 7/15 |
| MLP vs (ii) | +0.000 [−0.031, +0.032] | +0.010 [−0.054, +0.074] | 7/15 |

**Rule outcome: the rule fires with raw and shared-Platt outputs, but on near-ties, and its verdict reverses under per-condition calibration.**
- **Raw probabilities.** The gated network trails (ii) and (iii) by 0.002 on Brier and 0.003 on ECE, and leads by 0.010 on AUROC. None of these differences is distinguishable from zero.
- **Platt-recalibrated probabilities.** The gated network beats (iii) on all three endpoints and still trails (ii) on Brier and ECE.
- **Per-condition Platt.** Under 20× repeated CV the rule **does not fire**: the gated network edges both baselines on all three endpoints, again by near-ties (ΔBrier vs (ii) −0.007 [−0.040, +0.025]).
- **The verdict depends on calibration and CV scheme:**

  | Calibration | 20× repeated CV | LOSO pooled | Single 5-fold |
  |---|---|---|---|
  | Raw | fires | fires | fires |
  | Shared Platt | fires | fires | fires |
  | Per-condition Platt | does not fire | does not fire | fires |

  A point-estimate rule with no inferential margin flips on noise when the underlying differences are this small. The flip is the clearest evidence that the models are indistinguishable.
- **Pooled LOSO and single 5-fold estimates (Table 3).** These agree that the CIs overlap completely.
- **The first evaluation, pre-fix.** It reported the gated network as worse on all three endpoints. That came from the implementation errors in Appendix A and is superseded.
- **The defensible statement:** *against baselines trained or built for missing EDA, the gated network shows no advantage. The data are compatible with differences of up to about ±0.037 Brier in either direction. Equivalence was not tested.*

**Table 3. Pooled out-of-fold estimates (LOSO), raw, with participant-bootstrap 95% CIs.**

| Model | Complete Brier | Reduced Brier [CI] | Reduced AUROC [CI] | Reduced within-subject AUROC | Reduced ECE_SWEEP [CI] | Reduced calibration slope / intercept |
|---|---|---|---|---|---|---|
| (i) Complete-trained, NaN at test | 0.070 | 0.296 [0.271, 0.316] | 0.852 [0.752, 0.952] | 0.918 | 0.302 [0.268, 0.325] | 0.96 / **+3.10** |
| (ii) NaN-trained | 0.090 | 0.146 [0.089, 0.224] | 0.860 [0.747, 0.945] | 0.870 | 0.057 [0.031, 0.154] | 0.59 / −0.28 |
| (iii) ECG-only | 0.142 | 0.142 [0.084, 0.223] | 0.869 [0.759, 0.952] | 0.899 | 0.066 [0.030, 0.165] | 0.55 / −0.38 |
| (iv) Mean-imputation + per-condition Platt | 0.070 | 0.155 [0.072, 0.244] | 0.868 [0.771, 0.964] | 0.933 | 0.120 [0.046, 0.211] | 0.50 / −0.46 |
| HR-only | 0.178 | 0.178 [0.110, 0.261] | 0.820 [0.708, 0.946] | 0.867 | 0.120 [0.043, 0.214] | 0.44 / −0.40 |
| Gated + dropout | 0.122 | 0.146 [0.093, 0.208] | 0.873 [0.793, 0.941] | 0.903 | 0.087 [0.046, 0.166] | 0.43 / −0.16 |
| Gated, no dropout | 0.128 | 0.196 [0.132, 0.261] | 0.839 [0.764, 0.916] | 0.888 | 0.158 [0.094, 0.211] | 0.23 / +0.02 |
| Non-gated MLP | 0.131 | 0.145 [0.090, 0.210] | 0.878 [0.795, 0.947] | 0.899 | 0.100 [0.048, 0.173] | 0.42 / −0.25 |

The single random 5-fold partition gives the same picture (`results/baselines.json`, scheme `kfold5`).

**Deployment view.** A single model has to serve both conditions. The per-configuration system scores 0.070 when EDA is present and 0.142 when it is missing: it uses the complete-trained LightGBM with EDA (same model as row (i)) and switches to (iii) when EDA drops out. The single gated network scores 0.122 and 0.146. **Switching between per-configuration models matches or beats the single gated network in both conditions.**

### 5.2 Gating vs training with missingness
- **Gated vs MLP, untuned.** At default settings the gated network and a plain MLP trained with the same EDA masking are indistinguishable. ΔBrier is +0.002 [−0.016, +0.020] and ΔAUROC is −0.000 [−0.025, +0.024]: among the narrowest intervals in the study (only the larger capacity-sweep pairs are narrower).
- **Capacity sweep** (20× repeated CV, raw; `results/repeated_cv_capacity.json`):

  | Gated width (parameters) | MLP width (parameters) | ΔBrier gated − MLP [95% CI] | ΔAUROC [95% CI] |
  |---|---|---|---|
  | 8 (273) | 8 (281) | −0.001 [−0.024, +0.022] | +0.002 [−0.033, +0.036] |
  | 16 (801) | 25 (808) | +0.003 [−0.015, +0.021] | −0.002 [−0.026, +0.022] |
  | 32 (2,625) | 84 (2,637) | +0.002 [−0.014, +0.017] | −0.004 [−0.029, +0.021] |
  | 64 (9,345) | 300 (9,333) | +0.005 [−0.012, +0.023] | −0.005 [−0.026, +0.016] |

- **Nested tuning** (20× repeated CV, 20 distinct partitions, reduced condition, raw; `results/repeated_cv_tuned.json`):

  | Tuned model | Brier | AUROC | ECE_SWEEP |
  |---|---|---|---|
  | Gated network + dropout | 0.151 | 0.883 | 0.129 |
  | Non-gated MLP | 0.146 | 0.889 | 0.128 |
  | (ii) NaN-trained LightGBM | 0.148 | 0.879 | 0.128 |
  | (iii) ECG-only LightGBM | 0.148 | 0.883 | 0.135 |

  | Comparison (tuned) | ΔBrier [95% CI] | ΔAUROC [95% CI] |
  |---|---|---|
  | Gated vs (ii) | +0.004 [−0.028, +0.036] | +0.003 [−0.081, +0.088] |
  | Gated vs (iii) | +0.004 [−0.026, +0.034] | −0.000 [−0.076, +0.075] |
  | Gated vs MLP | +0.005 [−0.019, +0.029] | −0.006 [−0.058, +0.046] |
  | MLP vs (ii) | −0.001 [−0.032, +0.029] | +0.009 [−0.067, +0.086] |

  - **Rule outcome.** The decision rule fires again under tuning, with both raw and Platt outputs.
  - **Effect of tuning.** Tuning moves no model by more than 0.005 Brier.
  - **Configurations chosen.** For the gated network, embedding 16, 100 epochs, learning rate 3e-3, weight decay 1e-3 (25/100 folds). For the MLP, the largest width (84), with weight decay 1e-3. For the LightGBM models, shallow trees (depth 2–4, 50 trees, learning rate 0.05).
- **Training with missingness is the active ingredient.** Models never trained with missing EDA degrade sharply when EDA is removed: the complete-trained LightGBM goes to Brier 0.296 and the gated network without dropout to 0.196. Every model trained or built for missingness lands at 0.142–0.155, whether by masking (ii), by gated dropout, by MLP masking, by per-configuration training (iii), or by per-condition recalibration (iv).
- **Dropout within the gated network.** Dropout improves reduced-condition Brier by −0.038 [−0.089, +0.014]. The interval includes zero; the improvement appears in 11 of 15 participants.

### 5.3 Heart-rate dependence

**Table 4. LOSO, reduced condition, raw.**

| Model | Reduced AUROC [95% CI] | Within-subject AUROC |
|---|---|---|
| Gated + dropout | 0.873 [0.793, 0.941] | 0.903 |
| HR-only | 0.820 [0.708, 0.946] | 0.867 |
| Gated, HR removed | 0.692 [0.552, 0.814] | 0.662 |
| Gated, HR-residualized (HR removed + within-participant residualization) | 0.600 [0.505, 0.702] | 0.616 |
| (ii) NaN-trained, HR removed | 0.681 [0.526, 0.829] | 0.668 |
| (iii) ECG-only, HR removed | 0.674 [0.529, 0.818] | 0.674 |

- **A single feature does most of the work.** Heart rate alone reaches 0.82 AUROC.
- **Removing heart rate costs about half the above-chance AUROC in both model families** (0.87 → 0.67–0.69).
- **Removing within-participant heart-rate covariation leaves little.** HRV features that remain after it is removed carry little signal (0.60). Most of what HRV adds without EDA is heart rate expressed in other units.

### 5.4 Participant-level heterogeneity
- **Figure 1** (`results/fig_per_subject_forest.png`) shows, for each participant, the within-subject AUROC under LOSO with EDA missing, next to that participant's heart-rate response (stress-minus-baseline drop in mean NN interval: 12–448 ms).
- **Performance tracks heart-rate response.** Spearman ρ between response magnitude and AUROC (n = 15):

  | Model | ρ |
  |---|---|
  | Gated network | 0.88 |
  | NaN-trained LightGBM | 0.76 |
  | HR-only | 0.92 |

  The 5-fold check gives ρ = 0.67–0.91 across models.
- **Strong responders are near-perfect; weak responders vary.**
  - The 7 participants with a drop of 239 ms or more score at least 0.90 in every model that has heart rate (0.99–1.00 for the networks and HR-only).
  - With heart rate removed, the worst of those same 7 strong responders falls to 0.36–0.48 in every model.
  - In the 4 weakest responders (S15, S9, S10, S2; 12–51 ms), performance is erratic and model-dependent. For S10, the NaN-trained LightGBM's ranking is inverted (AUROC 0.16).
- **Pooled AUROC hides some of this.** It mixes between-participant offsets with within-participant ranking, which is why within-subject AUROC is reported throughout.
- **Clinical reading.** A detector that relies on heart rate when EDA is missing fails for exactly the people whose heart rate does not move. At the intended clinical boundary, tachycardia is shared by panic, SVT, AF with rapid ventricular response and ischaemia; heart-rate reliance is least informative there.

### 5.5 Calibration under missing EDA
- **The complete-trained LightGBM fails in calibration-in-the-large.** With EDA missing, its reduced-condition intercept is +3.10 while its slope stays near 1 (0.96). Discrimination is largely kept (AUROC 0.85), but the model systematically under-predicts stress. Missing EDA is routed down the "no EDA response" branches, so the model reads the absence of the sensor as the absence of arousal. That is the mechanism behind the earlier drafts' headline.
- **Every better model is over-confident without EDA.** Calibration slopes are 0.42–0.59.
- **The shared calibrator over-corrects the networks.** Fit on complete-condition predictions and applied to both conditions, it pushes their reduced-condition slopes above 1 (1.24–2.31 across schemes) without improving Brier.
- **Per-condition calibration fixes this** (LOSO, pooled, reduced condition):

  | Model | Raw: slope / intercept / ECE | Per-condition Platt: slope / intercept / ECE | Brier, raw → per-condition |
  |---|---|---|---|
  | Gated + dropout | 0.43 / −0.16 / 0.087 | 0.92 / +0.05 / 0.045 | 0.146 → 0.147 |
  | Non-gated MLP | 0.42 / −0.25 / 0.100 | 1.00 / +0.04 / 0.049 | 0.145 → 0.144 |
  | (ii) NaN-trained | 0.59 / −0.28 / 0.057 | 0.82 / −0.05 / 0.062 | 0.146 → 0.153 |
  | **(i) Complete-trained** | 0.96 / **+3.10** / 0.302 | 1.13 / **+0.09** / 0.112 | **0.296 → 0.163** |

  - For the networks, a per-condition calibrator fixes the over-confidence at no cost in Brier.
  - **Recalibration alone mostly rescues the complete-trained LightGBM:** Brier 0.296 → 0.163 and intercept +3.10 → +0.09. Its failure under missing EDA was almost entirely calibration-in-the-large, which a calibrator fit on reduced-condition predictions repairs.
  - This strengthens the principal finding. What matters is that *something* in the pipeline has seen missing EDA (the training, the calibrator, or a per-configuration model), not the architecture.
  - *Per-fold calibration slopes (about 29 windows per held-out participant set) are unstable and are not reported; slopes are given for pooled predictions only.*
- **ECE estimator lesson.** The originally implemented "debiased ECE" took the minimum over six bin counts of a zero-clipped estimate. Nearly half of its reduced-condition per-fold values (18/40) were exactly zero, and it ranked the HR-removed gated network as the best, although by conventional equal-width ECE it was the second-worst (0.229; only the complete-trained LightGBM was worse at 0.307; the others 0.18–0.20). With ECE_SWEEP, low-discrimination models can still post competitive ECE: under LOSO the HR-removed NaN-trained LightGBM has the lowest ECE_SWEEP (0.042) at AUROC 0.68, with slope 0.42. **ECE without discrimination and slope is uninformative, and at about 40–90 windows per evaluation set its estimator choice can reverse a ranking.**

### 5.6 Selective prediction (abstention)
- **Figure 2** (`results/fig_risk_coverage_reduced.png`) shows risk–coverage curves with EDA missing (LOSO, pooled, raw).
- **The pooled curves suggested a difference that repeated CV does not support.** Pooled area under the risk–coverage curve (AURC, lower is better) looked better for the networks (gated 0.085, MLP 0.082) than for (ii) (0.102) and (iii) (0.162). The ECG-only LightGBM's most confident 10% of windows were wrong 23% of the time. Under 20× repeated CV, none of these differences is distinguishable from zero:

  | Comparison | ΔAURC [95% CI], raw | ΔAURC, per-condition Platt |
  |---|---|---|
  | Gated vs (ii) | −0.012 [−0.078, +0.054] | −0.009 [−0.063, +0.045] |
  | Gated vs (iii) | −0.026 [−0.113, +0.061] | −0.010 [−0.064, +0.045] |
  | Gated vs MLP | −0.000 [−0.024, +0.023] | +0.001 [−0.021, +0.023] |

- **Reading.** Abstaining on low-confidence windows lowers error for every missingness-aware model: from about 0.20 at full coverage to about 0.10 at 60% coverage. There is no evidence that any model abstains better than another.
- **Clinical caution.** The weak heart-rate responders (§5.4) contribute confident errors. Abstention based on model confidence will not catch them.
- **Methods note.** The pooled-vs-repeated contrast is another instance of the small-n lesson: a single pooled split produced an apparent 2× difference that disappeared under repeated partitions.

### 5.7 How the earlier headline arose (summary; details in Appendix A)
- **The earlier headline** was an ECE-ratio sequence of 4.23× → 2.21× → 1.61× (LightGBM → gated network → gated network + dropout). It compared the gated network against the complete-trained LightGBM. On the frozen table the same script gives **4.50× → 2.02× → 1.15×** (Appendix B).
- **Why the ratio flattered the gated network.** It rewards whichever model is worse-calibrated on complete data, and the comparator was never trained with missing EDA.
- **What the fair baselines show instead.** Once missingness is handled fairly (§5.1–5.2), the gated network's apparent advantage disappears.

## 6. Discussion
- **Principal finding.** For wearable stress detection with a missing EDA sensor, what matters is whether the model was trained or built for missingness, not whether it has a modality gate.
  - A gated network, a plain MLP with the same masking, a NaN-trained LightGBM and an ECG-only LightGBM are statistically indistinguishable.
  - A per-configuration system matches or beats a single gated network in both sensor conditions.
  - Prior work's gating claims should be read against this: the comparator matters more than the architecture.
- **Mechanism.** Without EDA, all models fall back on heart rate, and per-participant performance tracks the size of each participant's heart-rate response.
  - In WESAD, most participants have large TSST heart-rate responses, which makes the task look easier than it is for the minority who do not.
  - For the motivating clinical problem this is the central caution. Heart-rate fallback is least informative where panic and cardiac tachyarrhythmias overlap.
- **Calibration.** Two failure modes appear, and one protocol issue:
  - Models never trained with missingness fail in calibration-in-the-large: they read missing EDA as "calm".
  - All models are over-confident without EDA.
  - Calibration has to be fit per sensor configuration. Done that way, the gated network and the MLP are well calibrated without EDA, and even the complete-trained LightGBM recovers most of its Brier.
  - Calibration matters for safety only through abstention (Lechuga López et al.). Abstention halves the error of every missingness-aware model at 60% coverage, but no model abstains detectably better than another (§5.6). Confident errors concentrate in weak heart-rate responders.
- **Methodological lessons for small-n wearable studies** (Appendix A gives each error, its effect, and its regression test):
  1. Repeated CV must be shown to produce distinct participant partitions.
  2. Missing-sensor conditions should be scored on the same windows.
  3. An ECE estimator's bias and ranking behaviour must be checked at the actual sample size.
  4. Feature tables should be frozen by hash.
  5. Baselines should be given the same missingness training as the proposed model.
- **What this does not show:**
  - panic-vs-cardiac differentiation;
  - performance in patients or in ambulatory, non-laboratory signals;
  - performance on wrist-only (E4) sensor configurations (data are available; Appendix G).

## 7. Limitations
- **Sample size.** 15 participants; each test fold holds about 3. Every between-model CI in Table 2 spans roughly ±0.03 Brier; differences smaller than that cannot be resolved.
- **Proxy task.** Laboratory stress in healthy young adults (TSST), not panic, with a fixed condition order (§3).
- **Simulated missingness.** EDA is removed completely and synthetically. Real sensor loss is partial, intermittent and correlated with motion.
- **Feature-level, ultra-short windows.** The features are hand-crafted, and HRV beyond mean NN and RMSSD is unreliable at 60 s. Results may differ for end-to-end signal models.
- **Limited tuning.** Nested tuning covers small grids only (16 LightGBM and 24 network configurations; §4.3), and the networks are small (273–9,345 parameters). Tuning changed no model by more than 0.005 Brier.
- **No external validation.** Everything is within WESAD.
- **Transductive residualization.** The HR-residualized analysis uses each test participant's own unlabelled windows.
- **Analytic flexibility.** Several analyses were added after the first results were known (§4.7). They are labelled as exploratory, and no conclusion rests on them alone.
- **Implementation history.** The pipeline contained implementation errors that were found and fixed after the decision rule's first evaluation (Appendix A). The rule's second evaluation therefore uses analysis code finalised after its first outcome was known; the rule's text and endpoints did not change apart from the ECE substitution.

## 8. Conclusion
- **What we found.** In this laboratory proxy setting, a modality-gated network trained with modality dropout does not outperform reduced-sensor or missingness-trained LightGBM baselines, or a plain MLP trained with the same simulated sensor loss, when EDA is missing.
  - This holds untuned and tuned, and across four matched model sizes.
  - The pre-stated decision rule fires with raw and shared-calibrated outputs on near-ties, and does not fire under per-condition calibration. The verdict is not robust because the models are indistinguishable.
  - The earlier reported gating advantage came from comparing against a model never trained with missing EDA.
- **Mechanism.** Without EDA, every model relies on heart rate, and participants whose heart rate does not respond to the stressor are where models fail.
- **Next step.** Real cardiac-vs-panic differentiation needs a clinically labelled cohort with paired events. That is the next step, not something proxy data can approximate.

## 9. Reporting, reproducibility, disclosures (TRIPOD+AI)
- **Reporting guideline.** TRIPOD+AI (BMJ 2024;385:e078378) applies: this is model development with internal validation on proxy data. Complete the 27-item checklist in the appendix. Items still to finalise:
  - per-participant window counts and class balance (`results/per_subject_loso.csv` has them);
  - feature-extraction specifics (now in §4.1);
  - hyperparameters and the tuning procedure (§4.3);
  - compute: CPU only, with runtimes from the logs.
- **Reproducibility.**
  - Frozen feature table hash: `results/feature_table_manifest.json`.
  - Package versions: `requirements.txt`.
  - Seeds: fixed per (repeat, fold).
  - Scripts: `eval_core.py`, `run_baselines.py`, `run_repeated_cv.py`, `make_figures.py`.
  - Tests: `tests/test_eval_core.py`.
  - Decision history: `results/decision_timeline.md`.
  - **When git is initialised, tag the current state as what it is** (e.g. `post-rerun-2026-10-01`). It is not a pre-registration.
- **Ethics.** Public, de-identified datasets (WESAD; MIT-BIH for the companion note). If the venue requires it, state that this is non-human-subjects research.
- **Conflict of interest.** See page 1. The paper motivates a commercial application, so the disclosure is on page 1, not in the acknowledgements.
- **Leaky-split and superseded numbers.**
  - The record-level leak (Appendix A, item 1) and every superseded headline are listed with their corrected values in Appendix A.
  - If any superseded number appeared publicly (preprint, website, pitch deck, grant, investor material), state it alongside the correction.

## 10. Title, abstract and scope

### 10.1 Title options (all answer the question the paper asks)
- *Missingness-Aware Training, Not Modality Gating, Determines Performance When EDA Drops Out: A Subject-Level Evaluation on WESAD* (current working title)
- *Modality Gating Adds Nothing Over Missingness-Aware Training When Wearable EDA Drops Out: A Subject-Level Evaluation on WESAD*
- *A Gating Advantage Under Missing Wearable EDA Was a Baseline Artefact: A Subject-Level Evaluation on WESAD*

Avoid "disappears": the CIs allow small differences in either direction (§5.1).

### 10.2 Abstract (draft)
> **Background.** Wearable systems meant to distinguish acute psychological from cardiac events will face missing sensors, and calibrated uncertainty under missingness is needed for safe abstention. Modality gating with modality dropout has been proposed for this, usually evaluated against baselines never trained with missing inputs.
>
> **Methods.** On WESAD (15 participants; stress vs baseline; 434 sixty-second windows), we scored every held-out window with and without its six EDA-dependent features. We compared a modality-gated network with modality dropout against:
> - a similar-size non-gated MLP (1,025 vs 801 parameters) trained with the same masking, plus a capacity sweep at four exactly matched sizes;
> - LightGBM trained with EDA masking, ECG-only LightGBM, mean-imputation with recalibration, and a heart-rate-only model;
> - nested-tuned versions of the main models.
>
> We used 20× repeated subject-level 5-fold CV with 20 distinct participant partitions and Bouckaert–Frank corrected intervals, plus leave-one-subject-out for per-participant analysis. A decision rule written before the baselines were run (not independently timestamped) required the gated network to beat the masked and ECG-only baselines on reduced-condition Brier, ECE and AUROC.
>
> **Results.** With EDA missing:
> - The gated network was indistinguishable from NaN-trained LightGBM (ΔBrier +0.002, 95% CI −0.031 to +0.035), from ECG-only LightGBM (+0.002, −0.033 to +0.037), and from the non-gated MLP (+0.002, −0.016 to +0.020; similar at all four sizes and after nested tuning, +0.005, −0.019 to +0.029).
> - The decision rule fired on near-ties with raw outputs. It did not fire under per-condition recalibration, under which all models were well calibrated.
> - The gated network's earlier reported advantage came from comparison with a LightGBM never trained with missing EDA. That model under-predicted stress when EDA was absent (Brier 0.296; calibration intercept +3.1).
> - Removing heart rate reduced AUROC from about 0.87 to 0.67–0.69 in both model families.
> - Per-participant AUROC tracked each participant's heart-rate response to the stressor (Spearman ρ 0.76–0.92).
> - All models were over-confident without EDA (calibration slope 0.42–0.59). Per-condition recalibration corrected this, and alone reduced the complete-trained LightGBM's Brier from 0.296 to 0.163.
> - Abstaining on low-confidence windows reduced error for every missingness-aware model, with no detectable difference between models.
>
> **Interpretation.** Training with simulated sensor loss, not modality gating, determined performance under missing EDA, and discrimination depended on heart rate. These results come from 15 healthy laboratory participants and do not evaluate differentiation of panic attacks from cardiac events.
>
> **Disclosure.** The author is a co-founder of E14 Health.

### 10.3 Scope
- **This preprint:** the missing-EDA evaluation (§§4–5).
- **Moved to appendices:**
  - the H2a mechanism test (Appendix C);
  - the data-gap search (Appendix D; H-B2 is reported there as historical and unverifiable, recommended for omission);
  - interpretability (Appendix E; to be regenerated on the frozen table).
- **Cut:**
  - **H-C1 (device tiers).** Its tier B was identical to the reduced condition, it used the complete-trained zero-fill LightGBM, and its script no longer runs on the current code base. The real chest-vs-wrist tier experiment is future work (Appendix G).
  - **The dataset-identity audit (WESAD × MIT-BIH)** becomes a separate short note after the full, non-`--fast` audit run (Appendix F).

---

## Appendix A. Implementation errors found, their effect, and the regression test for each

| # | Error | Effect on reported results | Fix / regression test |
|---|---|---|---|
| 1 | Record-level train/test split put the same participant in train and test (found before 2026-09-20) | Inflated all early H-B1 numbers; superseded by subject-level CV. **[List any leaked number that appeared publicly, with its corrected value]** | Subject-level splits everywhere; `test_partition_covers_each_subject_once` |
| 2 | "20× repeated CV" shuffled rows before `GroupKFold`, which ignores row order: 1 distinct participant partition in 20 | First rule evaluation's Bouckaert–Frank p-values (0.49–0.56) were invalid | Explicit participant permutation per repeat; `test_repeated_partitions_differ` |
| 3 | "Debiased ECE" = minimum over 6 bin counts of `max(raw − ad-hoc bias, 0)` | 18/40 reduced per-fold values exactly 0; ranked the worst-calibrated model best; the first rule evaluation's ECE comparison was uninformative | ECE_SWEEP (Roelofs); `test_ece_sweep_not_clipped_and_small_when_calibrated` |
| 4 | Complete and reduced conditions scored on disjoint random halves of each test fold | Extra sampling noise; ratio and ΔECE compared different windows | Paired scoring; `test_paired_tiers_same_windows` |
| 5 | Platt protocol described in Methods but applied to one model only, and fit on in-sample predictions | Methods text did not match the code | Inner-CV out-of-fold Platt for every model; `test_fit_predict_never_trains_on_test` |
| 6 | HR-residualized variant kept `hrv_mean_nn` as an input and residualized across pooled participants | Its AUROC (0.878) was read as "HRV beyond HR"; after the fix it is 0.600 | HR removed and within-participant residualization; `test_hr_resid_removes_hr` |
| 7 | Feature cache rebuilt on 2026-09-30 (445 → 434 WESAD windows) while 2026-09-20 numbers were kept; locked-comparison script loaded MIT-BIH rows | Earlier ECE-ratio table could not be reproduced | Hash-checked frozen table; every script refuses a different table |
| 8 | `requirements.txt` pins did not match the installed environment | Reproducibility | Pins updated to the versions that ran |
| 9 | Earlier drafts called the headline baseline "zero-fill"; the locked script delivered NaN | Label only (both encodings give Brier ≈ 0.30 under missing EDA) | Both encodings reported separately (Table 3) |

**First vs second evaluation of the decision rule:**

| | First (pre-fix, `results/archive/*_prefix.*`) | Second (post-fix) |
|---|---|---|
| Gated vs (ii), Brier | +0.014 (B–F p 0.56; invalid, error 2) | +0.002 [−0.031, +0.035] |
| Gated vs (iii), Brier | +0.015 (p 0.49; invalid) | +0.002 [−0.033, +0.037] |
| Gated vs (ii)/(iii), AUROC | gated lower | gated higher by 0.010 (CI spans 0) |
| Rule | fires: "worse on all three" | fires on near-ties |

## Appendix B. Legacy ECE-ratio analyses (regenerated on the frozen table; `results/locked_comparison_rerun.log`, `results/locked_per_modality_rerun.log`)

*These compare against the complete-trained LightGBM and are superseded as evidence about gating (§5.7). They use the old protocol: a single `GroupKFold` partition, unpaired 50% masking, and equal-width 10-bin ECE.*

| Condition | Ratio, mean of per-fold ratios | 95% CI | Earlier draft (445-window table) |
|---|---|---|---|
| Complete-trained LightGBM (NaN at test) | 4.50× | [0.05, 8.96] | 4.23× |
| Gated, no dropout | 2.02× | [0.57, 3.46] | 2.21× |
| Gated + dropout p = 0.5 | 1.15× | [0.67, 1.63] | 1.61× |

- **Dropout vs no dropout.** Dropout has the lower ratio in 4 of 5 folds (paired t p = 0.095; Wilcoxon W = 1, p = 0.125). The earlier "5/5 folds, W = 0, d_z = −1.19" does not reproduce.
- **Dropout-rate sweep.** p = 0.2 / 0.4 / 0.5 / 0.6 gives 1.32× / 1.35× / 1.15× / 1.13×, with all CIs overlapping: no dose–response.
- **Ablation: which modality is dropped in training.** Dropping EDA in training (the modality missing at test) gives 1.15× [0.67, 1.63]; dropping ECG instead (the wrong modality) gives 2.48× [0.26, 4.70]. The right-modality variant wins in 4/5 folds (p = 0.18). This is a sanity check: training must simulate the sensor that actually goes missing.
- **Why the ratio is not used as evidence.** It rewards whichever model is worse-calibrated on complete data, and it is unstable when complete-condition ECE is small.

## Appendix C. H2a mechanism test (EDA–HR coupling lag), subject-level reanalysis
- **Earlier report.** Window-level, n = 445 (old table): d = 0.06, p = 0.94. This was pseudoreplicated: windows within a participant are correlated.
- **Subject-level analysis on the frozen table** (`results/h2a_subject_level.json`):
  - Per-participant mean `coupling_lag_sec` under stress minus baseline: mean −1.13 s, 95% CI [−2.76, +0.49].
  - d_z = −0.39, 95% CI [−0.91, +0.15]; t(14) = −1.49, p = 0.16; Wilcoxon p = 0.39.
  - 8 of 15 participants show a longer lag under stress.
- **Equivalence.** No smallest effect of interest is specified. Choosing one now, after seeing the result, would be post hoc. The 90% CI of the mean change is [−2.47, +0.20] s. Equivalence would therefore be declared only for bounds wider than ±2.47 s (d_z > 0.84), and not below that.
- **Measurement reliability.** Within a single resting participant, the lag varies from window to window with SD 8.3 s (median across participants; range 5.9–10.6 s). That is comparable to the ±15 s search range. Across all windows the lag is centred near zero (median −0.75 s, IQR −6.95 to +3.65 s). A single 60-s cross-correlation peak is mostly noise.
- **Reading.** H2a is **not testable with this feature at this window length**. The result is uninformative, neither a null nor an effect. The earlier window-level "null" (d = 0.06) should not be cited. A real test would need longer windows (e.g. 5 min) or lag estimates time-locked to individual skin-conductance responses.

## Appendix D. H-B2 (historical, unverifiable) and the data-gap search
- **H-B2 is reported as historical and is not part of this paper's evidence.**
  - Earlier drafts reported that intra-WESAD contrastive ECG↔EDA pretraining gave no measurable downstream benefit at about 15 participants.
  - The pretraining code is not present in the project, so the result cannot be reproduced on the frozen feature table.
  - It is therefore **removed from all claims** (Contributions, Results, Discussion, Abstract). It is mentioned here only so that readers of earlier drafts can see where it went.
  - **Recommendation:** omit it entirely from the posted preprint unless the code is recovered and the analysis re-run.
- **Data-gap search** (`results/hb2_search.md`). This was a dated **rapid structured web search** on 2026-10-01; it is not a formal systematic review. Six queries were run over dataset repositories and the literature, and every candidate was opened.
  - **Panic-only wearable data:** the Korea University digital-phenotyping dataset (Fitbit heart rate, self-reported panic); Oh et al. 2025 (wearable ECG); Tsai et al. 2022 (smartwatch, weekly self-report).
  - **Cardiac-only data:** MIT-BIH, PTB-XL, Shaoxing–Ningbo, and a 12-lead ACS dataset (2026).
  - **Paired data:** none. No dataset paired confirmed panic and confirmed cardiac events in the same participants with wearable signals.
  - **How to state it in the paper:** "as of 2026-10-01, in a rapid search," not as an exhaustive finding. Re-run the queries before posting if more than a few weeks pass.

## Appendix E. Interpretability (descriptive; regenerated on the frozen table)
`run_interpretability_frozen.py`, `results/interpretability_frozen.json`. All attributions are out-of-fold under LOSO: each participant's attributions come from a model that never saw them.

**Share of mean |attribution| by feature group:**

| Model, condition | Heart rate | Other HRV | EDA | Coupling | Top 3 features |
|---|---|---|---|---|---|
| (ii) NaN-trained LightGBM, complete (SHAP) | 53% | 17% | 28% | 2% | mean NN, EDA tonic slope, EDA tonic mean |
| (ii) NaN-trained LightGBM, reduced (SHAP) | 68% | 22% | 8%* | 1%* | mean NN, pNN50, RMSSD |
| (iii) ECG-only LightGBM (SHAP) | 69% | 31% | — | — | mean NN, pNN50, RMSSD |
| Gated network, complete (Integrated Gradients) | 35% | 35% | 24% | 6% | mean NN, pNN50, EDA tonic slope |
| Gated network, reduced (Integrated Gradients) | 49% | 51% | 0%† | 0%† | mean NN, pNN50, RMSSD |

\*Under the reduced condition the EDA features are missing. Their SHAP values attribute the model's response to the *absence* of EDA (the missing-value branch), not to EDA content.

†The EDA gate is closed, so these are zero by construction.

- **Integrated Gradients settings:** 256-step midpoint rule; baseline = training median of every feature; logit output. Completeness error |ΣIG − (f(x) − f(baseline))|: median 0.004, 95th percentile 0.020, maximum 0.092 logit.
- **Reading.**
  - Heart rate is the top feature for every model and condition.
  - In the gated network, "other HRV" (pNN50, RMSSD) receives about half the attribution when EDA is missing. But §5.3 shows that removing within-participant heart-rate covariation from those features collapses AUROC to 0.60. Their attribution is therefore largely heart rate expressed in correlated units, a known limitation of per-feature attribution on correlated inputs.
  - The group shares are descriptive. Causal statements rest on the removal experiments in §5.3, not on attributions.
- **Do not interpret:** LF/HF ranks (unreliable at 60 s).
- **Superseded.** The earlier (2026-09-20, 445-window, single split) numbers, about 50% HR / 44% EDA SHAP, are superseded.

## Appendix F. Companion note: WESAD × MIT-BIH dataset-identity audit (separate paper)
- **Status.** The `--fast` run (2026-09-30) landed at Branch C (partial / inconclusive):
  - subject-level identity AUC 0.83, lower CI 0.68 (BORDERLINE under the lower-CI rule);
  - MMD R = 1.67;
  - CORAL lowered identity AUC by 0.087 with within-dataset clinical AUCs retained;
  - all p-values at the Monte Carlo floor.
- **Required before writing the note:**
  - the full run (10,000 MMD and 1,000 probe permutations, all 8 harmonization stages, leave-one-stage-out, Holm correction);
  - null-relative thresholds documented as a post-hoc amendment;
  - an explanation of the C1 subject pool (38, or 35 in Step 4, of 55);
  - the MIT-BIH reconciliation (48 records → 40 participants analysed).
- **Earlier analysis text.** The full audit write-up is in the archived outline, §5.6.
- **Note.** A full identity-probe run (`python -m audit.step4_identity_probe`) was in progress on 2026-10-01; its output is not yet reviewed.

## Appendix G. Pending items before posting
1. ~~Tuned results~~: done 2026-10-01 (§5.2, abstract).
2. ~~Per-condition Platt for all models~~: done (§4.4, §5.1, §5.5).
3. ~~Risk–coverage curves~~: done (§5.6, Figure 2).
4. ~~Feature-extraction documentation and coupling decision~~: done (§3, §4.1, §4.2). Coupling features were always removed with EDA.
5. ~~Regenerate interpretability~~: done (Appendix E). ~~H-B2~~: reported as historical, removed from claims (Appendix D). ~~H2a smallest effect of interest~~: resolved. No smallest effect of interest is set; H2a is reported as untestable at 60 s (Appendix C).
6. ~~Data-gap search~~: done as a dated rapid search (Appendix D). Re-run the queries if posting is delayed.
7. Author decisions: COI statement (page 1) and the public-number disclosure (Appendix A, item 1).
8. Optional, strengthens the paper:
   - real wrist-only tiers from the E4 channels (raw `.pkl` data are available at the path in `run_full_audit.sh`);
   - BVP and respiration features;
   - an external dataset.
9. `git init`; tag the current state as post-rerun (not a pre-registration).

## Appendix H. Change log (review traceability)
| Review item | Where addressed |
|---|---|
| Round 3, R1 C1 (single partition) | Appendix A #2; §4.5 |
| R1 C2 (Platt not implemented) | Appendix A #5; §4.4; §5.5 |
| R1 C3 (ECE estimator) | Appendix A #3; §5.5 |
| R1 C4 / R1+R2 §2 (rule provenance) | Page 1; §4.6; §7 |
| R1 C5 (overclaim "as good or better") | §5.1 defensible statement; §10.1 |
| R1 C6 (per-participant) | §5.4, Figure 1. *Correction:* the pooled failure for S9/S10 was partly a between-participant offset; within-participant, the networks rank S9/S10 well |
| R1 C7 / R2 finding 2 (HR mechanism) | §5.3; Appendix A #6 |
| R1 C8 (two feature tables) | Appendix A #7; §3 |
| R1 M1 (paired masking) | §4.2; Appendix A #4 |
| R1+R2 H-C1 | Cut; §10.3 |
| R1 M2 (gating vs model family) | §5.2 (MLP, capacity sweep, tuning) |
| R1 M3 (WESAD construct validity) | §3; §7 |
| R1 M4 (ultra-short HRV) | §4.1 |
| R1 M5 / M7 (H2a) | Appendix C |
| R1 M6 / M8 (audit) | Appendix F |
| R2 stale framing (Introduction / Discussion / Conclusion / abstract / ratio block / Appendix 11) | Rewritten; ratio block in Appendix B |
| R2 number fixes | Superseded by the re-run tables |
| R2 finding 3 (calibration without discrimination) | §5.5, reframed as the estimator lesson |
| Remaining items (2026-10-01 late) | per-condition Platt §4.4/§5.1/§5.5; selective prediction §5.6 and Figure 2; feature extraction §4.1; interpretability Appendix E |
| Earlier rounds (C1–C5, MA1–MA10, m1–m9) | See archived outline and `peer-review.md` / `peer_review_updated.md` |
