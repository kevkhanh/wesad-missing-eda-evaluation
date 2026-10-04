# Preprint Outline: Evaluation Pitfalls in a 15-Participant WESAD Study of Missing Wearable EDA: A Cardiac-Motor Fallback, and No Detectable Benefit of Hard Modality Masking
*(working title, see §10.1 for alternatives; target: arXiv preprint, then a negative-results / evaluation workshop or ML4H Findings)*

**Status of this outline (2026-10-02, revision 9).**
- Revision 9 answers review 9 (the expert feedback received on revision 8; its items are traced in `supplement_review_traceability.md`). Changes: (1) the masking and gate effects are read under the same three inference schemes, and the masking benefit for the networks is stated as unresolved at n = 15 (the tree result is the robust one); (2) §5.13 carries the design caveat and a post hoc no-added-masking variant (Tables 7c, 7d); (3) the per-configuration finding ("two simple trees and a switch") is in the abstract, Discussion and Conclusion with its caveats; (4) the Discussion is made consistent with the Results (calibration, abstention, robustness), with new post hoc intercept-only and slope-only recalibration runs (§5.2, Table 4b); (5) the abstract is complete and leads with the df-14 equivalence value; the negative movement partial is interpreted (§5.7); Appendix H is a reusable checklist. **Revision 8 was edited in place and no snapshot of it exists**; the latest archive is the revision-7 file below.
- Revision 8 answers the second round of review (equivalence reporting, informative missingness, abstract ordering, attribution of review rounds). Previous version archived at `archive/e14-preprint-outline_2026-10-02_rev7-c3written-pre-review8.md`; the revision-6 version is `archive/e14-preprint-outline_2026-10-02_rev6-pre-round5.md`. Review traceability (which change answers which review item) is in `supplement_review_traceability.md`, not in this document.
- References were re-checked on 2026-10-01 (`results/citation_check_2026-10-01.md`) and, for the items flagged in later review, on 2026-10-02.
- Numbers are cross-checked by script: `python check_numbers.py` ties each table cell and claim to a results-file key and writes `results/number_check_report.md`; every decimal, percentage and ratio in the body is either tied to a results key or listed with its reason in `check_numbers.CONSTANTS`.
- **Every number in §5 comes from the frozen feature table** (SHA-256 `df483a7abae0…`, `results/feature_table_manifest.json`) or, where stated, from the extended table (SHA-256 `3b384d7f66c7…`, `results/extended_table_manifest.json`). The extended table was re-extracted from the raw WESAD files; its chest baseline/stress rows reproduce the frozen table exactly (434/434 windows, 0 mismatched values).

**Conflict-of-interest statement (page 1).** The author is a co-founder of E14 Health, a company whose product thesis depends on wearable physiological monitoring for acute-event differentiation. No result from this project, including any superseded number listed in Appendix A, has appeared in E14 Health investor, accelerator, grant, marketing or recruitment materials, talks, websites or social media. No co-author or statistician without equity has yet checked the primary analysis; an independent check is listed as open in Appendix G.

**Non-clinical disclaimer (page 1, repeated in the abstract).** This paper studies a stress-vs-baseline detector trained on 15 WESAD participants (young, healthy, laboratory setting). It does not evaluate, and should not be read as evidence for, differentiating panic attacks from cardiac events in patients.

**Provenance (page 1).**
- **Decision rule.** The falsification rule in §4.6 was written in this analysis outline on 2026-10-01, before the reduced-sensor baselines were run, when the gated model's earlier point estimates were already known. It was not independently timestamped: the project was not under version control and no OSF registration exists. It was evaluated twice, before and after the implementation fixes in Appendix A; both outcomes are reported.
- **Later analyses (2026-10-02).** The movement-vs-HR paired contrast, the responders-vs-movers correlations and the abstention action model (Table 8) were added after review 8, and the intercept-only and slope-only recalibration (Table 4b) and the no-added-masking informative-missingness variant (Tables 7c, 7d) after review 9; all are post hoc. The factorial cell, equivalence tests, participant-level inference, order-confound checks, sensitivity analyses, wrist replication, posture/movement analysis and minimal re-implementation (§4.7) were all specified and run after the primary results were known. The equivalence margins were fixed in code before these runs (`run_repeated_cv.SESOI`), but the comparisons they apply to were already known, so every equivalence result is **post hoc**. None of these analyses was pre-registered.
- **Round-5 analyses (operating points, §5.9; informative missingness, §5.13).** Their design, margins and predicted outcomes were written in `prespec_round5.md` (2026-10-02, 05:46 UTC) before any code for them existed. The file's SHA-256 was recorded in `results/prespec_round5_manifest.json`, and both scripts refuse to run if the file has changed since. This is still not an independent timestamp (no version control yet); every prediction is reported as confirmed or not.
- **Configuration files.** `configs/preregistration.yaml` (audit thresholds) was likewise written without version control. The word "pre-registered" is not used for anything in this paper.

**Framing.** This is a methods and evaluation paper motivated by a clinical differentiation problem. It does not claim to solve that problem. Each Results subsection carries its own proxy-data caveat.

---

## Contributions (stated against prior work)

**Not claimed as new:**
- modality dropout and missing-modality training as techniques, and as the standard baseline in missing-modality work outside wearables (ModDrop, Neverova et al., *IEEE TPAMI* 2016; Ma et al., CVPR 2022; Woo et al., AAAI 2023; Ramazanova et al., CVPR Workshops 2025);
- the observation that training regime can matter more than fusion architecture under missing modalities (Ma et al., CVPR 2022: multi-task training "contributes more than the fusion strategy"), and that complex multimodal architectures often fail to beat simple baselines under standardised evaluation (SimBaMM; Rheude, Eils & Wild, arXiv 2512.22991);
- modality gating and masked late fusion for missing modalities (SimMLM, Li, Chen & Han, ICCV 2025, uses a *learnable* gate; ours is a fixed hard mask, see §4.3);
- calibration under modality removal as a problem framing (Ma et al., "Calibrating Multimodal Learning," ICML 2023);
- calibration and selective prediction in multimodal clinical classifiers (Lechuga López, Shamout & Rudner, arXiv 2603.02719).

**Claimed** (in order of how much weight we put on them):
1. **An evaluation-pitfalls case study for small-n wearable studies.** Appendix A lists nine implementation errors found in our own pipeline, each paired with the check that catches it and a regression test. Five of them changed or could have changed the headline comparison:
   - "repeated" CV that reused one subject partition (#2);
   - a zero-clipped minimum-over-bins ECE estimator that ranked the worst-calibrated model as best (#3);
   - complete and reduced conditions scored on disjoint windows (#4);
   - a feature table silently replaced between runs (#7);
   - an early record-level leak (#1), caught before any comparison reported here.

   A further one (#6) changed the reading of the heart-rate mechanism, and the other three (#5, #8, #9) were mismatches between the description and the code; we count the five above as the ones that bear on the headline.

   The decision rule's two evaluations (before and after the fixes) are both reported, and the corrected feature table is reproduced exactly from raw data. Each individual error is well known in the literature; what we add is the documented instance and **Appendix H, a one-page checklist (check → what it catches → how to run it)** that ties every error to an executable test or script in the released tree, so that a small-n study can run the same checks on its own pipeline.
2. **Without EDA, the models detect a cardiac-motor response, shown at subject level and checked against the condition-order and posture/movement confounds.**
   - Without EDA, removing heart rate removes about half of every model's above-chance pooled AUROC (§5.5).
   - HR-only discrimination is close to a monotone function of each participant's standardised HR response, so we report that correlation descriptively. The informative quantity is the *non-HR* models' collapse when HR is removed.
   - Order-balanced stress-vs-amusement and negative-control analyses (§5.6) show that the heart-rate mechanism is not a time-since-start artefact: it survives an order-balanced stress-vs-amusement contrast, and baseline-vs-stress models do not flag the later calm conditions as stress.
   - It is **not separable from movement and posture** (§5.6E). The TSST is performed standing and speaking; chest-accelerometer movement alone discriminates stress from baseline better than heart rate does, and heart rate adds only a small within-subject increment beyond it. We therefore call the fallback signal a cardiac-motor response, not a cardiac one.
3. **A negative result on hard modality masking (post hoc equivalent to a masked MLP within ±0.02 Brier), with the full mask × training-masking factorial.** Every comparison in this contribution is post hoc (§4.7), so none of it is confirmatory, the checks in §5.4 reuse the same 15 participants (sensitivity analyses, not replications), and the equivalence tests are not adjusted for multiplicity.
   - With EDA missing at test time, a modality-gated network trained with modality dropout shows no detectable advantage over a capacity-matched MLP trained with the same masking. Post hoc, the two are equivalent within ±0.02 Brier and ±0.03 AUROC under the widest interval scheme (B–F df 14: TOST p = 0.033 and 0.015; df 99: 0.025 and 0.009). The honest summary is "equivalent at about ±0.02, and fragile": the Brier bound attained is 0.018 against a margin of 0.02, and the equivalence tests are many and unadjusted (§7).
   - Against reduced-sensor LightGBM baselines, the data bound the difference only to about ±0.03 Brier, which cannot exclude practically relevant differences.
   - In the never-masked regime the gate does not substitute for masking: gated vs MLP, both never masked, ΔBrier −0.011 [−0.067, +0.045]. **For trees, exposure to missing EDA is essential:** (ii) vs (i) is −0.148 [−0.259, −0.037], and the interval excludes 0 under every inference scheme. **For the networks the masking benefit (−0.044) is directionally consistent (12/15 participants) but not resolved at n = 15:** its Bouckaert–Frank interval is [−0.088, −0.001] at df 99 and [−0.091, +0.003] at df 14, and only the narrowest, participant-level scheme excludes 0. The gate main effect is read under the same three schemes (§5.2, Table 4); only the participant-level scheme shows it to be within ±0.02.
   - **A practical corollary.** The per-configuration system (a complete-trained tree when EDA is present, an ECG-only tree when it is absent) had the lowest point-estimate Brier in all 19 informative-missingness scenarios and the lowest false-alarm rate with EDA present, and tied the other systems without EDA (§5.3, §5.9, §5.13). Nothing in this study favours a fusion architecture over "two simple trees and a switch"; the mixed-test design favours that system, and its margin over the best masked tree is small (see §5.13).

   The null was close to expected by construction: the "gate" multiplies the EDA embedding by the missingness bit and has no learned parameters, and an MLP given a missingness indicator can learn the same switch (§4.3). The contribution is the documented evidence and the factorial, not a surprise. It is consistent with Ma et al. (CVPR 2022) and SimBaMM, and adds a matched-capacity mask-vs-indicator ablation with calibration and selective-prediction endpoints that CognifySSL and PULSE do not report.

---

## 1. Introduction
- **Clinical motivation.** Panic attacks and acute cardiac arrhythmias such as paroxysmal supraventricular tachycardia (PSVT) overlap in symptoms, and cardiac events misattributed as panic are a documented, costly error (Lessmeier et al., *Arch Intern Med* 1997). Detectors meant for this setting should abstain or escalate when uncertain, which requires them to stay calibrated when a sensor drops out; this paper studies only that calibration question, on a laboratory stress proxy (§6 discusses what the proxy cannot say).
- **The data ceiling, stated immediately.** In a dated rapid search (2026-10-01; `results/hb2_search.md`, Appendix D) we found no public dataset that pairs confirmed panic attacks and confirmed cardiac events in the same participants with the same wearable sensors.
  - Panic-only wearable datasets exist, with self-reported panic and consumer heart rate (e.g. the Korea University digital-phenotyping dataset; Jang et al., *Sci Data* 2024;11:1264; current Zenodo record 10.5281/zenodo.14190138).
  - Cardiac-only datasets exist (12-lead or Holter).
  - This paper therefore studies a proxy question that the data can answer.
- **The question.** When the EDA sensor is missing at test time, does a modality-gated network trained with modality dropout give better-calibrated and more discriminative stress predictions than models built or trained for the reduced sensor set? And if not, is it the gate or the masking that matters?
- **What we found** (written as a candid result):
  - The gated network's advantage in our earlier drafts was an artefact of the comparator: a LightGBM trained only on complete data and given missing EDA at test time.
  - Against comparators that have seen missing EDA, we find no evidence that the gate adds anything. Against a matched MLP with the same masking, the two are equivalent within a small post-hoc margin. Against reduced-sensor LightGBM, the data cannot exclude differences of about ±0.03 Brier.
  - In the never-masked regime the gate does not substitute for masking.
  - Without EDA, all models rely on heart rate, and per-participant performance tracks how much each participant's heart rate responded to the stressor.
  - Along the way we made nine implementation errors, five of which moved or could have moved the headline comparison. We report them as the paper's most transferable contribution.
- **Why report it.** Missing-modality training is a standard baseline in vision and language. In the wearable missing-modality work closest to ours it is not always present: CognifySSL v2.0 compares only against self-supervised methods (CroSSL, MU-MAE, COCOA), with no zero-fill, imputation or masked supervised baseline. PULSE, by contrast, does include a model trained on the deployment sensors only. We do not claim that the wearable literature as a whole uses weak baselines.

## 2. Related Work
- **Wearable stress and panic detection.** WESAD (Schmidt et al., ICMI 2018) is the standard laboratory stress benchmark. Wearable panic-attack work forecasts attacks over days to a week:
  - from consumer-wearable heart rate, sleep and activity plus questionnaires, with self-reported labels (Tsai et al., *JMIR Med Inform* 2022;10(2):e33063: 7-day horizon, best random-forest accuracy 81.3%);
  - from wearable-ECG HRV (Oh et al., *J Med Internet Res* 2025;27:e69045);
  - see the scoping review by Zahed et al. (*Digital Health* 2025; 7 studies).

  This paper works at the 60-s window level and makes no claim beyond the proxy task.
- **Missing-modality robustness.**
  - ModDrop (Neverova et al., *IEEE TPAMI* 38(8):1692–1706, 2016) is the canonical modality-dropout technique.
  - Ma et al. (CVPR 2022, pp. 18177–18186) found that Transformers are sensitive to missing modalities, that the best fusion strategy is dataset-dependent, and that multi-task training contributes more than the fusion strategy. Modality dropout is a named baseline in Woo et al. (AAAI 2023), in multimodal prompt-learning work (ACL 2024), in MMP (2024), and in Ramazanova et al. (CVPR Workshops 2025), where the modality-dropping baseline outperforms training-free approaches. Our gating result is a wearable-physiology instance of this pattern.
  - SimBaMM (Rheude, Eils & Wild, arXiv 2512.22991) re-implemented 19 multimodal methods on nine datasets under standardised tuning and testing and found that more complex architectures often fail to beat simpler baselines. It supports our general thesis and lowers the novelty of "architecture does not matter under fair evaluation".
  - CognifySSL v2.0 (Cisse & Ishimaru, *Sensors* 2026;26(8):2419) is a self-supervised detect-and-repair framework with random modality dropout, evaluated on WESAD under LOSO. It reports accuracy and macro-F1 (and reconstruction RMSE), but no AUROC, Brier or quantitative calibration. Its baselines are CroSSL, MU-MAE and COCOA; its dropout evidence is a self-ablation (85.5% vs 74.1% F1 under partial input). Its full-modality set is described as IMU, ECG, EDA and temperature; respiration appears only in reconstruction. Results are from a single seed.
  - PULSE (Zhao et al., arXiv 2510.24058; CVPR 2026 Workshop) reports 0.994 AUROC on WESAD baseline-vs-stress without EDA at inference, via privileged-information distillation. It is the same binary task under LOSO, but with a different protocol:
    - student sensors are chest ECG plus wrist BVP, ACC and temperature;
    - every channel is z-scored per subject with parameters from that subject's **baseline windows** (label-informed and transductive);
    - S12 is a fixed validation fold, leaving 13 test subjects;
    - AUROC is averaged per subject.

    Its no-teacher model trained on the deployment sensors alone already reaches 0.963. Our closest metric, LOSO within-subject AUROC without EDA (also a per-subject average), is 0.87–0.90 (Table 5). The gap is therefore mostly protocol, not a sign that our features are uniquely weak. §5.4 reruns our comparison with PULSE-style normalisation.
- **Learnable gating (not what we test).** SimMLM (Li, Chen & Han, ICCV 2025; arXiv 2507.19264) combines a learnable gating network (DMoME) with a ranking loss (MoFe), trains with random modality dropout, and reports ECE and SCE on BraTS 2018, UPMC Food-101 and avMNIST. Our "gate" has no learned parameters (§4.3), so our results say nothing about learnable gating of the SimMLM kind; they concern hard modality masking (masked late fusion) against a missingness indicator.
- **Calibration under missing or corrupted modalities.** Ma et al. (ICML 2023, PMLR 202; arXiv 2306.01265) show that confidence can *increase* when modalities are removed and propose the CML regulariser. Our complete-trained LightGBM shows the opposite failure: it confidently under-predicts stress when EDA is missing (§5.7).
- **Tabular model families.** Gradient-boosted trees routinely match or beat neural networks on small tabular data (Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 Datasets & Benchmarks). We see this in the complete condition (§5.3).
- **Evaluation methodology.**
  - Subject-level CV and pseudoreplication.
  - Corrected repeated-CV inference (Nadeau & Bengio 2003; Bouckaert & Frank, PAKDD 2004) and its limits: naive CV intervals undercover (Bates, Hastie & Tibshirani, *JASA* 2024;119(546):1434–1445).
  - Equivalence testing (TOST) and the correlated Bayesian t-test for CV (Corani & Benavoli 2015).
  - Cluster bootstrap with few clusters (Cameron & Miller, *J Hum Resour* 2015).
  - ECE estimator bias (Roelofs et al., AISTATS 2022); SmoothECE (Błasiok & Nakkiran, ICLR 2024).
  - Selective prediction and E-AURC (Geifman, Uziel & El-Yaniv, ICLR 2019).
  - TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378).
- **Equity of wearable access.** Wearable ownership and performance are unequal across age and other groups (Zinzuwadia & Singh, *Lancet Digit Health* 2022;4(12):e856–e857), which matters for the target population (§9).
- **Shortcut learning and dataset identity** (Zech et al., *PLOS Med* 2018; Rabanser et al., NeurIPS 2019; DeGrave et al., *Nat Mach Intell* 2021) matter for the companion audit note (Appendix F), not for this paper's within-WESAD results.

## 3. Data
- **WESAD** (Schmidt et al., 2018):
  - 15 participants (12M/3F, 27.5 ± 2.4 years).
  - Chest RespiBAN (ECG, EDA, respiration, EMG, temperature, ACC at 700 Hz) and wrist Empatica E4 (BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz).
  - **Primary analyses:** chest ECG and chest EDA; baseline vs stress (TSST).
  - **Secondary analyses** also use amusement and both meditation blocks (§5.6), chest accelerometry (§5.6E) and the wrist E4 channels (§5.10).
- **Windows** (`real_data_loaders._window_wesad_subject`; the same rules in `extract_extended.py`):
  - Non-overlapping 60-s windows on the 700 Hz label stream; a window is kept only if its label is constant across all 60 s.
  - At most the first 20 windows per condition per participant. After the label-constancy filter the cap is never reached: baseline yields 18–19 windows and stress 9–11; amusement 5–6 and meditation 10–12 (two blocks).
  - Primary table: 434 windows, 27–30 per participant, 35% stress, frozen by hash. It replaces a 445-window table used by the 2026-09-20 runs that no longer exists (Appendix A).
  - Extended table: 683 windows (282 baseline, 152 stress, 79 amusement, 170 meditation) with each window's session time, signal-quality indices, movement, and wrist features.
- **Construct validity:**
  - **TSST is not panic.** It is a social-evaluative task with public speaking and mental arithmetic, so part of the heart-rate rise reflects speech, respiration and motor activity rather than anxiety. Panic onset is typically uncued and often accompanied by hyperventilation.
  - **TSST heart-rate reactivity is heterogeneous** across studies (meta-analytic pre–post HR effect size 0.85 for the traditional TSST; Helminen, Morton, Wang & Felver, *Psychosom Med* 2021;83(3):200–211; verified 2026-10-02). The four weak responders here (27%) are therefore plausible, not anomalous.
  - **Posture and movement.** The TSST is performed standing, speaking and gesturing; baseline, amusement and meditation are seated or reclined. Any heart-rate rise under stress therefore mixes psychological arousal with orthostatic and motor load. A chest accelerometer records trunk tilt and movement but cannot tell standing from upright sitting, so this confound can be bounded but not removed (§5.6E).
  - **Fixed condition order.** In WESAD the baseline always comes first. Stress and amusement were swapped between the two protocol versions (here 7 participants had stress before amusement, 8 after), and each was followed by meditation. Baseline-vs-stress is therefore confounded with time on electrode, tonic EDA drift and habituation. §5.6 tests this.
  - **Population.** Young, healthy and mostly male; generalisation to older adults and people with cardiac comorbidity is untested.
- **MIT-BIH Arrhythmia** is used only in the companion dataset-identity audit (Appendix F).

## 4. Methods

### 4.1 Features
- **ECG (7 features):** mean NN interval (`hrv_mean_nn`, equivalent to heart rate), SDNN, RMSSD, pNN50, LF power, HF power, LF/HF.
  - **Reliability at 60 s.** Only mean NN and RMSSD are reasonably supported at ultra-short windows; SDNN, pNN50, LF and LF/HF are not (Task Force 1996, *Circulation* 93:1043, asks for about 2 minutes for LF and prefers 5). LF-based results are not interpreted, and §5.4 reruns the comparison with mean NN and RMSSD only.
- **EDA (4 features):** tonic mean, tonic slope, phasic mean amplitude, phasic rate.
- **Coupling (2 features):** ECG–EDA peak cross-correlation and its lag (`coupling_lag_sec`, the H2a feature; Appendix C).
- **Extraction specifics** (`feature_extraction.py`, `stage1_pipeline.py`; neurokit2 0.2.13):
  - **ECG/HRV.** R-peaks from `neurokit2.ecg_peaks` (default detector) on the raw 700 Hz chest ECG, with no separate cleaning step and no mains notch. HRV from `nk.hrv_time` / `nk.hrv_frequency` with Task Force bands (LF 0.04–0.15 Hz, HF 0.15–0.40 Hz). A band-pass fallback detector exists but was not triggered on this table (0% missing HRV).
  - **EDA.** Tonic level from a 2nd-order Butterworth low-pass at 0.5 Hz (not cvxEDA); tonic features are its mean and linear slope. Phasic = residual; skin-conductance responses by peak detection (height ≥ 5% of the phasic SD, spacing ≥ 0.8 s), summarised as mean amplitude and rate per minute.
  - **Coupling.** Instantaneous HR from a 5–15 Hz band-pass R-peak detector, cross-correlated with z-scored EDA within ±15 s; outputs the peak |correlation| and its lag. Undefined in 0.2% of windows.
  - **Wrist (E4).** The same 13-feature schema from wrist signals: HRV from BVP beats (`nk.ppg_clean` then `nk.ppg_peaks`, 64 Hz), EDA features on the 4 Hz wrist EDA, coupling from PPG-derived HR. A beat-detection-free spectral HR (dominant 0.7–3.5 Hz BVP frequency) is computed only to check wrist–chest agreement.
- **Signal quality (descriptive).** No window is excluded. For each window we record the fraction of NN intervals outside 300–2000 ms and the fraction of successive NN changes > 20%, from chest and wrist beats, plus chest and wrist accelerometer magnitude SD (§5.11).

### 4.2 Missing-EDA simulation (paired)
- Every held-out window is scored twice: once with all features (complete condition) and once with all **six** EDA-dependent features (four EDA plus two coupling) missing (reduced condition). Missingness is complete and MCAR by construction.
- The coupling features need the EDA channel, so they are treated as part of the EDA modality throughout: removed in the reduced condition, in the EDA branch of the gated network, and masked with EDA in all missingness training.
- How each model receives missing EDA:

  | Model | Missing-EDA encoding |
  |---|---|
  | LightGBM | native NaN, zero-fill, or training-mean imputation, as named |
  | Gated network | closed EDA gate |
  | MLP | inputs are robust-standardised ((x − training median)/IQR), so "zeroed EDA" is training-**median** imputation, plus an EDA-present indicator |

### 4.3 Models
| Name | Description |
|---|---|
| (i) Complete-trained LightGBM, NaN at test (in effect zero-fill) | trained on complete windows; missing EDA delivered as NaN. **Checked 2026-10-02 (`tests/test_round5.py`): LightGBM maps NaN to 0 at prediction for a feature that had no missing values in training, so for the four EDA features this is zero-fill, i.e. a tonic EDA of 0 µS, outside the observed range; the two coupling features, undefined in 0.2% of windows, use the native missing branch only in folds whose training set contains such a window.** The label "NaN at test" is kept for continuity with the scripts. *This is the comparator behind earlier drafts' headline* (Appendix A) |
| Complete-trained LightGBM, zero-fill at test | as above with EDA = 0 |
| (ii) NaN-trained LightGBM | EDA set to NaN on a random 50% of training windows |
| (iii) ECG-only LightGBM | trained on ECG features only. With (i) it forms the per-configuration deployment: full model when EDA is present, ECG-only model when absent |
| (iv) Mean-imputation + per-condition Platt | |
| HR-only | LightGBM on `hrv_mean_nn` alone |
| **Gated network** (hard mask + masking) | two branches (ECG 7→16→16, EDA 6→16→16), summed, ReLU, linear head. The "gate" is a hard mask: the EDA embedding is multiplied by the EDA-present bit (0 or 1), with no learned gating parameters. Modality dropout p = 0.5 zeroes that bit during training; 801 parameters |
| Gated network, no dropout (gate, no masking) | trained on complete data only |
| **Non-gated MLP** (no gate, masking) | [ECG, EDA median-imputed, EDA-present indicator] → 32 → 16 → 1, with the same 50% EDA masking; 1,025 parameters (the capacity sweep uses size-matched pairs) |
| **Non-gated MLP, no masking** (no gate, no masking) | as above, trained on complete data only, so its indicator is never 0 in training |
| HR-removed variants | gated network, (ii) and (iii) with `hrv_mean_nn` removed |
| HR-residualised gated network | `hrv_mean_nn` removed; every other ECG feature residualised on heart rate *within each participant* (subject mean retained). Label-free but transductive |

- The four network rows form a 2 × 2 factorial: gate {yes, no} × training-time EDA masking {50%, none}.
- **Terminology, and why the null was expected.** We keep the name "gated network" for continuity with earlier drafts, but it is masked late fusion: the gate is the missingness bit itself. The masked MLP receives the same bit as an input feature and, with one hidden layer, can learn to suppress the median-imputed EDA inputs when the bit is 0. Both models can therefore represent nearly the same function, and the comparison tests an architectural inductive bias (hard-wired vs learned switching), not a learned gating mechanism. Equivalence was the expected outcome; the contribution is measuring it with matched capacity and calibration endpoints.
- **Training (untuned defaults):** LightGBM with 50 trees, depth 3, learning rate 0.1; networks with full-batch Adam, learning rate 1e-2, 100 epochs, no early stopping.
- **Nested tuning:** each outer training fold selects hyperparameters by inner subject-level 4-fold CV, minimising the mean of complete- and reduced-condition out-of-fold Brier.

  | Family | Grid | Configurations |
  |---|---|---|
  | LightGBM | trees {50, 200} × depth {2, 4} × learning rate {0.05, 0.1} × min child samples {10, 30} | 16 |
  | Networks | width {3 matched sizes} × epochs {100, 300} × learning rate {3e-3, 1e-2} × weight decay {0, 1e-3} | 24 |

- **Capacity sweep:** gated embedding width 8/16/32/64 (273/801/2,625/9,345 parameters) vs MLP hidden width 8/25/84/300 (281/808/2,637/9,333 parameters).
- **Sensitivity sets** (§5.4): masking rate p ∈ {0.1, 0.25, 0.5, 0.75} for the gated network, MLP and (ii); HRV restricted to mean NN and RMSSD; per-subject z-scoring of every feature, either label-free (each subject's own mean/SD over all its windows) or PULSE-style (mean/SD of the subject's baseline windows, label-informed and transductive, an optimistic upper bound).

### 4.4 Calibration
Three outputs are reported for every model. All calibrators are fit on inner-CV out-of-fold predictions from the training participants only.
- **Raw:** the model's own probabilities.
- **Shared Platt:** one calibrator fit on complete-condition predictions and applied to both conditions. Baseline (iv) uses per-condition calibrators by design.
- **Per-condition Platt:** a separate calibrator for each sensor configuration.

Isotonic regression is not used, because about 12 training participants is too few.

### 4.5 Evaluation design
- **Primary inference:** 20× repeated subject-level 5-fold CV, each repeat with its own random partition of participants (20 distinct partitions, stored in `results/repeated_cv.json`). 100 (repeat, fold) pairs, about 12 training and 3 test participants each. Every metric and comparison is computed in **both** conditions.
- **Pooled estimates and per-participant analysis:** leave-one-subject-out (LOSO; 15 folds), and a single random 5-fold partition as a check.
- **Metrics:**
  - Brier score (primary); log-loss; AUROC pooled across held-out participants; **within-subject AUROC** (mean of per-participant AUROCs);
  - equal-mass ECE_SWEEP (Roelofs et al. 2022; no bias subtraction, no clipping) and, as a binning-free check, SmoothECE (Błasiok & Nakkiran 2024; our implementation, reflected Gaussian kernel, fixed-point bandwidth);
  - calibration intercept and slope;
  - **selective prediction:** AURC and error at 60% and 80% coverage, with two references computed from each model's own full-coverage error rate r: random abstention (AURC = r) and the oracle that abstains on errors first (AURC* = r + (1 − r) ln(1 − r)); E-AURC = AURC − AURC*. Ties are averaged over orderings.
- **Inference** (A − B; negative ΔBrier means A is better). **Primary scheme: the Bouckaert–Frank corrected t with df = 99** over the 100 repeated-CV fold differences; every CI quoted in the abstract and main text uses it unless labelled otherwise. The df = 14 and participant-level intervals are sensitivity analyses, and we flag any conclusion that differs between schemes. **Equivalence claims are read under the widest scheme (B–F df = 14)**, because an anti-conservative interval makes a TOST easier to pass; Table 3 shows all schemes.
  - **Bouckaert–Frank corrected t** over the 100 fold differences: SE² = (1/100 + n_test/n_train)·var(d), with df = 99 as proposed and, as a sensitivity analysis, df = 14 (the number of independent units is about 15 participants; this widens intervals by about 8%). The correction is a heuristic here: with 15 clusters there is no simulation evidence for nominal coverage, and naive CV intervals are known to undercover (Bates et al. 2024).
  - **Participant-level analysis (sensitivity):** each participant's metric is averaged over the 20 repeats, and the 15 paired differences get a t interval (df 14) and an exact sign-flip test (2^15 patterns). Participants in the same fold share a model, so this is approximate too.
  - **Post-hoc equivalence:** TOST at a smallest effect size of interest of ±0.02 Brier (about 9% of the prevalence-only Brier of 0.35 × 0.65 = 0.2275, i.e. about 0.09 Brier skill) and ±0.03 AUROC, for both inference schemes. We also report the smallest symmetric bound at which TOST would pass (the larger |bound| of the 90% CI), which needs no SESOI, the minimum detectable difference at 80% power, and a ROPE probability (correlated Bayesian t-test for fold differences; flat-prior t posterior for participant differences).
  - **Factorial contrasts** per fold and per participant: interaction (G − G_nomask) − (M − M_nomask), and the main effects of gate and of masking (each averaged over the other factor).
- **Pooled LOSO CIs:** participant-cluster bootstrap, B = 2,000, percentile and BCa (acceleration from the leave-one-participant-out jackknife), for both conditions. With 15 clusters these are approximate (Cameron & Miller 2015) and labelled as such.
- **Software and reproducibility:** feature tables hash-checked by every script; package versions in `requirements.txt` (Python 3.13.13, CPU); fixed seeds per (repeat, fold); 44 automated tests, including a regression test for every implementation error in Appendix A, tests of the round-5 threshold, net-benefit, abstention and missingness-mechanism code, and tests of the review-9 recalibrators and no-added-masking variant.

### 4.6 Decision rule (provenance on page 1)
> If the gated network (with dropout) does not beat (ii) NaN-trained LightGBM and (iii) ECG-only LightGBM on reduced-condition Brier **and** ECE **and** AUROC, the architectural claim is not supported, and the paper is reported as an evaluation paper.

- **Inferential gap.** The rule compares point estimates with no inferential margin. We report CIs and the post-hoc equivalence analysis alongside it.
- **Endpoint substitution.** The rule's ECE endpoint was originally a mis-implemented "debiased ECE" (Appendix A); it is evaluated with ECE_SWEEP. Because per-fold ECE on about 87 windows is biased upward, the rule is also reported without its ECE criterion (§5.1).

### 4.7 Which analyses were pre-specified
Every analysis falls into one of five classes:
- **Pre-specified:** written in the analysis outline before the reduced-sensor baselines were run on 2026-10-01 (not independently timestamped).
- **Pre-specified (round 5):** designed, with margins and predicted outcomes, in `prespec_round5.md` before any code for it existed (hash-checked, not independently timestamped; §4.8). The primary results were known, but nothing about operating points or informative missingness had been computed.
- **Correction:** a fix to an implementation error found after the first evaluation of the decision rule (Appendix A).
- **Reviewer-requested:** asked for in one of the author's own internal, AI-assisted review rounds (not external peer review) before it was run, but specified after the primary results were known.
- **Exploratory:** added by us after seeing results.

Only pre-specified analyses can confirm anything. No claim rests on an exploratory analysis alone.

*(Supplementary Table S11.)*

**Consequence for the decision rule.** The rule's verdict changes across calibration modes and CV schemes (§5.1), and the mode under which it does not fire is exploratory. We draw no conclusion from the rule. The conclusion rests on the paired CIs and the (post-hoc) equivalence analysis.

### 4.8 Round-5 analyses: operating points and informative missingness
Both use the frozen table, the 20 stored partitions and the model seeds of the primary run, and the inference of §4.5 (Bouckaert–Frank df 99 primary). Full design: `prespec_round5.md`.
- **Operating points (`run_operating_points.py`, §5.9).** Systems: gated network, masked MLP, (ii), and the per-configuration system ((i) with EDA, (iii) without). Thresholds are chosen per condition on the inner out-of-fold predictions of the *training* participants only (the same predictions that fit Platt): a high-sensitivity point (target sensitivity 0.90) and a high-specificity point (target specificity 0.90). We report held-out sensitivity, specificity, PPV and NPV; false alarms per hour of rest, (1 − specificity) × 60 for 60-s windows; and, at 60% and 80% coverage, the abstention rate separately in stress and baseline windows. Decision curves (net benefit on per-condition-Platt probabilities) are drawn for thresholds 0.05–0.60, with inference only at 0.20, 0.35 and 0.50. Margins: ±0.05 sensitivity or specificity, ±0.01 net benefit. Raw per-fold Brier in this run reproduces the primary run exactly (1,000 of 1,000 model × fold × condition values).
- **Informative missingness (`run_mnar.py`, §5.13).** Each test window's six EDA features are missing with a window-specific probability (overall 30%), with the same random draw for every system, so missing and present windows are mixed in one test set. Training windows get "natural" missingness from a training mechanism, on top of each model's own 50% training-time masking. Mechanisms: MCAR; **label-driven** (missingness odds ratio, stress vs baseline, of 2, 4 or 8); and **motion-driven** (logistic in log chest-accelerometer SD, solved to give the same odds ratios). Scenarios: S0 MCAR→MCAR; S1 trained and tested under the same MNAR mechanism; S2 trained MNAR, tested MCAR (the mechanism changes); S3 trained MCAR, tested MNAR. Label-driven odds ratio 4, S1 and S2, are the primary scenarios; the rest are descriptive. A **shortcut index** measures how far each model reads missingness itself as evidence: the mean, over held-out baseline windows, of p(EDA removed) − p(EDA present). **Post hoc variant (review 9; not in the prespecification):** the same run with each system's own 50% MCAR training masking removed (modality dropout p = 0, no random NaN masking), because that added masking dilutes the label–missingness association in training and can hide a shortcut (§5.13).

## 5. Results

*Every result is within WESAD (stress vs baseline in healthy young adults, plus amusement and meditation in §5.6). None of it concerns panic or cardiac events.*

### 5.1 Primary comparison with EDA missing

**Table 1. 20× repeated subject-level 5-fold CV, raw probabilities: mean across 100 folds (SD for Brier and AUROC).**

| Model | Reduced Brier | Reduced AUROC | Reduced ECE_SWEEP* | Reduced log-loss | Complete Brier | Complete AUROC |
|---|---|---|---|---|---|---|
| Gated network + dropout (gate, masking) | 0.148 (0.075) | 0.892 (0.075) | 0.126 | 0.555 | 0.130 | 0.930 |
| Non-gated MLP, same masking (no gate, masking) | 0.146 (0.075) | 0.892 (0.075) | 0.125 | 0.580 | 0.141 | 0.922 |
| Gated network, no dropout (gate, no masking) | 0.186 (0.094) | 0.877 (0.093) | 0.170 | 0.980 | 0.139 | 0.928 |
| Non-gated MLP, no masking (no gate, no masking) | 0.197 (0.095) | 0.864 (0.098) | 0.171 | 1.205 | 0.166 | 0.907 |
| (i) Complete-trained LightGBM, NaN at test | 0.294 (0.041) | 0.871 (0.105) | 0.293 | 1.061 | 0.072 | 0.963 |
| (ii) NaN-trained LightGBM | 0.146 (0.084) | 0.882 (0.100) | 0.123 | 0.498 | 0.093 | 0.952 |
| (iii) ECG-only LightGBM | 0.146 (0.083) | 0.882 (0.103) | 0.123 | 0.523 | 0.146 | 0.882 |
| (iv) Mean-imputation + per-condition Platt | 0.161 (0.104) | 0.886 (0.101) | 0.141 | 0.567 | 0.072 | 0.963 |
| HR-only | 0.179 (0.087) | 0.854 (0.100) | 0.138 | 0.606 | 0.179 | 0.854 |

\*Per-fold ECE on about 87 windows is biased upward relative to the pooled LOSO estimates in §5.8. SmoothECE gives the same ordering (`results/repeated_cv.json`).

- **Log-loss points the other way, without inferential support.** Reduced-condition log-loss is lower for (ii) (0.498) than for the gated network (0.555) or the masked MLP (0.580), but the paired differences are imprecise: gated − (ii) +0.056 [−0.112, +0.225], MLP − (ii) +0.082 [−0.096, +0.260], gated − MLP −0.025 [−0.158, +0.107] (Bouckaert–Frank). Log-loss is dominated by a few confidently wrong windows, and the networks are the more over-confident models without EDA (calibration slopes 0.42–0.43 vs 0.59 for (ii), §5.8). We keep Brier as the primary endpoint, as specified, and read the log-loss column as a secondary signal that the networks' errors are more confident.

**Table 2. Paired differences (A − B), reduced condition, raw.** Bouckaert–Frank 95% CI with df = 99 and df = 14; participant-level 95% CI (t, df 14) with the number of participants for whom A is better.

| Comparison | ΔBrier, B–F df 99 | ΔBrier, B–F df 14 | ΔBrier, participant-level | A better | ΔAUROC, B–F df 99 |
|---|---|---|---|---|---|
| Gated vs (ii) | +0.002 [−0.031, +0.035] | [−0.033, +0.038] | +0.003 [−0.018, +0.023] | 7/15 | +0.010 [−0.058, +0.077] |
| Gated vs (iii) | +0.002 [−0.033, +0.037] | [−0.036, +0.040] | +0.002 [−0.020, +0.024] | 6/15 | +0.010 [−0.062, +0.082] |
| Gated vs (iv) | −0.013 [−0.073, +0.047] | [−0.077, +0.052] | −0.012 [−0.072, +0.048] | 6/15 | +0.006 [−0.065, +0.076] |
| Gated vs HR-only | −0.030 [−0.077, +0.016] | [−0.081, +0.020] | −0.030 [−0.076, +0.017] | 8/15 | +0.038 [−0.039, +0.115] |
| Gated vs MLP | +0.002 [−0.016, +0.020] | [−0.017, +0.021] | +0.002 [−0.004, +0.008] | 7/15 | −0.000 [−0.025, +0.024] |
| MLP vs (ii) | +0.000 [−0.031, +0.032] | [−0.034, +0.034] | +0.000 [−0.021, +0.022] | 7/15 | +0.010 [−0.054, +0.074] |
| Gated vs Gated, no masking | −0.038 [−0.089, +0.014] | [−0.093, +0.018] | −0.037 [−0.060, −0.015] | 11/15 | +0.015 [−0.036, +0.065] |
| MLP vs MLP, no masking | −0.051 [−0.102, +0.001] | [−0.106, +0.005] | −0.051 [−0.083, −0.019] | 12/15 | +0.028 [−0.034, +0.090] |
| Gated, no masking vs MLP, no masking | −0.011 [−0.067, +0.045] | [−0.072, +0.049] | −0.012 [−0.030, +0.007] | 10/15 | +0.014 [−0.034, +0.061] |
| Gated vs (i) | −0.146 [−0.244, −0.048] | [−0.252, −0.040] | −0.146 [−0.216, −0.076] | 14/15 | +0.021 [−0.053, +0.094] |
| (ii) vs (i) | −0.148 [−0.259, −0.037] | [−0.268, −0.028] | −0.149 [−0.230, −0.067] | 14/15 | +0.011 [−0.038, +0.060] |

- **The data bound, stated per comparison**:
  - Gated vs (ii) and vs (iii): the Bouckaert–Frank intervals allow differences up to 0.035 and 0.037 Brier in either direction.
  - Gated vs (iv): up to 0.073 Brier, because mean imputation plus recalibration is noisier across folds.
  - Gated vs the matched MLP: up to 0.020 Brier.
- **Precision.** The minimum detectable ΔBrier at 80% power is 0.047 for gated vs (ii) and 0.026 for gated vs MLP. ±0.035 Brier is about 15% of the prevalence-only Brier and spans Brier skill from about 0.19 to 0.50 around the gated network's 0.148, which is practically meaningful. Non-significance alone is weak evidence of equivalence.
- **Participant-level vs Bouckaert–Frank.** The participant-level intervals are narrower (e.g. gated vs (ii): [−0.018, +0.023] vs [−0.031, +0.035]). They do not include the fold-to-fold variance of retraining, so we treat them as the lower end of the plausible width and Bouckaert–Frank (df 14) as the upper end.

**Table 3. Post-hoc equivalence (reduced condition, raw).** SESOI ±0.02 Brier, ±0.03 AUROC. "Bound attained" is the smallest symmetric margin at which TOST would pass (α = 0.05). **For equivalence claims the widest scheme, B–F with df = 14, is the one to read**: an anti-conservative interval makes TOST easier to pass (§4.5).

| Comparison | Brier TOST p (B–F df 99 / df 14 / participant) | Brier bound attained (B–F df 99 / df 14 / participant) | ROPE P(\|Δ\| < 0.02), B–F | AUROC TOST p (B–F df 99 / df 14) | AUROC bound attained (B–F df 99 / df 14) |
|---|---|---|---|---|---|
| Gated vs MLP | 0.025 / 0.033 / <0.001 | 0.017 / 0.018 / 0.007 | 0.97 | 0.009 / 0.015 | 0.021 / 0.022 |
| Gated vs (ii) | 0.142 / 0.150 / 0.047 | 0.030 / 0.031 / 0.020 | 0.77 | 0.275 / 0.279 | 0.066 / 0.070 |
| Gated vs (iii) | 0.153 / 0.160 / 0.052 | 0.031 / 0.033 / 0.020 | 0.74 | 0.290 / 0.293 | 0.070 / 0.074 |
| MLP vs (ii) | 0.108 / 0.117 / 0.034 | 0.027 / 0.028 / 0.018 | 0.79 | 0.265 / 0.269 | 0.063 / 0.066 |
| Gated vs (iv) | 0.403 / 0.405 / 0.386 | 0.063 / 0.066 / 0.061 | 0.45 | 0.248 / 0.253 | 0.065 / 0.068 |

- **Reading.** Gated ≈ MLP passes equivalence at ±0.02 Brier and ±0.03 AUROC under every scheme, including the widest one (B–F df 14: p = 0.033 for Brier, 0.015 for AUROC; the Brier bound attained is 0.018, so the margin is passed narrowly). Gated vs (ii)/(iii) reaches ±0.02 Brier only under the participant-level scheme, and only just (p = 0.047 and 0.052); the Bouckaert–Frank bound is about ±0.03; and AUROC equivalence fails at ±0.03 (bounds attained 0.066–0.070). That split is the true shape of the evidence: **the gate adds nothing over a matched masked MLP; against reduced-sensor trees we can only say there is no detectable difference.**

**Decision-rule outcome: the rule fires on near-ties, and its verdict depends on calibration and CV scheme.**

  *(Supplementary Table S1.)*

- With raw outputs the gated network trails (ii) and (iii) by 0.002 on Brier and 0.003 on ECE and leads by 0.010 on AUROC. Under per-condition Platt it edges both on all three endpoints (ΔBrier vs (ii) −0.007 [−0.040, +0.025]).
- Dropping the ECE criterion does not change any verdict.
- The first, pre-fix evaluation reported the gated network as worse on all three endpoints; that came from the implementation errors in Appendix A and is superseded.

### 5.2 Gate × masking factorial

**Figure 1** (`results/fig_factorial_reduced.png`): reduced-condition Brier for the four network cells, with LightGBM (i)/(ii) as the tree reference.

**Table 4. Factorial contrasts, reduced condition, 20× repeated CV.** Negative = the first-named factor (or, for the interaction, masking in the gated network more than in the MLP) lowers Brier. The count is the number of participants with a negative value.

| Contrast | Raw: B–F 95% CI (df 99) | Raw: B–F 95% CI (df 14) | Raw: participant-level 95% CI (A better) | Per-condition Platt: B–F 95% CI (df 99) |
|---|---|---|---|---|
| Interaction (G − G_nomask) − (M − M_nomask) | +0.013 [−0.042, +0.069] | +0.013 [−0.046, +0.073] | +0.014 [−0.007, +0.034] (6/15) | +0.000 [−0.027, +0.027] |
| Main effect of gate | −0.005 [−0.036, +0.026] | −0.005 [−0.038, +0.029] | −0.005 [−0.014, +0.004] (10/15) | +0.002 [−0.016, +0.019] |
| Main effect of masking | −0.044 [−0.088, −0.001] | −0.044 [−0.091, +0.003] | −0.044 [−0.070, −0.019] (12/15) | −0.013 [−0.040, +0.014] |
| Gate, never masked: gated vs MLP, no masking | −0.011 [−0.067, +0.045] | −0.011 [−0.072, +0.049] | −0.012 [−0.030, +0.007] (10/15) | +0.001 [−0.027, +0.030] |
| Gate, masked: gated vs MLP | +0.002 [−0.016, +0.020] | +0.002 [−0.017, +0.021] | +0.002 [−0.004, +0.008] (7/15) | +0.002 [−0.013, +0.016] |
| Masking, gated network | −0.038 [−0.089, +0.014] | −0.038 [−0.093, +0.018] | −0.037 [−0.060, −0.015] (11/15) | −0.013 [−0.041, +0.015] |
| Masking, MLP | −0.051 [−0.102, +0.001] | −0.051 [−0.106, +0.005] | −0.051 [−0.083, −0.019] (12/15) | −0.013 [−0.045, +0.019] |

- **Cells (raw, reduced Brier):** gated + masking 0.148; MLP + masking 0.146; gated, no masking 0.186; MLP, no masking 0.197. The never-masked tree (i) is at 0.294 and the masked tree (ii) at 0.146.
- **Both effects, read under the same three schemes.** Earlier revisions read the gate null under the widest scheme and the masking effect under the narrowest; that was a double standard. Side by side (Table 4):
  - *Participant-level (narrowest, df 14 on 15 participant means).* Masking −0.044 [−0.070, −0.019], 12/15 participants, sign-flip p = 0.002. Gate −0.005 [−0.014, +0.004], TOST at ±0.02 p = 0.001. Both effects are "demonstrated": masking helps and the gate is within ±0.02.
  - *Bouckaert–Frank, df 99.* Masking [−0.088, −0.001], excluding 0 by 0.001; gate [−0.036, +0.026], TOST p = 0.164 (not equivalent within ±0.02).
  - *Bouckaert–Frank, df 14 (the scheme read for equivalence).* Masking [−0.091, +0.003] **includes 0**; gate [−0.038, +0.029], TOST p = 0.172.
  - Within a single network the masking effect includes 0 under both Bouckaert–Frank schemes (gated [−0.089, +0.014] at df 99 and [−0.093, +0.018] at df 14; MLP [−0.102, +0.001] and [−0.106, +0.005]).
  - **At one standard, neither "masking helps the networks" nor "the gate does not" is demonstrated:** under the narrow scheme both are; under the widest neither is. What is robust is the **tree** result, (ii) vs (i) −0.148 [−0.259, −0.037] (df 14: [−0.268, −0.028]; participant-level [−0.230, −0.067], 14/15), which excludes 0 under every scheme. We therefore state the masking benefit for the networks as "directionally consistent (−0.04, 12/15 participants) but not resolved at n = 15", and the gate null as "no detectable effect", with equivalence resting on the gated-vs-MLP comparison (Table 3), not on the gate main effect.
- **The gate does not substitute for masking.** In the never-masked regime the gated network beats the never-masked MLP by only −0.011 [−0.067, +0.045] (participant-level −0.012 [−0.030, +0.007]; 10/15). The interaction is +0.013 [−0.042, +0.069]. We cannot exclude a modest substitution benefit (up to about 0.03 at participant level), but it is small next to the masking effect.
- **An apparent gate effect in the never-masked regime is a model-family effect.** The never-masked gated network (0.196 LOSO) beats the never-masked tree (0.296). The never-masked MLP behaves like the never-masked gated network (0.197 vs 0.186 in repeated CV), so the gap is between networks with median imputation and a tree whose NaN branch was never trained, not between gate and no gate.
- **Under per-condition recalibration** the factorial flattens: the masking main effect shrinks to −0.013 [−0.040, +0.014] and the interaction to 0.000. For trees the practical statement is that *something* in the pipeline has seen missing EDA: the training data, a per-configuration model, or the calibrator. These routes have different deployment costs (§5.3, §5.7). For the networks, the evidence that this exposure matters at all is the directionally consistent, unresolved −0.044 above.
- **Intercept-only recalibration (post hoc; `run_intercept_recal.py`, `results/intercept_recal.json`): the base-rate reading holds for the tree and fails for the networks.** The earlier draft read the masking effect as "mainly calibration-in-the-large". We tested it by fitting, per condition and on the training participants' inner out-of-fold predictions only, a single logit shift (slope fixed at 1), and compared it with full per-condition Platt (intercept and slope) and, as the complement, a slope-only rescaling (intercept fixed at 0; added once the intercept-only result was in). AUROC is unchanged by any of them.

**Table 4b. Reduced-condition Brier under three per-condition calibrations (20× repeated CV, same partitions and seeds as Table 1).**

| Calibration | Gated | Gated, no masking | MLP | MLP, no masking | (i) | (ii) | Masking main effect, networks [B–F df 99] | Tree masking, (ii) − (i) [B–F df 99] |
|---|---|---|---|---|---|---|---|---|
| Raw | 0.148 | 0.186 | 0.146 | 0.197 | 0.294 | 0.146 | −0.044 [−0.088, −0.001] | −0.148 [−0.259, −0.037] |
| Intercept-only | 0.147 | 0.185 | 0.145 | 0.197 | 0.157 | 0.151 | −0.045 [−0.093, +0.004] | −0.007 [−0.050, +0.037] |
| Slope-only | 0.142 | 0.156 | 0.141 | 0.156 | 0.198 | 0.147 | −0.014 [−0.041, +0.013] | −0.052 [−0.115, +0.012] |
| Per-condition Platt | 0.142 | 0.155 | 0.141 | 0.154 | 0.162 | 0.149 | −0.013 [−0.040, +0.014] | −0.012 [−0.051, +0.026] |

  - **Trees: calibration-in-the-large.** The never-masked tree (i) needs a mean logit shift of +2.98 (SD 0.64 over folds); applying it takes its Brier from 0.294 to 0.157, as good as full Platt (0.162), and the tree masking effect falls from −0.148 to −0.007.
  - **Networks: a scale effect, not an intercept effect.** Intercept-only recalibration removes none of the networks' masking main effect (−0.044 → −0.045), whereas slope-only rescaling removes 68% (→ −0.014) and full Platt 71% (→ −0.013). The never-masked networks' mean intercept shifts are near zero (gated −0.05, MLP +0.08, with fold SDs of 1.7 and 1.1 logits), and their fitted slopes are 0.20 and 0.17, against 0.35 and 0.32 for the masked networks: all four networks are strongly over-confident without EDA, and the never-masked ones more so. The masking effect on the networks is mostly the difference in over-confidence, not in base rate. We therefore withdraw "mainly calibration-in-the-large" as a description of the networks' masking effect. It remains an accurate description of the never-masked tree's failure (§5.8): slope-only rescaling leaves (i) at 0.198 and its tree masking effect at −0.052 [−0.115, +0.012], while the logit shift alone recovers it.
  - **Discrimination.** With EDA missing, every reasonable model's reduced AUROC lies between 0.854 and 0.892 (Table 1), and no paired AUROC difference in Table 2 is distinguishable from zero. The intervals are wide: masking vs no masking is +0.015 [−0.036, +0.065] (gated) and +0.028 [−0.034, +0.090] (MLP), so a discrimination gain of 0.06–0.09 cannot be excluded.
  - Under label-free per-subject normalisation (§5.4) even the never-masked tree (i) no longer fails without EDA, though that normalisation is not deployable as specified (it uses the participant's whole session).

### 5.3 Complete condition and the deployment trade-off

*(Supplementary Table S2.)*

- **The networks underperform trees when EDA is present.** Complete-condition Brier is 0.072 for (i), 0.093 for (ii), 0.130 for the gated network and 0.141 for the masked MLP. Gated − (i) is +0.058; the B–F interval includes zero, but the participant-level interval does not (2/15 participants favour the gated network). Nested tuning does not close the gap (tuned gated 0.129, tuned MLP 0.139, tuned (ii) 0.098).
- **The networks barely use EDA on the default features.** Adding EDA improves the gated network by about 0.019 Brier (0.148 → 0.130) and LightGBM by about 0.074 ((iii) 0.146 → (i) 0.072). This deficit is largely a feature-scaling problem, not a capacity one: under label-free per-subject z-scoring (§5.4) the networks use EDA almost as well as the trees (complete-condition Brier: gated 0.052, MLP 0.050, (ii) 0.046; gated − (ii) +0.005 [−0.017, +0.027]), and the gated-vs-MLP null still holds there (complete +0.002 [−0.011, +0.015]; reduced −0.002 [−0.015, +0.011]). So the null is not only a statement about networks that ignore EDA.
- **The cost of masking for trees.** Masking 50% of training EDA costs LightGBM +0.021 complete-condition Brier ((ii) − (i), [−0.031, +0.073]). The cost grows with the masking rate (complete Brier 0.074 / 0.079 / 0.093 / 0.128 at p = 0.1 / 0.25 / 0.5 / 0.75), while the reduced-condition benefit saturates early (0.153 / 0.149 / 0.146 / 0.145). The networks pay no such cost (gated complete 0.133 / 0.134 / 0.130 / 0.126). The training masking rate should match expected deployment missingness, not default to 50%.
- **Deployment view.** A per-configuration system (complete-trained LightGBM with EDA, switching to ECG-only LightGBM without it) scores 0.072 / 0.146; the single gated network scores 0.130 / 0.148. The reduced-condition difference is −0.002 [−0.037, +0.033] ((iii) − gated); the complete-condition difference is the Gated vs (i) row of Supplementary Table S2. Switching between per-configuration models matches the gated network without EDA and is better with it, at the cost of maintaining two models.
  - **Evidence for "two simple trees and a switch".** Across the study the per-configuration system is never detectably worse than any fusion architecture and is nominally best: complete-condition Brier 0.072 against 0.130 (gated), 0.141 (MLP) and 0.093 ((ii)); lowest point-estimate Brier in all 19 mixed-test scenarios (§5.13); and the fewest false alarms with EDA present at the high-sensitivity point, 6.9 against 16.2 per hour of rest for the gated network (§5.9). Without EDA it is tied with every other system (Tables 1, 6).
  - **The caveats stay attached.** (a) Most of these margins are point estimates: gated − (i) in the complete condition is +0.058 [−0.012, +0.127] under Bouckaert–Frank (the participant-level interval excludes 0), and the per-configuration lead in the mixed-test run is not distinguishable from 0 against (ii) in any of the 19 scenarios under Bouckaert–Frank. (b) The mixed-test design favours it, because it applies the complete-trained tree to every window that has EDA. (c) Its lead over (ii) shrinks from 0.024–0.029 to at most 0.012 when (ii) is not given an extra 50% training mask (§5.13, Table 7d): this is consistent with the tree's masking cost (§5.3) rather than the architecture explaining it, though we did not test that separately. (d) It needs two models and an EDA-presence switch, and a labelled calibration set for each configuration if it is to be recalibrated. (e) All of this is stress-vs-baseline on 15 laboratory participants.

### 5.4 Robustness of the gating null

*These checks reuse the same 15 participants and partitions, so they are sensitivity analyses, not replications. The two per-subject-normalisation bullets below use the participant's whole session (both conditions) to centre every feature, which is not available at deployment and implicitly encodes the stress/baseline mix; read them as an **upper bound** on what better-scaled features can do, not as deployable robustness evidence. A causal (expanding, past-only) normalisation is the deployable test and has not been run (Appendix G).*

*(Supplementary Table S3.)*

- **Capacity sweep** (gated vs MLP at four matched sizes, 273–9,345 parameters): every ΔBrier lies within ±0.024; TOST at ±0.02 passes at the three larger sizes and narrowly misses at the smallest (p = 0.052).
- **Nested tuning** moves no model by more than 0.004 Brier; tuned gated vs (ii) +0.004 [−0.028, +0.036], vs MLP +0.005 [−0.019, +0.029]. The modal gated configuration was chosen in only 25/100 folds: the tuning surface is flat. With tuning, the rule fires under all three calibration modes.
- **Masking rate.** Gated ≈ MLP at every rate (ΔBrier +0.000 to +0.002).
- **HRV restricted to mean NN and RMSSD.** Dropping the four features that are unreliable at 60 s *improves* both networks (gated 0.148 → 0.140, MLP 0.146 → 0.142) and leaves the trees unchanged. Gated vs MLP stays equivalent (−0.003 [−0.018, +0.013]). Gated vs (ii) moves to −0.007 [−0.034, +0.020], and with raw outputs the decision rule no longer fires. This is another near-tie flip.
- **Per-subject normalisation.** Label-free per-subject z-scoring raises every model's reduced AUROC to about 0.93–0.94; PULSE-style baseline-referenced z-scoring raises it to about 0.95. Under both, the gated network, MLP, (ii) and (iii) lie within 0.006 Brier of each other, and gated vs (ii) passes TOST at ±0.02. The negative result is not an artefact of a weak-feature regime. *(Under baseline-referenced normalisation, (iv) degenerates (Brier 0.606), because training-mean imputation inserts a stress-level EDA value into every reduced window; we do not interpret it.)*
- **Per-subject normalisation is where the networks use EDA.** With EDA present, label-free z-scoring brings the networks to within 0.005 Brier of the masked tree (complete Brier: gated 0.052, MLP 0.050, (ii) 0.046, (i) 0.040; gated − (ii) +0.005 [−0.017, +0.027], participant-level +0.005 [−0.014, +0.024]), against a +0.037 gap on the default features (Supplementary Table S2). Gated and MLP stay within 0.002 Brier in both conditions (complete +0.002 [−0.011, +0.015], TOST p = 0.004). This is the regime in which the gate has work to do, and it still adds nothing detectable.
- **The calibration-in-the-large failure disappears.** Under label-free z-scoring the never-masked tree (i) reaches reduced Brier 0.073, level with (ii) ((ii) − (i) −0.000 [−0.013, +0.012]): when every feature is centred on the participant's own mean, a missing EDA value no longer shifts the predicted base rate. Under baseline-referenced z-scoring, (i) partly fails again (0.099; (ii) − (i) −0.030 [−0.068, +0.008]). A plausible reason, not tested: the tree's "NaN" EDA is delivered as 0 (§4.3), and after centring 0 equals the participant's own mean.

### 5.5 Heart-rate dependence

**Table 5. LOSO, reduced condition, raw, participant-bootstrap 95% CIs (BCa).**

| Model | Reduced AUROC [95% CI] | Within-subject AUROC [95% CI] |
|---|---|---|
| Gated + dropout | 0.873 [0.771, 0.934] | 0.903 [0.816, 0.960] |
| Non-gated MLP | 0.878 [0.775, 0.937] | 0.899 [0.809, 0.957] |
| (ii) NaN-trained | 0.860 [0.709, 0.938] | 0.870 [0.702, 0.943] |
| (iii) ECG-only | 0.869 [0.695, 0.940] | 0.899 [0.731, 0.961] |
| HR-only | 0.820 [0.673, 0.931] | 0.867 [0.749, 0.952] |
| Gated, HR removed | 0.692 [0.540, 0.824] | 0.662 [0.508, 0.793] |
| Gated, HR-residualised (HR removed + within-participant residualisation) | 0.600 [0.500, 0.699] | 0.616 [0.516, 0.716] |
| (ii) NaN-trained, HR removed | 0.681 [0.494, 0.828] | 0.668 [0.486, 0.807] |
| (iii) ECG-only, HR removed | 0.674 [0.493, 0.815] | 0.674 [0.503, 0.809] |

- **A single feature does most of the work.** Heart rate alone reaches 0.82 AUROC; it carries about 86% of the gated network's above-chance pooled AUROC and about 91% of its within-subject AUROC.
- **Removing heart rate costs about half the above-chance pooled AUROC in every family** (retained: gated 51%, (ii) 50%, (iii) 47%) and about 60% of the within-subject above-chance AUROC (gated 0.903 → 0.662). Discrimination does not collapse to chance (0.67–0.69 pooled).
- **Removing within-participant heart-rate covariation leaves little** (HR-residualised gated network 0.600). Most of what HRV adds without EDA is heart rate expressed in other units.

### 5.6 Condition-order, posture and movement confounds

Results from `results/order_confound.json` (LOSO, extended table, raw).

**A. Stress vs amusement (order-balanced: 7 participants had stress first, 8 amusement first).**

*(Supplementary Table S4.)*

- **Discrimination without EDA survives the order-balanced contrast** (reduced within-subject AUROC 0.89–0.95 for the missingness-aware models and HR-only), and does not depend on which protocol version a participant had.
- **It is again heart rate.** HR-only alone reaches 0.938 within-subject; removing HR drops the models to 0.59–0.70. Heart rate is higher under the TSST than during amusement (stress-minus-amusement mean NN, median −216 ms).
- The HR mechanism is therefore not an order artefact. Note that this contrast tests a different question (TSST vs a positive-valence film, not stress vs rest).

**B. Negative control: baseline-vs-stress models applied to later, calm conditions.** Fraction of windows predicted "stress" (p ≥ 0.5), held-out participants, raw.

*(Supplementary Table S5.)*

- **The detectors are not time-since-start detectors.** Applied to the held-out participant's later, calm conditions, every missingness-aware model labels amusement and meditation windows as stress at about its baseline false-positive rate (0.10–0.28 vs 0.14–0.22). Stress windows are labelled at 0.65–0.75 (0.86–0.88 with EDA present). At participant level, at most 4 of 15 participants have more than half of any calm block labelled stress (stress: 9–13 of 15).
- (i) with EDA missing labels nothing as stress (its calibration-in-the-large failure, §5.8).

**C. Baseline vs meditation (both low arousal, at different times).** Complete condition, LOSO.

*(Supplementary Table S6.)*

- **No evidence of tonic EDA drift.** EDA alone does not separate baseline from meditation (within-subject AUROC 0.42, CI includes 0.5). Neither does heart rate (0.44).
- The ECG-only model and the gated network do separate them (within-subject 0.75–0.76; the complete-trained LightGBM 0.62), most likely through RMSSD and the other HRV features, which guided slow breathing changes. This is a physiological difference, not drift; meditation is not a pure null for ECG.

**D. Drift within the baseline block.** Within the 20-minute baseline, neither tonic EDA nor mean NN trends with time (median per-participant Spearman ρ +0.02 and −0.05; 8/15 and 7/15 positive). The one clear order effect is participant-level: S10's heart rate was already elevated at the start of the session (§5.7).

**E. Posture and movement** (`run_posture_movement.py`, `results/posture_movement.json`; LOSO on the 434 primary windows, LightGBM with default settings, participant-cluster percentile bootstrap, B = 2,000). Chest-ACC features per window: movement (magnitude SD and per-axis SD) and posture (per-axis mean, i.e. the gravity vector).

*(Supplementary Table S7.)*

- **Movement discriminates better than heart rate.** Chest movement alone reaches 0.945 pooled AUROC, against 0.820 for HR-only. The paired contrast (`run_movement_checks.py`, `results/movement_checks.json`; post hoc, participant-cluster percentile bootstrap, B = 2,000) is movement-only − HR-only +0.125 [+0.005, +0.243] pooled AUROC, +0.091 [+0.003, +0.188] within-subject AUROC and −0.104 [−0.185, −0.031] Brier. The lower bounds sit close to 0, so "better" is narrow with 15 participants. Trunk tilt alone does not separate stress from baseline (0.574), because WESAD participants sat upright or reclined in different ways at baseline.
- **Heart rate adds little beyond movement.** ACC + HR − ACC: within-subject AUROC +0.027 [+0.006, +0.051], pooled AUROC +0.016 [−0.029, +0.057], Brier −0.001 [−0.034, +0.043]. Adding all ECG features gives within-subject +0.032 [+0.010, +0.060].
- **Residualising heart rate on movement and posture is not identifiable here.** Within each participant, ACC features explain a median 91% of the variance in mean NN (range 51–98%), and the residual separates stress from baseline only weakly (within-subject AUROC 0.546 [0.516, 0.581], against 0.933 for raw mean NN). Because movement and condition are nearly collinear within each participant (movement within-subject AUROC 0.925 as a single feature), this residualisation removes the condition signal whatever its cause, and the regression on about 29 windows with seven predictors overfits. It shows that the two cannot be separated in this design, not that heart rate carries no information.
- **Lower-motion stress windows.** Keeping only stress windows whose movement does not exceed the participant's highest baseline value leaves 75 of 152 stress windows; in the 10 participants with at least three such windows, heart rate still separates stress from baseline (within-subject AUROC 0.873 [0.732, 0.976], vs 0.900 on all their windows). Movement still differs in this subset (0.856), so it is a partial, not a matched, control.
- **Stress vs amusement is not posture-matched either.** Trunk tilt separates stress from amusement within every participant (within-subject AUROC 0.970), so the order-balanced contrast in A does not remove the posture difference. The heart-rate difference is large (stress − baseline median 19.7 bpm, range 1.4–61.6; above 10 bpm in 10/15 participants), but a chest accelerometer cannot tell how much of it is orthostatic.
- **Reading.** Without EDA, the models detect a response that is cardiac and motor at once. Heart rate carries a small, detectable within-subject increment over movement, and it survives in lower-motion stress windows, but this design cannot separate psychological arousal from the standing, speaking and gesturing of the TSST. Contribution 2 is stated accordingly.

### 5.7 Participant-level heterogeneity (exploratory)
- **Figure 2** (`results/fig_per_subject_forest.png`) shows each participant's within-subject AUROC under LOSO with EDA missing, next to that participant's HR response (stress-minus-baseline drop in mean NN: 12–448 ms; 0.45–13.1 pooled within-condition SDs).
- **The HR-only correlation is close to mechanical.** For the HR-only model, within-subject AUROC is the AUC of mean NN separating that participant's stress and baseline windows, a monotone function of the standardised separation. We therefore report it descriptively, with CIs, and lead with the non-HR models:

  *(Supplementary Table S8.)*

  The three HR-containing correlations share participants and are not independent replications.
- **Responders, not movers (post hoc, `results/movement_checks.json`).** The "responders vs non-responders" pattern could in principle be a movers vs non-movers pattern, since chest movement discriminates so well (§5.6E). We correlated each participant's within-subject AUROC with their standardised movement response (stress − baseline chest-ACC magnitude SD over the pooled within-condition SD; median 1.73, range 0.15–4.17). For the gated network the correlation with HR response is 0.88 [0.66, 0.96] and with movement response 0.19 [−0.36, 0.64]; for the MLP 0.81 [0.51, 0.94] and 0.14 [−0.40, 0.61]; for HR-only 0.92 [0.77, 0.97] and 0.37 [−0.17, 0.74]. Controlling for movement response leaves the gated network's HR-response partial correlation at 0.92, while the movement-response partial given HR is −0.59. **A negative partial is expected from a model that reads heart rate:** AUROC rises with HR response (0.88), HR response and movement response are positively correlated (0.50), so a participant who moved a lot for a given HR response is one whose heart rate rose little relative to their movement, and a model that reads heart rate classifies such a participant worse (the MLP's partial is −0.51, (iii)'s −0.11 and, for HR-only, which has no movement input at all, −0.25, which gives a sense of how large a partial can be by construction and noise at n = 15). We read it as that interpretation or noise, not as evidence that movement degrades the models, and we did not test it further. The one model with a movement-response correlation whose interval excludes 0 is (ii) (0.54 [0.03, 0.82]), one of eight comparisons with no adjustment. HR response and movement response are themselves only moderately correlated (0.50 [−0.02, 0.80]). With 15 participants these are descriptive, but they do not support the movers-vs-non-movers reading of the heterogeneity.
- **Descriptive reading.** An HR-reliant detector works to the extent that HR separates the conditions. In 4/15 participants (27%; S2, S9, S10, S15) heart rate barely moved (drop 12–51 ms), and performance there is erratic and model-dependent.
- **Within-subject AUROC precision.** With about 10 stress and 19 baseline windows, the Hanley–McNeil SE is about 0.07 at AUC 0.90 and 0.11 at AUC 0.70, before accounting for autocorrelation between consecutive windows. Per-participant AUROCs are noisy.
- **S10.** Its within-subject AUROC for the NaN-trained LightGBM is 0.16 (Hanley–McNeil SE 0.07, about 4.6 SE below 0.5). Consecutive windows are autocorrelated, so the effective sample is smaller than the window count and this SE is too small; the inversion is probably, not certainly, beyond noise. The extended table explains it. S10's chest-ECG heart rate during baseline was 99 bpm, the highest baseline of all 15 participants, against 106 bpm under stress and 76 bpm in amusement and meditation. R-peaks were clean (median 0% successive NN jumps > 20%), movement was low (chest ACC SD 0.007 g), and the wrist BVP agreed in the later conditions. Its baseline RMSSD (17 ms) was lower than under stress (34 ms), so tree models that learned "low RMSSD means stress" ranked S10's windows backwards (HR-removed models 0.19–0.22). The most plausible reading is anticipatory arousal at the start of the session, the first condition in every WESAD protocol: the condition-order confound (§5.6) visible in a single participant.
- **Clinical reading.** See §6 ("What the proxy cannot say").

### 5.8 Calibration under missing EDA
- **The complete-trained LightGBM fails in calibration-in-the-large.** With EDA missing, its reduced-condition intercept is +3.10 (slope 0.96; AUROC 0.85) while its slope stays near 1. It reads the absence of the sensor as the absence of arousal. That is the mechanism behind the earlier drafts' headline.
- **Every better model is over-confident without EDA** (calibration slopes 0.42–0.59).
- **The shared calibrator over-corrects the networks;** per-condition calibration fixes this (LOSO, pooled, reduced):

  *(Supplementary Table S9.)*

  - **Recalibration mostly rescues the complete-trained LightGBM:** it recovers 86% (Brier 0.296 → 0.163; best missingness-aware model 0.142) of the gap between its raw Brier and the best missingness-aware model. "Mostly", not fully: masking at training time, per-configuration models and recalibration are different mechanisms with different deployment costs, and we do not lump them.
- **ECE estimator lesson.** The originally implemented "debiased ECE" took the minimum over six bin counts of a zero-clipped estimate: 18/40 reduced-condition per-fold values were exactly zero, and it ranked the HR-removed gated network as best although by conventional ECE it was the second-worst. With ECE_SWEEP, low-discrimination models can still post competitive ECE (the HR-removed NaN-trained LightGBM has among the lowest ECE_SWEEP at AUROC 0.68). **ECE without discrimination and slope is uninformative, and at about 40–90 windows its estimator choice can reverse a ranking.** SmoothECE, which has no binning choices, agrees with ECE_SWEEP on the repeated-CV ordering.

### 5.9 Selective prediction (abstention)
- **Figure 3** (`results/fig_risk_coverage_reduced.png`): risk–coverage with EDA missing (LOSO, pooled, raw), with the random-abstention and oracle references.
- **Against references.** At full coverage the missingness-aware models err on 19–21% of windows (the random-abstention AURC). Abstaining down to 60% coverage roughly halves that (gated 0.096, MLP 0.092, (ii) 0.096, (iii) 0.107), while the oracle that abstains on errors first would reach 0 at 60% coverage. Excess AURC over the oracle (E-AURC) is 0.065 (gated), 0.062 (MLP), 0.078 ((ii)) and 0.141 ((iii)). Measured as (random − AURC)/(random − oracle), the gated network, MLP and (ii) capture about 60% of the possible benefit of abstention (63%, 64%, 59%) and (iii) about 20%, in pooled LOSO predictions.
- **No model abstains detectably better than another under repeated CV:**

  *(Supplementary Table S10.)*

- **The inverted-confidence result does not replicate.** In the pooled LOSO predictions, the ECG-only LightGBM's most confident 10% of windows were wrong 23% of the time, at or above its average error, echoing the class-dependent miscalibration failure of Lechuga López et al. Under repeated CV the gated-vs-(iii) AURC difference is −0.026 [−0.113, +0.061], so we do not claim it.
- **Clinical caution.** The weak heart-rate responders (§5.7) contribute confident errors; abstention on model confidence will not catch them.

**Abstention with an action model (post hoc; `run_abstention_action.py`, `results/abstention_action.json`; not in `prespec_round5.md`).** The risk–coverage analysis above scores abstention as removal. A detector that must "abstain or escalate" has to do something with the abstained windows. Table 8 scores, with EDA missing, sensitivity over all stress windows and false alarms per hour of rest over all baseline windows (per-condition Platt, alert threshold 0.5, confidence max(p, 1 − p), 20× repeated CV means; descriptive, no inference), when abstained windows are dropped, not alerted, or alerted.

**Table 8. Abstention under three actions, EDA missing: sensitivity / false alarms per hour of rest.**

| System | No abstention | 60% coverage, dropped | 60% coverage, abstain → no alert | 60% coverage, abstain → alert | 80% coverage, dropped | 80% coverage, abstain → no alert | 80% coverage, abstain → alert |
|---|---|---|---|---|---|---|---|
| Gated + dropout | 0.613 / 5.6 | 0.596 / 3.0 | 0.256 / 1.8 | 0.877 / 18.2 | 0.625 / 3.9 | 0.438 / 3.2 | 0.768 / 10.5 |
| Non-gated MLP, same masking | 0.625 / 5.6 | 0.654 / 3.1 | 0.277 / 1.9 | 0.887 / 18.6 | 0.649 / 3.9 | 0.455 / 3.1 | 0.777 / 10.7 |
| (ii) NaN-trained LightGBM | 0.618 / 6.8 | 0.519 / 4.5 | 0.200 / 3.0 | 0.899 / 16.9 | 0.615 / 5.6 | 0.412 / 4.8 | 0.775 / 11.1 |
| Per-configuration ((iii) without EDA) | 0.643 / 6.7 | 0.501 / 4.5 | 0.194 / 3.2 | 0.889 / 17.1 | 0.629 / 5.5 | 0.429 / 4.7 | 0.783 / 11.3 |

- **Dropping the abstained windows flatters the detector.** Treated as "no alert", abstention at 60% coverage cuts the gated network's sensitivity from 0.613 to 0.256, because the abstained windows are mostly stress windows (the asymmetry in the predictions below). The false alarms fall from 5.6 to 1.8 per hour, but at the price of missing most stress windows.
- **Treated as "alert", abstention buys sensitivity at a cost.** At 60% coverage the gated network reaches 0.877 sensitivity with 18.2 false alarms per hour of rest; at 80% coverage, 0.768 with 10.5. That is the same order as moving the threshold (Table 6), and we have not shown that abstention is a more efficient route than a lower threshold.
- **Reading.** "Abstention halves the error" (above) is a statement about the retained windows. As a safety mechanism, abstention has no free benefit: it moves along the same sensitivity / false-alarm trade-off, and which side of it you land on depends on the action. The four systems stay close to one another under every action, so the near-tie conclusions are unchanged.

**Operating points and decision curves (round 5, pre-specified; §4.8).**

**Table 6. Held-out operating points with EDA missing, 20× repeated CV (mean over 100 folds).** Thresholds chosen on training participants' out-of-fold predictions only. False alarms per hour of rest = (1 − specificity) × 60.

| System | High-sensitivity point: sensitivity / specificity | False alarms per hour of rest | High-specificity point: sensitivity / specificity | False alarms per hour of rest | Net benefit at threshold 0.35 |
|---|---|---|---|---|---|
| Gated + dropout | 0.904 / 0.594 | 24.3 | 0.556 / 0.921 | 4.7 | 0.208 |
| Non-gated MLP, same masking | 0.910 / 0.594 | 24.4 | 0.584 / 0.916 | 5.0 | 0.208 |
| (ii) NaN-trained LightGBM | 0.904 / 0.672 | 19.7 | 0.462 / 0.919 | 4.9 | 0.211 |
| Per-configuration ((iii) without EDA) | 0.905 / 0.650 | 21.0 | 0.490 / 0.916 | 5.1 | 0.210 |

- **Figure 4** (`results/fig_decision_curves.png`): decision curves with EDA missing and present, against treat-all and treat-none.
- **What the near-ties look like at the bedside.** Without EDA, every system that reaches 90% sensitivity on held-out participants raises 20–24 false alarms per hour of rest. At about 92% specificity it still raises about 5 per hour and misses about half of stress windows (sensitivity 0.46–0.58). The thresholds transfer well from training to held-out participants (achieved sensitivity 0.904–0.910, specificity 0.916–0.921). With EDA present the per-configuration system is clearly better at the high-sensitivity point: 6.9 false alarms per hour against 16.2 for the gated network.
- **Pre-stated predictions** (primary scheme: Bouckaert–Frank df 99):
  - *Gated ≈ MLP within ±0.05 sensitivity/specificity and ±0.01 net benefit:* **not confirmed** under the primary scheme. TOST passes only for specificity at the high-specificity point (p = 0.002); the other intervals are wider than the margins (net benefit at 0.35: −0.000 [−0.026, +0.026]). Under participant-level inference 6 of the 7 endpoints pass.
  - *Gated vs (ii) and vs per-configuration intervals include 0:* **confirmed** for all 14 endpoint × comparison pairs (net benefit at 0.35: gated − (ii) −0.004 [−0.040, +0.033]). At participant level the gated network has higher sensitivity than (ii) at the high-specificity point (+0.090 [+0.023, +0.158]).
  - *With EDA present, per-configuration net benefit exceeds the gated network's at 0.35:* **direction confirmed, not significant** under the primary scheme (gated − per-configuration −0.049 [−0.109, +0.011]; participant-level −0.049 [−0.093, −0.005]).
  - *At 60% coverage stress windows are abstained on more than baseline windows:* **confirmed** for every system (abstention rate in stress minus baseline windows +0.33 to +0.47, every interval above 0). Abstention preferentially drops stress windows: at 60% coverage the gated network keeps 38% of stress windows and 73% of baseline windows.
- **What this says about the ±0.02 Brier margin.** Even the gated-vs-MLP pair, equivalent at ±0.02 Brier, cannot be shown equivalent at ±0.01 net benefit with 15 participants. Against the trees the data allow net-benefit differences of up to about 0.04, i.e. four net true positives per 100 windows. The Brier margin is therefore not a stricter standard than a one-in-100 decision margin, and "no detectable difference" against the trees should be read at that coarser resolution.
- **Prevalence caveat.** Stress prevalence is 35% by design, so PPV, NPV and net benefit do not transfer to deployment; sensitivity, specificity and false alarms per hour are the portable quantities.

### 5.10 Wrist-only (E4) replication (not pre-registered)
Same models, same 434 windows and the same 20 partitions, with every feature computed from the wrist E4 instead of the chest (`results/repeated_cv_wrist.json`); the reduced condition removes the wrist EDA and coupling features.

- **The gating null replicates.** Gated vs MLP +0.001 [−0.013, +0.014] (TOST p = 0.003); never-masked gated vs never-masked MLP +0.005 [−0.034, +0.044]; interaction −0.005 [−0.044, +0.035]. Masking again carries the effect (main effect −0.026 [−0.056, +0.004]; participant-level −0.026 [−0.044, −0.007]).
- **Against trees the direction reverses but stays within noise.** The networks lead (ii) and (iii) by about 0.02 Brier (gated vs (ii) −0.021 [−0.052, +0.009]; 12/15 participants) and by about 0.035 AUROC; with raw outputs the decision rule does **not** fire. This is a further near-tie flip, not evidence for gating: the matched MLP leads by the same amount.
- **But the wrist "without EDA" signal is not cardiac.** Reduced-condition AUROC is 0.93 for the networks, higher than on the chest, while wrist HR-only reaches only 0.64. Wrist beat detection fails under the TSST: wrist-vs-chest HR median absolute error is 14.6 bpm during stress (0.6–2.2 bpm in the calm conditions), and 64% of successive wrist NN intervals jump by more than 20% under stress (41% at baseline). The wrist "HRV" features do not measure HRV: wrist RMSSD is uncorrelated with chest RMSSD (Spearman ρ −0.01 at baseline, 0.11 under stress) and tracks the beat-artefact fraction (ρ 0.83 and 0.73). The artefact fraction alone separates stress from baseline (pooled AUROC 0.81), as does chest movement (0.91). **Without EDA, the wrist models are largely detecting motion-corrupted PPG during speech and arithmetic**, a protocol shortcut that would not transfer to panic detection.
- **Reading.** The gating conclusion holds on a second sensor configuration. The mechanism does not: on the wrist the fallback is an artefact detector, not heart rate. Any wrist-based claim needs signal-quality gating first.

### 5.11 Signal quality and minimal re-implementation
- **Chest ECG quality.** No chest window has an implausible NN interval (outside 300–2000 ms). 3.5% of windows have more than 10% successive NN jumps above 20% (maximum 26%), so chest R-peak detection is clean and SQI gating would not change the chest results. Chest movement differs strongly between conditions (accelerometer-magnitude SD alone separates stress from baseline, pooled AUROC 0.91): TSST involves standing, speaking and gesturing. Movement is not a model input, but it is a plausible contaminant of any physiological feature computed during the stressor. Wrist PPG quality is poor under stress (§5.10).
- **Minimal re-implementation.** `reimpl/minimal_primary.py` rebuilds the primary comparison from the raw `.pkl` files without importing any project module. It uses its own Pan-Tompkins-style R-peak detector, time-domain HRV only, a different EDA decomposition (4th-order 1 Hz low-pass), no coupling features, mean/SD standardisation, its own gated network (721 parameters) and MLP (761), 200 epochs, and 5 seeds averaged, all under LOSO. The agreement criterion was fixed in the script before it was run.
  - Reduced-condition Brier: gated 0.135, MLP 0.135, (ii) 0.137, (iii) 0.130, vs 0.146 / 0.145 / 0.146 / 0.142 in the main pipeline (all within 0.03).
  - Every pairwise reduced-condition ΔBrier lies within ±0.007, with participant-bootstrap CIs that include 0 (gated − MLP −0.000 [−0.007, +0.008]; gated − (ii) −0.002 [−0.016, +0.012]).
  - It also reproduces the networks' complete-condition deficit (gated 0.129 vs (ii) 0.072).
  - Agreement criterion met. This is **not** the independent re-implementation the review asks for: it was written by the same author, with an AI assistant, after seeing the main code. That remains open (Appendix G).

### 5.12 How the earlier headline arose (details in Appendix A)
- **The earlier headline** was an ECE-ratio sequence of 4.23× → 2.21× → 1.61× (LightGBM → gated network → gated network + dropout), comparing against the complete-trained LightGBM. On the frozen table the same script gives 4.50× → 2.02× → 1.15× (Appendix B).
- **Why the ratio flattered the gated network.** It rewards whichever model is worse-calibrated on complete data, and the comparator was never trained with missing EDA.
- **What fair comparators show instead** (§5.1–5.2): the advantage was a comparator artefact.

### 5.13 Informative (label- or motion-driven) EDA missingness (round 5, pre-specified)
Design in §4.8; results in `results/mnar.json`, `results/mnar.log` and **Figure 5** (`results/fig_mnar_brier.png`). Each test set mixes windows with and without EDA (30% missing), so Brier here is a mixture over both conditions and is not comparable with the reduced-condition values of Table 1. Primary scenarios (label-driven, odds ratio 4; S1 and S2) are the only ones with inference; everything else is descriptive. Inference is Bouckaert–Frank df 99 unless labelled.

**Design caveat: this pre-specified run cannot detect the shortcut it was built to find, so "no mechanism effect detected" here means "not detectable in this design".** Every model is also trained with its own 50% MCAR EDA masking (gated and MLP: modality dropout; (ii): random NaN masking) on top of the 30% informative missingness. That random masking breaks the association between the label and the missingness bit in most training rows, so near-zero shortcut indices, and the failure of C3-P1 to separate the MLP from the hard mask, are what this design would produce whether or not the risk is real. The decisive version trains with only the informative mechanism and no added MCAR masking. We ran it after seeing these results (post hoc, `python run_mnar.py --no-added-masking`, `results/mnar_nomask.json`; reported after the pre-stated predictions below, with Tables 7c and 7d).

**Table 7a. Shortcut index by training mechanism, pre-specified run (own 50% MCAR training masking on top)** (mean over held-out baseline windows of p(EDA removed) − p(EDA present), raw; 95% CI).

| Training mechanism | Gated + dropout | Non-gated MLP | (ii) NaN-trained LightGBM | Per-configuration |
|---|---|---|---|---|
| MCAR | +0.008 [−0.120, +0.136] | −0.004 [−0.153, +0.146] | +0.023 [−0.023, +0.069] | +0.088 [−0.026, +0.202] |
| Label-driven, OR 2 | +0.031 [−0.097, +0.158] | +0.017 [−0.132, +0.166] | +0.044 [−0.007, +0.095] | +0.100 [−0.014, +0.214] |
| Label-driven, OR 4 | +0.047 [−0.081, +0.175] | +0.039 [−0.116, +0.194] | +0.062 [+0.011, +0.113] | +0.110 [−0.004, +0.225] |
| Label-driven, OR 8 | +0.067 [−0.062, +0.196] | +0.069 [−0.079, +0.217] | +0.081 [+0.025, +0.136] | +0.125 [+0.008, +0.241] |
| Motion-driven, OR 2 | +0.019 [−0.106, +0.145] | +0.004 [−0.148, +0.156] | +0.035 [−0.013, +0.083] | +0.095 [−0.019, +0.209] |
| Motion-driven, OR 4 | +0.027 [−0.100, +0.153] | +0.018 [−0.124, +0.161] | +0.046 [−0.003, +0.095] | +0.102 [−0.011, +0.215] |
| Motion-driven, OR 8 | +0.037 [−0.088, +0.161] | +0.036 [−0.113, +0.185] | +0.058 [+0.010, +0.107] | +0.108 [−0.007, +0.223] |

For the per-configuration system the index is a model switch (the complete-trained tree is used where EDA is present), not a learned shortcut; it is already +0.088 under MCAR.

**Table 7b. Mixed-test Brier, pre-specified run, raw, mean over 100 folds**, with the gated − MLP contrast (B–F 95% CI; TOST p at ±0.02 under B–F / participant-level).

| Scenario | Gated + dropout | Non-gated MLP | (ii) NaN-trained LightGBM | Per-configuration | Gated − MLP | TOST p (B–F / participant) |
|---|---|---|---|---|---|---|
| S0 MCAR → MCAR | 0.135 | 0.146 | 0.123 | 0.097 | −0.011 [−0.050, +0.028] | 0.324 / 0.075 |
| S1 label OR 4 → same | 0.133 | 0.141 | 0.121 | 0.097 | −0.008 [−0.049, +0.033] | 0.282 / 0.028 |
| S2 label OR 4 → MCAR | 0.131 | 0.140 | 0.123 | 0.096 | −0.009 [−0.046, +0.027] | 0.284 / 0.054 |
| S3 MCAR → label OR 4 | 0.142 | 0.152 | 0.126 | 0.100 | −0.010 [−0.054, +0.035] | 0.326 / 0.048 |

**Pre-stated predictions** (`prespec_round5.md`, written before the run):
- *C3-P1, shortcut index above 0 for the gated network, MLP and (ii) under label-driven training, rising with the odds ratio, near 0 under MCAR training:* **confirmed in direction, small and imprecise.** The gated network rises from +0.008 (MCAR) to +0.067 (OR 8) and the MLP from −0.004 to +0.069, but their intervals include 0 throughout. Only (ii) has an interval above 0, at OR 4 and OR 8. A window with missing EDA is, on average, scored at most 0.08 higher in p(stress) than the same baseline window with EDA present.
- *C3-P2, in S1 the gated network, MLP and (ii) have lower Brier than the per-configuration system:* **not confirmed.** The per-configuration system has the lowest Brier in every one of the 19 scenarios, by 0.024 to 0.029 over the best other system (S1: 0.097 against 0.121 for (ii), 0.133 and 0.141). The prediction's own caveat (a shifted prior in the complete-trained tree) did not materialise as a handicap. The mixed-test design also favours that system, since it applies the complete-trained tree to every window that has EDA.
- *C3-P3, in S2 the three systems have higher Brier than in S0:* **not confirmed.** Changes from S0 are −0.004 (gated), −0.005 (MLP) and +0.001 ((ii)), with every interval including 0. In the two primary scenarios no system differs from its S0 value by more than 0.005, and in none of the 18 non-S0 scenarios by more than 0.012.
- *C3-P4, gated − MLP within ±0.02 Brier (TOST) in both primary scenarios:* **not confirmed under the primary scheme** (p = 0.282 in S1 and 0.284 in S2, because the B–F intervals are about ±0.04 wide in the mixed design). At participant level it passes in S1 (p = 0.028) and just fails in S2 (p = 0.054).

**Post hoc variant: no added MCAR masking (the decisive version; `run_mnar.py --no-added-masking`, `results/mnar_nomask.json`, Figure 5b `results/fig_mnar_brier_nomask.png`).** Same partitions, seeds, mechanisms, per-configuration system and inference, but the gated network and the MLP get no modality dropout (p = 0) and (ii) no random NaN masking: the only missing EDA a system sees in training is the training mechanism's own (30%). Not in `prespec_round5.md`; descriptive apart from the intervals shown.

**Table 7c. Shortcut index by training mechanism, no added masking** (same definition and layout as Table 7a; the per-configuration column is identical to Table 7a, since that system has no added masking in either run).

| Training mechanism | Gated, no added masking | MLP, no added masking | (ii), no added masking | Per-configuration |
|---|---|---|---|---|
| MCAR | +0.005 [−0.137, +0.148] | −0.007 [−0.163, +0.149] | +0.053 [−0.017, +0.123] | +0.088 [−0.026, +0.202] |
| Label-driven, OR 2 | +0.053 [−0.089, +0.195] | +0.053 [−0.113, +0.219] | +0.106 [+0.027, +0.186] | +0.100 [−0.014, +0.214] |
| Label-driven, OR 4 | +0.113 [−0.042, +0.267] | +0.108 [−0.069, +0.286] | +0.159 [+0.069, +0.249] | +0.110 [−0.004, +0.225] |
| Label-driven, OR 8 | +0.174 [+0.027, +0.322] | +0.162 [−0.003, +0.327] | +0.222 [+0.125, +0.318] | +0.125 [+0.008, +0.241] |
| Motion-driven, OR 2 | +0.031 [−0.106, +0.168] | +0.023 [−0.134, +0.181] | +0.087 [+0.019, +0.155] | +0.095 [−0.019, +0.209] |
| Motion-driven, OR 4 | +0.062 [−0.074, +0.198] | +0.051 [−0.112, +0.213] | +0.122 [+0.052, +0.192] | +0.102 [−0.011, +0.215] |
| Motion-driven, OR 8 | +0.099 [−0.042, +0.241] | +0.101 [−0.071, +0.272] | +0.178 [+0.111, +0.246] | +0.108 [−0.007, +0.223] |

**Table 7d. Mixed-test Brier, no added masking, raw, mean over 100 folds**, with the gated − MLP contrast (B–F 95% CI; TOST p at ±0.02 under B–F / participant-level) and (ii) − per-configuration (B–F 95% CI).

| Scenario | Gated, no added masking | MLP, no added masking | (ii), no added masking | Per-configuration | Gated − MLP | TOST p (B–F / participant) | (ii) − per-configuration |
|---|---|---|---|---|---|---|---|
| S0 MCAR → MCAR | 0.149 | 0.173 | 0.104 | 0.097 | −0.025 [−0.074, +0.025] | 0.575 / 0.737 | +0.007 [−0.023, +0.037] |
| S1 label OR 4 → same | 0.142 | 0.156 | 0.094 | 0.097 | −0.014 [−0.067, +0.038] | 0.413 / 0.183 | −0.003 [−0.032, +0.026] |
| S2 label OR 4 → MCAR | 0.148 | 0.164 | 0.102 | 0.096 | −0.017 [−0.066, +0.033] | 0.445 / 0.300 | +0.005 [−0.020, +0.031] |
| S3 MCAR → label OR 4 | 0.157 | 0.179 | 0.109 | 0.100 | −0.023 [−0.076, +0.031] | 0.542 / 0.643 | +0.009 [−0.026, +0.043] |

- **The design caveat was right: the shortcut is larger, and detectable, once the added MCAR masking is removed.** At label-driven OR 8 the index is +0.174 for the gated network (pre-specified run +0.067; interval now above 0), +0.162 for the MLP (+0.069; lower bound −0.003) and +0.222 for (ii) (+0.081). (ii)'s interval is above 0 at every label-driven and motion-driven odds ratio, and the gated network's only at label OR 8. Under label-driven missingness at OR 4–8 a baseline window whose EDA is missing is scored on average 0.11–0.22 higher in p(stress) by the three learned systems (motion-driven: 0.05–0.18), two to three times the pre-specified run. C3-P1 (above 0, rising with the odds ratio, near 0 under MCAR) is therefore confirmed in this variant for the gated network (+0.005 → +0.053 → +0.113 → +0.174) and the MLP (−0.007 → +0.053 → +0.108 → +0.162) in direction, with imprecise intervals, and clearly for (ii).
- **The shortcut costs little Brier even so.** In the 18 non-S0 scenarios no system differs from its S0 value by more than 0.025. Where training and test mechanisms match (S1) the learned systems *gain* (label OR 8: gated −0.017, MLP −0.025, (ii) −0.020), because missingness is then a legitimate feature; where the mechanism shifts (S2, trained informative, tested MCAR) the changes are −0.001 (gated), −0.009 (MLP) and −0.002 ((ii)) at OR 4, all with intervals including 0, and the largest S3 (trained MCAR, tested informative) increase is +0.011 (gated, label OR 8). The risk is a few hundredths of probability on a baseline window, not a Brier regime change, but it is a risk this design could detect and the pre-specified one could not.
- **The gate is not shown to be better or worse than the MLP here.** Gated − MLP is −0.014 to −0.026 across the 19 scenarios, with B–F intervals of about ±0.05 and TOST failing under both schemes in all four scenarios of Table 7d. The MLP is the weaker model in this variant (S0 Brier 0.173 against 0.146 in the pre-specified run), so the difference is not about the missingness mechanism (it is there under MCAR), is consistent with the MLP learning less from an indicator that is 0 in only 30% of training rows (not tested), and is not evidence for the gate (the interval includes 0 everywhere).
- **The per-configuration lead over (ii) mostly disappears.** In the pre-specified run that system led the best other system by 0.024–0.029 in every scenario. Without the added 50% masking, (ii) is within 0.012 of it in every scenario and lower in 3 of the 19 (S1 label OR 4, label OR 8 and motion OR 8: by 0.003, 0.010 and 0.001), with every B–F interval including 0 (Table 7d, last column). The earlier margin is consistent with the masking cost to the tree, as in §5.3 (complete-condition Brier grows with the masking rate): with the training masking rate closer to the 30% deployment missingness, (ii) is nearly as good (not tested separately). The per-configuration system remains the lowest in 16 of 19 scenarios, and ahead of both networks in all 19 at participant level.

**Reading.**
- **Label- or motion-correlated missingness at odds ratios up to 8 barely moves any system in the pre-specified design.** Brier changes by at most 0.012 from the MCAR case, and the shortcut index is a few hundredths of probability. That is what the design would show if the risk were absent and also if it were present but diluted by the added random masking (see the caveat and the variant above). There is no sign that the MLP exploits it more than the hard mask (index at OR 8: MLP +0.069, gated +0.067; no added masking: MLP +0.162, gated +0.174).
- **The gated − MLP gap is the same in every scenario (−0.003 to −0.013, MCAR included).** It sits on the rows that have EDA: in S0 the two differ by 0.014 there (0.133 vs 0.147) and are close on rows with missing EDA (0.138 vs 0.142). It is the complete-condition difference of Supplementary Table S2, not a missingness-mechanism effect. What this run cannot do is re-establish equivalence of the two networks in a mixed design: the contrast is imprecise there, and the pre-specified prediction failed.
- **Consequence for the principal finding.** Nothing here contradicts it, and nothing re-establishes it. In the pre-specified run "no mechanism effect detected" is a statement about a design that could not detect one. In the decisive variant a shortcut is detectable for the learned systems, it moves Brier by at most 0.025, and the gate-versus-MLP contrast is too imprecise (about ±0.05) to say anything in either direction. The equivalence of §5.1 was established for complete, MCAR EDA loss only.
- **Scope.** The mechanisms are simulated, the odds ratios stop at 8, and missingness is all-or-nothing for the six EDA features. Real EDA dropout (electrode contact loss during movement, sweat saturation) can be stronger and can depend on the signal itself.

### 5.14 Arrhythmia failure-mode probe on MIT-BIH (round 6, specified in `prespec_round6_mitbih.md`; added after external feedback)
Design in `prespec_round6_mitbih.md` (SHA-256 `0f0aecb676fe…`, `results/prespec_round6_manifest.json`; written before `run_mitbih_failure.py` existed, not independently timestamped); results in `results/mitbih_failure.json`, `results/mitbih_failure.log` and per-window `results/mitbih_failure_windows.csv`. Systems trained on all 434 WESAD windows (seeds 0-4, per-condition Platt) are scored with EDA missing on 1,067 non-noise 60-s windows from 43 MIT-BIH clusters (non-paced records; 338 normal-sinus control windows from 23 records, 130 AF/flutter from 6, 35 VT from 11, 12 SVT from 6). Annotated rate is beats per minute; the extractor's heart rate tracks it closely (Spearman 0.994, median absolute error 0.4 bpm).

- **The primary endpoint could not be tested.** The pre-specified floor was at least 30 windows from at least 5 records with annotated rate of 120 bpm or more; the database has 1 such window (rate distribution: median 74 bpm, 971 windows below 100, 95 at 100-119, 1 at or above 120). P1 and P2 are therefore **not testable**, and no sustained tachyarrhythmia at the rates that motivated the question is in the data.
- **P3, confirmed in point estimate.** Calibrated p(stress) rises with annotated rate: Spearman 0.68 (bit-masked network), 0.56 (MLP), 0.83 ((ii)), 0.78 ((iii)) and 0.91 (HR-only); the record-cluster interval for the MLP is [0.32, 0.73], for the others the lower bound is at least 0.50. For the two HR-removed contrasts it is -0.03 and 0.10.
- **P4, confirmed.** On normal-sinus windows the flag rate (p >= 0.5) is 0.03 (bit-masked network), 0.09 (MLP), 0.01 ((ii), (iii) and HR-only), at most 0.17 for the HR-removed contrasts.
- **Descriptive (P5).** At 100-119 bpm (95 windows, 10 clusters, 76 of the windows from records 213, 215 and 233) the bit-masked network and MLP flag 1.00, (ii) 0.94, (iii) 0.93 and HR-only 0.44. Flag rates by rhythm: AF/flutter 0.98 (bit-masked), 1.00 (MLP), 0.86 ((ii)), 0.85 ((iii)), 0.02 (HR-only); VT 1.00, 1.00, 0.91, 0.89, 0.09; SVT (12 windows) 0.92, 0.92, 0.67, 0.67, 0.17.
- **Reading (revised after external feedback): an out-of-distribution failure to abstain, not arrhythmia detection.** The systems have no rhythm class and no OOD check; the multi-feature systems respond to irregularity features that in WESAD never occur as AF produces them, and do so confidently (mean calibrated p on the 130 AF/flutter windows: bit-masked network 0.90, MLP 0.94, (iii) 0.69). Normal-sinus windows from the same database are flagged only 1-9%, which argues against pure sensor/population shift (identity AUC 0.83) without excluding that the irregularity features are out of range.
- **100-119 bpm band split by rhythm (post hoc, not in the prespecification).** The band holds 5 pure-sinus windows (all from record 213) and 90 non-sinus windows (ectopy or an annotated arrhythmia; 20 carry an SVT, AF/flutter or VT tag), 76 of the 95 from records 213, 215 and 233. Flag rates in the 90 non-sinus windows: bit-masked network 1.00, MLP 1.00, (ii) 0.97, (iii) 0.96, HR-only 0.44; in the 5 sinus windows 1.00, 1.00, 0.40, 0.40, 0.40 (too few to interpret). Sinus tachycardia cannot be separated from arrhythmia in this band, so its flagging is not attributed to rate. The SVT sample (12 windows, 6 records) is too small to support any statement about SVT; the Discussion restricts the claim to AF/flutter.
- **Earlier wording (superseded).** The flagging is not only a rate effect. AF/flutter windows have a median annotated rate of 86 bpm, and the HR-containing systems with other HRV features still flag them (the bit-masked network flags 0.98 of the 121 AF/flutter windows below 100 bpm, the MLP 1.00, (iii) 0.83, HR-only 0.00). The rhythm-irregularity features (RMSSD, pNN50, LF/HF) are driving this. So "an HR-driven model" understates it: with EDA missing these systems call irregular rhythms stress-like even at ordinary rates, and call 100-119 bpm stress-like through rate.
- **Caveats (stated in advance and still applying).** Different sensor and sampling rate, patients instead of healthy volunteers, no EDA or stress labels, dataset identity is separable (companion audit AUC 0.83), windows within a record are autocorrelated and a few records dominate (213, 215 and 233 give 76 of the 95 windows at 100-119 bpm, AF comes from 6 records), and non-normal-sinus windows at rates below 100 are also flagged at 0.41 (bit-masked network), so the probe shows that these systems should not be read as arrhythmia-safe, not how often a given arrhythmia would be misread.

## 6. Discussion
- **Principal finding.** For wearable stress detection with a missing EDA sensor, we find no evidence that hard modality masking adds anything once the pipeline has seen missing EDA. Because the mask has no learned parameters and the comparator MLP sees the same missingness bit, this was the expected outcome (§4.3); the result says nothing about learnable gating.
  - Against a capacity-matched MLP with the same masking, the gate is equivalent within ±0.02 Brier and ±0.03 AUROC (post hoc; read under the widest scheme, B–F df 14, TOST p = 0.033 and 0.015), at four sizes, after tuning, at four masking rates and with a reduced HRV set. The honest summary is "equivalent at about ±0.02, and fragile": the Brier bound attained is 0.018, and many unadjusted equivalence tests were run (§7). Per-subject normalisation gives the same null, but it is a non-deployable upper bound (§5.4), so we do not count it as robustness evidence.
  - In the never-masked regime the gate does not substitute for masking.
  - **The masking effect is not at the same evidential level as the tree result.** For trees, exposure to missing EDA is essential ((ii) vs (i) −0.148, excluding 0 under every scheme). For the networks the masking benefit (−0.044) is directionally consistent but not resolved at n = 15: under B–F df 14 its interval is [−0.091, +0.003] (§5.2).
  - Against reduced-sensor LightGBM, the data allow differences of about ±0.03 Brier. We call this "no detectable difference", not equivalence.
  - This is the wearable-physiology instance of a pattern already reported in vision and language (Ma et al. 2022) and across multimodal benchmarks (SimBaMM): the training regime and the comparator matter more than the fusion architecture.
- **What "missingness-aware" means here.** Three different mechanisms reach similar reduced-condition Brier: training-time masking, per-configuration models, and per-condition recalibration (for the networks the masking route is itself unresolved at n = 15, §5.2). They differ in cost: masking costs trees complete-condition accuracy, per-configuration systems require maintaining several models, and recalibration requires labelled data from each sensor configuration.
- **What a wearable-detector builder should take from this: two simple trees and a switch.** A complete-trained LightGBM when EDA is present and an ECG-only LightGBM when it is absent had the lowest point-estimate Brier in all 19 informative-missingness scenarios (by 0.024–0.029 over the best other system), the lowest false-alarm rate with EDA present at the high-sensitivity point (6.9 against 16.2 per hour of rest for the gated network) and tied every other system without EDA (§5.3, §5.9, §5.13). No fusion architecture tested beat it. Attached caveats: the mixed-test design favours it; its lead over the masked tree (ii) is not distinguishable from 0 under Bouckaert–Frank and falls to at most 0.012 when (ii) gets no extra training mask (Table 7d); the margins over the networks in the paired design are point estimates under Bouckaert–Frank; and everything is stress-vs-baseline on 15 laboratory participants. For E14's own pipeline the implication is to build and calibrate per-sensor-configuration models before investing in a fusion architecture.
- **Informative missingness (§5.13).** In the pre-specified run, label- or motion-correlated EDA loss at odds ratios up to 8 moved no system's Brier by more than 0.012 and the gated-vs-MLP gap was the same in every scenario. That run also gave every system its own 50% random training mask, which dilutes the label–missingness association, so "no mechanism effect detected" there means "not detectable in this design". In the post hoc variant without the added mask the shortcut is detectable (baseline windows with missing EDA scored 0.11–0.22 higher at label odds ratio 4–8 by the learned systems; interval above 0 for (ii) throughout and for the gated network at OR 8), and it still moves Brier by at most 0.025. Three of the four pre-stated predictions were not confirmed. The gate-versus-MLP equivalence of §5.1 is neither contradicted nor re-established under these mechanisms (the mixed-design intervals are about ±0.05 wide).
- **The networks are the weak link with EDA present, on the default features.** Trees beat the networks by 0.04–0.06 Brier when EDA is available (gated − (i) +0.058; gated − (ii) +0.037, whose participant-level CI includes 0). Under label-free per-subject normalisation the gap nearly closes (gated − (ii) +0.005) and the gated-vs-MLP null still holds (§5.4). That normalisation uses the whole session and is not deployable, so this shows the null is not only a property of networks that ignore EDA when the features are well scaled, not that it holds in deployment. Whether it holds for signal-level models or longer windows remains open.
- **Mechanism.** Without EDA, all models fall back on heart rate. The mechanism survives an order-balanced stress-vs-amusement contrast and a negative control on later calm conditions (§5.6), so it is not a time-on-electrode artefact. It is, however, inseparable from movement and posture: chest movement alone discriminates better than heart rate, and heart rate adds only a small within-subject increment beyond it (§5.6E). We call it a cardiac-motor response. On the wrist, the fallback is a PPG-artefact detector (§5.10). Per-participant performance tracks HR response, which in TSST is heterogeneous.
- **Calibration.** Models never exposed to missingness fail without EDA, in two different ways. The never-masked tree fails in calibration-in-the-large: it reads the absent sensor as absent arousal, and a single logit shift per condition recovers it (Brier 0.294 → 0.157, §5.2 Table 4b). The never-masked networks do not: an intercept-only recalibration removes none of their masking effect, while a slope-only rescaling removes 68% of it, so the networks' masking benefit is mostly a difference in over-confidence (and, by Bouckaert–Frank df 14, it is not resolved at n = 15). Discrimination without EDA is similar across all reasonable models (AUROC 0.854–0.892), though the AUROC intervals cannot exclude a gain of 0.06–0.09. All models are over-confident without EDA, and per-condition calibration fixes the slope. Abstention does not repair this: it has no free benefit (Table 8). On the retained windows it roughly halves error for every missingness-aware model without separating them, but as a safety mechanism it only moves along the sensitivity / false-alarm trade-off, which depends on what happens to the abstained windows (dropped, not alerted, alerted), and it preferentially abstains on stress windows.
- **Methodological lessons for small-n wearable studies** (Appendix A gives each error, its effect and its regression test):
  1. Repeated CV must be shown to produce distinct participant partitions.
  2. Missing-sensor conditions should be scored on the same windows.
  3. An ECE estimator's bias and ranking behaviour must be checked at the actual sample size.
  4. Feature tables should be frozen by hash and, ideally, re-derived from raw data.
  5. Baselines should be given the same missingness exposure as the proposed model.
  6. "Indistinguishable" needs an equivalence analysis and an honest minimum detectable difference; with 15 participants that is about 0.05 Brier.
  7. A point-estimate decision rule flips on near-ties; it flipped here under calibration mode, CV scheme, tuning and feature set.
  8. A physiological "mechanism" in a laboratory protocol needs a movement and posture control; here movement out-discriminated heart rate.
  9. Read both sides of an argument under the same inference scheme: we read the gate null under the widest interval and the masking effect under the narrowest, and the conclusion changed when we did not.
  10. A test for a shortcut must not contain a mechanism that destroys it: random training masking on top of informative missingness made the shortcut check unable to fail.
  11. Test a mechanism claim directly before stating it: "mostly calibration-in-the-large" was true for the tree and false for the networks once intercept-only and slope-only recalibrations were run.
- **What the proxy cannot say (clinical reading, now probed).** With EDA missing, WESAD-trained systems label most AF/flutter windows "stress" with high confidence (mean calibrated p 0.90 for the bit-masked network): a failure to abstain on out-of-distribution input, which abstention on the model's own confidence cannot repair. The data support this for AF/flutter only; the SVT sample is 12 windows and MIT-BIH holds one window at 120 bpm or more, so nothing is shown about paroxysmal SVT or other sustained tachycardias (§5.14). The information that separates panic from cardiac events (abrupt onset and offset, rates above about 150 bpm, RR regularity, P-wave morphology) is not what the models were trained to use. This is not a validation of any panic-vs-cardiac classifier.
- **What this does not show:** panic-vs-cardiac differentiation; performance in patients, in ambulatory signals, or with realistic (partial, intermittent, motion-correlated) sensor loss; and behaviour of networks large or well-specified enough to exploit EDA.

## 7. Limitations
- **Sample size.** 15 participants; each test fold holds about 3. The minimum detectable ΔBrier against tree baselines is about 0.047.
- **Inference.** Bouckaert–Frank is a heuristic correction with 15 clusters and is probably anti-conservative; we report df = 14 and participant-level intervals alongside it. Bootstrap CIs with 15 clusters are approximate even with BCa.
- **Post-hoc equivalence.** The ±0.02 Brier margin was fixed in code before the runs but after the comparisons were known; its only anchor is a fraction (about 9%) of the prevalence-only Brier, not a clinical consequence. It is looser than the ±0.01 net-benefit margin of §5.9, at which gated ≈ MLP was **not** confirmed under the primary scheme. Dozens of equivalence tests are reported (Tables 3 and S3, §5.9, §5.13) with no multiplicity adjustment, and several pass narrowly (p = 0.048 at the largest model size; the headline gated-vs-MLP Brier bound attained is 0.018 against a margin of 0.02 under df 14). The honest summary is "equivalent at about ±0.02, and fragile".
- **Proxy task.** Laboratory stress in healthy young adults (TSST), not panic, with a fixed baseline-first order (§3, §5.6). The stressor is performed standing and speaking, so heart rate is confounded with posture and movement (§5.6E).
- **Masking effect for the networks is not resolved.** Under Bouckaert–Frank df 14 the masking main effect on the networks is [−0.091, +0.003] and within each network it includes 0 at both Bouckaert–Frank dfs; only the participant-level scheme excludes 0. The tree result ((ii) vs (i) −0.148) is the robust one (§5.2).
- **The informative-missingness check is design-limited.** The pre-specified run added 50% random training masking to every model, which dilutes the label–missingness association; its null is "not detectable in this design". The no-added-masking variant is post hoc, the odds ratios stop at 8, and the gated-vs-MLP contrast in the mixed design is only resolved to about ±0.05 (§5.13).
- **The per-configuration system's advantage is unadjudicated.** Its lead over the masked tree is not distinguishable from 0 under Bouckaert–Frank in any of the 19 mixed-test scenarios, is favoured by the mixed-test design, and shrinks to at most 0.012 when the tree gets no extra training mask (§5.13).
- **Simulated missingness.** EDA removed completely, MCAR in the primary analysis and label- or motion-correlated at odds ratios up to 8 in §5.13. Real sensor loss is partial, intermittent and can be stronger and signal-dependent; the gate-versus-MLP equivalence is not established for mixed test sets (§5.13).
- **Feature-level, ultra-short windows, small networks.** Hand-crafted features; HRV beyond mean NN and RMSSD is unreliable at 60 s; networks of at most 9,345 parameters, trained full-batch without early stopping. On the default features they underuse EDA; per-subject normalisation largely fixes that (§5.4), but no signal-level model was tested.
- **No external validation.** Everything is within WESAD; the wrist replication uses the same participants.
- **Transductive analyses.** HR-residualisation and per-subject normalisation use each test participant's own windows (the whole session, both conditions); the PULSE-style variant also uses their labels. They are upper bounds, not deployable results.
- **Analytic flexibility and residual error risk.** Nine implementation errors were found across revisions (five affecting the headline comparison); the decision rule was written after the gated model's estimates were known; many analyses were reviewer-requested or exploratory. The minimal re-implementation is author-side, not independent.

## 8. Conclusion
- **What we found.** In this laboratory proxy setting, a modality-gated network trained with modality dropout showed no detectable advantage when EDA was missing. Against a matched MLP with the same masking it was equivalent within a small post-hoc margin (about ±0.02 Brier, fragile, read under the widest interval scheme). Against reduced-sensor LightGBM the data could not exclude differences of about ±0.03 Brier. Without masking, the gate did not substitute for it. For trees, exposure to missing EDA was essential under every inference scheme; for the networks the masking benefit was directionally consistent but unresolved at n = 15, and it was mostly a difference in over-confidence, not base rate.
- **What to build.** A complete-trained tree when EDA is present and an ECG-only tree when it is absent was the nominally best system throughout (lowest point-estimate Brier in all 19 informative-missingness scenarios; fewest false alarms with EDA present; tied without EDA). For a wearable detector the practical answer from this study is two simple trees and a switch, not a fusion architecture, with the mixed-test caveat attached.
- **How the earlier claim arose.** The earlier apparent gating advantage came from comparing against a model never exposed to missing EDA, and from evaluation errors we document.
- **Mechanism.** Without EDA, every model's discrimination depends mainly on each participant's heart-rate response. This holds in an order-balanced stress-vs-amusement contrast and is not explained by time since the start of the session, but it cannot be separated from the movement and posture of the TSST.
- **Next step.** Real cardiac-vs-panic differentiation needs a clinically labelled cohort with paired events and rhythm-level features. Any further confirmatory analysis on new data should be registered on OSF before it is run.

## 9. Reporting, reproducibility, disclosures (TRIPOD+AI)
- **Reporting guideline.** TRIPOD+AI (BMJ 2024;385:e078378): model development with internal validation on proxy data. Items still to finalise: the 27-item checklist; per-participant window counts and class balance (`results/per_subject_loso.csv`); compute (CPU only; runtimes in the logs).
- **Reproducibility.**
  - Frozen and extended tables: hashes in `results/feature_table_manifest.json` and `results/extended_table_manifest.json`; the extended table reproduces the frozen one from raw WESAD.
  - Seeds fixed per (repeat, fold); the 20 partitions are stored in `results/repeated_cv.json`; per-participant results in `results/*per_subject*.csv`.
  - Scripts and order of execution: `README.md`.
  - Decision history: `results/decision_timeline.md`.
  - **Release (blocking, author action):** initialise git, commit the full tree, tag it `post-rerun-2026-10-02` (not a pre-registration), and release code, partitions, per-participant results and, if the WESAD licence allows derived features, the feature tables with their hashes.
- **Ethics.** Public, de-identified datasets (WESAD; MIT-BIH for the companion note).
- **Data use (page-1 COI context).** WESAD is used here for research only, under its original terms (Schmidt et al., ICMI 2018; distributed by the University of Siegen, mirrored by UCI, which defers to the original licence). No model, feature table or result derived from WESAD is used in any E14 Health product; commercial development uses different datasets. Any released WESAD-derived feature tables carry the WESAD citation and are offered under the same terms as WESAD itself, for research use.
- **Conflict of interest.** See page 1.
- **Population and equity.** The target population (older adults, people with cardiac comorbidity) is not represented, and wearable access and performance are unequal across groups (Zinzuwadia & Singh 2022).
- **Superseded numbers.** Every superseded headline is listed with its corrected value in Appendix A. None appeared publicly (page 1).

## 10. Title, abstract and scope

### 10.1 Title options
- *Evaluation Pitfalls in a 15-Participant WESAD Study of Missing Wearable EDA: A Cardiac-Motor Fallback, and No Detectable Benefit of Hard Modality Masking* (current working title, revision 9: leads with the contributions ranked first, §Contributions)
- *Hard Modality Masking Adds Nothing Detectable to Missingness-Aware Training When Wearable EDA Drops Out: A 15-Participant WESAD Case Study in Evaluation Pitfalls* (revision 7 title; leads with the contribution ranked third)
- *Masked Late Fusion vs a Missingness Indicator Under Wearable EDA Dropout: Evaluation Pitfalls from a 15-Participant WESAD Study*
- *No Evidence That Modality Gating Adds to Missingness-Aware Training When Wearable EDA Drops Out* (revision 6 title; "gating" overstates a fixed mask)
- *A Modality-Masking Advantage Under Missing EDA Was a Baseline Artefact: Fair Comparators, a Cardiac-Motor Fallback, and Nine Implementation Errors*

Avoid "determines", "not gating" and "disappears": the design shows absence of a detectable gate effect, conditional on masking, on one dataset.

### 10.2 Abstract (draft)
> **Background.** Wearable stress detectors face missing sensors, and safe abstention needs predictions that stay calibrated when a sensor drops out. Modality gating or masking with modality dropout has been proposed for this.
>
> **Methods.** On WESAD (15 participants; stress vs baseline; 434 sixty-second windows), we scored every held-out window with and without its six EDA-dependent features. We compared a network that hard-masks the EDA branch (a fixed gate with no learned parameters) and is trained with modality dropout against a capacity-matched MLP with a missingness indicator and the same masking, the never-masked versions of both (a 2 × 2 gate × masking factorial), and LightGBM baselines trained with masking, on ECG only, or with mean imputation and recalibration, plus a per-configuration system (a complete-trained tree with EDA, an ECG-only tree without it). We used 20× repeated subject-level 5-fold CV with Bouckaert–Frank (df 99 and the wider df 14, which we read for equivalence) and participant-level inference, post-hoc equivalence tests (±0.02 Brier, ±0.03 AUROC), leave-one-subject-out for per-participant and confound analyses, post-hoc intercept-only and slope-only recalibration, and a study of operating points and of label- or motion-driven EDA missingness (specified before it was run; its no-added-masking variant is post hoc). The analyses followed internal, AI-assisted review rounds on one small dataset; only the round-5 analyses were specified before they were run.
>
> **Results.** *Pitfalls.* We document nine implementation errors in our own pipeline, five of which moved or could have moved the headline comparison, and give a one-page checklist that catches each. An earlier apparent gating advantage came from a comparator never exposed to missing EDA (Brier 0.296, calibration intercept +3.1). *Mechanism.* Without EDA, removing heart rate removed about half of every model's above-chance AUROC, and this held in an order-balanced stress-vs-amusement contrast. But chest movement alone discriminated stress from baseline better than heart rate (AUROC 0.945 vs 0.820; paired difference narrowly above 0), so the fallback is a cardiac-motor response to the standing, speaking TSST, not a cardiac one. On wrist-only features the no-EDA signal was largely motion-corrupted PPG. *Gating.* With EDA missing, the gated network was post hoc equivalent to the masked MLP within ±0.02 Brier, but narrowly and fragilely: ΔBrier +0.002 (95% CI −0.016 to +0.020), TOST p = 0.033 under the widest scheme (B–F df 14; 0.025 at df 99), Brier bound attained 0.018, with many unadjusted equivalence tests. The ±0.02 margin has no clinical anchor and is looser than the ±0.01 net-benefit margin, at which the pre-specified equivalence was **not** confirmed. Against NaN-trained and ECG-only LightGBM the data could not exclude differences of about ±0.035 Brier. *Masking and calibration.* For trees, exposure to missing EDA was essential: the masked tree beat the never-masked tree by 0.148 Brier, an interval that excludes 0 under every inference scheme, and the never-masked tree's failure was a base-rate shift that a single logit shift per condition removes (Brier 0.294 to 0.157). For the networks the masking benefit (−0.044, 12/15 participants) was directionally consistent but not resolved at n = 15 (B–F df 14: −0.091 to +0.003); it was mostly a difference in over-confidence, not base rate (slope-only rescaling removed 68% of it, intercept-only none). With EDA present, LightGBM beat the networks by 0.04–0.06 Brier on default features. *Per-configuration system.* The per-configuration system had the lowest point-estimate Brier in all 19 informative-missingness scenarios (by 0.024–0.029 over the best other system) and the fewest false alarms with EDA present (6.9 vs 16.2 per hour of rest for the gated network at the high-sensitivity point), and tied the other systems without EDA; the mixed-test design favours it, and its lead over the masked tree fell to at most 0.012 when that tree got no extra training mask. *Informative missingness.* In the pre-specified run, label- or motion-correlated EDA loss (odds ratio up to 8) moved no system's Brier by more than 0.012 and three of four pre-stated predictions failed, but that design added random 50% training masking that dilutes any shortcut; without it, baseline windows with missing EDA were scored 0.11–0.22 higher in p(stress) at label odds ratio 4–8, a detectable shortcut that still moved Brier by at most 0.025. *Operating points.* Without EDA, every system that reached 90% sensitivity raised 20–24 false alarms per hour of rest (stress prevalence is 35% by design, so predictive values do not transfer). Abstention is not free: treating abstained windows as "no alert" cut the gated network's sensitivity from 0.613 to 0.256 at 60% coverage, and treating them as "alert" bought sensitivity at 18.2 false alarms per hour (post hoc).
>
> **Interpretation.** In this setting a fixed hard mask adds nothing detectable to a masked MLP that sees the same missingness bit, which was expected by construction and does not test learnable gating. What survives under every inference scheme is that a tree never exposed to missing EDA fails and a tree trained with missing EDA does not; the practical answer for a wearable detector, nominally rather than demonstrably, is two simple trees and a switch, not a fusion architecture, subject to the caveats above. The transferable results beyond that are the evaluation errors and the confound analysis. These results come from 15 healthy laboratory participants and do not evaluate differentiation of panic attacks from cardiac events.
>
> **Disclosure.** The author is a co-founder of E14 Health.

### 10.3 Scope
- **This preprint:** the missing-EDA evaluation (§§4–5) and the evaluation-pitfalls case study (Appendix A).
- **Appendices:** H2a (Appendix C); the data-gap search (Appendix D; H-B2 reported as historical and recommended for omission); interpretability (Appendix E).
- **Cut:** H-C1 device tiers (its tier B was identical to the reduced condition; the wrist replication in §5.10 replaces it), and the WESAD × MIT-BIH audit (separate note, Appendix F).

---

## Supplementary tables

*At submission these go to a separate supplement. Headers are unchanged, so `check_numbers.py` still ties every cell.*

**Table S1. Decision-rule verdict by calibration mode and CV scheme (§5.1).**

| Calibration | 20× repeated CV | Without ECE criterion (20× CV) | LOSO pooled | Single 5-fold |
|---|---|---|---|---|
| Raw | fires | fires | fires | fires |
| Shared Platt | fires | fires | fires | fires |
| Per-condition Platt | does not fire | does not fire | does not fire | fires |

**Table S2. Complete condition (EDA present), paired differences, 20× repeated CV, raw.**

| Comparison | ΔBrier, B–F 95% CI | ΔBrier, participant-level (A better) | ΔAUROC, B–F 95% CI |
|---|---|---|---|
| Gated vs (i) | +0.058 [−0.012, +0.127] | +0.058 [+0.010, +0.105] (2/15) | −0.033 [−0.080, +0.014] |
| Gated vs (ii) | +0.037 [−0.033, +0.106] | +0.037 [−0.016, +0.089] (3/15) | −0.022 [−0.079, +0.035] |
| MLP vs (ii) | +0.048 [−0.037, +0.132] | +0.048 [−0.011, +0.106] (4/15) | −0.030 [−0.099, +0.039] |
| (ii) vs (i) | +0.021 [−0.031, +0.073] | +0.021 [−0.007, +0.049] (6/15) | −0.011 [−0.052, +0.030] |
| Gated vs MLP | −0.011 [−0.053, +0.031] | −0.011 [−0.031, +0.010] (8/15) | +0.008 [−0.025, +0.041] |
| Gated, no masking vs MLP, no masking | −0.028 [−0.073, +0.017] | −0.028 [−0.047, −0.008] (13/15) | +0.021 [−0.017, +0.060] |

**Table S3. Gated − MLP and gated − (ii), reduced condition, raw, 20× repeated CV, under each variation.**

| Variation | Gated − MLP ΔBrier, reduced [B–F 95% CI] | Gated − (ii) ΔBrier, reduced [B–F 95% CI] | Note |
|---|---|---|---|
| Default (Table 2) | +0.002 [−0.016, +0.020] | +0.002 [−0.031, +0.035] | |
| Capacity: 273 vs 281 parameters | −0.001 [−0.024, +0.022] | — | TOST p 0.052 |
| Capacity: 801 vs 808 parameters | +0.003 [−0.015, +0.021] | — | TOST p 0.030 |
| Capacity: 2,625 vs 2,637 parameters | +0.002 [−0.014, +0.017] | — | TOST p 0.011 |
| Capacity: 9,345 vs 9,333 parameters | +0.005 [−0.012, +0.023] | — | TOST p 0.048 |
| Nested tuning | +0.005 [−0.019, +0.029] | +0.004 [−0.028, +0.036] | |
| Masking rate 0.1 | +0.000 [−0.031, +0.032] | +0.002 [−0.036, +0.041] | |
| Masking rate 0.25 | +0.001 [−0.022, +0.025] | +0.002 [−0.030, +0.033] | |
| Masking rate 0.5 | +0.002 [−0.016, +0.020] | +0.002 [−0.031, +0.035] | |
| Masking rate 0.75 | +0.002 [−0.016, +0.019] | +0.001 [−0.034, +0.037] | |
| HRV = mean NN + RMSSD only | −0.003 [−0.018, +0.013] | −0.007 [−0.034, +0.020] | rule does not fire (raw) |
| Per-subject z-score (label-free) | −0.002 [−0.015, +0.011] | +0.002 [−0.012, +0.015] | gated reduced AUROC 0.939 |
| Per-subject z-score on baseline windows (PULSE-style) | +0.001 [−0.012, +0.014] | −0.004 [−0.023, +0.014] | gated reduced AUROC 0.949 |
| Wrist E4 features (§5.10) | +0.001 [−0.013, +0.014] | −0.021 [−0.052, +0.009] | gated reduced AUROC 0.933 |

**Table S4. Stress vs amusement, LOSO, extended table, raw (§5.6A).**

| Model | Complete pooled AUROC | Reduced pooled AUROC [95% CI] | Reduced within-subject AUROC [95% CI] | Reduced within-subject AUROC: stress first / amusement first |
|---|---|---|---|---|
| Gated + dropout | 0.907 | 0.866 [0.799, 0.931] | 0.914 [0.839, 0.976] | 0.889 / 0.937 |
| Non-gated MLP | 0.872 | 0.857 [0.781, 0.925] | 0.894 [0.799, 0.972] | 0.875 / 0.910 |
| (ii) NaN-trained | 0.925 | 0.909 [0.857, 0.954] | 0.936 [0.871, 0.989] | 0.969 / 0.908 |
| (iii) ECG-only | 0.911 | 0.911 [0.860, 0.958] | 0.946 [0.892, 0.997] | 0.953 / 0.940 |
| HR-only | 0.851 | 0.851 [0.786, 0.911] | 0.938 [0.851, 1.000] | 0.952 / 0.925 |
| Gated, HR removed | 0.713 | 0.583 [0.467, 0.687] | 0.595 [0.471, 0.713] | 0.669 / 0.529 |
| (ii), HR removed | 0.799 | 0.666 [0.554, 0.777] | 0.684 [0.548, 0.815] | 0.688 / 0.681 |
| (iii), HR removed | 0.680 | 0.680 [0.573, 0.785] | 0.703 [0.579, 0.818] | 0.717 / 0.691 |

**Table S5. Negative control: fraction of windows predicted "stress" (p ≥ 0.5), held-out participants, raw (§5.6B).**

| Model, condition | Baseline (first) | Stress | Amusement | Meditation 1 | Meditation 2 (last) |
|---|---|---|---|---|---|
| Gated + dropout, EDA missing | 0.14 | 0.70 | 0.14 | 0.23 | 0.21 |
| Non-gated MLP, EDA missing | 0.15 | 0.73 | 0.15 | 0.26 | 0.17 |
| (ii) NaN-trained, EDA missing | 0.18 | 0.72 | 0.18 | 0.15 | 0.10 |
| (iii) ECG-only, ECG only | 0.17 | 0.75 | 0.19 | 0.21 | 0.14 |
| HR-only, HR only | 0.22 | 0.65 | 0.28 | 0.13 | 0.13 |
| Gated + dropout, EDA present | 0.16 | 0.86 | 0.10 | 0.14 | 0.11 |
| (ii) NaN-trained, EDA present | 0.13 | 0.86 | 0.13 | 0.12 | 0.08 |
| (i) Complete-trained, EDA present | 0.09 | 0.88 | 0.27 | 0.09 | 0.07 |
| (i) Complete-trained, EDA missing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Table S6. Baseline vs meditation, complete condition, LOSO (§5.6C).**

| Model | Pooled AUROC | Within-subject AUROC [95% CI] |
|---|---|---|
| EDA-only LightGBM | 0.455 | 0.423 [0.287, 0.556] |
| HR-only | 0.426 | 0.438 [0.300, 0.575] |
| (iii) ECG-only | 0.732 | 0.750 [0.635, 0.850] |
| (i) Complete-trained | 0.627 | 0.622 [0.478, 0.754] |
| Gated network (no masking needed: EDA present) | 0.726 | 0.758 [0.681, 0.833] |

**Table S7. Stress vs baseline from movement, posture and heart rate, LOSO.**

| Feature set | Pooled AUROC [95% CI] | Within-subject AUROC [95% CI] | Brier |
|---|---|---|---|
| Movement only | 0.945 [0.910, 0.980] | 0.958 [0.937, 0.978] | 0.074 |
| Posture only | 0.574 [0.432, 0.714] | 0.581 [0.413, 0.753] | 0.279 |
| ACC (movement + posture) | 0.928 [0.886, 0.968] | 0.944 [0.903, 0.974] | 0.093 |
| HR only | 0.820 [0.694, 0.947] | 0.867 [0.765, 0.959] | 0.178 |
| ACC + HR | 0.944 [0.898, 0.983] | 0.971 [0.948, 0.992] | 0.092 |
| ECG only | 0.869 [0.756, 0.953] | 0.899 [0.789, 0.971] | 0.142 |
| ECG + ACC | 0.952 [0.910, 0.986] | 0.976 [0.959, 0.991] | 0.083 |

**Table S8. Spearman correlation of within-subject AUROC with HR response (§5.7).**

| Model | ρ with raw response [Fisher-z 95% CI] | ρ with standardised response [Fisher-z 95% CI] |
|---|---|---|
| Gated, HR removed | 0.70 [0.30, 0.89] | 0.68 [0.26, 0.89] |
| (ii), HR removed | 0.70 [0.29, 0.89] | 0.83 [0.56, 0.94] |
| (iii), HR removed | 0.62 [0.16, 0.86] | 0.75 [0.39, 0.91] |
| Gated network | 0.88 [0.67, 0.96] | 0.88 [0.66, 0.96] |
| Non-gated MLP | 0.88 [0.66, 0.96] | 0.81 [0.51, 0.94] |
| (ii) NaN-trained | 0.76 [0.41, 0.92] | 0.90 [0.72, 0.97] |
| (iii) ECG-only | 0.88 [0.68, 0.96] | 0.94 [0.83, 0.98] |
| HR-only | 0.92 [0.77, 0.97] | 0.92 [0.77, 0.97] |

**Table S9. Calibration under missing EDA, LOSO pooled, reduced (§5.8).**

| Model | Raw: slope / intercept / ECE | Per-condition Platt: slope / intercept / ECE | Brier, raw → per-condition |
|---|---|---|---|
| Gated + dropout | 0.43 / −0.16 / 0.087 | 0.92 / +0.05 / 0.045 | 0.146 → 0.147 |
| Non-gated MLP | 0.42 / −0.25 / 0.100 | 1.00 / +0.04 / 0.049 | 0.145 → 0.144 |
| (ii) NaN-trained | 0.59 / −0.28 / 0.057 | 0.82 / −0.05 / 0.062 | 0.146 → 0.153 |
| **(i) Complete-trained** | 0.96 / +3.10 / 0.302 | 1.13 / +0.09 / 0.112 | 0.296 → 0.163 |

**Table S10. Selective-prediction contrasts, 20× repeated CV, raw (§5.9).**

| Comparison | ΔAURC [95% CI], raw | ΔE-AURC [95% CI], raw |
|---|---|---|
| Gated vs (ii) | −0.012 [−0.078, +0.054] | −0.012 [−0.074, +0.051] |
| Gated vs (iii) | −0.026 [−0.113, +0.061] | −0.026 [−0.112, +0.060] |
| Gated vs MLP | −0.000 [−0.024, +0.023] | −0.002 [−0.026, +0.023] |

**Table S11. Status of every analysis (§4.7).**

| Analysis | Status | Note |
|---|---|---|
| Baselines (i)–(iv), HR-only | Pre-specified | |
| HR-removed and HR-residualised variants | Reviewer-requested | |
| Decision rule (gated vs (ii), (iii) on Brier, ECE, AUROC) | Pre-specified | Point-estimate rule; also reported without ECE |
| 20× repeated CV with Bouckaert–Frank inference; sign-flip test | Pre-specified | Partitioning corrected (Appendix A #2) |
| Shared Platt, fit on training participants only | Pre-specified | Implementation corrected (Appendix A #5) |
| ECE_SWEEP; paired scoring; fixed HR-residualisation; frozen table | Correction | Appendix A #3, #4, #6, #7 |
| Non-gated MLP with the same masking; nested tuning; HR-removed LightGBM | Reviewer-requested | |
| Capacity sweep; per-condition Platt; risk–coverage | Exploratory | |
| LOSO, within-subject AUROC, per-participant / HR-response analysis | Reviewer-requested / exploratory | HR-response correlation is exploratory |
| H2a subject-level reanalysis | Reviewer-requested | Appendix C |
| **Never-masked MLP (2 × 2 factorial) and factorial contrasts** | Reviewer-requested | |
| **Post-hoc TOST / ROPE equivalence** | Reviewer-requested | margins fixed in code before the run |
| **B–F with df = 14; participant-level inference** | Reviewer-requested | |
| **Complete-condition paired CIs, all models, untuned and tuned** | Reviewer-requested | |
| **Order-confound checks** (stress vs amusement, negative control, baseline vs meditation) | Reviewer-requested | |
| **Standardised HR response with CIs** | Reviewer-requested / exploratory | |
| Masking-rate sweep; HRV subset; per-subject normalisation | Reviewer-requested | |
| SmoothECE, E-AURC, BCa bootstrap (B = 2,000), signal quality, S10 | Reviewer-requested | |
| Wrist (E4) replication | Reviewer-requested | not pre-registered |
| Minimal re-implementation (author-side) | Reviewer-requested | not independent |
| Interpretability (Appendix E) | Exploratory | Descriptive only |
| Posture and movement (§5.6E) | Reviewer-requested | specified after every other result was known |
| **Operating points, abstention by class, decision curves (§5.9)** | Pre-specified (round 5) | `prespec_round5.md`; predictions reported as confirmed or not |
| **Movement-vs-HR paired contrast; responders-vs-movers correlations (§5.6E, §5.7)** | Reviewer-requested | post hoc, `run_movement_checks.py` |
| **Abstention with an action model (Table 8, §5.9)** | Reviewer-requested | post hoc, `run_abstention_action.py`; not in `prespec_round5.md` |
| **Informative (label- or motion-driven) missingness (§5.13)** | Pre-specified (round 5) | `prespec_round5.md`; label-driven OR 4, S1 and S2, primary; other scenarios descriptive |
| **Informative missingness without added MCAR masking (Tables 7c, 7d, §5.13)** | Reviewer-requested | post hoc, `run_mnar.py --no-added-masking`; not in `prespec_round5.md` |
| **Intercept-only and slope-only per-condition recalibration (Table 4b, §5.2)** | Reviewer-requested | post hoc, `run_intercept_recal.py`; the slope-only arm was added after the intercept-only result |
| **Factorial read under B–F df 14 and participant-level alongside df 99 (Table 4)** | Reviewer-requested | stored intervals of `run_repeated_cv.py`; no new run |

---

## Appendix A. Implementation errors found, their effect, and the regression test for each

| # | Error | Effect on reported results | Fix / regression test |
|---|---|---|---|
| 1 | Record-level train/test split put the same participant in train and test (found before 2026-09-20) | Inflated all early H-B1 numbers; superseded by subject-level CV. **No leaky-split or other superseded number appeared publicly** (author confirmation, 2026-10-02; `results/decision_timeline.md` row 19) | Subject-level splits everywhere; `test_partition_covers_each_subject_once` |
| 2 | "20× repeated CV" shuffled rows before `GroupKFold`, which ignores row order: 1 distinct participant partition in 20 | First rule evaluation's Bouckaert–Frank p-values (0.49–0.56) were invalid | Explicit participant permutation per repeat; `test_repeated_partitions_differ` |
| 3 | "Debiased ECE" = minimum over 6 bin counts of `max(raw − ad-hoc bias, 0)` | 18/40 reduced per-fold values exactly 0; ranked the worst-calibrated model best | ECE_SWEEP (Roelofs); `test_ece_sweep_not_clipped_and_small_when_calibrated` |
| 4 | Complete and reduced conditions scored on disjoint random halves of each test fold | Extra sampling noise; ratio and ΔECE compared different windows | Paired scoring; `test_paired_tiers_same_windows` |
| 5 | Platt protocol described in Methods but applied to one model only, and fit on in-sample predictions | Methods text did not match the code | Inner-CV out-of-fold Platt for every model; `test_fit_predict_never_trains_on_test` |
| 6 | HR-residualised variant kept `hrv_mean_nn` and residualised across pooled participants | Its AUROC (0.878) was read as "HRV beyond HR"; after the fix it is 0.600 | HR removed, within-participant residualisation; `test_hr_resid_removes_hr` |
| 7 | Feature cache rebuilt on 2026-09-30 (445 → 434 WESAD windows) while 2026-09-20 numbers were kept; locked-comparison script loaded MIT-BIH rows | Earlier ECE-ratio table could not be reproduced | Hash-checked frozen table; every script refuses a different table; round 4: table reproduced exactly from raw data |
| 8 | `requirements.txt` pins did not match the installed environment | Reproducibility | Pins updated to the versions that ran |
| 9 | Earlier drafts called the headline baseline "zero-fill"; the locked script delivered NaN | Label only (both give Brier ≈ 0.30 under missing EDA) | Both encodings reported separately |
| 9b | Found in review 8: the "NaN at test" comparator is zero-fill for the four EDA features, because LightGBM sends NaN to 0 when training had no missing values | Same Brier under both labels; explains why the failure is so large (EDA = 0 µS is outside the training range) and why per-subject centring cures it (0 then equals the subject's mean) | Relabelled in §4.3; `test_complete_trained_lightgbm_nan_is_zero_fill_for_never_missing_features` |

**First vs second evaluation of the decision rule:**

| | First (pre-fix, `results/archive/*_prefix.*`) | Second (post-fix) |
|---|---|---|
| Gated vs (ii), Brier | +0.014 (B–F p 0.56; invalid, error 2) | +0.002 [−0.031, +0.035] |
| Gated vs (iii), Brier | +0.015 (p 0.49; invalid) | +0.002 [−0.033, +0.037] |
| Gated vs (ii)/(iii), AUROC | gated lower | gated higher by 0.010 (CI spans 0) |
| Rule | fires: "worse on all three" | fires on near-ties |

## Appendix B. Legacy ECE-ratio analyses (regenerated on the frozen table; `results/locked_comparison_rerun.log`, `results/locked_per_modality_rerun.log`)

*These compare against the complete-trained LightGBM and are superseded as evidence about gating. They use the old protocol: a single `GroupKFold` partition, unpaired 50% masking, and equal-width 10-bin ECE.*

| Condition | Ratio, mean of per-fold ratios | 95% CI (Wald, as originally computed) | Earlier draft (445-window table) |
|---|---|---|---|
| Complete-trained LightGBM (NaN at test) | 4.50× | [0.05, 8.96] | 4.23× |
| Gated, no dropout | 2.02× | [0.57, 3.46] | 2.21× |
| Gated + dropout p = 0.5 | 1.15× | [0.67, 1.63] | 1.61× |

- **Ratio intervals.** A Wald interval on a ratio with a noisy denominator is inappropriate. On the log scale the LightGBM ratio is a geometric mean of 3.39× with 95% CI [1.13, 10.20] (t on log per-fold ratios, n = 5). Either way the ratio is retired as a metric.
- **Dropout vs no dropout.** Dropout has the lower ratio in 4 of 5 folds (paired t p = 0.095; Wilcoxon W = 1, p = 0.125). The earlier "5/5 folds, W = 0" could never have been significant at 0.05: with n = 5 the smallest attainable two-sided Wilcoxon p is 0.0625. The original evidence was inadequate by construction.
- **Dropout-rate sweep.** p = 0.2 / 0.4 / 0.5 / 0.6 gives 1.32× / 1.35× / 1.15× / 1.13×, all CIs overlapping: no dose–response.
- **Ablation: which modality is dropped in training.** Dropping EDA (the modality missing at test) gives 1.15× [0.67, 1.63]; dropping ECG gives 2.48× [0.26, 4.70]; the right-modality variant wins in 4/5 folds (p = 0.18).

## Appendix C. H2a mechanism test (EDA–HR coupling lag), subject-level reanalysis
- **Earlier report.** Window-level, n = 445 (old table): d = 0.06, p = 0.94. Pseudoreplicated.
- **Subject-level analysis on the frozen table** (`results/h2a_subject_level.json`): mean stress-minus-baseline change −1.13 s, 95% CI [−2.76, +0.49]; d_z = −0.39, 95% CI [−0.91, +0.15]; t(14) = −1.49, p = 0.16; Wilcoxon p = 0.39; 8 of 15 participants show a longer lag under stress.
- **Equivalence.** No smallest effect of interest is specified. The 90% CI of the mean change is [−2.47, +0.20] s, so equivalence would be declared only for bounds wider than ±2.47 s (d_z > 0.84).
- **Measurement reliability.** Within a resting participant the lag varies with SD 8.3 s (median; range 5.9–10.6 s). A lag drawn uniformly from the ±15 s search window has SD 30/√12 = 8.66 s; the observed 8.3 s is 96% of that. The observed variability is essentially that of a random lag, so a single 60-s cross-correlation peak carries almost no information.
- **Reading.** H2a is **not testable with this feature at this window length**. The earlier window-level "null" should not be cited.

## Appendix D. H-B2 (historical, unverifiable) and the data-gap search
- **H-B2 is reported as historical and is not part of this paper's evidence.** The contrastive-pretraining code is not present, so the result cannot be reproduced. **Recommendation:** omit it from the posted preprint.
- **Data-gap search** (`results/hb2_search.md`): a dated rapid structured web search on 2026-10-01, not a systematic review.
  - Panic-only wearable data: the Korea University digital-phenotyping dataset (Jang et al., *Sci Data* 2024; Fitbit heart rate, self-reported panic; Zenodo 10.5281/zenodo.14190138); Oh et al. 2025; Tsai et al. 2022.
  - Cardiac-only data: MIT-BIH, PTB-XL, Shaoxing–Ningbo, a 12-lead ACS dataset (2026).
  - Paired data: none found. State "as of 2026-10-01, in a rapid search"; re-run before posting if more than a few weeks pass.

## Appendix E. Interpretability (descriptive; regenerated on the frozen table)
`run_interpretability_frozen.py`, `results/interpretability_frozen.json`. All attributions are out-of-fold under LOSO.

| Model, condition | Heart rate | Other HRV | EDA | Coupling | Top 3 features |
|---|---|---|---|---|---|
| (ii) NaN-trained LightGBM, complete (SHAP) | 53% | 17% | 28% | 2% | mean NN, EDA tonic slope, EDA tonic mean |
| (ii) NaN-trained LightGBM, reduced (SHAP) | 68% | 22% | 8%* | 1%* | mean NN, pNN50, RMSSD |
| (iii) ECG-only LightGBM (SHAP) | 69% | 31% | — | — | mean NN, pNN50, RMSSD |
| Gated network, complete (Integrated Gradients) | 35% | 35% | 24% | 6% | mean NN, pNN50, EDA tonic slope |
| Gated network, reduced (Integrated Gradients) | 49% | 51% | 0%† | 0%† | mean NN, pNN50, RMSSD |

\*Under the reduced condition the EDA SHAP values attribute the response to the *absence* of EDA. †Zero by construction (gate closed).

- **Integrated Gradients settings:** 256-step midpoint rule; baseline = training median of every feature; logit output. Completeness error: median 0.004, 95th percentile 0.020, maximum 0.092 logit.
- **Reading.** Heart rate is the top feature everywhere. The "other HRV" attribution is largely heart rate in correlated units (§5.5). Group shares are descriptive; causal statements rest on the removal experiments. LF/HF ranks are not interpreted.
- The earlier (2026-09-20, 445-window, single split) attributions are superseded.

## Appendix F. Companion note: WESAD × MIT-BIH dataset-identity audit (separate paper)
- **Status.** The `--fast` run (2026-09-30) landed at Branch C (partial / inconclusive): subject-level identity AUC 0.83, lower CI 0.68; MMD R = 1.67; CORAL lowered identity AUC by 0.087 (to be confirmed against the step-8 log); all p-values at the Monte Carlo floor.
- **Required before writing the note:** the full run; null-relative thresholds documented as a post-hoc amendment; the C1 subject pool (38, or 35 in Step 4, of 55); the MIT-BIH reconciliation (48 records → 40 participants).

## Appendix G. Pending items before posting
**Blocking (author actions):**
1. `git init`; commit; tag `post-rerun-2026-10-02`; release code, partitions, per-participant results and (licence permitting) feature tables.
2. ~~Read the licence wording in `wesad_readme.pdf`~~ **Done 2026-10-04:** the readme has no licence terms beyond "please reference the publication"; the distribution page's terms (scientific, non-commercial use with credit) govern and say nothing about redistributing derived data. Window-level feature tables stay withheld (hashes published); see `DATA_NOTICE.md`.

**Open, author decision before posting:**
- Confirm how review 9 is attributed in §4.7 and Table S11. "Reviewer-requested" is defined there as an internal, AI-assisted review round; if the feedback came from a human expert, change the definition or add a class.

**Open, strengthens the paper:**
3. An independent re-implementation of the primary comparison by someone who has not seen the code (the §5.11 re-implementation is author-side).
4. An independent statistical check by a co-author or statistician without equity.
5. A second dataset under a protocol registered on OSF before running (e.g. a public E4 stress dataset).
6. Realistic missingness (partial, intermittent, signal-dependent EDA loss). Label- and motion-correlated all-or-nothing loss is done (§5.13); real dropout is not.
6a. A causal per-subject normalisation (expanding, past-only baseline) to replace the whole-session version.
6c. A seated-stressor dataset (e.g. SWELL-KW) as the proper control for the standing, speaking TSST, and a check of what the WESAD literature already reports for accelerometer features.
6d. A packaged, installable version of the Appendix H checks (the table maps each check to a test or script in the tree; it is not yet a library).
7. Signal-level models or longer windows (per-subject normalisation already brings feature-level networks close to the trees with EDA present, §5.4).

**Done:** MIT-BIH arrhythmia failure-mode probe (§5.14; primary endpoint untestable, floor not met), intercept-only and slope-only recalibration (§5.2, Table 4b), informative missingness without added MCAR masking (§5.13, Tables 7c, 7d), same-scheme reading of the factorial, reusable checklist (Appendix H), movement-vs-HR paired contrast, responders-vs-movers correlations, abstention action model, relabelling of baseline (i) as zero-fill, equivalence under df 14, factorial cell, equivalence, df = 14 and participant-level inference, complete-condition CIs (untuned and tuned), order-confound checks, standardised HR response, masking-rate / HRV-subset / normalisation sensitivity, SmoothECE, E-AURC, BCa bootstrap, signal quality, S10, wrist replication, minimal re-implementation, citation fixes, disclosure placeholder, posture/movement analysis, complete-condition CIs under per-subject normalisation.

**Round 5 (pre-specified in `prespec_round5.md`):** clinical operating points and decision curves are done (§5.9, Table 6, Figure 4). Informative missingness has been run and reported (§5.13, Table 7, Figure 5): of the four predictions only C3-P1 is confirmed, in direction and imprecisely; the abstract, §6 and §7 carry a sentence each.

## Appendix H. Checklist for small-n wearable evaluation (check → what it catches → how to run it)

Each row is an executable test or script in the released tree (`python -m pytest -q tests` runs the tests; scripts are listed in `README.md`). "Error" refers to Appendix A.

| # | Check | What it catches | How to run it |
|---|---|---|---|
| 1 | Each repeat draws its own participant partition; log the number of distinct partitions | "Repeated" CV that reuses one partition, so its p-values are invalid (error 2) | `tests/test_eval_core.py::test_repeated_partitions_differ`; `config.n_distinct_partitions` in `results/repeated_cv.json` |
| 2 | Every participant is in exactly one test fold; calibrators and tuning never touch test participants | Record-level leakage; calibration fit on in-sample predictions (errors 1, 5) | `test_partition_covers_each_subject_once`, `test_fit_predict_never_trains_on_test` |
| 3 | Score all sensor conditions on the same windows | Complete and reduced conditions compared on disjoint windows (error 4) | `test_paired_tiers_same_windows` |
| 4 | ECE estimator is unclipped, near 0 when calibrated, and agrees with a binning-free SmoothECE at the real n | An ECE that ranks the worst-calibrated model best (error 3) | `test_ece_sweep_not_clipped_and_small_when_calibrated`, `test_smooth_ece_calibrated_vs_miscalibrated` |
| 5 | Hash the feature table, make every script refuse a different one, re-derive it from raw files | A table silently replaced between runs (error 7) | `python extract_extended.py` (reproduces the frozen table); `eval_core.load_wesad_frozen()` |
| 6 | An ablation actually removes what it claims to (heart rate removed, residualised within participant) | A "beyond heart rate" claim that still contained heart rate (error 6) | `test_hr_resid_removes_hr`, `test_zero_features_removes_inputs` |
| 7 | The label of a baseline is what the code delivers (NaN, zero-fill, mean) | A "NaN" comparator that is zero-fill; descriptions that do not match code (errors 9, 9b) | `test_complete_trained_lightgbm_nan_is_zero_fill_for_never_missing_features` |
| 8 | Every baseline gets the same missingness exposure as the proposed model; the never-masked cell never sees a missing indicator | A comparator artefact that manufactures an advantage (§5.12) | `test_mlp_no_mask_never_sees_missing_indicator`; the 2 × 2 factorial in `run_repeated_cv.py` |
| 9 | Interval schemes behave (df 14 wider than df 99, TOST consistent with the 90% CI, direction of participant counts); read both sides of an argument under the same scheme | A double standard between a null and an effect (§5.2); equivalence passed under an anti-conservative interval | `test_df14_interval_wider_than_df99`, `test_tost_matches_ci90`, `test_participant_level_counts_direction`; Table 3 and Table 4 |
| 10 | Within-subject AUROC ignores between-subject offsets; sign-flip test is exact | Pooled AUROC driven by between-participant differences | `test_within_subject_auroc_ignores_between_subject_offsets`, `test_sign_flip_exact` |
| 11 | A shortcut check contains no mechanism that destroys the shortcut: run it with and without added random masking | A null that the design would give whether or not the risk is real (§5.13) | `python run_mnar.py` and `python run_mnar.py --no-added-masking`; `test_no_added_masking_variant_removes_every_extra_training_mask` |
| 12 | Decompose a calibration claim: intercept-only, slope-only and full Platt | "Calibration-in-the-large" asserted for a model whose problem was scale (§5.2) | `python run_intercept_recal.py`; `test_intercept_recalibrator_recovers_a_pure_shift`, `test_slope_recalibrator_recovers_a_pure_rescale_and_keeps_zero_intercept` |
| 13 | Control a physiological mechanism for movement, posture and condition order | A "cardiac" signal that is movement and posture (§5.6) | `python run_order_confound.py`, `python run_posture_movement.py`, `python run_movement_checks.py` |
| 14 | Tie every number in the text to a results key | Stale or mistyped numbers after a rerun | `python check_numbers.py` (writes `results/number_check_report.md`) |

