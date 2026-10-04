# Citation check: 2026-10-01

Each reference in `e14-preprint-outline.md` was checked against its arXiv page, the publisher or proceedings page, PubMed Central, or the full text. Corrections have been applied to the outline.

## Verified, with corrections applied

| Reference as cited | Verified as | Correction |
|---|---|---|
| ModDrop, Neverova et al., TPAMI **2015** | "ModDrop: Adaptive Multi-Modal Gesture Recognition," *IEEE TPAMI* 38(8):1692–1706, **2016**; arXiv 1501.00102 | Year 2015 → 2016; volume and pages added |
| CognifySSL v2.0, *Sensors* 2026 (WESAD IMU/ECG/EDA/TEMP) | Cisse & Ishimaru, "Detect and Repair: Robust Self-Supervised Wearable Sensing Under Missing Modalities," *Sensors* 26:2419 (2026). WESAD modalities: ECG, EDA, respiration, temperature, ACC. Reports F1; **no AUROC, no calibration** | Authors and title added; respiration added; metrics stated |
| PULSE, arXiv 2510.24058, "0.994 AUROC without EDA" | Zhao, Pendiyala, Mortazavi, Yan, "PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors…" (2025). 0.994 AUROC / 0.988 AUPRC on WESAD stress monitoring without EDA; uses modality dropout and distillation | Number confirmed. **Task and CV protocol not verified to match ours**; flagged in text |
| SimMLM, arXiv 2507.19264, "learnable gating; reports ECE across missing-modality settings" | Li, Chen, Han, ICCV 2025. Quote: "a dynamic, learnable gating mechanism…"; Table 4 reports ECE/SCE averaged across 15 full and missing-modality settings on **BraTS 2018** (brain MRI) | Venue added; dataset domain stated |
| Ma et al., "Calibrating Multimodal Learning," ICML 2023 | PMLR 202:23429–23450 (oral). CML regularizer; confidence can increase when modalities are corrupted | Proceedings details added |
| Lechuga López, Shamout & Rudner, arXiv 2603.02719 | "An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification" (ICU); **not** a missing-modality paper | None (the outline already described it correctly) |
| arXiv 2603.13373, "device type as a fairness axis, citing Adler 2024 / Yfantidou 2023" | Roy, Sharma, Adler, Sen, Choudhury, Salekin, "Ethical Fairness in Ubiquitous Health Sensing without Known Attributes." Mentions "device-level variations" and "sensor type, device availability." **Does not cite Adler et al. 2024, Yfantidou et al. 2023, or Zinzuwadia & Singh** | Authors added; the claim narrowed to "device-level variation" (the earlier outline's citation chain is retracted) |
| Zinzuwadia & Singh, *Lancet Digit Health* 2022 | "Wearable devices—addressing bias and inequity," *Lancet Digit Health* 4(12):e856, 2022. About under-representation in wearable research, not device type per se | Wording changed to "unequal wearable access" |
| Błasiok & Nakkiran **2023** | "Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing," **ICLR 2024** | Year → ICLR 2024 |
| Geifman et al. **2018** (AURC) | Geifman, Uziel, El-Yaniv, "Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers," **ICLR 2019** | Year → ICLR 2019 |
| Grinsztajn et al., NeurIPS 2022 | "Why do tree-based models still outperform deep learning on typical tabular data?" NeurIPS 2022 **Datasets & Benchmarks** track | Track added |
| "Wearable panic work is largely next-day risk prediction from single-channel heart rate with self-report labels" (no citation) | Tsai et al., *JMIR Med Inform* 10(2):e33063, 2022: **7-day** horizon, smartwatch HR/sleep/activity **plus questionnaires**, weekly self-report labels. Oh et al., *J Med Internet Res* 27:e69045, 2025: wearable-ECG HRV. Scoping review: Zahed et al., *Digital Health* 2025 (doi 10.1177/20552076251390475) | The claim was inaccurate ("next-day," "single-channel"); rewritten with these citations |

## Verified, no change needed

| Reference | Verified as |
|---|---|
| Schmidt et al., ICMI 2018 | "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection," ICMI '18, doi 10.1145/3242969.3242985 |
| Roelofs et al., AISTATS 2022 | "Mitigating Bias in Calibration Error Estimation," PMLR 151:4036–4054. Defines ECE_SWEEP as the largest equal-mass bin count preserving monotonicity, matching our implementation |
| Bouckaert & Frank, PAKDD 2004 | "Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms," LNCS, pp. 3–12 |
| Varoquaux 2018 | "Cross-validation failure: Small sample sizes lead to large error bars," *NeuroImage* 180:68–77 |
| TRIPOD+AI, BMJ 2024;385:e078378 | Collins et al., 16 April 2024; 27-item checklist |

## Not individually web-checked (standard references; confirm in the final bibliography)
- Nadeau & Bengio, "Inference for the Generalization Error," *Machine Learning* 52:239–281, 2003.
- Task Force of the ESC and NASPE, "Heart rate variability: standards of measurement…," *Circulation* 93(5):1043–1065, 1996.
- Zech et al., *PLOS Med* 2018. Rabanser et al., NeurIPS 2019. DeGrave et al., *Nat Mach Intell* 2021. These are cited only for the companion audit note (Appendix F).

## Sources
- https://arxiv.org/abs/1501.00102 ; https://dl.acm.org/doi/10.1109/TPAMI.2015.2461544
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13119957/ (CognifySSL v2.0)
- https://arxiv.org/abs/2510.24058 (PULSE)
- https://arxiv.org/abs/2507.19264 ; https://arxiv.org/html/2507.19264 (SimMLM)
- https://proceedings.mlr.press/v202/ma23i.html (Ma et al.)
- https://arxiv.org/abs/2603.02719 (Lechuga López et al.)
- https://arxiv.org/abs/2603.13373 ; https://arxiv.org/html/2603.13373 (Roy et al.)
- https://proceedings.iclr.cc/paper_files/paper/2024/file/06cf4bae7ccb6ea37b968a394edc2e33-Paper-Conference.pdf (SmoothECE)
- https://dblp.org/rec/conf/iclr/GeifmanUE19.html (AURC)
- https://proceedings.mlr.press/v151/roelofs22a.html (ECE_SWEEP)
- https://neurips.cc/virtual/2022/poster/55627 (Grinsztajn et al.)
- https://link.springer.com/chapter/10.1007/978-3-540-24775-3_3 (Bouckaert & Frank)
- https://www.sciencedirect.com/science/article/abs/pii/S1053811917305311 (Varoquaux)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/ (TRIPOD+AI)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8889475/ (Tsai et al.) ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12526660/ (Oh et al.) ; https://journals.sagepub.com/doi/10.1177/20552076251390475 (Zahed et al.)
