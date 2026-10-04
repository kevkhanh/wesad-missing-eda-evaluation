"""Dataset-identity audit for the E14 Preprint 1 pipeline.

Implements the full 10-step protocol from dataset_identity_preprint_01.md.
Each step is one module. Everything writes to results/audit/.

  step0_freeze              env pin + prove EDA/resp missingness is
                             dataset-collinear (Driver #1)
  step1_splits              build C1/C2/C3/C4/N1/N2 comparison sets
                             (subject-grouped)
  step2_common_modality     F_ECG restriction + EDA-ablation impact report
  step3_shortcut            frozen production-model shortcut test (RUN FIRST)
  step4_identity_probe      P1-P4 identity probes with subject-level CV,
                             cluster-bootstrap CI, subject-level permutation
  step5_mmd                 MMD^2 with block permutation + MMDAgg fallback
                             + heterogeneity ratio R vs N1/N2
  harmonize                 H0-H7 raw-signal and feature-level transforms
  step6_harmonization       cumulative ladder + leave-one-stage-out
                             attribution
  step7_within_dataset      C3 (WESAD baseline vs stress, LOSO) and C4
                             (MIT-BIH NSR vs arrhythmic, GroupKFold),
                             plus feature-elimination sweep (Branch D check)
  step8_alignment           conditional CORAL diagnostic on matched-normal
                             (C1) windows only
  step9_decision            Branch A/B/C/D decision + preprint claim patch
  run_all                   orchestrator that runs Steps 0->3->...->9
"""
