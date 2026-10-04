# Round-4 rerun summary (2026-10-02, `peer_reviewe_v2.md`)

Authoritative numbers are in the JSON files named below; the outline
(`e14-preprint-outline.md`, revision 6) is tied to them by `check_numbers.py`
(593 PASS, 0 FAIL, 0 untraced).

## Reproduction
- Default 20× repeated CV reproduces every round-3 Table 1/2 number exactly (same seeds).
- LOSO point estimates identical; CIs now B = 2,000 with BCa.
- `extract_extended.py` reproduces the frozen table from raw WESAD `.pkl`: 434/434 rows, 0 mismatched values.
- Author-side minimal re-implementation (`reimpl/minimal_primary.py`): all pairwise reduced ΔBrier within ±0.007, CIs include 0; agreement criterion met. Not independent.

## Main new results (`repeated_cv.json`, raw, reduced condition)
- Factorial cells: gated+mask 0.148, MLP+mask 0.146, gated no-mask 0.186, MLP no-mask 0.197; tree (i) 0.294, (ii) 0.146.
- Gate never masked: −0.011 [−0.067, +0.045]; interaction +0.013 [−0.042, +0.069]; masking main effect −0.044 [−0.088, −0.001] (participant 12/15, sign-flip p 0.002); gate main effect −0.005.
- Post-hoc TOST (±0.02 Brier, ±0.03 AUROC): gated vs MLP passes (p 0.025 / 0.009); gated vs (ii)/(iii) bound ≈ 0.030–0.031 (B–F), passes only at participant level (p 0.047 / 0.052); AUROC fails (bounds 0.066–0.070). MDD 0.047 vs (ii), 0.026 vs MLP.
- Complete condition: (i) 0.072, (ii) 0.093, gated 0.130, MLP 0.141; gated − (i) +0.058 [−0.012, +0.127], participant [+0.010, +0.105]. Tuned gated 0.129, MLP 0.139, (ii) 0.098.
- Masking-rate sweep: LightGBM complete Brier 0.074/0.079/0.093/0.128 at p 0.1/0.25/0.5/0.75; networks no cost.
- HRV subset improves networks (0.140/0.142); rule no longer fires (raw). Per-subject normalisation: all ≈ 0.07 Brier, AUROC ≈ 0.94–0.95, still indistinguishable.

## Order confound (`order_confound.json`)
- Stress vs amusement (order-balanced, 7 vs 8): reduced within-subject AUROC 0.89–0.95; HR-only 0.938; HR removed 0.59–0.70.
- Negative control: calm conditions flagged at 0.10–0.28 vs baseline 0.14–0.22, stress 0.65–0.75.
- Baseline vs meditation: EDA-only 0.42 (CI includes 0.5) → no tonic-drift evidence.
- S10 inversion explained: baseline HR 99 bpm (highest of 15), anticipatory arousal at session start.

## Wrist (`repeated_cv_wrist.json`, `participant_analyses.json`)
- Gating null replicates (gated vs MLP +0.001 [−0.013, +0.014]).
- No-EDA wrist AUROC 0.93 but wrist HR-only 0.64: wrist "HRV" tracks PPG artefact fraction (ρ 0.73–0.83), uncorrelated with chest RMSSD; wrist HR error 14.6 bpm under stress. Shortcut, not cardiac.
