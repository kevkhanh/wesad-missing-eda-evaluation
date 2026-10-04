# Review traceability (supplement to e14-preprint-outline.md)

Moved out of the outline body in revision 7 (2026-10-02). Not part of the paper.

## Change log

**Round 4 (`peer_reviewe_v2.md`, 2026-10-02):**

| Review item | Where addressed |
|---|---|
| F-block-1 (leaked-number placeholder) | Page 1 COI; Appendix A #1; decision timeline row 19 |
| F-block-2 (version control, release) | §9; Appendix G (author action); `README.md` rewritten as release documentation |
| M1 (causal title, "determines") | Title; Contributions; §1; §6; §8; §10 |
| M2 (missing factorial cell) | §4.3; §5.2; Figure 1 |
| M3 (no equivalence analysis) | §4.5; Table 3 |
| M4 (B–F heuristic; df; participant-level) | §4.5; Table 2; §7 |
| M5 (network deficit with EDA; complete-condition CIs) | §5.3; Table 5; tuned complete-condition Brier |
| M6 (condition-order confound) | §3; §5.6 |
| M7 (HR-response correlation near-tautological) | §5.7 (standardised response, CIs, non-HR models first) |
| M8 (broad literature claim) | §1 "Why report it"; §2 |
| M9 (residual error risk) | §5.11; untraced tokens resolved in `check_numbers.py`; Appendix G |
| Minor 1 ("±0.037") | §5.1 data-bound bullets |
| Minor 2 (within-subject AUROC precision; S10) | §5.7 |
| Minor 3 (bootstrap) | §4.5 (B = 2,000, BCa) |
| Minor 4 (selective prediction references; inverted confidence) | §4.5; §5.9; Figure 3 |
| Minor 5 (cost of masking; masking rate) | §5.3 |
| Minor 6 (MLP encoding) | §4.2 |
| Minor 7 (rule without ECE) | §5.1 |
| Minor 8 (lag noise floor) | Appendix C |
| Minor 9 (legacy Wilcoxon) | Appendix B |
| Minor 10 (ratio CI) | Appendix B |
| Minor 11 (HRV at 60 s) | §5.4 |
| Minor 12 (signal quality) | §4.1; §5.11 |
| Minor 13 (PULSE; per-subject normalisation) | §2; §5.4 |
| Minor 14 (SmoothECE) | §4.5; §5.8 |
| §8 citations | Roy et al. removed (it is a demographic-free fairness method, not about device access); Korea dataset cited via the *Sci Data* descriptor and current Zenodo record; CognifySSL (accuracy + F1, RESP ambiguity) and PULSE (13 test subjects, baseline-referenced normalisation, no-teacher 0.963) corrected; Ma et al. CVPR 2022, SimBaMM, Woo et al. AAAI 2023, Ramazanova et al. CVPRW 2025 added. Verified 2026-10-02 against arXiv 2512.22991, arXiv 2603.13373, arXiv 2510.24058, the CVF open-access pages and the *Sci Data* record |
| §9 (construct validity, Lessmeier context, clinical extrapolation) | §1; §3; §5.7 |
| §10 (provenance, OSF for future confirmatory work, COI management) | Page 1; §8; Appendix G |
| Release of derived data (follow-up to F-block-2; author decision 2026-10-02) | §9 "Data use": WESAD used for research only; commercial work uses other datasets; feature tables released only after the `wesad_readme.pdf` licence wording is recorded (Appendix G item 2) |

**Earlier rounds:** see the archived outlines and `peer-review.md`, `peer_review_updated.md`, `peer_review_round3.md`.

**Round 5 (combined revision list, 2026-10-02):**

| Item | Where addressed |
|---|---|
| A3 ("confirmatory") | Contribution 3 now "robust negative result ... post hoc" |
| A4 (abstract wrist sentence) | Abstract: "motion-corrupted PPG during speech and arithmetic" |
| B1 (gate is a hard mask) | Title; Contributions; §2 SimMLM bullet; §4.3 terminology bullet; §6; abstract |
| B2 (calibration-in-the-large) | §5.2 last bullet; §6 Calibration; abstract |
| B3 (clinical framing) | §1 cut to two sentences; abstract Background; PSVT point moved to §6 "What the proxy cannot say" |
| B4 (log-loss) | §5.1 bullet under Table 1 with paired CIs |
| B5 (0.06 -> 0.04–0.06) | §6; abstract |
| B6 (five vs nine errors) | Contribution 1; §1; §7; abstract |
| B7 (PULSE AUROC) | §2 PULSE paragraph (within-subject LOSO 0.87–0.90) |
| B8 (S10 4.6 SE) | §5.7 |
| B9 (structure) | Round tags and review codes removed from the body; Appendix H moved here; secondary tables moved to the supplementary-tables section; §4.5 names the primary inference scheme |
| B10 (Helminen 2021) | §3: verified *Psychosom Med* 83(3):200–211, HR ES 0.85 (traditional TSST); unsourced "trained / less neurotic" clause removed |
| C1 (posture and movement) | `run_posture_movement.py`; §3; §5.6E; Supplementary Table S7; Contribution 2; §6; §7; §8; abstract |
| C2 (gating where networks use EDA) | §5.3; §5.4 (complete-condition paired CIs under per-subject normalisation); §6; §7 |
| C4 (operating points, decision curves) | `prespec_round5.md`; `run_operating_points.py`; §4.8; §5.9 Table 6, Figure 4 |
| C3 (informative missingness) | `prespec_round5.md`; `run_mnar.py` run; §5.13 text pending |
| D | Not done (Appendix G) |

**Review 9 (expert feedback on revision 8, 2026-10-02):**

| Item | Where addressed |
|---|---|
| 1 (double standard: gate read at df 14, masking at participant level) | §5.2 "Both effects, read under the same three schemes", Table 4 (df-14 column); Contribution 3; §6; §7; abstract. Masking benefit for networks stated as unresolved; tree result (−0.148) is the robust one |
| 2 (§5.13 cannot detect the shortcut) | §5.13 design caveat; post hoc `run_mnar.py --no-added-masking`, Tables 7c/7d, Fig 5b; §4.8; §6; §7; abstract. Shortcut becomes detectable; per-config lead over (ii) shrinks to ≤0.012 |
| 3 (per-configuration result under-stated) | Contribution 3 corollary; §5.3 evidence and caveats; §6 "two simple trees and a switch"; §8 "What to build"; abstract |
| 4 (Discussion contradicts Results: calibration, abstention, robustness) | §6 Calibration and principal-finding bullets rewritten; new `run_intercept_recal.py` (Table 4b): intercept-only fixes the tree, not the networks (slope-only removes 68%) |
| 5 (abstract incomplete) | §10.2 Results rewritten (masking/calibration, per-configuration, MNAR, 20–24 false alarms/h; equivalence led with df-14 p = 0.033) |
| Negative movement partial | §5.7 |
| Equivalence fragile / df-14 lead | Contribution 3; §6; §7; abstract |
| Pitfalls checklist | Appendix H |

