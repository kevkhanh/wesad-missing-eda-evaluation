# E14 robustness: missing-EDA evaluation on WESAD

Code and derived data for the preprint drafted in `e14-preprint-outline.md`
(see `paper/main.pdf`; the long supplement is the outline), plus the separate WESAD × MIT-BIH dataset-identity
audit (`audit/`, companion note).

Nothing in this repository was pre-registered. `configs/preregistration.yaml`
is a thresholds file written without version control; see the outline's
page-1 provenance note and `results/decision_timeline.md`.

## Data

- WESAD raw `.pkl` files (Schmidt et al., ICMI 2018) are **not** redistributed.
  Set `WESAD_ROOT` to a folder containing `S2.pkl … S17.pkl`.
- Derived feature tables:
  - primary (frozen): `../stage3_hb1_dataset.csv`, SHA-256 in
    `results/feature_table_manifest.json` (434 baseline/stress windows);
  - extended (round 4): `results/extended_features.csv`, SHA-256 in
    `results/extended_table_manifest.json` (683 windows: + amusement,
    meditation, session timing, signal quality, wrist E4 features). Its
    manifest records that its chest baseline/stress rows reproduce the frozen
    table exactly from the raw files.
- Licence and data use: see `DATA_NOTICE.md`. WESAD terms (checked 2026-10-04): scientific,
  non-commercial use with credit to Schmidt et al. (ICMI 2018); redistribution of derived data is not
  addressed, so the window-level feature tables are withheld and only their hashes are published.
  Code is MIT-licensed (`LICENSE`). WESAD-derived models and features are not used commercially.
- MIT-BIH (round-6 probe only): set `MITBIH_ROOT` to a folder with the `.dat/.hea/.atr` files.

## Quick check by an outside reader (no WESAD download needed for the first two)

```bash
pip install -r requirements.txt
python -m pytest -q tests     # 44 tests, ~1 min: a regression test for each pipeline error
python check_numbers.py          # ties the outline's numbers to the stored results files
```

To re-run the primary comparison you need the official WESAD files (see below). Corrections and
independent re-runs are welcome via GitHub issues.

## Reproducing the paper's numbers (CPU, Python 3.13; `pip install -r requirements.txt`)

```bash
export WESAD_ROOT=/path/to/wesad_pkls
python extract_extended.py                     # extended table + check vs frozen table (~1 min)
python run_repeated_cv.py                      # Tables 1-2, factorial, equivalence (20x5 CV)
python run_repeated_cv.py --set tuned          # nested tuning
python run_repeated_cv.py --set capacity       # matched-size gated vs MLP
python run_repeated_cv.py --set hrv_subset     # mean NN + RMSSD only
python run_repeated_cv.py --set maskrate       # training masking-rate sweep
python run_repeated_cv.py --transform subjnorm_all
python run_repeated_cv.py --transform subjnorm_baseline
python run_repeated_cv.py --table wrist        # E4 wrist replication
python run_baselines.py                        # LOSO / single 5-fold pooled panels (BCa CIs)
python run_baselines.py --table wrist --schemes loso
python run_order_confound.py                   # condition-order checks
python run_posture_movement.py                  # posture/movement confound (raw .pkl ACC; ~40 min, bootstrap-bound)
python run_participant_analyses.py             # HR response, S10, signal quality, small stats
python run_operating_points.py                 # round 5 (prespec_round5.md): operating points, decision curves (~4 min)
python run_mnar.py                             # round 5 (prespec_round5.md): label/motion-driven EDA missingness (~10 min)
python run_mnar.py --no-added-masking          # review 9, post hoc: same, without each system's own 50% MCAR training mask (~5 min)
python run_intercept_recal.py                  # review 9, post hoc: intercept-only / slope-only / Platt recalibration (~4 min)
python run_mitbih_failure.py                   # round 6 (prespec_round6_mitbih.md): arrhythmia failure-mode probe on MIT-BIH (~2 min)
python reimpl/minimal_primary.py               # author-side minimal re-implementation
python make_figures.py
python check_numbers.py                        # ties outline numbers to results files
python -m pytest -q tests
```

Seeds: model seed `200 + repeat*5 + fold`; partition seed `1000 + repeat`
(the 20 partitions are stored in `results/repeated_cv.json` → `config.partitions`).

## Layout

```
eval_core.py              models, frozen-table loading, paired tiers, OOF Platt
run_metrics.py            Brier, ECE_SWEEP, SmoothECE, AURC/E-AURC, slopes
run_repeated_cv.py        repeated CV + Bouckaert-Frank, participant-level, TOST/ROPE
run_baselines.py          LOSO / 5-fold pooled panels, participant bootstrap (BCa)
run_order_confound.py     stress-vs-amusement, negative control, baseline-vs-meditation
run_posture_movement.py   chest-ACC movement/posture vs heart rate (C1)
run_participant_analyses.py
run_operating_points.py   sens/spec at OOF-chosen thresholds, abstention by class, net benefit (C4)
run_mnar.py               informative (label-/motion-driven) EDA missingness, shortcut index (C3)
prespec_round5.md         pre-specification for the two scripts above (SHA-256 in results/)
extract_extended.py       raw .pkl -> extended feature table (chest + wrist)
reimpl/minimal_primary.py minimal re-implementation (no project imports)
check_numbers.py          outline <-> results cross-check
tests/                    regression tests (one per implementation error in Appendix A)
audit/                    dataset-identity audit (companion note)
results/                  every number in the outline; archive/ holds superseded outputs
```
