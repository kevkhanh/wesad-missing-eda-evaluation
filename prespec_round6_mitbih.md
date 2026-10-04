# Pre-specification, round 6: arrhythmia failure-mode check on MIT-BIH (written 2026-10-04, before any code for it existed)

Status of this file: written after the primary WESAD results were known, before `run_mitbih_failure.py`
existed. Its SHA-256 is recorded in `results/prespec_round6_manifest.json` and the script refuses to run
if the file has changed. This is NOT an independent timestamp (the repository was created the same day).
The check is **not** a clinical validation: it is a probe of what the WESAD-trained, EDA-missing systems
do when given arrhythmia ECG, to test the claim in the Discussion that an HR-driven model would call a
tachyarrhythmia "stress-like".

## Question
With EDA missing, how often do the WESAD-trained systems output "stress" on MIT-BIH windows with a fast
ventricular or supraventricular rate, and does that depend on heart rate?

## Systems (trained on all 434 frozen WESAD windows, per-condition Platt from inner OOF, reduced tier only)
Hard-masked network + dropout (`gated_dropout_p0.5`), masked MLP (`mlp_nongated_dropout_p0.5`),
NaN-trained LightGBM (`lgbm_nan_training_p0.5`), ECG-only LightGBM (`lgbm_ecg_only_per_config`), HR-only
(`hr_only`), and two HR-removed contrasts (`gated_dropout_hr_removed`, `lgbm_ecg_only_hr_removed`).
Seeds 0-4; neural predictions are averaged over seeds, LightGBM is run per seed and averaged.
Alert threshold: calibrated probability >= 0.5 (per-condition Platt, reduced tier); raw >= 0.5 as a check.

## Test windows
MIT-BIH Arrhythmia Database, all non-paced records (the loader's exclusion list), lead MLII, 360 Hz,
non-overlapping 60-s windows, windows containing noise/unreadable symbols (`~`, `U`, `|`) excluded.
Annotated heart rate of a window = number of annotated beats in the window (beat symbols
`N L R B A a J S V r F e j n E / f Q`), i.e. beats per minute. Features come from the same extractor as
WESAD with EDA absent. Grouping uses annotations only (never the model outputs):
- rate bands (any rhythm): < 100 bpm, 100-119 bpm, >= 120 bpm;
- rhythm groups (window contains the rhythm tag, or the rhythm in force at its start): normal sinus only
  (rhythm `(N`, all beats `N`, >= 30 beats; the control), SVT `(SVTA`, AF/flutter `(AFIB`/`(AFL`, VT `(VT`.
Windows are clustered by record; records 201/202 are one cluster (same patient).

## Primary endpoint
Flag rate (fraction of windows with calibrated p >= 0.5) among windows with annotated rate >= 120 bpm,
per system, with a record-cluster bootstrap 95% interval (B = 2,000). **Floor, fixed in advance:** if there
are fewer than 30 such windows or fewer than 5 records contributing, the predictions below are reported as
not testable and only descriptive numbers are given.

## Pre-stated predictions
- **P1.** For the hard-masked network, the masked MLP, (ii), (iii) and HR-only, the flag rate at >= 120 bpm
  is at least 0.50 (and so at least as high as these systems' WESAD stress-window flag rate, 0.65-0.75).
- **P2.** The HR-removed contrasts flag at least 0.20 less than their HR-containing counterparts
  (`gated_dropout_hr_removed` vs `gated_dropout_p0.5`; `lgbm_ecg_only_hr_removed` vs `lgbm_ecg_only_per_config`).
- **P3.** Across all MIT-BIH windows, the Spearman correlation of calibrated p(stress) with annotated rate is
  >= 0.50 for the HR-containing systems (record-cluster bootstrap interval reported).
- **P4 (control, descriptive).** On normal-sinus windows the flag rate is <= 0.35 for every system
  (WESAD baseline false-positive rate is 0.14-0.22; a looser bound allows for sensor and population shift).
- **P5 (descriptive).** SVT, AF/flutter and VT windows are reported by rhythm and by rate band, because
  a rhythm label alone does not fix the heart rate.

## What would count as the claim failing
If P1 fails (flag rate < 0.50 at >= 120 bpm for the HR-containing systems), the Discussion's statement that an
HR-driven model would call a tachyarrhythmia "stress-like" is not supported on this probe and the paper says so.

## Caveats stated in advance
Different sensor and sampling rate (360 Hz ambulatory MLII vs 700 Hz chest ECG); patients, not healthy
volunteers; no EDA and no stress labels; WESAD and MIT-BIH are separable by dataset identity (companion audit,
identity AUC 0.83), so part of any flag/no-flag difference can be domain shift; R-peak detection can be less
reliable on wide-complex rhythms; windows within a record are autocorrelated; the 0.5 threshold is on
probabilities calibrated to a 35%-stress laboratory sample. The probe shows what these systems do, not what
a wearable could do, and does not evaluate any panic-versus-cardiac classifier.
