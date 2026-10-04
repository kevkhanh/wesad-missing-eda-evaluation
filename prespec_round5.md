# Pre-specification, round 5: operating points (C4) and informative missingness (C3)

**Written:** 2026-10-02T05:46Z, before any code for these analyses existed and before any
of them was run. **Not independently timestamped**: the project is not yet under version
control. The SHA-256 of this file is recorded by `run_operating_points.py` and `run_mnar.py`
in every output they write; any edit after the runs would change that hash. If this file is
committed to git before the outputs are, the commit order is the timestamp.

**Status of what is already known.** All round-1 to round-4 results are known, including
the per-fold raw and calibrated Brier/AUROC of every model in the default set. Nothing about
operating points, net benefit or MNAR missingness has been computed on this data.

Shared settings for both analyses:
- Frozen chest table (434 windows, SHA-256 `df483a7a…`), the same 20 participant partitions
  (`subject_kfold(..., seed=1000 + repeat)`) and the same model seeds as `run_repeated_cv.py`.
- Models: gated network + dropout (G), non-gated MLP with the same masking (M), NaN-trained
  LightGBM (ii), and the **per-configuration system (PC)**: complete-trained LightGBM (i)
  when EDA is present, ECG-only LightGBM (iii) when it is absent.
- Inference exactly as in `run_repeated_cv.py`: Bouckaert–Frank over the 100 fold
  differences (df 99, df 14 sensitivity), and participant-level t (df 14) with exact
  sign-flip on per-participant means over repeats.

---

## C4. Clinical operating points, abstention by class, decision curves

**Question.** Do the Brier/AUROC near-ties translate into near-ties at clinically readable
operating points, and is ±0.02 Brier a decision-relevant margin?

**Thresholds are chosen on training participants only**, from the inner subject-level
4-fold out-of-fold predictions already used to fit Platt, separately for each condition
(complete, reduced):
- high-sensitivity point: threshold = 10th percentile of OOF scores among stress windows
  (positive if p ≥ threshold), target sensitivity 0.90;
- high-specificity point: threshold = 90th percentile of OOF scores among baseline windows
  (positive if p > threshold), target specificity 0.90;
- fixed 0.5 on per-condition-Platt probabilities (check).

**Per fold and condition we report:** sensitivity, specificity, PPV, NPV at each point;
false alarms per hour of rest = (1 − specificity) × 60; missed stress windows per 10 minutes
of stress = (1 − sensitivity) × 10; at 60% and 80% coverage (confidence = max(p, 1 − p),
per-condition Platt), sensitivity and specificity among retained windows and the abstention
rate separately in stress and baseline windows.

**Decision curves.** Net benefit NB(pₜ) = TP/n − FP/n · pₜ/(1 − pₜ) on per-condition-Platt
probabilities, pₜ = 0.05 … 0.60 in steps of 0.05, with treat-all and treat-none. **Inference
only at three pre-stated thresholds, pₜ ∈ {0.20, 0.35, 0.50}.** Raw-probability curves are
supplementary.

**Comparisons (A − B):** G − M, G − (ii), G − PC, M − PC, (ii) − PC, in both conditions.
(In the reduced condition PC = (iii).)

**Margins (fixed now).** Sensitivity and specificity ±0.05; net benefit ±0.01 (one net true
positive per 100 windows). Equivalence by TOST at α = 0.05.

**Predictions (written before running).**
- C4-P1. In the reduced condition, G − M is within ±0.05 sensitivity and specificity at both
  operating points, and within ±0.01 NB at all three pₜ.
- C4-P2. In the reduced condition, G − (ii) and G − PC intervals include 0 for sensitivity,
  specificity and NB; we make no equivalence prediction for them.
- C4-P3. In the complete condition, PC has higher NB than G at pₜ = 0.35 (it has lower Brier).
- C4-P4. At 60% coverage, stress windows are abstained on at a higher rate than baseline
  windows for every model (the weak heart-rate responders).

**Check.** Raw per-fold Brier from this run must reproduce `results/repeated_cv_per_fold.csv`
for G, M, (i), (ii), (iii) to within 1e-9; otherwise the run is invalid.

**Caveat (fixed now).** Prevalence is 35% by design, so PPV, NPV and NB do not transfer to
deployment; sensitivity and specificity are the portable quantities.

---

## C3. Informative (MNAR) EDA missingness

**Question.** If EDA dropout is correlated with the label, do models that see missingness
(G, M through its indicator or closed gate, (ii) through its NaN branch) learn "EDA missing
→ stress"? Does that look like a gain when the mechanism is stable and turn into a loss when
it changes? Does gated ≈ MLP still hold?

**Mixed test sets.** Unlike the paired design, each test window has its EDA (all six
EDA-dependent features) missing with a window-specific probability; every model sees the
same random draw. Overall missing rate 30% in training data and in test data.

**Mechanisms** (calibrated on the full 434-window table; the label is used only to set the
simulation parameters):
- **MCAR:** p = 0.30 for every window.
- **Label-driven (primary):** p(missing | stress) and p(missing | baseline) with overall rate
  0.30 and odds ratio OR ∈ {2, 4, 8}.
- **Motion-driven (secondary):** logit p = a + b · z, z = standardised log chest-accelerometer
  magnitude SD (`chest_acc_mag_sd`, joined from the extended table, which reproduces the
  frozen table row for row); a, b solved so the overall rate is 0.30 and the implied
  stress-vs-baseline missingness OR equals 2, 4 or 8.

**Training.** "Natural" missingness is applied once to the training participants' windows
under the training mechanism (seeded per fold), then every model trains as in the default
run: G and M keep 50% MCAR EDA dropout on top, (ii) keeps 50% MCAR NaN masking on top. In PC,
(i) is trained on the training windows that have EDA, (iii) on all training windows.

**Scenarios.**

| Scenario | Training mechanism | Test mechanism |
|---|---|---|
| S0 sanity | MCAR | MCAR |
| S1 matched | MNAR (mechanism, OR) | same MNAR |
| S2 shortcut break | MNAR (mechanism, OR) | MCAR |
| S3 MCAR-trained, deployed under MNAR | MCAR | MNAR (mechanism, OR) |

for both mechanisms and OR ∈ {2, 4, 8}: 19 scenarios.

**Primary scenarios for inference:** label-driven OR = 4, S1 and S2. All other scenarios are
reported descriptively (point estimates and B–F intervals, no claims).

**Metrics.** Brier (primary), AUROC, log-loss on the mixed test set, raw probabilities
(primary); secondary: Platt fit on inner-OOF predictions whose inner test windows carry the
training mechanism's missingness. **Shortcut index:** for each model and training mechanism,
the mean over test **baseline** windows of p(EDA removed) − p(EDA present) (paired, raw);
also reported for stress windows.

**Comparisons (A − B):** G − M, G − (ii), G − PC, M − PC, (ii) − PC. Same margins as the
main paper: ±0.02 Brier, ±0.03 AUROC.

**Predictions (written before running).**
- C3-P1. Shortcut index > 0 for G, M and (ii) under label-driven training, increasing with
  OR; near 0 under MCAR training.
- C3-P2. In S1 (label, OR 4), G, M and (ii) have lower Brier than PC.
- C3-P3. In S2 (label, OR 4), the Brier of G, M and (ii) is higher than in S0.
- C3-P4. G − M is within ±0.02 Brier (TOST) in both primary scenarios.
- No prediction is made for S3 or for the motion mechanism.

**Known caveat (written now).** PC is not fully immune: under label-driven missingness the
EDA-present training windows have lower stress prevalence, so (i) learns a shifted prior.

**Reporting.** Every prediction is reported as confirmed or not confirmed, whichever way it
goes. Any analysis added after the runs is labelled exploratory.
