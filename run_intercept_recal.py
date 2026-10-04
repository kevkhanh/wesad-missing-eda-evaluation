"""
Review 9, item 1 / Appendix G 6b (post hoc): does an intercept-only (prior-shift)
per-condition recalibration remove the masking effect?

The outline's reading was "training-time masking mainly teaches a model what base
rate to predict when EDA is absent". If that is right, shifting each model's logit
by one constant per sensor configuration, fit on the training participants'
inner out-of-fold predictions, should close most of the gap between never-masked
and masked models; full Platt (intercept + slope) removed about 71% of the
masking main effect, so the intercept-only shift is compared with it.

Models (reduced = EDA missing, complete = EDA present, same windows):
  gated_dropout_p0.5, gated_no_dropout, mlp_nongated_dropout_p0.5,
  mlp_nongated_no_dropout, lgbm_nan_at_test (i), lgbm_nan_training_p0.5 (ii).
Calibrations, each fit per condition on inner-CV OOF predictions of the training
participants only: raw; Platt (slope + intercept); intercept-only (slope fixed at
1, the ML offset b in sigmoid(logit p + b)); slope-only (intercept fixed at 0,
sigmoid(a * logit p); added after the intercept-only result to test the scale
reading).

Inference as in run_repeated_cv (Bouckaert-Frank df 99 and df 14, participant
level), same 20 partitions and model seeds as the primary run. AUROC is
invariant to either recalibration, so it is not re-reported here.

Not in prespec_round5.md; labelled post hoc / exploratory.

CLI:
    python run_intercept_recal.py            # 20 repeats
    python run_intercept_recal.py --repeats 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_core as ec
from run_metrics import PlattCalibrator
from run_repeated_cv import (MODEL_SEED_BASE, N_FOLDS, SESOI, bouckaert_frank,
                             participant_level)

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "results" / "intercept_recal.json"
EPS = 1e-6
MODELS = ["gated_dropout_p0.5", "gated_no_dropout", "mlp_nongated_dropout_p0.5",
          "mlp_nongated_no_dropout", "lgbm_nan_at_test", "lgbm_nan_training_p0.5"]
G, GN, M, MN = MODELS[:4]
T1, T2 = MODELS[4:]
CALIBS = ("raw", "platt_pc", "intercept", "slope")
TIERS = ("reduced", "complete")


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class InterceptRecalibrator:
    """sigmoid(logit p + b), b by Newton on the Bernoulli log-likelihood (slope = 1)."""

    def fit(self, y, p):
        z, y = _logit(p), np.asarray(y, float)
        b = 0.0
        for _ in range(50):
            s = 1 / (1 + np.exp(-(z + b)))
            g, h = np.sum(s - y), np.sum(s * (1 - s))
            if h < 1e-12:
                break
            step = np.clip(g / h, -2.0, 2.0)
            b -= step
            if abs(step) < 1e-8:
                break
        self.b_ = float(b)
        return self

    def predict(self, p):
        return 1 / (1 + np.exp(-(_logit(p) + self.b_)))


class SlopeRecalibrator:
    """sigmoid(a * logit p), intercept fixed at 0, a by Newton (the complement of the
    intercept-only shift: tests whether the scale of the logit, not its location,
    carries the masking effect)."""

    def fit(self, y, p):
        z, y = _logit(p), np.asarray(y, float)
        a = 1.0
        for _ in range(50):
            s = 1 / (1 + np.exp(-a * z))
            g, h = np.sum((s - y) * z), np.sum(s * (1 - s) * z * z)
            if h < 1e-12:
                break
            step = np.clip(g / h, -1.0, 1.0)
            a = max(a - step, 0.0)
            if abs(step) < 1e-8:
                break
        self.a_ = float(a)
        return self

    def predict(self, p):
        return 1 / (1 + np.exp(-self.a_ * _logit(p)))


def _fold_task(df, rep, fold_i, test_subj):
    import torch
    torch.set_num_threads(1)
    train, test = ec.split_by_subjects(df, test_subj)
    seed = MODEL_SEED_BASE + rep * N_FOLDS + fold_i
    fold_rows, subj_rows, shift_rows = [], [], []
    tiers = ec.paired_tiers(test)
    for name in MODELS:
        factory, _ = ec.MODELS[name]
        model = factory().fit(train, seed=seed)
        oof_y, oof = ec._inner_oof(factory, train, seed)
        cals = {t: {"platt_pc": PlattCalibrator().fit(oof_y, oof[t]),
                    "intercept": InterceptRecalibrator().fit(oof_y, oof[t]),
                    "slope": SlopeRecalibrator().fit(oof_y, oof[t])} for t in TIERS}
        for t in TIERS:
            shift_rows.append({"repeat": rep, "fold": fold_i, "model": name, "tier": t,
                               "intercept_shift": cals[t]["intercept"].b_,
                               "slope": cals[t]["slope"].a_})
            tdf = tiers[t]
            y = tdf.label.values
            p_raw = model.predict(tdf)
            for calib in CALIBS:
                p = p_raw if calib == "raw" else cals[t][calib].predict(p_raw)
                fold_rows.append({"repeat": rep, "fold": fold_i, "model": name, "calib": calib,
                                  "tier": t, "brier": float(np.mean((p - y) ** 2)),
                                  "mean_pred": float(np.mean(p)), "prevalence": float(np.mean(y))})
                for s_, g in tdf.assign(_p=p).groupby("subject_id"):
                    subj_rows.append({"repeat": rep, "model": name, "calib": calib, "tier": t,
                                      "subject_id": s_,
                                      "brier": float(np.mean((g._p - g.label) ** 2))})
    return fold_rows, subj_rows, shift_rows


def run(n_repeats: int = 20, n_jobs: int = 12) -> dict:
    from joblib import Parallel, delayed
    df = ec.load_wesad_frozen()
    n_subj = df.subject_id.nunique()
    tasks, n_tr, n_te = [], [], []
    for rep in range(n_repeats):
        for f, ts in enumerate(ec.subject_kfold(df.subject_id, N_FOLDS, seed=1000 + rep)):
            tasks.append((rep, f, ts))
            n_tr.append(n_subj - len(ts))
            n_te.append(len(ts))
    print(f"[intercept_recal] {len(tasks)} fold tasks x {len(MODELS)} models, n_jobs={n_jobs}",
          flush=True)
    outs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fold_task)(df, r, f, ts) for r, f, ts in tasks)
    folds = pd.DataFrame([r for o in outs for r in o[0]])
    subj = pd.DataFrame([r for o in outs for r in o[1]])
    shifts = pd.DataFrame([r for o in outs for r in o[2]])
    ntr, nte = float(np.mean(n_tr)), float(np.mean(n_te))

    contrasts = {
        "main_effect_masking": [(0.5, G), (-0.5, GN), (0.5, M), (-0.5, MN)],
        "main_effect_gate": [(0.5, G), (0.5, GN), (-0.5, M), (-0.5, MN)],
        "interaction_gate_x_masking": [(1, G), (-1, GN), (-1, M), (1, MN)],
        "masking_gated": [(1, G), (-1, GN)],
        "masking_mlp": [(1, M), (-1, MN)],
        "gate_masked": [(1, G), (-1, M)],
        "gate_never_masked": [(1, GN), (-1, MN)],
        "masking_tree_(ii)_minus_(i)": [(1, T2), (-1, T1)],
    }
    res = {"config": {"n_repeats": n_repeats, "n_subjects": int(n_subj), "models": MODELS,
                      "calibrations": CALIBS, "sesoi": SESOI, "post_hoc": True},
           "per_model": {}, "contrasts": {}, "intercept_shift": {}}

    def signed(frames, terms, col="brier"):
        out = None
        for w, n in terms:
            v = w * (frames[n][col] if col else frames[n])
            out = v if out is None else out + v
        return out

    for tier in TIERS:
        for calib in CALIBS:
            fc = folds[(folds.tier == tier) & (folds.calib == calib)]
            sc = subj[(subj.tier == tier) & (subj.calib == calib)]
            res["per_model"][f"{tier}:{calib}"] = {
                n: {"brier": float(g.brier.mean()), "mean_pred": float(g.mean_pred.mean()),
                    "prevalence": float(g.prevalence.mean())} for n, g in fc.groupby("model")}
            frames = {n: g.set_index(["repeat", "fold"]) for n, g in fc.groupby("model")}
            sm = {n: g.groupby("subject_id").brier.mean() for n, g in sc.groupby("model")}
            res["contrasts"][f"{tier}:{calib}"] = {
                cn: {"bf": bouckaert_frank(signed(frames, t).values, ntr, nte, n_subj,
                                           SESOI["brier"]),
                     "participant": participant_level(signed(sm, t, None), SESOI["brier"], True)}
                for cn, t in contrasts.items()}
        res["intercept_shift"][tier] = {
            n: {"mean": float(g.intercept_shift.mean()), "sd": float(g.intercept_shift.std()),
                "slope_mean": float(g.slope.mean()), "slope_sd": float(g.slope.std())}
            for n, g in shifts[shifts.tier == tier].groupby("model")}

    # Share of the raw masking main effect (reduced) removed by each recalibration.
    raw = res["contrasts"]["reduced:raw"]["main_effect_masking"]["bf"]["mean"]
    res["share_of_masking_main_effect_removed"] = {
        c: float(1 - res["contrasts"][f"reduced:{c}"]["main_effect_masking"]["bf"]["mean"] / raw)
        for c in ("platt_pc", "intercept", "slope")}

    OUT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
    folds.to_csv(OUT_PATH.with_name("intercept_recal_per_fold.csv"), index=False)
    subj.to_csv(OUT_PATH.with_name("intercept_recal_per_subject.csv"), index=False)
    _print(res)
    print(f"\n[intercept_recal] wrote {OUT_PATH}")
    return res


def _fmt(e):
    return f"{e['mean']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"


def _print(res):
    for k, pm in res["per_model"].items():
        print(f"\n{k}: " + "  ".join(f"{n.split('_')[0]}{'_nm' if 'no_' in n else ''} "
                                      f"{v['brier']:.3f}" for n, v in pm.items()))
    for k, cs in res["contrasts"].items():
        if not k.startswith("reduced"):
            continue
        print(f"\n=== {k} ===")
        for cn, v in cs.items():
            b, p = v["bf"], v["participant"]
            print(f"  {cn:32s} BF99 {_fmt(b)} | BF14 [{b['df_subjects']['ci95'][0]:+.3f}, "
                  f"{b['df_subjects']['ci95'][1]:+.3f}] | participant {_fmt(p)} "
                  f"({p['n_A_better']}/{p['n']})")
    print("\nshare of masking main effect removed:", res["share_of_masking_main_effect_removed"])
    print("intercept shifts (reduced):", json.dumps(res["intercept_shift"]["reduced"], indent=None))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    run(a.repeats, a.jobs)
