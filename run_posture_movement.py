"""
Round-5 C1: posture and movement confound in the heart-rate mechanism.

The TSST is performed standing and speaking; baseline, amusement and
meditation are seated (or reclined). Chest accelerometer movement alone
separates stress from baseline (pooled AUROC 0.91, §5.11), more than HR-only
(0.82). This script asks how much of the "heart-rate mechanism" survives
once movement and trunk posture are accounted for.

Inputs: the hash-checked extended table (eval_core.load_extended) plus
per-window chest-ACC features extracted here from the raw WESAD .pkl files
(same 60-s windows, keyed by subject_id + start_sample):
  posture  = per-axis mean of chest ACC (gravity vector -> trunk tilt)
  movement = magnitude SD (already in the extended table) + per-axis SD

Caveat built into the design: a chest accelerometer measures trunk tilt,
not standing vs upright sitting, so "posture" here cannot remove the
orthostatic component of HR during the (standing) TSST.

Analyses (LOSO, LightGBM with the project defaults: 50 trees, depth 3, lr 0.1):
  A. ACC-only / posture-only / movement-only baselines; HR-only; ACC + HR;
     ECG-only; ECG + ACC. Pooled and within-subject AUROC, Brier, with
     participant-cluster percentile bootstrap CIs (B = 2,000), and paired
     bootstrap CIs for (ACC + HR) - ACC and (ECG + ACC) - ACC.
  B. HR residualised on movement + posture within each participant
     (label-free, transductive OLS); within-subject AUROC of the residual.
  C. Low-motion stress windows: stress windows whose chest ACC magnitude SD
     is at or below that participant's maximum baseline value; HR-only and
     ACC within-subject AUROC on that subset.
  D. Stress vs amusement (both upright-seated or standing by chest tilt):
     within-subject AUROC of posture, movement and HR; stress-minus-amusement
     and stress-minus-baseline HR in bpm.

Output: results/posture_movement.json, results/posture_features.csv
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import eval_core as ec
from run_metrics import within_subject_auroc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "posture_movement.json"
POSTURE_CSV = RESULTS / "posture_features.csv"
WESAD_ROOT = os.environ.get("WESAD_ROOT", r"D:\wesad_pkls")
FS, WIN_S = 700, 60
N_BOOT = 2000
SEED = 300

POSTURE = ["acc_x_mean", "acc_y_mean", "acc_z_mean"]
MOVE = ["chest_acc_mag_sd", "acc_x_sd", "acc_y_sd", "acc_z_sd"]
ACC = MOVE + POSTURE
HR = ["hrv_mean_nn"]
ECG = list(ec.ECG_FEATURE_NAMES)

FEATURE_SETS = {
    "movement_only": MOVE,
    "posture_only": POSTURE,
    "acc_only": ACC,
    "hr_only": HR,
    "acc_plus_hr": ACC + HR,
    "ecg_only": ECG,
    "ecg_plus_acc": ECG + ACC,
}


# ---------------------------------------------------------------------------
# Posture / movement features from the raw files
# ---------------------------------------------------------------------------

def _subject_posture(sid: str, starts: list[int]) -> list[dict]:
    with open(Path(WESAD_ROOT) / f"{sid}.pkl", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    acc = np.asarray(d["signal"]["chest"]["ACC"], dtype=float)
    W = FS * WIN_S
    rows = []
    for st in starts:
        a = acc[st:st + W]
        m, sd = a.mean(0), a.std(0)
        rows.append({"subject_id": f"WESAD_{sid}", "start_sample": int(st),
                     "acc_x_mean": m[0], "acc_y_mean": m[1], "acc_z_mean": m[2],
                     "acc_x_sd": sd[0], "acc_y_sd": sd[1], "acc_z_sd": sd[2]})
    return rows


def posture_features(ext: pd.DataFrame) -> pd.DataFrame:
    if POSTURE_CSV.exists():
        return pd.read_csv(POSTURE_CSV)
    jobs = [(s.replace("WESAD_", ""), sorted(g.start_sample.astype(int).tolist()))
            for s, g in ext.groupby("subject_id")]
    out = Parallel(n_jobs=min(15, len(jobs)))(delayed(_subject_posture)(s, st) for s, st in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(POSTURE_CSV, index=False)
    return df


# ---------------------------------------------------------------------------
# LOSO + bootstrap
# ---------------------------------------------------------------------------

def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], seed: int) -> np.ndarray:
    m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                           random_state=seed, verbosity=-1, n_jobs=1)
    m.fit(tr[cols].values, tr["y"].values)
    return m.predict_proba(te[cols].values)[:, 1]


def loso(df: pd.DataFrame) -> pd.DataFrame:
    subjects = sorted(df.subject_id.unique())

    def one(i, s, name):
        tr, te = df[df.subject_id != s], df[df.subject_id == s]
        p = _fit_predict(tr, te, FEATURE_SETS[name], SEED + i)
        return pd.DataFrame({"row_id": te.row_id.values, "subject_id": s, "model": name,
                             "y": te.y.values, "p": p})
    outs = Parallel(n_jobs=15)(delayed(one)(i, s, n) for i, s in enumerate(subjects)
                               for n in FEATURE_SETS)
    return pd.concat(outs, ignore_index=True)


def _stats(y, p, sid):
    return (roc_auc_score(y, p), within_subject_auroc(y, p, sid), float(np.mean((p - y) ** 2)))


def bootstrap(pred: pd.DataFrame, contrasts: list[tuple[str, str]]) -> dict:
    """Participant-cluster percentile bootstrap for every model and paired
    contrast (same resampled participants for both arms)."""
    wide = pred.pivot_table(index=["row_id", "subject_id", "y"], columns="model", values="p").reset_index()
    subj = np.array(sorted(wide.subject_id.unique()))
    idx = {s: np.flatnonzero(wide.subject_id.values == s) for s in subj}
    models = [c for c in FEATURE_SETS if c in wide.columns]
    y_all = wide.y.values.astype(int)
    P = {m: wide[m].values for m in models}

    def compute(rows, sid):
        y = y_all[rows]
        return {m: _stats(y, P[m][rows], sid) for m in models}

    point = compute(np.arange(len(wide)), wide.subject_id.values)
    rng = np.random.default_rng(17)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(subj, len(subj), replace=True)
        rows = np.concatenate([idx[s] for s in pick])
        sid = np.concatenate([np.full(len(idx[s]), j) for j, s in enumerate(pick)])
        if len(np.unique(y_all[rows])) < 2:
            continue
        boots.append(compute(rows, sid))

    def ci(vals):
        return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

    out = {"models": {}, "contrasts": {}}
    keys = ["pooled_auroc", "within_subject_auroc", "brier"]
    for m in models:
        out["models"][m] = {k: {"est": float(point[m][j]), "ci95": ci([b[m][j] for b in boots])}
                            for j, k in enumerate(keys)}
    for a, b in contrasts:
        out["contrasts"][f"{a}__minus__{b}"] = {
            k: {"est": float(point[a][j] - point[b][j]),
                "ci95": ci([bb[a][j] - bb[b][j] for bb in boots])}
            for j, k in enumerate(keys)}
    return out


# ---------------------------------------------------------------------------
# B. HR residualised on movement + posture (within participant)
# ---------------------------------------------------------------------------

def hr_residualised(df: pd.DataFrame) -> dict:
    res = []
    r2 = []
    for s, g in df.groupby("subject_id"):
        X = np.column_stack([np.ones(len(g)), g[ACC].values])
        beta, *_ = np.linalg.lstsq(X, g["hrv_mean_nn"].values, rcond=None)
        fit = X @ beta
        resid = g["hrv_mean_nn"].values - fit
        ss = np.sum((g.hrv_mean_nn - g.hrv_mean_nn.mean()) ** 2)
        r2.append(1 - np.sum(resid ** 2) / ss)
        res.append(pd.DataFrame({"subject_id": s, "y": g.y.values, "resid": resid,
                                 "hr": g.hrv_mean_nn.values, "accsd": g.chest_acc_mag_sd.values}))
    r = pd.concat(res, ignore_index=True)
    # Lower NN interval = higher HR = stress, so score with the negated value.
    per = {s: {"hr": roc_auc_score(g.y, -g.hr), "hr_resid": roc_auc_score(g.y, -g.resid),
               "acc_sd": roc_auc_score(g.y, g.accsd)} for s, g in r.groupby("subject_id")}
    rng = np.random.default_rng(23)
    subj = np.array(list(per))
    out = {"r2_median": float(np.median(r2)), "r2_range": [float(min(r2)), float(max(r2))]}
    for k in ["hr", "hr_resid", "acc_sd"]:
        v = np.array([per[s][k] for s in subj])
        bs = [rng.choice(v, len(v)).mean() for _ in range(N_BOOT)]
        out[k] = {"within_subject_auroc": float(v.mean()),
                  "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "n_above_0.5": int((v > 0.5).sum()), "per_subject": {s: float(per[s][k]) for s in subj}}
    return out


# ---------------------------------------------------------------------------
# C. Low-motion stress windows
# ---------------------------------------------------------------------------

def low_motion(df: pd.DataFrame) -> dict:
    keep = []
    for s, g in df.groupby("subject_id"):
        thr = g.loc[g.y == 0, "chest_acc_mag_sd"].max()
        keep.append(g[(g.y == 0) | (g.chest_acc_mag_sd <= thr)])
    k = pd.concat(keep)
    n_stress = k[k.y == 1].groupby("subject_id").size().reindex(sorted(df.subject_id.unique())).fillna(0)
    use = n_stress[n_stress >= 3].index
    kk = k[k.subject_id.isin(use)]
    per = {s: {"hr": roc_auc_score(g.y, -g.hrv_mean_nn), "acc_sd": roc_auc_score(g.y, g.chest_acc_mag_sd)}
           for s, g in kk.groupby("subject_id")}
    full = {s: roc_auc_score(g.y, -g.hrv_mean_nn) for s, g in df[df.subject_id.isin(use)].groupby("subject_id")}
    rng = np.random.default_rng(29)
    out = {"n_stress_windows_kept": int(n_stress.sum()), "n_stress_windows_total": int((df.y == 1).sum()),
           "n_participants_ge3": int(len(use)), "stress_kept_per_subject": n_stress.astype(int).to_dict()}
    for key, vals in [("hr_low_motion", [per[s]["hr"] for s in use]),
                      ("acc_sd_low_motion", [per[s]["acc_sd"] for s in use]),
                      ("hr_all_windows_same_participants", [full[s] for s in use])]:
        v = np.array(vals)
        bs = [rng.choice(v, len(v)).mean() for _ in range(N_BOOT)]
        out[key] = {"within_subject_auroc": float(v.mean()),
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "n_above_0.5": int((v > 0.5).sum())}
    return out


# ---------------------------------------------------------------------------
# D. Posture by condition; stress vs amusement
# ---------------------------------------------------------------------------

def _tilt_deg(g: pd.DataFrame) -> np.ndarray:
    """Angle between the window's gravity vector and that participant's
    mean baseline gravity vector, in degrees."""
    v = g[POSTURE].values
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def posture_by_condition(ext4: pd.DataFrame) -> dict:
    rows = []
    for s, g in ext4.groupby("subject_id"):
        u = _tilt_deg(g)
        ref = {c: u[(g.condition == c).values].mean(0) for c in ["baseline", "stress", "amusement", "meditation"]}
        ref = {c: r / np.linalg.norm(r) for c, r in ref.items()}
        ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
        hr = g.groupby("condition").hrv_mean_nn.mean()
        bpm = 60000.0 / hr
        rows.append({"subject_id": s,
                     "tilt_stress_vs_baseline_deg": ang(ref["stress"], ref["baseline"]),
                     "tilt_stress_vs_amusement_deg": ang(ref["stress"], ref["amusement"]),
                     "tilt_amusement_vs_baseline_deg": ang(ref["amusement"], ref["baseline"]),
                     "bpm_stress_minus_baseline": float(bpm["stress"] - bpm["baseline"]),
                     "bpm_stress_minus_amusement": float(bpm["stress"] - bpm["amusement"])})
    t = pd.DataFrame(rows)
    sa = ext4[ext4.condition.isin(["stress", "amusement"])].copy()
    sa["y"] = (sa.condition == "stress").astype(int)
    per = {}
    for s, g in sa.groupby("subject_id"):
        u = _tilt_deg(g)
        amu = u[(g.y == 0).values].mean(0)
        amu = amu / np.linalg.norm(amu)
        tilt = np.degrees(np.arccos(np.clip(u @ amu, -1, 1)))
        per[s] = {"tilt_from_amusement": roc_auc_score(g.y, tilt),
                  "acc_sd": roc_auc_score(g.y, g.chest_acc_mag_sd),
                  "hr": roc_auc_score(g.y, -g.hrv_mean_nn)}
    out = {"per_subject": t.set_index("subject_id").to_dict(orient="index"),
           "median": t.drop(columns="subject_id").median().to_dict(),
           "range": {c: [float(t[c].min()), float(t[c].max())] for c in t.columns if c != "subject_id"},
           "stress_vs_amusement_within_subject_auroc": {
               k: float(np.mean([per[s][k] for s in per])) for k in ["tilt_from_amusement", "acc_sd", "hr"]},
           "n_bpm_stress_minus_amusement_gt10": int((t.bpm_stress_minus_amusement > 10).sum()),
           "n_bpm_stress_minus_baseline_gt10": int((t.bpm_stress_minus_baseline > 10).sum())}
    return out


def main() -> dict:
    ext4 = ec.load_extended("chest", conditions=("baseline", "stress", "amusement", "meditation"))
    pf = posture_features(ext4)
    ext4 = ext4.merge(pf, on=["subject_id", "start_sample"], how="left", validate="one_to_one")
    assert ext4[POSTURE].notna().all().all()

    df = ext4[ext4.condition.isin(["baseline", "stress"])].copy()
    df["y"] = (df.condition == "stress").astype(int)
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    assert len(df) == 434, len(df)

    pred = loso(df)
    res = {"n_windows": int(len(df)),
           "A_loso": bootstrap(pred, [("acc_plus_hr", "acc_only"), ("ecg_plus_acc", "acc_only"),
                                      ("acc_plus_hr", "hr_only"), ("acc_only", "hr_only")]),
           "B_hr_residualised_on_acc": hr_residualised(df),
           "C_low_motion": low_motion(df),
           "D_posture": posture_by_condition(ext4)}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    A = r["A_loso"]
    for m, v in A["models"].items():
        print(f"{m:15s} " + "  ".join(f"{k} {x['est']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]" for k, x in v.items()))
    for c, v in A["contrasts"].items():
        print(f"{c:28s} " + "  ".join(f"{k} {x['est']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]" for k, x in v.items()))
    print(json.dumps({k: v for k, v in r["B_hr_residualised_on_acc"].items() if k != "per_subject"}, default=str)[:900])
    print(json.dumps({k: v for k, v in r["C_low_motion"].items() if k != "stress_kept_per_subject"}))
    D = r["D_posture"]
    print(json.dumps({k: D[k] for k in ["median", "range", "stress_vs_amusement_within_subject_auroc",
                                        "n_bpm_stress_minus_amusement_gt10", "n_bpm_stress_minus_baseline_gt10"]}, indent=1))
