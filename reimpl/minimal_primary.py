"""
Minimal re-implementation of the primary comparison (peer_reviewe_v2.md M9).

What it is: a from-scratch pipeline for the paper's primary contrast
(gated network vs non-gated MLP vs NaN-trained LightGBM vs ECG-only
LightGBM, reduced-condition Brier, LOSO), reading the raw WESAD .pkl files.
It imports NOTHING from the project: windowing, R-peak detection, HRV, EDA
decomposition, standardisation, both networks and the evaluation loop are
written here, with deliberately different implementation choices:

  * R-peaks: Pan-Tompkins-style (5-20 Hz band-pass, derivative, squaring,
    150 ms integration, adaptive threshold, refractory 250 ms), not neurokit2.
  * HRV: time domain only (mean NN, SDNN, RMSSD, pNN50); no LF/HF.
  * EDA: 4th-order 1 Hz low-pass for the tonic level (not 0.5 Hz, 2nd
    order); SCRs = phasic peaks with prominence >= 0.01 uS.
  * No coupling features. EDA modality = 4 features.
  * Standardisation by training mean/SD (not median/IQR).
  * Networks: 200 epochs, Adam lr 5e-3, weight decay 1e-4; 5 seeds averaged.

What it is NOT: an independent re-implementation. It was written by the
same author (with an AI assistant) who had seen the main code. The review
asks for someone who has not; that remains open.

Agreement criterion (fixed before running): every pairwise reduced-
condition Brier difference among the four models within +/-0.02 of zero
with participant-bootstrap CIs that include 0, and each model's pooled
reduced Brier within 0.03 of the main pipeline's LOSO value.

Output: results/reimpl_minimal_primary.json
CLI:    python reimpl/minimal_primary.py
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(os.environ.get("WESAD_ROOT", r"D:\wesad_pkls"))
OUT = Path(__file__).resolve().parent.parent / "results" / "reimpl_minimal_primary.json"
FS = 700
WIN = 60 * FS
SEEDS = [11, 22, 33, 44, 55]


# ----------------------------------------------------------------------------- features
def rpeaks(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    b, a = butter(2, [5 / (fs / 2), 20 / (fs / 2)], btype="band")
    f = filtfilt(b, a, ecg)
    d = np.gradient(f) ** 2
    k = int(0.15 * fs)
    integ = np.convolve(d, np.ones(k) / k, mode="same")
    pk, _ = find_peaks(integ, height=0.35 * np.percentile(integ, 99), distance=int(0.25 * fs))
    # refine to the band-passed |ECG| maximum within +/-75 ms
    w = int(0.075 * fs)
    ref = [max(0, p - w) + int(np.argmax(np.abs(f[max(0, p - w):p + w]))) for p in pk]
    return np.unique(ref)


def hrv(peaks: np.ndarray, fs: int = FS) -> dict:
    nn = np.diff(peaks) / fs * 1000.0
    nn = nn[(nn > 300) & (nn < 2000)]
    if len(nn) < 5:
        return dict(mean_nn=np.nan, sdnn=np.nan, rmssd=np.nan, pnn50=np.nan)
    dd = np.diff(nn)
    return dict(mean_nn=nn.mean(), sdnn=nn.std(ddof=1), rmssd=np.sqrt(np.mean(dd ** 2)),
                pnn50=100 * np.mean(np.abs(dd) > 50))


def eda_feats(eda: np.ndarray, fs: int = FS) -> dict:
    b, a = butter(4, 1.0 / (fs / 2), btype="low")
    ton = filtfilt(b, a, eda)
    t = np.arange(len(ton)) / fs
    slope = np.polyfit(t, ton, 1)[0]
    ph = eda - ton
    pk, pr = find_peaks(ph, prominence=0.01, distance=int(1.0 * fs))
    return dict(scl=ton.mean(), scl_slope=slope, scr_rate=len(pk) / (len(eda) / fs) * 60,
                scr_amp=float(np.mean(pr["prominences"])) if len(pk) else 0.0)


ECG = ["mean_nn", "sdnn", "rmssd", "pnn50"]
EDA = ["scl", "scl_slope", "scr_rate", "scr_amp"]


def build_table() -> pd.DataFrame:
    rows = []
    for pkl in sorted(ROOT.glob("S*.pkl")):
        with open(pkl, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        ecg = np.asarray(d["signal"]["chest"]["ECG"]).ravel()
        eda = np.asarray(d["signal"]["chest"]["EDA"]).ravel()
        lab = np.asarray(d["label"]).ravel()
        cnt = {1: 0, 2: 0}
        for s in range(0, len(lab) - WIN + 1, WIN):
            u = np.unique(lab[s:s + WIN])
            if len(u) != 1 or u[0] not in (1, 2) or cnt[int(u[0])] >= 20:
                continue
            cnt[int(u[0])] += 1
            rows.append({"subject": pkl.stem, "y": int(u[0] == 2),
                         **hrv(rpeaks(ecg[s:s + WIN])), **eda_feats(eda[s:s + WIN])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- models
def _torch():
    import torch
    torch.set_num_threads(4)
    return torch


class Gated:
    def __init__(self, seed, d=16):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(seed)
        self.seed = seed
        self.e = nn.Sequential(nn.Linear(4, d), nn.ReLU(), nn.Linear(d, d))
        self.a = nn.Sequential(nn.Linear(4, d), nn.ReLU(), nn.Linear(d, d))
        self.h = nn.Linear(d, 1)
        self.params = list(self.e.parameters()) + list(self.a.parameters()) + list(self.h.parameters())

    def logit(self, xe, xa, gate):
        torch = _torch()
        return self.h(torch.relu(self.e(xe) + gate * self.a(xa))).squeeze(-1)

    def fit(self, xe, xa, y):
        torch = _torch()
        opt = torch.optim.Adam(self.params, lr=5e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(200):
            gate = (torch.rand(len(y), 1, generator=g) > 0.5).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.logit(xe, xa, gate), y)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict(self, xe, xa, present: bool):
        torch = _torch()
        with torch.no_grad():
            gate = torch.full((len(xe), 1), 1.0 if present else 0.0)
            return torch.sigmoid(self.logit(xe, xa, gate)).numpy()


class MLP:
    def __init__(self, seed, h=28):   # 761 vs 721 parameters (gated)
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(seed)
        self.seed = seed
        self.f = nn.Sequential(nn.Linear(9, h), nn.ReLU(), nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, 1))

    def fit(self, xe, xa, y):
        torch = _torch()
        opt = torch.optim.Adam(self.f.parameters(), lr=5e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(200):
            m = (torch.rand(len(y), 1, generator=g) > 0.5).float()
            out = self.f(torch.cat([xe, xa * m, m], 1)).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict(self, xe, xa, present: bool):
        torch = _torch()
        with torch.no_grad():
            m = torch.full((len(xe), 1), 1.0 if present else 0.0)
            return torch.sigmoid(self.f(torch.cat([xe, xa * m, m], 1)).squeeze(-1)).numpy()


def lgbm(seed):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                              random_state=seed, verbosity=-1, n_jobs=4)


# ----------------------------------------------------------------------------- LOSO
def loso(df: pd.DataFrame) -> pd.DataFrame:
    torch = _torch()
    out = []
    for s in sorted(df.subject.unique()):
        tr, te = df[df.subject != s], df[df.subject == s]
        mu, sd = tr[ECG + EDA].mean(), tr[ECG + EDA].std(ddof=0).replace(0, 1)
        z = lambda d: ((d[ECG + EDA] - mu) / sd).fillna(0.0)
        ztr, zte = z(tr), z(te)
        T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
        ytr = T(tr.y.values)
        rec = {"subject": te.subject.values, "y": te.y.values}
        for name, cls in (("gated", Gated), ("mlp", MLP)):
            for cond, present in (("complete", True), ("reduced", False)):
                rec[f"{name}_{cond}"] = np.mean([
                    cls(sd_).fit(T(ztr[ECG]), T(ztr[EDA]), ytr).predict(T(zte[ECG]), T(zte[EDA]), present)
                    for sd_ in SEEDS], axis=0)
        # (ii) NaN-trained LightGBM, (iii) ECG-only LightGBM
        p2c, p2r, p3 = [], [], []
        for sd_ in SEEDS:
            X = tr[ECG + EDA].copy()
            X.loc[np.random.default_rng(sd_).random(len(X)) < 0.5, EDA] = np.nan
            m2 = lgbm(sd_).fit(X, tr.y)
            Xr = te[ECG + EDA].copy(); Xr[EDA] = np.nan
            p2c.append(m2.predict_proba(te[ECG + EDA])[:, 1]); p2r.append(m2.predict_proba(Xr)[:, 1])
            p3.append(lgbm(sd_).fit(tr[ECG], tr.y).predict_proba(te[ECG])[:, 1])
        rec["ii_complete"], rec["ii_reduced"] = np.mean(p2c, 0), np.mean(p2r, 0)
        rec["iii_complete"] = rec["iii_reduced"] = np.mean(p3, 0)
        out.append(pd.DataFrame(rec))
        print(f"  {s} done", flush=True)
    return pd.concat(out, ignore_index=True)


def boot_diff(pred, a, b, n_boot=2000, seed=5):
    rng = np.random.default_rng(seed)
    subj = pred.subject.unique()
    g = {s: d for s, d in pred.groupby("subject")}
    def brier_diff(d):
        return float(np.mean((d[a] - d.y) ** 2) - np.mean((d[b] - d.y) ** 2))
    vals = [brier_diff(pd.concat([g[s] for s in rng.choice(subj, len(subj))])) for _ in range(n_boot)]
    return {"diff": brier_diff(pred), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def main():
    from sklearn.metrics import roc_auc_score
    df = build_table()
    print(f"[reimpl] {len(df)} windows, {df.subject.nunique()} subjects, "
          f"{df.y.mean():.3f} stress; missing HRV rows {int(df[ECG].isna().any(axis=1).sum())}")
    pred = loso(df)
    models = ["gated", "mlp", "ii", "iii"]
    res = {"n_windows": int(len(df)), "n_subjects": int(df.subject.nunique()),
           "per_model": {}, "pairwise_reduced_brier": {}}
    for m in models:
        res["per_model"][m] = {c: {"brier": float(np.mean((pred[f"{m}_{c}"] - pred.y) ** 2)),
                                   "auroc": float(roc_auc_score(pred.y, pred[f"{m}_{c}"]))}
                               for c in ("complete", "reduced")}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            res["pairwise_reduced_brier"][f"{a}_vs_{b}"] = boot_diff(pred, f"{a}_reduced", f"{b}_reduced")
    main_loso = json.loads((OUT.parent / "baselines.json").read_text(encoding="utf-8"))
    ref = {"gated": "gated_dropout_p0.5", "mlp": "mlp_nongated_dropout_p0.5",
           "ii": "lgbm_nan_training_p0.5", "iii": "lgbm_ecg_only_per_config"}
    res["main_pipeline_loso_reduced_brier"] = {
        m: main_loso["schemes"]["loso"]["models"][k]["raw"]["reduced"]["pooled"]["brier"] for m, k in ref.items()}
    pw_ok = all(abs(v["diff"]) <= 0.02 and v["ci95"][0] <= 0 <= v["ci95"][1]
                for v in res["pairwise_reduced_brier"].values())
    lvl_ok = all(abs(res["per_model"][m]["reduced"]["brier"] - res["main_pipeline_loso_reduced_brier"][m]) <= 0.03
                 for m in models)
    res["agreement"] = {"pairwise_within_0.02_and_ci_includes_0": pw_ok,
                        "levels_within_0.03_of_main": lvl_ok}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    pred.to_csv(OUT.with_name("reimpl_minimal_primary_predictions.csv"), index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
