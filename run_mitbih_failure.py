"""
Round 6: arrhythmia failure-mode probe, as pre-specified in prespec_round5-style file
prespec_round6_mitbih.md (SHA-256 recorded in results/prespec_round6_manifest.json and
re-checked here).

WESAD-trained systems (trained on all 434 frozen windows) are scored on MIT-BIH windows with EDA
missing (the reduced tier only). Reports the fraction of windows called "stress" (calibrated p >= 0.5)
by annotated rate band and rhythm group, with record-cluster bootstrap intervals.

    PYTHONIOENCODING=utf-8 python run_mitbih_failure.py            # full run
    PYTHONIOENCODING=utf-8 python run_mitbih_failure.py --smoke    # 2 records, 1 seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import eval_core as ec
import real_data_loaders as rdl
from feature_extraction import ALL_FEATURE_NAMES, EDA_FEATURE_NAMES, extract_full_features
from stage1_pipeline import SubjectWindow

HERE = Path(__file__).resolve().parent
PRESPEC = HERE / "prespec_round6_mitbih.md"
PRESPEC_MANIFEST = HERE / "results" / "prespec_round6_manifest.json"
OUT_JSON = HERE / "results" / "mitbih_failure.json"
OUT_CSV = HERE / "results" / "mitbih_failure_windows.csv"

MITDB = os.environ.get("MITBIH_ROOT", r"D:\Trần An Khánh\E14\mitdb")
SYSTEMS = {
    "hard_mask_network": "gated_dropout_p0.5",
    "masked_mlp": "mlp_nongated_dropout_p0.5",
    "lgbm_ii_nan_trained": "lgbm_nan_training_p0.5",
    "lgbm_iii_ecg_only": "lgbm_ecg_only_per_config",
    "hr_only": "hr_only",
    "hard_mask_network_hr_removed": "gated_dropout_hr_removed",
    "lgbm_ecg_only_hr_removed": "lgbm_ecg_only_hr_removed",
}
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ")
NOISE_SYMBOLS = {"~", "U", "|"}
MIN_WINDOWS, MIN_RECORDS = 30, 5
SEEDS = [0, 1, 2, 3, 4]
B = 2000


def prespec_sha() -> str:
    sha = hashlib.sha256(PRESPEC.read_bytes()).hexdigest()
    recorded = json.loads(PRESPEC_MANIFEST.read_text(encoding="utf-8"))["sha256"]
    if sha != recorded:
        raise RuntimeError("prespec_round6_mitbih.md changed after its hash was recorded; "
                           "refusing to run a 'pre-specified' analysis against it.")
    return sha


def build_windows(records, window_sec=60):
    """One row per non-noise 60-s MIT-BIH window, with annotation-derived rate and rhythm group."""
    rows, wins = [], []
    for rec in records:
        ecg, fs, ann = rdl._load_mitbih_record(rec, MITDB)
        W = int(fs * window_sec)
        symbols = np.asarray(ann.symbol)
        samples = np.asarray(ann.sample)
        aux = np.asarray(ann.aux_note)
        events = [(samples[i], aux[i].strip()) for i in range(len(aux)) if aux[i].strip()]
        cluster = rdl.MITBIH_SHARED_SUBJECT.get(rec, rec)
        for start in range(0, len(ecg) - W + 1, W):
            stop = start + W
            in_win = (samples >= start) & (samples < stop)
            syms = symbols[in_win]
            if any(s in NOISE_SYMBOLS for s in syms):
                continue
            tags = {t for s, t in events if start <= s < stop}
            cur = "(N"
            for s, t in events:
                if s <= start:
                    cur = t
                else:
                    break
            tags.add(cur)
            n_beats = int(sum(s in BEAT_SYMBOLS for s in syms))
            if n_beats < 10:   # unreadable / asystole-like window: no meaningful rate
                continue
            if tags == {"(N"} and n_beats >= 30 and all(s == "N" for s in syms if s in BEAT_SYMBOLS):
                rhythm = "normal_sinus"
            else:
                rhythm = "other"
            rows.append({
                "record": rec, "cluster": cluster, "start_s": start / fs, "n_beats": n_beats,
                "rate_bpm": float(n_beats) * 60.0 / window_sec,
                "rhythm_group": rhythm,
                "is_svt": "(SVTA" in tags, "is_afib_flutter": bool(tags & {"(AFIB", "(AFL"}),
                "is_vt": "(VT" in tags,
            })
            wins.append(SubjectWindow(subject_id=f"MITBIH_{cluster}", dataset="MITBIH", label="x",
                                      fs=fs, ecg=ecg[start:stop].astype(float), eda=None, resp=None,
                                      meta={"record": rec}))
    meta = pd.DataFrame(rows)
    feats = pd.DataFrame([extract_full_features(w) for w in wins])
    return meta, feats


def rate_band(r):
    return "<100" if r < 100 else ("100-119" if r < 120 else ">=120")


def cluster_boot(values: np.ndarray, clusters: np.ndarray, stat, rng):
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    out = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        out.append(stat(idx))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    sha = prespec_sha()
    t0 = time.time()

    records = rdl._list_mitbih_records(MITDB)
    if args.smoke:
        records = records[:2]
    print(f"[{time.time()-t0:5.0f}s] {len(records)} MIT-BIH records; building windows ...", flush=True)
    meta, feats = build_windows(records)
    print(f"[{time.time()-t0:5.0f}s] {len(meta)} windows from {meta.cluster.nunique()} clusters", flush=True)

    train = ec.load_wesad_frozen()
    med = train[ALL_FEATURE_NAMES].median()
    test = feats.copy()
    for c in EDA_FEATURE_NAMES:          # harmless fill so the (unused) complete tier cannot fail
        test[c] = med[c]
    test["label"] = 0
    test["row_id"] = np.arange(len(test))
    test["subject_id"] = meta["cluster"].values
    test["has_ecg"] = True

    seeds = SEEDS[:1] if args.smoke else SEEDS
    p_cal, p_raw = {}, {}
    for key, model in SYSTEMS.items():
        cal, raw = [], []
        for s in seeds:
            out = ec.fit_predict(model, train, test, seed=s)
            red = out[out.tier == "reduced"].sort_values("row_id")
            cal.append(red["p_platt_pc"].values)
            raw.append(red["p_raw"].values)
            print(f"[{time.time()-t0:5.0f}s] {key} seed {s}", flush=True)
        p_cal[key], p_raw[key] = np.mean(cal, axis=0), np.mean(raw, axis=0)

    for k in SYSTEMS:
        meta[f"p_{k}"] = p_cal[k]
        meta[f"praw_{k}"] = p_raw[k]
    meta["hrv_mean_nn_ms"] = feats["hrv_mean_nn"].values
    meta["feature_rate_bpm"] = 60000.0 / feats["hrv_mean_nn"].values
    meta["rate_band"] = meta["rate_bpm"].map(rate_band)
    meta.to_csv(OUT_CSV, index=False)

    rng = np.random.default_rng(20261004)
    clusters = meta["cluster"].values

    def flag_rate(mask, col, thr=0.5):
        m = np.asarray(mask)
        sel = np.where(m)[0]
        if len(sel) == 0:
            return {"n": 0}
        v = (meta[col].values[sel] >= thr).astype(float)
        cl = clusters[sel]
        lo, hi = cluster_boot(v, cl, lambda i: v[i].mean(), rng) if len(np.unique(cl)) > 1 else (np.nan, np.nan)
        return {"n": int(len(sel)), "n_clusters": int(len(np.unique(cl))), "flag_rate": float(v.mean()),
                "ci95": [lo, hi]}

    groups = {
        "rate<100": meta.rate_band == "<100",
        "rate100-119": meta.rate_band == "100-119",
        "rate>=120": meta.rate_band == ">=120",
        "normal_sinus": meta.rhythm_group == "normal_sinus",
        "SVT": meta.is_svt, "AF_flutter": meta.is_afib_flutter, "VT": meta.is_vt,
        "all": np.ones(len(meta), bool),
    }
    result = {"prespec_sha256": sha, "n_windows": int(len(meta)), "n_clusters": int(meta.cluster.nunique()),
              "seeds": seeds, "systems": {}}
    high = groups["rate>=120"]
    testable = bool(high.sum() >= MIN_WINDOWS and meta.cluster[high].nunique() >= MIN_RECORDS)
    result["primary_testable"] = testable
    result["n_high_rate"] = int(high.sum())
    result["n_high_rate_clusters"] = int(meta.cluster[high].nunique())

    for k in SYSTEMS:
        col, rcol = f"p_{k}", f"praw_{k}"
        sysres = {"flag_rate_calibrated": {g: flag_rate(m, col) for g, m in groups.items()},
                  "flag_rate_raw": {g: flag_rate(m, rcol) for g, m in groups.items()}}
        rho = spearmanr(meta[col], meta["rate_bpm"]).statistic
        idx_all = np.arange(len(meta))
        lo, hi = cluster_boot(None, clusters,
                              lambda i: spearmanr(meta[col].values[i], meta["rate_bpm"].values[i]).statistic, rng)
        sysres["spearman_p_vs_rate"] = {"rho": float(rho), "ci95": [lo, hi]}
        sysres["mean_p_by_band"] = {b: float(meta.loc[meta.rate_band == b, col].mean())
                                    for b in ["<100", "100-119", ">=120"] if (meta.rate_band == b).any()}
        result["systems"][k] = sysres

    hr_sys = ["hard_mask_network", "masked_mlp", "lgbm_ii_nan_trained", "lgbm_iii_ecg_only", "hr_only"]
    fr = lambda k: result["systems"][k]["flag_rate_calibrated"]["rate>=120"].get("flag_rate", np.nan)
    sp = lambda k: result["systems"][k]["spearman_p_vs_rate"]["rho"]
    nsr = lambda k: result["systems"][k]["flag_rate_calibrated"]["normal_sinus"].get("flag_rate", np.nan)
    result["predictions"] = {
        "testable": testable,
        "P1_flag_rate_ge_0.50_at_rate_ge_120": {k: fr(k) >= 0.50 for k in hr_sys} if testable else None,
        "P2_hr_removed_flags_at_least_0.20_less": (
            {"hard_mask_network": fr("hard_mask_network") - fr("hard_mask_network_hr_removed") >= 0.20,
             "lgbm_ecg_only": fr("lgbm_iii_ecg_only") - fr("lgbm_ecg_only_hr_removed") >= 0.20}
            if testable else None),
        "P3_spearman_ge_0.50": {k: sp(k) >= 0.50 for k in hr_sys},
        "P4_normal_sinus_flag_rate_le_0.35": {k: nsr(k) <= 0.35 for k in SYSTEMS},
    }
    # How well does the extractor's heart rate track the annotated rate? (R-peak detection check)
    ok = np.isfinite(meta["feature_rate_bpm"])
    result["feature_vs_annotated_rate"] = {
        "spearman": float(spearmanr(meta.loc[ok, "feature_rate_bpm"], meta.loc[ok, "rate_bpm"]).statistic),
        "median_abs_error_bpm": float(np.median(np.abs(meta.loc[ok, "feature_rate_bpm"] - meta.loc[ok, "rate_bpm"]))),
        "n_missing_features": int((~ok).sum()),
    }
    # WESAD reference flag rates for the same trained systems are in the main results (Table S5).
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["n_windows", "n_clusters", "primary_testable", "n_high_rate",
                                             "n_high_rate_clusters", "predictions",
                                             "feature_vs_annotated_rate"]}, indent=2))
    print(f"done in {time.time()-t0:.0f}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
