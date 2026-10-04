"""
Round-4 extended feature extraction from the raw WESAD .pkl archives.

Why (peer_reviewe_v2.md):
  M6   order-confound checks need amusement and meditation windows, with
       their position in the session (the frozen table has neither).
  E4   wrist-only replication: the same 13-feature schema from the Empatica
       E4 (BVP-derived HRV at 64 Hz, wrist EDA at 4 Hz).
  m2   S10 investigation: movement (chest ACC) and R-peak plausibility.
  m12  signal-quality index: fraction of physiologically implausible NN
       intervals per window.

The chest features use exactly the code that built the frozen table
(`real_data_loaders` windowing rules, `feature_extraction.extract_full_features`),
and the baseline/stress rows are verified against the frozen table before
the manifest is written, so the extended table is a superset of it.

Windows: non-overlapping 60 s on the 700 Hz label stream, label constant
across the window, at most the first 20 per condition (same rules as the
frozen table), for baseline (1), stress (2), amusement (3), meditation (4).
Wrist signals are cut at the same wall-clock interval (t0 = start/700 s).

Output:  results/extended_features.csv, results/extended_table_manifest.json

CLI:     python extract_extended.py [--jobs 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from feature_extraction import ALL_FEATURE_NAMES, ECG_FEATURE_NAMES, _eda_features, extract_full_features
from stage1_pipeline import SubjectWindow

HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "results" / "extended_features.csv"
MANIFEST = HERE / "results" / "extended_table_manifest.json"
WESAD_ROOT = os.environ.get("WESAD_ROOT", r"C:\Users\Lenovo\Downloads\wesad_pkls")
FS_CHEST, FS_BVP, FS_WEDA, FS_WACC = 700, 64, 4, 32
WIN_S = 60
CAP = 20
CONDITIONS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def _nn_quality(peaks: np.ndarray, fs: float) -> dict:
    """Signal-quality index from detected beats (review m12)."""
    if len(peaks) < 3:
        return {"n_beats": int(len(peaks)), "frac_nn_implausible": float("nan"),
                "frac_nn_jump20": float("nan")}
    nn = np.diff(peaks) / fs * 1000.0
    jump = np.abs(np.diff(nn)) / nn[:-1]
    return {"n_beats": int(len(peaks)),
            # Outside 300-2000 ms (200-30 bpm): not a plausible sinus NN interval.
            "frac_nn_implausible": float(np.mean((nn < 300) | (nn > 2000))),
            # Successive change > 20%: the usual ectopy/artefact screen.
            "frac_nn_jump20": float(np.mean(jump > 0.2)) if len(jump) else float("nan")}


def _hrv_from_peaks(peaks: np.ndarray, fs: float) -> dict:
    import neurokit2 as nk
    out = {k: float("nan") for k in ECG_FEATURE_NAMES}
    if len(peaks) < 4:
        return out
    try:
        t = nk.hrv_time(peaks, sampling_rate=fs, show=False)
        f = nk.hrv_frequency(peaks, sampling_rate=fs, show=False)
    except Exception:
        return out

    def g(df, k):
        v = df[k].values[0] if k in df.columns else np.nan
        return float(v) if np.isfinite(v) else float("nan")
    out.update(hrv_mean_nn=g(t, "HRV_MeanNN"), hrv_sdnn=g(t, "HRV_SDNN"),
               hrv_rmssd=g(t, "HRV_RMSSD"), hrv_pnn50=g(t, "HRV_pNN50"),
               hrv_lf_power=g(f, "HRV_LF"), hrv_hf_power=g(f, "HRV_HF"))
    lf, hf = out["hrv_lf_power"], out["hrv_hf_power"]
    out["hrv_lf_hf_ratio"] = float(lf / hf) if hf and hf > 0 else float("nan")
    return out


def _coupling_from_hr(peaks: np.ndarray, fs_peaks: float, eda: np.ndarray, fs_eda: float,
                      max_lag_s: float = 15.0) -> tuple[float, float]:
    """Wrist analogue of stage1_pipeline.compute_coupling_lag: instantaneous
    HR from PPG beats, interpolated onto the EDA grid, cross-correlated with
    z-scored EDA within +/-15 s. Returns (peak |corr| signed, lag s)."""
    from scipy.signal import correlate
    if len(peaks) < 3 or len(eda) < fs_eda * 3 or np.nanstd(eda) < 1e-9:
        return float("nan"), float("nan")
    rr = np.diff(peaks) / fs_peaks
    hr = 60.0 / rr
    grid = np.arange(len(eda)) / fs_eda
    hr_g = np.interp(grid, peaks[1:] / fs_peaks, hr, left=hr[0], right=hr[-1])
    if np.nanstd(hr_g) < 1e-6:
        return float("nan"), float("nan")
    hr_g = (hr_g - hr_g.mean()) / hr_g.std()
    e = (eda - eda.mean()) / eda.std()
    n = len(e)
    corr = correlate(e, hr_g, mode="full") / n
    lags = np.arange(-n + 1, n)
    m = np.abs(lags) <= int(round(max_lag_s * fs_eda))
    corr, lags = corr[m], lags[m]
    i = int(np.nanargmax(np.abs(corr)))
    return float(corr[i]), float(lags[i] / fs_eda)


def _spectral_hr(bvp: np.ndarray, fs: float) -> float:
    """Dominant BVP frequency in 0.7-3.5 Hz (42-210 bpm), Welch. A beat-
    detection-free HR estimate used only to check wrist-chest agreement."""
    from scipy.signal import butter, filtfilt, welch
    if len(bvp) < fs * 30:
        return float("nan")
    b, a = butter(3, [0.7 / (fs / 2), 3.5 / (fs / 2)], btype="band")
    f, P = welch(filtfilt(b, a, bvp), fs=fs, nperseg=1024, nfft=8192)
    m = (f > 0.7) & (f < 3.5)
    return float(60.0 * f[m][np.argmax(P[m])])


def _blocks(label: np.ndarray) -> list[tuple[int, int, int]]:
    """(raw_label, start, stop) for each contiguous run of the label stream."""
    ch = np.r_[0, np.flatnonzero(np.diff(label)) + 1, len(label)]
    return [(int(label[a]), int(a), int(b)) for a, b in zip(ch[:-1], ch[1:])]


def extract_subject(sid: str, pkl: str) -> list[dict]:
    import neurokit2 as nk
    with open(pkl, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    ch, wr = d["signal"]["chest"], d["signal"]["wrist"]
    ecg = np.asarray(ch["ECG"]).ravel()
    eda = np.asarray(ch["EDA"]).ravel()
    resp = np.asarray(ch["Resp"]).ravel()
    acc = np.asarray(ch["ACC"], dtype=float)
    label = np.asarray(d["label"]).ravel()
    bvp = np.asarray(wr["BVP"]).ravel()
    weda = np.asarray(wr["EDA"]).ravel()
    wacc = np.asarray(wr["ACC"], dtype=float)

    # Condition blocks in session order (meditation occurs twice).
    blocks = [b for b in _blocks(label) if b[0] in CONDITIONS]
    block_start = {CONDITIONS[l]: [] for l in CONDITIONS}
    for lab, a, b in blocks:
        block_start[CONDITIONS[lab]].append((a, b))
    stress_start = block_start["stress"][0][0]
    amuse_start = block_start["amusement"][0][0]

    W = FS_CHEST * WIN_S
    N = min(len(ecg), len(eda), len(resp), len(label))
    counts = {c: 0 for c in CONDITIONS.values()}
    rows = []
    for start in range(0, N - W + 1, W):
        seg = label[start:start + W]
        u = np.unique(seg)
        if len(u) != 1 or int(u[0]) not in CONDITIONS:
            continue
        cond = CONDITIONS[int(u[0])]
        if counts[cond] >= CAP:
            continue
        counts[cond] += 1
        stop = start + W
        win = SubjectWindow(subject_id=f"WESAD_{sid}", dataset="WESAD", label=cond,
                            fs=FS_CHEST, ecg=ecg[start:stop].astype(float),
                            eda=eda[start:stop].astype(float),
                            resp=resp[start:stop].astype(float))
        row = extract_full_features(win)
        row["condition"] = cond
        row["start_sample"] = int(start)
        row["t_start_min"] = start / FS_CHEST / 60.0
        row["window_index_in_condition"] = counts[cond] - 1
        blk = [i for i, (a, b) in enumerate(block_start[cond]) if a <= start < b]
        row["condition_block"] = int(blk[0]) if blk else -1
        row["protocol_stress_before_amusement"] = bool(stress_start < amuse_start)

        # Chest SQI + movement.
        try:
            _, info = nk.ecg_peaks(win.ecg, sampling_rate=FS_CHEST)
            rp = np.asarray(info["ECG_R_Peaks"])
        except Exception:
            rp = np.array([], dtype=int)
        row.update({f"chest_{k}": v for k, v in _nn_quality(rp, FS_CHEST).items()})
        mag = np.linalg.norm(acc[start:stop], axis=1)
        row["chest_acc_mag_mean"] = float(mag.mean())
        row["chest_acc_mag_sd"] = float(mag.std())

        # Wrist (E4) at the same wall-clock interval.
        t0 = start / FS_CHEST
        b0, w0, a0 = int(round(t0 * FS_BVP)), int(round(t0 * FS_WEDA)), int(round(t0 * FS_WACC))
        bseg = bvp[b0:b0 + FS_BVP * WIN_S].astype(float)
        eseg = weda[w0:w0 + FS_WEDA * WIN_S].astype(float)
        wrow = {k: float("nan") for k in ALL_FEATURE_NAMES}
        pk = np.array([], dtype=int)
        if len(bseg) == FS_BVP * WIN_S:
            try:
                # Standard neurokit2 pipeline: clean (Elgendi band-pass), then detect.
                _, pinfo = nk.ppg_peaks(nk.ppg_clean(bseg, sampling_rate=FS_BVP),
                                        sampling_rate=FS_BVP)
                pk = np.asarray(pinfo["PPG_Peaks"])
            except Exception:
                pk = np.array([], dtype=int)
            wrow.update(_hrv_from_peaks(pk, FS_BVP))
        if len(eseg) == FS_WEDA * WIN_S:
            wrow.update(_eda_features(eseg, FS_WEDA))
            c, lag = _coupling_from_hr(pk, FS_BVP, eseg, FS_WEDA)
            wrow["coupling_peak_corr"], wrow["coupling_lag_sec"] = c, lag
        row["wrist_spectral_hr_bpm"] = _spectral_hr(bseg, FS_BVP)
        row.update({f"wrist_{k}": v for k, v in wrow.items()})
        row.update({f"wrist_{k}": v for k, v in _nn_quality(pk, FS_BVP).items()})
        wm = np.linalg.norm(wacc[a0:a0 + FS_WACC * WIN_S], axis=1) / 64.0  # E4 ACC unit 1/64 g
        row["wrist_acc_mag_sd"] = float(wm.std()) if len(wm) else float("nan")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_against_frozen(ext: pd.DataFrame) -> dict:
    """Chest baseline/stress rows must reproduce the frozen table exactly."""
    import eval_core as ec
    fz = ec.load_wesad_frozen()
    fz = fz.assign(condition=np.where(fz["label"] == 1, "stress", "baseline"))
    fz["k"] = fz.groupby(["subject_id", "condition"]).cumcount()
    ex = ext[ext.condition.isin(["baseline", "stress"])].copy()
    ex["k"] = ex.groupby(["subject_id", "condition"]).cumcount()
    m = fz.merge(ex, on=["subject_id", "condition", "k"], suffixes=("_f", "_e"))
    diffs = {}
    for c in ALL_FEATURE_NAMES:
        a, b = m[f"{c}_f"].values, m[f"{c}_e"].values
        same = np.isclose(a, b, rtol=1e-9, atol=1e-12) | (np.isnan(a) & np.isnan(b))
        diffs[c] = int((~same).sum())
    return {"n_frozen": int(len(fz)), "n_extended_bs": int(len(ex)), "n_matched": int(len(m)),
            "n_mismatched_values_per_feature": diffs,
            "reproduces_frozen_table": bool(len(m) == len(fz) == len(ex)
                                            and not any(diffs.values()))}


def main(jobs: int = 4) -> None:
    from joblib import Parallel, delayed
    pkls = sorted((p.stem, str(p)) for p in Path(WESAD_ROOT).glob("S*.pkl"))
    print(f"[extended] {len(pkls)} subjects from {WESAD_ROOT}", flush=True)
    outs = Parallel(n_jobs=jobs, verbose=5)(delayed(extract_subject)(s, p) for s, p in pkls)
    ext = pd.DataFrame([r for o in outs for r in o])
    ext = ext.sort_values(["subject_id", "start_sample"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ext.to_csv(OUT_CSV, index=False)
    check = verify_against_frozen(ext)
    manifest = {
        "path": str(OUT_CSV.relative_to(HERE)).replace("\\", "/"),
        "sha256": _sha256(OUT_CSV),
        "n_rows": int(len(ext)),
        "rows_per_condition": ext["condition"].value_counts().to_dict(),
        "n_subjects": int(ext["subject_id"].nunique()),
        "wesad_root": WESAD_ROOT,
        "verification_vs_frozen_table": check,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Round-4 superset of the frozen table: adds amusement/meditation "
                 "windows, session timing, chest SQI/ACC and wrist (E4) features. "
                 "The primary analyses still read the frozen table."),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    main(ap.parse_args().jobs)
