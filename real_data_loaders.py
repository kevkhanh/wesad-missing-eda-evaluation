"""
Real (and synthetic-fallback) data loaders for WESAD and MIT-BIH.

Both loaders return lists of stage1_pipeline.SubjectWindow instances so
downstream feature extraction and audit code doesn't care where the samples
came from. If the raw datasets aren't present on disk, the loaders emit
clearly-labelled synthetic windows so the pipeline is testable in CI /
without downloads. Every synthetic path prints a WARNING so nobody
accidentally reports metrics on synthetic data as real.

Data layout (override via env vars WESAD_ROOT / MITBIH_ROOT):
    <WESAD_ROOT>/S2/S2.pkl, S3/S3.pkl, ...
    <MITBIH_ROOT>/100.dat + 100.hea + 100.atr, 101.*, ...

MIT-BIH audit-specific rules (from dataset_identity_preprint_01.md Step 1):
  * paced records 102, 104, 107, 217 excluded (AAMI convention)
  * records 201 and 202 share one group ID (same patient)
  * NSR-pure windows: rhythm annotation is "(N" throughout AND every beat
    symbol in the window is "N" AND no noise/quality annotations
  * Arrhythmic windows: >=50% under (AFIB / (AFL / (SVTA / (VT / (VFL /
    (B / (T, OR contains >=3 V/A-class ectopic beats
"""

import os
import pickle
import warnings
from typing import List, Optional

import numpy as np

from stage1_pipeline import SubjectWindow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WESAD_ROOT = os.environ.get("WESAD_ROOT", "/mnt/user-data/uploads/WESAD")
MITBIH_ROOT = os.environ.get("MITBIH_ROOT", "/mnt/user-data/uploads/mit-bih")

WESAD_FS = 700
MITBIH_FS = 360

WESAD_LABEL_BASELINE = 1
WESAD_LABEL_STRESS = 2

MITBIH_PACED_RECORDS = {"102", "104", "107", "217"}
MITBIH_SHARED_SUBJECT = {"201": "P201_202", "202": "P201_202"}

ARRHYTHMIC_RHYTHM_TAGS = {"(AFIB", "(AFL", "(SVTA", "(VT", "(VFL", "(B", "(T"}
ECTOPIC_BEAT_SYMBOLS = set("VAaJSFEjeLRQ/f")


# ---------------------------------------------------------------------------
# WESAD
# ---------------------------------------------------------------------------

def _list_wesad_subjects(root=WESAD_ROOT):
    """Accept both the canonical WESAD layout (<root>/S2/S2.pkl) and a flat
    layout (<root>/S2.pkl). Some downloads unpack one way, some the other."""
    if not os.path.isdir(root):
        return []
    out = []
    seen = set()
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            pkl = os.path.join(full, f"{name}.pkl")
            if os.path.isfile(pkl) and name not in seen:
                out.append((name, pkl))
                seen.add(name)
        elif name.endswith(".pkl") and name.startswith("S"):
            sid = name[:-4]
            if sid not in seen:
                out.append((sid, full))
                seen.add(sid)
    return out


def _load_wesad_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    chest = data["signal"]["chest"]
    return {
        "ecg":   np.asarray(chest["ECG"]).flatten(),
        "eda":   np.asarray(chest["EDA"]).flatten(),
        "resp":  np.asarray(chest["Resp"]).flatten(),
        "label": np.asarray(data["label"]).flatten(),
    }


def _window_wesad_subject(
    subject_id: str,
    signals: dict,
    fs: int = WESAD_FS,
    window_sec: int = 60,
    per_condition_cap: int = 20,
) -> List[SubjectWindow]:
    """Non-overlapping windows, homogeneous label (100%), per audit doc Step 1."""
    W = int(fs * window_sec)
    labels_of_interest = {WESAD_LABEL_BASELINE: "baseline", WESAD_LABEL_STRESS: "stress"}
    windows_by_label: dict[str, list[SubjectWindow]] = {v: [] for v in labels_of_interest.values()}

    N = min(len(signals["ecg"]), len(signals["eda"]),
            len(signals["resp"]), len(signals["label"]))
    for start in range(0, N - W + 1, W):
        stop = start + W
        seg_lbl = signals["label"][start:stop]
        u = np.unique(seg_lbl)
        if len(u) != 1:
            continue
        raw = int(u[0])
        if raw not in labels_of_interest:
            continue
        name = labels_of_interest[raw]
        if len(windows_by_label[name]) >= per_condition_cap:
            continue
        windows_by_label[name].append(SubjectWindow(
            subject_id=f"WESAD_{subject_id}",
            dataset="WESAD",
            label=name,
            fs=fs,
            ecg=signals["ecg"][start:stop].astype(float),
            eda=signals["eda"][start:stop].astype(float),
            resp=signals["resp"][start:stop].astype(float),
            meta={"start_sample": start, "raw_label": raw},
        ))
    out = []
    for lst in windows_by_label.values():
        out.extend(lst)
    return out


def load_wesad_windows(
    root: str = WESAD_ROOT,
    window_sec: int = 60,
    per_condition_cap: int = 20,
    synthetic_if_missing: bool = True,
    synthetic_n_subjects: int = 6,
) -> List[SubjectWindow]:
    subs = _list_wesad_subjects(root)
    if not subs:
        if not synthetic_if_missing:
            raise FileNotFoundError(f"No WESAD subjects under {root}")
        warnings.warn(
            f"[WESAD] no real data at {root} — emitting SYNTHETIC windows. "
            "Do not report metrics from this run as real.",
            stacklevel=2,
        )
        return _synthesize_wesad_windows(
            n_subjects=synthetic_n_subjects, window_sec=window_sec
        )

    all_windows = []
    for sid, pkl in subs:
        try:
            sigs = _load_wesad_pkl(pkl)
        except Exception as e:
            warnings.warn(f"[WESAD] skipped {sid}: {e!r}")
            continue
        all_windows.extend(_window_wesad_subject(
            sid, sigs, window_sec=window_sec, per_condition_cap=per_condition_cap
        ))
    return all_windows


def _synthesize_wesad_windows(n_subjects=6, window_sec=60, fs=WESAD_FS, seed=0):
    """Physiologically-plausible synthetic ECG+EDA+Resp windows.

    Baseline windows: slower HR (~70 bpm), lower EDA tonic. Stress windows:
    faster HR (~95 bpm), higher tonic and more phasic events. These are not
    real physiology; they exist so the pipeline can be exercised end-to-end
    without WESAD downloads.
    """
    rng = np.random.default_rng(seed)
    windows = []
    N = int(fs * window_sec)
    t = np.arange(N) / fs
    for i in range(n_subjects):
        sid = f"SYN{i:02d}"
        subject_hr_offset = rng.normal(0, 3)
        for lbl, hr_bpm, eda_base, eda_gain in [
            ("baseline", 70 + subject_hr_offset, 3.0, 0.3),
            ("stress",   95 + subject_hr_offset, 6.0, 1.0),
        ]:
            for _ in range(6):
                hr_hz = hr_bpm / 60.0
                ecg = np.zeros(N)
                beat_period_s = 1.0 / hr_hz
                for beat_t in np.arange(0.2, window_sec, beat_period_s):
                    idx = int(beat_t * fs)
                    if 0 <= idx < N - int(0.15 * fs):
                        span = slice(idx, idx + int(0.03 * fs))
                        ecg[span] += 1.0
                        span2 = slice(idx + int(0.03 * fs), idx + int(0.06 * fs))
                        ecg[span2] -= 0.3
                ecg += 0.05 * rng.standard_normal(N)
                phasic_events = rng.poisson(lam=eda_gain * 2)
                eda = eda_base + 0.05 * np.sin(2 * np.pi * 0.02 * t) + \
                      0.02 * rng.standard_normal(N)
                for _e in range(phasic_events):
                    center = int(rng.uniform(0, N))
                    width = int(fs * 2.0)
                    for k in range(-width, width):
                        if 0 <= center + k < N:
                            eda[center + k] += eda_gain * np.exp(
                                -(k / (fs * 0.8)) ** 2
                            )
                resp = 0.5 * np.sin(2 * np.pi * (0.25 + 0.05 * rng.standard_normal()) * t)
                windows.append(SubjectWindow(
                    subject_id=f"WESAD_{sid}", dataset="WESAD",
                    label=lbl, fs=fs, ecg=ecg, eda=eda, resp=resp,
                    meta={"synthetic": True},
                ))
    return windows


# ---------------------------------------------------------------------------
# MIT-BIH
# ---------------------------------------------------------------------------

def _list_mitbih_records(root=MITBIH_ROOT):
    if not os.path.isdir(root):
        return []
    seen = set()
    for name in sorted(os.listdir(root)):
        stem, _, ext = name.rpartition(".")
        if ext == "dat" and stem.isdigit() and stem not in MITBIH_PACED_RECORDS:
            seen.add(stem)
    return sorted(seen)


def _load_mitbih_record(rec_id: str, root: str):
    import wfdb

    rec = wfdb.rdrecord(os.path.join(root, rec_id))
    ann = wfdb.rdann(os.path.join(root, rec_id), "atr")
    try:
        ch = rec.sig_name.index("MLII")
    except ValueError:
        ch = 0
    return rec.p_signal[:, ch], int(rec.fs), ann


def _window_mitbih_record(
    rec_id: str,
    ecg: np.ndarray,
    fs: int,
    ann,
    window_sec: int = 60,
    per_rhythm_cap: int = 20,
) -> List[SubjectWindow]:
    """NSR-pure vs arrhythmic window construction (audit doc Step 1)."""
    W = int(fs * window_sec)
    subject_id = f"MITBIH_{MITBIH_SHARED_SUBJECT.get(rec_id, rec_id)}"
    windows = {"nsr": [], "arrhythmic": []}

    symbols = np.asarray(ann.symbol)
    samples = np.asarray(ann.sample)
    aux = np.asarray(ann.aux_note) if hasattr(ann, "aux_note") else np.array([""] * len(samples))

    rhythm_events = [(samples[i], aux[i].strip()) for i in range(len(aux)) if aux[i].strip()]

    def _rhythm_at(sample_idx):
        cur = "(N"
        for s, tag in rhythm_events:
            if s <= sample_idx:
                cur = tag
            else:
                break
        return cur

    N = len(ecg)
    for start in range(0, N - W + 1, W):
        stop = start + W
        beat_mask = (samples >= start) & (samples < stop)
        w_syms = symbols[beat_mask]
        rhythms_in_window = set()
        for s, tag in rhythm_events:
            if start <= s < stop:
                rhythms_in_window.add(tag)
        anchor_rhythm = _rhythm_at(start)
        rhythms_in_window.add(anchor_rhythm)

        noise = any(sy in ("~", "U", "|") for sy in w_syms)
        if noise:
            continue

        if (rhythms_in_window == {"(N"}
                and len(w_syms) >= 30
                and all(sy == "N" for sy in w_syms)):
            label = "nsr"
        else:
            arr_tags = rhythms_in_window & ARRHYTHMIC_RHYTHM_TAGS
            n_ectopic = sum(1 for sy in w_syms if sy in ECTOPIC_BEAT_SYMBOLS)
            if arr_tags or n_ectopic >= 3:
                label = "arrhythmic"
            else:
                continue

        if len(windows[label]) >= per_rhythm_cap:
            continue
        windows[label].append(SubjectWindow(
            subject_id=subject_id,
            dataset="MITBIH",
            label=label,
            fs=fs,
            ecg=ecg[start:stop].astype(float),
            eda=None,
            resp=None,
            meta={"record": rec_id, "series": "200" if int(rec_id) >= 200 else "100"},
        ))
    return windows["nsr"] + windows["arrhythmic"]


def load_mitbih_windows(
    root: str = MITBIH_ROOT,
    window_sec: int = 60,
    per_rhythm_cap: int = 20,
    synthetic_if_missing: bool = True,
    synthetic_n_records: int = 12,
) -> List[SubjectWindow]:
    records = _list_mitbih_records(root)
    if not records:
        if not synthetic_if_missing:
            raise FileNotFoundError(f"No MIT-BIH records under {root}")
        warnings.warn(
            f"[MIT-BIH] no real data at {root} — emitting SYNTHETIC windows. "
            "Do not report metrics from this run as real.",
            stacklevel=2,
        )
        return _synthesize_mitbih_windows(
            n_records=synthetic_n_records, window_sec=window_sec
        )

    out = []
    for rec_id in records:
        try:
            ecg, fs, ann = _load_mitbih_record(rec_id, root)
        except Exception as e:
            warnings.warn(f"[MIT-BIH] skipped {rec_id}: {e!r}")
            continue
        out.extend(_window_mitbih_record(
            rec_id, ecg, fs, ann, window_sec=window_sec, per_rhythm_cap=per_rhythm_cap
        ))
    return out


def _synthesize_mitbih_windows(n_records=12, window_sec=60, fs=MITBIH_FS, seed=1):
    """Synthetic MIT-BIH-shaped windows: ECG only, no EDA.

    We deliberately encode two structural fingerprints from the audit doc:
      * 360 Hz sampling (differs from WESAD's 700 Hz)
      * a small periodic artifact at 0.167 Hz (the tape-transport artifact)
    ...so the identity probe on synthetic data isn't spuriously chance.
    """
    rng = np.random.default_rng(seed)
    windows = []
    N = int(fs * window_sec)
    t = np.arange(N) / fs
    for i in range(n_records):
        rec_id = str(200 + i) if i < 6 else str(100 + i)
        if rec_id in MITBIH_PACED_RECORDS:
            continue
        subject_id = f"MITBIH_{MITBIH_SHARED_SUBJECT.get(rec_id, rec_id)}"
        for lbl, hr_bpm in [("nsr", 68 + rng.normal(0, 4)),
                            ("arrhythmic", 110 + rng.normal(0, 10))]:
            for _ in range(4):
                hr_hz = hr_bpm / 60.0
                ecg = np.zeros(N)
                beat_period_s = 1.0 / hr_hz
                jitter = rng.normal(0, 0.02, size=200)
                bt = 0.2
                bi = 0
                while bt < window_sec and bi < len(jitter):
                    idx = int((bt + jitter[bi]) * fs)
                    if 0 <= idx < N - int(0.15 * fs):
                        ecg[idx:idx + int(0.03 * fs)] += 1.0
                        ecg[idx + int(0.03 * fs):idx + int(0.06 * fs)] -= 0.3
                    if lbl == "arrhythmic" and rng.random() < 0.15:
                        bt += beat_period_s * rng.uniform(0.4, 1.5)
                    else:
                        bt += beat_period_s
                    bi += 1
                ecg += 0.03 * np.sin(2 * np.pi * 0.167 * t)  # tape artifact
                ecg += 0.04 * rng.standard_normal(N)
                windows.append(SubjectWindow(
                    subject_id=subject_id, dataset="MITBIH", label=lbl,
                    fs=fs, ecg=ecg, eda=None, resp=None,
                    meta={"record": rec_id, "synthetic": True,
                          "series": "200" if int(rec_id) >= 200 else "100"},
                ))
    return windows


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all_windows(
    wesad_root: Optional[str] = None,
    mitbih_root: Optional[str] = None,
    window_sec: int = 60,
    synthetic_if_missing: bool = True,
) -> List[SubjectWindow]:
    """Return a flat list of WESAD + MIT-BIH windows."""
    w = load_wesad_windows(
        root=wesad_root or WESAD_ROOT,
        window_sec=window_sec,
        synthetic_if_missing=synthetic_if_missing,
    )
    m = load_mitbih_windows(
        root=mitbih_root or MITBIH_ROOT,
        window_sec=window_sec,
        synthetic_if_missing=synthetic_if_missing,
    )
    return w + m


if __name__ == "__main__":
    print("WESAD:")
    for sid, _ in _list_wesad_subjects()[:3]:
        print(" ", sid)
    print("MIT-BIH:")
    for r in _list_mitbih_records()[:5]:
        print(" ", r)
    print("Synthetic smoke:")
    ws = _synthesize_wesad_windows(n_subjects=1)
    ms = _synthesize_mitbih_windows(n_records=2)
    print(f"  WESAD windows: {len(ws)}, MIT-BIH windows: {len(ms)}")
