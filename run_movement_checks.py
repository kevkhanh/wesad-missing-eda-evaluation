"""
Review 8, item 6: two cheap checks on the movement finding (post hoc;
exploratory; written after the primary results and the posture analysis).

  1. Paired participant-cluster bootstrap (B = 2,000) of movement-only minus
     HR-only LOSO AUROC (pooled and within-subject) and Brier. Table S7 shows
     overlapping separate CIs; this is the paired contrast the text needs.
  2. Are the "heart-rate responders vs non-responders" movers vs non-movers?
     Spearman correlation across the 15 participants between each model's
     within-subject AUROC (reduced condition, LOSO, results/per_subject_loso.csv)
     and (a) the standardised HR response, (b) the standardised movement
     response (stress - baseline difference in chest ACC magnitude SD over
     the pooled within-condition SD), plus the correlation of (a) with (b).

Output: results/movement_checks.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import eval_core as ec
import run_posture_movement as rp
from run_participant_analyses import MODELS, _spearman_ci

R = Path(__file__).resolve().parent / "results"
OUT = R / "movement_checks.json"


def paired_contrast() -> dict:
    ext4 = ec.load_extended("chest", conditions=("baseline", "stress", "amusement", "meditation"))
    pf = rp.posture_features(ext4)
    ext4 = ext4.merge(pf, on=["subject_id", "start_sample"], how="left", validate="one_to_one")
    df = ext4[ext4.condition.isin(["baseline", "stress"])].copy()
    df["y"] = (df.condition == "stress").astype(int)
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    assert len(df) == 434
    pred = rp.loso(df)
    return rp.bootstrap(pred, [("movement_only", "hr_only"), ("acc_only", "hr_only")])["contrasts"]


def movement_response() -> pd.Series:
    ext = ec.load_extended("chest", conditions=("baseline", "stress"))
    out = {}
    for s, g in ext.groupby("subject_id"):
        b = g[g.condition == "baseline"]["chest_acc_mag_sd"].astype(float)
        st = g[g.condition == "stress"]["chest_acc_mag_sd"].astype(float)
        pooled = np.sqrt(((len(b) - 1) * b.var(ddof=1) + (len(st) - 1) * st.var(ddof=1)) / (len(b) + len(st) - 2))
        out[s] = float((st.mean() - b.mean()) / pooled)
    return pd.Series(out)


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        beta = np.cov(a, b)[0, 1] / np.var(b, ddof=1)
        return a - beta * (b - b.mean())
    return float(stats.pearsonr(resid(rx, rz), resid(ry, rz))[0])


def correlations() -> dict:
    ps = pd.read_csv(R / "per_subject_loso.csv")
    red = ps[ps.tier == "reduced"]
    hr = red.drop_duplicates("subject_id").set_index("subject_id")
    hr_std = -hr["hr_response_standardised"]
    mv = movement_response().reindex(hr_std.index)
    out = {"movement_response_standardised": {"min": float(mv.min()), "median": float(mv.median()), "max": float(mv.max())},
           "hr_vs_movement_response": _spearman_ci(hr_std.values, mv.values), "per_model": {}}
    for key, short in MODELS.items():
        g = red[red.model == key].set_index("subject_id").reindex(hr_std.index)["auroc_raw"].values
        out["per_model"][short] = {
            "with_hr_response": _spearman_ci(hr_std.values, g),
            "with_movement_response": _spearman_ci(mv.values, g),
            "partial_hr_given_movement": partial_spearman(hr_std.values, g, mv.values),
            "partial_movement_given_hr": partial_spearman(mv.values, g, hr_std.values)}
    return out


def main() -> dict:
    res = {"note": "Post hoc, exploratory (review 8).", "movement_vs_hr_paired": paired_contrast(),
           "response_correlations": correlations()}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    for c, v in r["movement_vs_hr_paired"].items():
        print(c, {k: (round(x["est"], 3), [round(y, 3) for y in x["ci95"]]) for k, x in v.items()})
    rc = r["response_correlations"]
    print("HR vs movement response", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rc["hr_vs_movement_response"].items() if k in ("rho", "p")})
    for m, v in rc["per_model"].items():
        print(f"{m:18s} rho_hr {v['with_hr_response']['rho']:+.2f}  rho_mv {v['with_movement_response']['rho']:+.2f}  "
              f"partial_hr|mv {v['partial_hr_given_movement']:+.2f}  partial_mv|hr {v['partial_movement_given_hr']:+.2f}")
