"""
Participant-level and small statistical items from peer_reviewe_v2.md.

  M7      HR-response analysis, reframed: raw and STANDARDISED response
          (delta mean NN / pooled within-condition SD); Spearman rho with
          Fisher-z and participant-bootstrap 95% CIs; the HR-removed models'
          dependence on HR response (the non-tautological quantity).
          Exploratory throughout.
  m2      S10: Hanley-McNeil SE of its within-subject AUROC; heart rate by
          condition from chest ECG and wrist BVP; movement; beat plausibility.
  m8      H2a lag noise floor: SD of U(-15, 15) vs the observed within-rest SD.
  m9      Legacy Wilcoxon with n = 5: smallest attainable two-sided p.
  m10     Legacy 4.50x LightGBM ECE ratio: Wald vs log-scale (geometric) CI.
  m12     Signal quality: fraction of windows with implausible NN intervals
          or > 20% successive NN jumps (chest), and wrist-chest HR agreement.

Inputs: results/per_subject_loso.csv (run_baselines.py), the frozen and
extended tables, results/h2a_subject_level.json, the legacy locked log.
Output: results/participant_analyses.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import eval_core as ec

R = Path(__file__).resolve().parent / "results"
OUT = R / "participant_analyses.json"
MODELS = {"gated_dropout_p0.5": "gated", "lgbm_nan_training_p0.5": "ii",
          "hr_only": "hr_only", "lgbm_ecg_only_per_config": "iii",
          "mlp_nongated_dropout_p0.5": "mlp",
          "gated_dropout_hr_removed": "gated_hr_removed",
          "lgbm_nan_training_hr_removed": "ii_hr_removed",
          "lgbm_ecg_only_hr_removed": "iii_hr_removed"}


def _spearman_ci(x, y, n_boot=5000, seed=3):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(stats.spearmanr(x, y)[0])
    n = len(x)
    z, se = np.arctanh(rho), 1.0 / np.sqrt(n - 3)
    fisher = [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) > 2:
            bs.append(stats.spearmanr(x[i], y[i])[0])
    return {"rho": rho, "fisher_z_ci95": fisher,
            "bootstrap_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
            "p": float(stats.spearmanr(x, y)[1]), "n": n}


def hr_response() -> dict:
    ps = pd.read_csv(R / "per_subject_loso.csv")
    red = ps[ps.tier == "reduced"]
    hr = red.drop_duplicates("subject_id").set_index("subject_id")
    raw = -hr["delta_mean_nn_ms_stress_minus_baseline"]          # drop in NN, ms
    std = -hr["hr_response_standardised"]                         # in pooled-SD units
    out = {"response_ms": {"min": float(raw.min()), "max": float(raw.max())},
           "response_standardised": {"min": float(std.min()), "median": float(std.median()),
                                     "max": float(std.max())},
           "weak_responders_lt_60ms": sorted(raw[raw < 60].index.str.replace("WESAD_", "").tolist()),
           "per_model": {}}
    for key, short in MODELS.items():
        g = red[red.model == key].set_index("subject_id").reindex(raw.index)
        out["per_model"][short] = {
            "raw_response": _spearman_ci(raw.values, g["auroc_raw"].values),
            "standardised_response": _spearman_ci(std.values, g["auroc_raw"].values),
            "within_subject_auroc_mean": float(g["auroc_raw"].mean()),
        }
    return out


def s10() -> dict:
    ps = pd.read_csv(R / "per_subject_loso.csv")
    red = ps[(ps.tier == "reduced") & (ps.subject_id == "WESAD_S10")].set_index("model")
    ext = ec.load_extended("chest", ("baseline", "stress", "amusement", "meditation"))
    e = ext[ext.subject_id == "WESAD_S10"]
    by = e.groupby("condition").agg(
        chest_bpm=("hrv_mean_nn", lambda v: float(np.mean(60000 / v))),
        wrist_bpm=("wrist_hrv_mean_nn", lambda v: float(np.nanmean(60000 / v))),
        rmssd_ms=("hrv_rmssd", "mean"),
        chest_acc_sd_g=("chest_acc_mag_sd", "mean"),
        nn_jump20=("chest_frac_nn_jump20", "mean"),
        nn_jump20_median=("chest_frac_nn_jump20", "median"),
        t_start_min=("t_start_min", "mean"))
    allsub = ext.groupby(["subject_id", "condition"])["hrv_mean_nn"].mean().unstack()
    rank_baseline_hr = int((60000 / allsub["baseline"]).rank(ascending=False)["WESAD_S10"])
    out = {"within_subject_auroc_reduced": {
               m: {"auroc": float(red.loc[m, "auroc_raw"]),
                   "se_hanley_mcneil": float(red.loc[m, "auroc_se_hanley_mcneil"]),
                   "z_vs_0.5": float((red.loc[m, "auroc_raw"] - 0.5) / red.loc[m, "auroc_se_hanley_mcneil"])}
               for m in MODELS if m in red.index},
           "by_condition": by.round(4).to_dict(orient="index"),
           "baseline_hr_rank_among_15_highest_first": rank_baseline_hr}
    return out


def quality() -> dict:
    ext = ec.load_extended("chest", ("baseline", "stress", "amusement", "meditation"))
    bs = ext[ext.condition.isin(["baseline", "stress"])]
    out = {"chest": {
        "frac_windows_any_implausible_nn": float((bs.chest_frac_nn_implausible > 0).mean()),
        "frac_windows_jump20_gt_10pct": float((bs.chest_frac_nn_jump20 > 0.10).mean()),
        "median_frac_nn_jump20": float(bs.chest_frac_nn_jump20.median()),
        "max_frac_nn_jump20": float(bs.chest_frac_nn_jump20.max()),
        "n_windows": int(len(bs))}}
    agree = {}
    for c, g in ext.groupby("condition"):
        cb, wb = 60000 / g.hrv_mean_nn, 60000 / g.wrist_hrv_mean_nn
        ok = np.isfinite(wb)
        agree[c] = {"pearson_r": float(np.corrcoef(cb[ok], wb[ok])[0, 1]),
                    "median_abs_err_bpm": float(np.median(np.abs(cb[ok] - wb[ok]))),
                    "spectral_median_abs_err_bpm": float(np.median(np.abs(cb - g.wrist_spectral_hr_bpm))),
                    "wrist_median_frac_nn_jump20": float(g.wrist_frac_nn_jump20.median()),
                    "n": int(ok.sum())}
    out["wrist_vs_chest_hr"] = agree
    return out


def wrist_diagnostics() -> dict:
    """§5.10: is the wrist 'without EDA' signal cardiac? Correlation of wrist
    and chest HRV per condition, wrist RMSSD vs the beat-artefact fraction,
    and univariate pooled AUROC (stress vs baseline) of artefact and movement."""
    from sklearn.metrics import roc_auc_score
    ext = ec.load_extended("chest", ("baseline", "stress"))
    y = (ext.condition == "stress").astype(int)
    out = {"spearman_wrist_vs_chest": {}, "spearman_wrist_rmssd_vs_artefact": {}}
    for c, g in ext.groupby("condition"):
        out["spearman_wrist_vs_chest"][c] = {
            f: float(stats.spearmanr(g[f], g[f"wrist_{f}"], nan_policy="omit")[0])
            for f in ("hrv_rmssd", "hrv_sdnn", "hrv_mean_nn")}
        out["spearman_wrist_rmssd_vs_artefact"][c] = float(
            stats.spearmanr(g["wrist_hrv_rmssd"], g["wrist_frac_nn_jump20"], nan_policy="omit")[0])
    auc = lambda v: float(roc_auc_score(y[v.notna()], v[v.notna()]))
    out["univariate_pooled_auroc"] = {"wrist_frac_nn_jump20": auc(ext.wrist_frac_nn_jump20),
                                      "chest_acc_mag_sd": auc(ext.chest_acc_mag_sd),
                                      "wrist_acc_mag_sd": auc(ext.wrist_acc_mag_sd)}
    return out


def small_stats() -> dict:
    h2a = json.loads((R / "h2a_subject_level.json").read_text(encoding="utf-8"))
    log = (R / "locked_comparison_rerun.log").read_text(encoding="utf-8", errors="replace")
    blk = log[log.index("LightGBM baseline (locked folds)"):]
    ratios = np.array([float(x) for x in re.findall(r"'([\d.]+)'", blk.split("Per-fold ratios:")[1].splitlines()[0])])
    n = len(ratios)
    m, sd = ratios.mean(), ratios.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    lr = np.log(ratios)
    gm = float(np.exp(lr.mean()))
    lci = np.exp(lr.mean() + np.array([-1, 1]) * t * lr.std(ddof=1) / np.sqrt(n))
    return {
        "h2a_lag_noise_floor": {
            "sd_uniform_pm15_sec": float(30 / np.sqrt(12)),
            "observed_within_rest_sd_sec_median": h2a["within_subject_sd_lag_baseline_sec"]["median"],
            "ratio_observed_to_uniform": float(h2a["within_subject_sd_lag_baseline_sec"]["median"] / (30 / np.sqrt(12)))},
        "legacy_wilcoxon_n5_min_two_sided_p": float(stats.wilcoxon(np.arange(1, 6)).pvalue),
        "legacy_lgbm_ratio": {"per_fold": ratios.tolist(), "arithmetic_mean": float(m),
                              "wald_ci95": [float(m - t * sd / np.sqrt(n)), float(m + t * sd / np.sqrt(n))],
                              "geometric_mean": gm, "log_scale_ci95": [float(lci[0]), float(lci[1])]},
    }


def main():
    res = {"note": "Exploratory (HR response) and descriptive items; review round 4.",
           "hr_response": hr_response(), "s10": s10(), "signal_quality": quality(), "wrist_diagnostics": wrist_diagnostics(),
           "small_stats": small_stats()}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
