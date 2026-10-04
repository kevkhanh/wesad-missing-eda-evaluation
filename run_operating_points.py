"""
Round 5, C4: clinical operating points, abstention by class and decision
curves, as pre-specified in prespec_round5.md (SHA-256 recorded in
results/prespec_round5_manifest.json and re-checked here).

Same frozen table, the same 20 participant partitions and the same model
seeds as run_repeated_cv.py, so raw per-fold Brier must reproduce
results/repeated_cv_per_fold.csv exactly (checked and written to the output).

Systems: gated + dropout (G), non-gated MLP with the same masking (M),
NaN-trained LightGBM (ii), and the per-configuration system (PC: complete-
trained LightGBM (i) with EDA, ECG-only LightGBM (iii) without it).

Operating thresholds are chosen per condition on the inner subject-level OOF
predictions over training participants only (the same OOF predictions that
fit Platt), never on the held-out participants:
  sens90: p >= 10th percentile of OOF stress scores (target sensitivity 0.90)
  spec90: p >  90th percentile of OOF baseline scores (target specificity 0.90)
  p05:    per-condition-Platt probability >= 0.5 (check)
Net benefit is computed on per-condition-Platt probabilities; inference only
at pt in {0.20, 0.35, 0.50}.

CLI:
    python run_operating_points.py              # 20 repeats
    python run_operating_points.py --repeats 1  # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_core as ec
from run_metrics import (compute_panel, confusion_at, net_benefit, net_benefit_contrib,
                         selective_by_class, threshold_for_sensitivity,
                         threshold_for_specificity)
from run_repeated_cv import (G, L1, L2, L3, M, MODEL_SEED_BASE, N_FOLDS,
                             bouckaert_frank, participant_level)

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "results" / "operating_points.json"
PRESPEC = HERE / "prespec_round5.md"
PRESPEC_MANIFEST = HERE / "results" / "prespec_round5_manifest.json"

BASE_MODELS = [G, M, L1, L2, L3]
SYSTEMS = {"gated": {"complete": G, "reduced": G},
           "mlp": {"complete": M, "reduced": M},
           "lgbm_nan_training": {"complete": L2, "reduced": L2},
           "per_config": {"complete": L1, "reduced": L3}}
COMPARISONS = [("gated", "mlp"), ("gated", "lgbm_nan_training"), ("gated", "per_config"),
               ("mlp", "per_config"), ("lgbm_nan_training", "per_config")]
TIERS = ("reduced", "complete")
PT_GRID = [round(x, 2) for x in np.arange(0.05, 0.601, 0.05)]
PT_INFER = [0.2, 0.35, 0.5]
OPS = ("sens90", "spec90", "p05")
# Margins fixed in prespec_round5.md.
MARGIN = {**{f"{op}_{s}": 0.05 for op in OPS for s in ("sens", "spec")},
          **{f"nb_{pt}": 0.01 for pt in PT_INFER}}
INFER_METRICS = ([f"{op}_{s}" for op in OPS for s in ("sens", "spec", "ppv", "npv")]
                 + [f"nb_{pt}" for pt in PT_INFER]
                 + [f"sel{c}_{s}" for c in (60, 80)
                    for s in ("sens", "spec", "abstain_stress", "abstain_baseline")])
PARTICIPANT_METRICS = ([f"{op}_{s}" for op in ("sens90", "spec90") for s in ("sens", "spec")]
                       + [f"nb_{pt}" for pt in PT_INFER])


def prespec_sha() -> str:
    sha = hashlib.sha256(PRESPEC.read_bytes()).hexdigest()
    recorded = json.loads(PRESPEC_MANIFEST.read_text(encoding="utf-8"))["sha256"]
    if sha != recorded:
        raise RuntimeError("prespec_round5.md changed after its hash was recorded; "
                           "refusing to run a 'pre-specified' analysis against it.")
    return sha


def _system_metrics(y, p_raw, p_pc, oof_y, oof_p) -> tuple[dict, dict]:
    thr = {"sens90": threshold_for_sensitivity(oof_y, oof_p, 0.9),
           "spec90": threshold_for_specificity(oof_y, oof_p, 0.9)}
    out = {"thr_sens90": thr["sens90"], "thr_spec90": thr["spec90"],
           "prevalence": float(np.mean(y))}
    for k, v in confusion_at(y, p_raw, thr["sens90"]).items():
        out[f"sens90_{k}"] = v
    for k, v in confusion_at(y, p_raw, thr["spec90"], strict=True).items():
        out[f"spec90_{k}"] = v
    for k, v in confusion_at(y, p_pc, 0.5).items():
        out[f"p05_{k}"] = v
    for cov in (60, 80):
        for k, v in selective_by_class(y, p_pc, cov / 100).items():
            out[f"sel{cov}_{k}"] = v
    for pt in PT_GRID:
        out[f"nb_{pt}"] = net_benefit(y, p_pc, pt)
        out[f"nb_raw_{pt}"] = net_benefit(y, p_raw, pt)
        out[f"nb_treat_all_{pt}"] = net_benefit(y, np.ones_like(p_pc), pt)
    return out, thr


def _fold_task(df: pd.DataFrame, rep: int, fold_i: int, test_subj):
    import torch
    torch.set_num_threads(1)
    train_df, test_df = ec.split_by_subjects(df, test_subj)
    seed = MODEL_SEED_BASE + rep * N_FOLDS + fold_i
    preds, oofs, check = {}, {}, []
    for name in BASE_MODELS:
        out, oof = ec.fit_predict(name, train_df, test_df, seed=seed, return_oof=True)
        preds[name], oofs[name] = out, oof
        for tier in TIERS:
            sub = out[out.tier == tier]
            check.append({"repeat": rep, "fold": fold_i, "model": name, "tier": tier,
                          "brier": compute_panel(sub.y.values, sub.p_raw.values,
                                                 smooth=False).brier})
    fold_rows, subj_rows = [], []
    for sysname, by_tier in SYSTEMS.items():
        for tier in TIERS:
            name = by_tier[tier]
            sub = preds[name][preds[name].tier == tier]
            oof_y, oof = oofs[name]
            m, thr = _system_metrics(sub.y.values, sub.p_raw.values, sub.p_platt_pc.values,
                                     oof_y, oof[tier])
            fold_rows.append({"repeat": rep, "fold": fold_i, "system": sysname,
                              "tier": tier, **m})
            for s_, g in sub.groupby("subject_id"):
                y, pr, pc = g.y.values, g.p_raw.values, g.p_platt_pc.values
                row = {"repeat": rep, "system": sysname, "tier": tier, "subject_id": s_}
                c1 = confusion_at(y, pr, thr["sens90"])
                c2 = confusion_at(y, pr, thr["spec90"], strict=True)
                row.update({"sens90_sens": c1["sens"], "sens90_spec": c1["spec"],
                            "spec90_sens": c2["sens"], "spec90_spec": c2["spec"]})
                for pt in PT_INFER:
                    row[f"nb_{pt}"] = float(np.mean(net_benefit_contrib(y, pc, pt)))
                subj_rows.append(row)
    return fold_rows, subj_rows, check


def _verify(check_df: pd.DataFrame) -> dict:
    ref = pd.read_csv(HERE / "results" / "repeated_cv_per_fold.csv")
    ref = ref[ref.calib == "raw"][["repeat", "fold", "model", "tier", "brier"]]
    m = check_df.merge(ref, on=["repeat", "fold", "model", "tier"], suffixes=("", "_ref"))
    dmax = float(np.abs(m.brier - m.brier_ref).max()) if len(m) else float("nan")
    return {"n_checked": int(len(m)), "n_expected": int(len(check_df)),
            "max_abs_brier_diff": dmax,
            "reproduces_repeated_cv": bool(len(m) == len(check_df) and dmax < 1e-9)}


def run(n_repeats: int = 20, n_jobs: int = 12) -> dict:
    from joblib import Parallel, delayed
    sha_pre = prespec_sha()
    df = ec.load_wesad_frozen()
    n_subj = df.subject_id.nunique()
    tasks, n_train, n_test = [], [], []
    for rep in range(n_repeats):
        for fold_i, ts in enumerate(ec.subject_kfold(df.subject_id, N_FOLDS, seed=1000 + rep)):
            tasks.append((rep, fold_i, ts))
            n_train.append(n_subj - len(ts))
            n_test.append(len(ts))
    print(f"[operating_points] {len(tasks)} fold tasks, n_jobs={n_jobs}", flush=True)
    outs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fold_task)(df, r, f, ts) for r, f, ts in tasks)
    folds = pd.DataFrame([r for o in outs for r in o[0]])
    subj = pd.DataFrame([r for o in outs for r in o[1]])
    check = pd.DataFrame([r for o in outs for r in o[2]])
    ntr, nte = float(np.mean(n_train)), float(np.mean(n_test))

    res = {"config": {"n_repeats": n_repeats, "n_folds": N_FOLDS, "n_subjects": int(n_subj),
                      "systems": SYSTEMS, "pt_grid": PT_GRID, "pt_inference": PT_INFER,
                      "margins": MARGIN, "prespec_sha256": sha_pre,
                      "feature_table_sha256": json.loads(
                          ec.MANIFEST_PATH.read_text(encoding="utf-8"))["sha256"]},
           "verification": _verify(check)}
    metric_cols = [c for c in folds.columns
                   if c not in ("repeat", "fold", "system", "tier")]
    for tier in TIERS:
        ft = folds[folds.tier == tier]
        st = subj[subj.tier == tier]
        res[f"per_system_{tier}"] = {
            s: {f"{c}_mean": float(g[c].mean()) for c in metric_cols}
            for s, g in ft.groupby("system")}
        frames = {s: g.set_index(["repeat", "fold"]) for s, g in ft.groupby("system")}
        smeans = {s: g.groupby("subject_id")[PARTICIPANT_METRICS].mean()
                  for s, g in st.groupby("system")}
        res[f"pairwise_{tier}"] = {}
        for a, b in COMPARISONS:
            e = {m: bouckaert_frank((frames[a][m] - frames[b][m]).values, ntr, nte, n_subj,
                                    MARGIN.get(m)) for m in INFER_METRICS}
            e["participant"] = {m: participant_level(smeans[a][m] - smeans[b][m],
                                                     MARGIN.get(m), lower_is_better=False)
                                for m in PARTICIPANT_METRICS}
            res[f"pairwise_{tier}"][f"{a}__vs__{b}"] = e
        # C4-P4: within-system abstention asymmetry (stress minus baseline).
        res[f"abstention_asymmetry_{tier}"] = {
            s: {f"sel{c}": bouckaert_frank(
                (g[f"sel{c}_abstain_stress"] - g[f"sel{c}_abstain_baseline"]).values,
                ntr, nte, n_subj) for c in (60, 80)}
            for s, g in ft.groupby("system")}

    OUT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
    folds.to_csv(OUT_PATH.with_name("operating_points_per_fold.csv"), index=False)
    subj.to_csv(OUT_PATH.with_name("operating_points_per_subject.csv"), index=False)
    _print(res)
    _figure(folds)
    print(f"\n[operating_points] wrote {OUT_PATH}")
    return res


def _fmt(e: dict) -> str:
    return f"{e['mean']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"


def _print(res: dict) -> None:
    v = res["verification"]
    print(f"\nverification vs repeated_cv: {v}")
    for tier in TIERS:
        print(f"\n=== {tier} ===")
        for s, r in res[f"per_system_{tier}"].items():
            print(f"  {s:18s} sens90: sens {r['sens90_sens_mean']:.3f} spec {r['sens90_spec_mean']:.3f}"
                  f" | spec90: sens {r['spec90_sens_mean']:.3f} spec {r['spec90_spec_mean']:.3f}"
                  f" | NB .2/.35/.5 {r['nb_0.2_mean']:.3f}/{r['nb_0.35_mean']:.3f}/{r['nb_0.5_mean']:.3f}"
                  f" | abst60 S/B {r['sel60_abstain_stress_mean']:.2f}/{r['sel60_abstain_baseline_mean']:.2f}")
        for k, e in res[f"pairwise_{tier}"].items():
            print(f"  {k}")
            for m in PARTICIPANT_METRICS:
                pe = e["participant"][m]
                print(f"    {m:13s} BF {_fmt(e[m])} TOST {e[m].get('tost_p', float('nan')):.3f}"
                      f" | participant {_fmt(pe)} TOST {pe.get('tost_p', float('nan')):.3f}")
        for s, a in res[f"abstention_asymmetry_{tier}"].items():
            print(f"  abstention stress-baseline {s:18s} 60%: {_fmt(a['sel60'])}")


def _figure(folds: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"gated": "Gated + dropout", "mlp": "MLP, same masking",
              "lgbm_nan_training": "(ii) NaN-trained LightGBM",
              "per_config": "Per-configuration (i)/(iii)"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, tier in zip(axes, ("reduced", "complete")):
        ft = folds[folds.tier == tier]
        for s, lab in labels.items():
            g = ft[ft.system == s]
            ax.plot(PT_GRID, [g[f"nb_{pt}"].mean() for pt in PT_GRID], label=lab)
        ax.plot(PT_GRID, [ft[f"nb_treat_all_{pt}"].mean() for pt in PT_GRID],
                "k--", lw=1, label="Treat all")
        ax.axhline(0, color="grey", lw=1, label="Treat none")
        for pt in PT_INFER:
            ax.axvline(pt, color="grey", lw=0.5, ls=":")
        ax.set_title(f"EDA {'missing' if tier == 'reduced' else 'present'}")
        ax.set_xlabel("Threshold probability")
    axes[0].set_ylabel("Net benefit (mean over 100 folds)")
    axes[0].set_ylim(-0.05, 0.4)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "results" / "fig_decision_curves.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    run(a.repeats, a.jobs)
