"""
Review 8, item 7: an action model for abstention (post hoc; exploratory;
NOT in prespec_round5.md).

Section 5.9 scores abstention as removal: the most uncertain windows are
dropped and error is computed on the rest. For a detector that must "abstain
or escalate", what matters is what happens to the abstained windows. With
EDA missing (reduced condition), per-condition-Platt probabilities, alert
threshold 0.5, and confidence = max(p, 1 - p) as in the risk-coverage
analysis, at 60% and 80% coverage we score:

  none        no abstention
  remove      abstained windows dropped (the section 5.9 scoring)
  no_alert    abstained windows are NOT alerted (abstain -> no alert)
  alert       abstained windows ARE alerted    (abstain -> escalate)

Each reports sensitivity over all stress windows and false alarms per hour of
rest = (1 - specificity over all baseline windows) x 60 (60-s windows).
Ties at the confidence cut-off are split fractionally, as in
run_metrics.selective_by_class. Same 20 partitions and seeds as the primary
run. Descriptive means over the 100 folds; no inference.

Output: results/abstention_action.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_core as ec
from run_operating_points import BASE_MODELS, SYSTEMS
from run_repeated_cv import MODEL_SEED_BASE, N_FOLDS

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "abstention_action.json"
COVERAGES = (60, 80)


def _weights(p: np.ndarray, coverage: float) -> np.ndarray:
    conf = np.round(np.maximum(p, 1 - p), 12)
    k = int(np.ceil(coverage * len(p)))
    cut = np.sort(conf)[::-1][k - 1]
    above, tied = conf > cut, conf == cut
    w = above.astype(float)
    w[tied] = (k - above.sum()) / tied.sum()
    return w


def scores(y: np.ndarray, p: np.ndarray, coverage: float | None) -> dict:
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    pred = (p >= 0.5).astype(float)
    out = {}
    if coverage is None:
        out["none"] = (float(np.sum(pred * pos) / pos.sum()), float(np.sum(pred * neg) / neg.sum() * 60))
        return out
    w = _weights(p, coverage)
    out["remove"] = (float(np.sum(w * pred * pos) / np.sum(w * pos)),
                     float(np.sum(w * pred * neg) / np.sum(w * neg) * 60))
    out["no_alert"] = (float(np.sum(w * pred * pos) / pos.sum()), float(np.sum(w * pred * neg) / neg.sum() * 60))
    a = w * pred + (1 - w)
    out["alert"] = (float(np.sum(a * pos) / pos.sum()), float(np.sum(a * neg) / neg.sum() * 60))
    return out


def _fold(df, rep, fold_i, test_subj):
    import torch
    torch.set_num_threads(1)
    train_df, test_df = ec.split_by_subjects(df, test_subj)
    seed = MODEL_SEED_BASE + rep * N_FOLDS + fold_i
    preds = {n: ec.fit_predict(n, train_df, test_df, seed=seed) for n in BASE_MODELS}
    rows = []
    for sysname, by_tier in SYSTEMS.items():
        out = preds[by_tier["reduced"]]
        sub = out[out.tier == "reduced"]
        y, p = sub.y.values, sub.p_platt_pc.values
        for cov in (None, *COVERAGES):
            for scheme, (sens, fa) in scores(y, p, None if cov is None else cov / 100).items():
                rows.append({"system": sysname, "coverage": 100 if cov is None else cov, "scheme": scheme,
                             "sensitivity": sens, "false_alarms_per_hour": fa})
    return rows


def main() -> dict:
    from joblib import Parallel, delayed
    df = ec.load_wesad_frozen()
    tasks = [(rep, fi, ts) for rep in range(20)
             for fi, ts in enumerate(ec.subject_kfold(df.subject_id, N_FOLDS, seed=1000 + rep))]
    outs = Parallel(n_jobs=12, verbose=5)(delayed(_fold)(df, r, f, ts) for r, f, ts in tasks)
    t = pd.DataFrame([r for o in outs for r in o])
    res = {"note": "Post hoc, exploratory (review 8). Reduced condition, per-condition Platt, threshold 0.5.",
           "systems": {}}
    for s, g in t.groupby("system"):
        res["systems"][s] = {f"{c}_{k}": {"sensitivity": float(h.sensitivity.mean()),
                                          "false_alarms_per_hour": float(h.false_alarms_per_hour.mean())}
                             for (c, k), h in g.groupby(["coverage", "scheme"])}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for s, v in res["systems"].items():
        print(s)
        for k, e in v.items():
            print(f"  {k:12s} sens {e['sensitivity']:.3f}  FA/h {e['false_alarms_per_hour']:.1f}")
    return res


if __name__ == "__main__":
    main()
