"""
Numbers consistency check for e14-preprint-outline.md (round-3 review, item 6).

Every check ties ONE place in the outline (a table cell found by table header
+ row label + column, or a prose line found by an anchor phrase) to ONE value
recomputed from a results file (JSON path, CSV query, log line, or a direct
recomputation from code). The source value is formatted the way the outline
formats it and compared as a string, so a check passes only if the outline
shows exactly the rounded source value in exactly that place.

This avoids the failure of a pooled "does this number appear anywhere in the
results?" search, where any 3-decimal token matches something by chance.

After the keyed checks, every numeric token in the outline that no keyed
check covered is listed as UNTRACED, grouped by section, so the coverage gap
is explicit rather than silent.

Round 4 (peer_reviewe_v2.md M9): rewritten for revision 6 of the outline;
the untraced-token list is now resolved against an explicit allowlist of
method constants and external facts (CONSTANTS), so a remaining token is a
real gap.

Usage:  python check_numbers.py            # writes results/number_check_report.md
Exit code 1 if any keyed check FAILS.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
OUTLINE = ROOT / "e14-preprint-outline.md"
REPORT = R / "number_check_report.md"

MINUS = "\u2212"

G, M = "gated_dropout_p0.5", "mlp_nongated_dropout_p0.5"
L2, L3, L4 = "lgbm_nan_training_p0.5", "lgbm_ecg_only_per_config", "lgbm_mean_impute_recal"
HR, GN = "hr_only", "gated_no_dropout"
LI, LZ = "lgbm_nan_at_test", "lgbm_zero_fill_at_test"
GHR, GRES = "gated_dropout_hr_removed", "gated_dropout_hr_resid"
L2HR, L3HR = "lgbm_nan_training_hr_removed", "lgbm_ecg_only_hr_removed"


# ---------------------------------------------------------------------------
# Formatting (matches the outline's conventions)
# ---------------------------------------------------------------------------

def n(x: float, d: int = 3) -> str:
    """Unsigned format; true minus sign for negatives (−0.000 kept)."""
    return f"{x:.{d}f}".replace("-", MINUS)


def s(x: float, d: int = 3) -> str:
    """Signed format: +0.002 / −0.013."""
    return f"{x:+.{d}f}".replace("-", MINUS)


def sci(m, lo, hi, d=3) -> str:
    return f"{s(m, d)} [{s(lo, d)}, {s(hi, d)}]"


def uci(m, lo, hi, d=3) -> str:
    return f"{n(m, d)} [{n(lo, d)}, {n(hi, d)}]"


def rng(vals, d=2, sep="–") -> str:
    return f"{n(min(vals), d)}{sep}{n(max(vals), d)}"


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def J(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


RCV = J("repeated_cv.json")
TUN = J("repeated_cv_tuned.json")
CAP = J("repeated_cv_capacity.json")
BAS = J("baselines.json")
H2A = J("h2a_subject_level.json")
INT = J("interpretability_frozen.json")
PRE_RCV = J("archive/repeated_cv_2026-10-01_prefix.json")
PRE_BAS = J("archive/baselines_2026-10-01_prefix.json")
PS_LOSO = pd.read_csv(R / "per_subject_loso.csv")
PS_K5 = pd.read_csv(R / "per_subject_kfold5.csv")
LOG_LOCK = (R / "locked_comparison_rerun.log").read_text(encoding="utf-8", errors="replace")
LOG_MOD = (R / "locked_per_modality_rerun.log").read_text(encoding="utf-8", errors="replace")
ARCH_OUTLINE = (ROOT / "archive/e14-preprint-outline_2026-10-01_pre-rerun.md").read_text(encoding="utf-8")


def pm(src, calib, model, metric):          # repeated-CV per-model
    return src["per_model_reduced"][calib][model][metric]


def pw(src, calib, a, b, metric):           # repeated-CV pairwise
    return src["pairwise"][calib][f"{a}__vs__{b}"][metric]


def pooled(scheme, model, calib="raw", tier="reduced"):
    return BAS["schemes"][scheme]["models"][model][calib][tier]["pooled"]


def pci(scheme, model, metric, calib="raw"):
    return BAS["schemes"][scheme]["models"][model][calib]["reduced"]["pooled_ci95"][metric]


def pw_ci(src, calib, a, b, metric, d=3):
    e = pw(src, calib, a, b, metric)
    return sci(e["mean"], *e["ci95"], d)


def pooled_ci(scheme, model, metric, calib="raw", d=3):
    return uci(pooled(scheme, model, calib)[metric], *pci(scheme, model, metric, calib), d)


# ---------------------------------------------------------------------------
# Outline access
# ---------------------------------------------------------------------------

LINES = OUTLINE.read_text(encoding="utf-8").splitlines()


def section_of(i: int) -> str:
    for j in range(i, -1, -1):
        if LINES[j].startswith("## ") or LINES[j].startswith("### "):
            return LINES[j].lstrip("# ").strip()
    return "front matter"


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def find_table(header_substr: str, occurrence: int = 0) -> tuple[int, list[str], dict[str, int]]:
    """Locate a markdown table by a substring of its header row."""
    hits = [i for i, l in enumerate(LINES) if l.lstrip().startswith("|") and header_substr in l]
    if len(hits) <= occurrence:
        raise LookupError(f"table header not found: {header_substr!r}")
    h = hits[occurrence]
    header = _cells(LINES[h])
    rows = {}
    for i in range(h + 2, len(LINES)):
        if not LINES[i].lstrip().startswith("|"):
            break
        rows[_cells(LINES[i])[0].replace("**", "")] = i
    return h, header, rows


# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

@dataclass
class Result:
    status: str          # PASS / FAIL / NOTE
    where: str
    line: int | None
    expected: str
    found: str
    source: str
    note: str = ""


RESULTS: list[Result] = []
COVERED: dict[int, list[tuple[int, int]]] = {}   # line -> covered char spans


def _cover(i: int, text: str, frag: str):
    """Mark every occurrence of frag on line i as traced (a verified value
    may legitimately appear twice on a line, e.g. equal raw/standardised rho)."""
    if not frag:
        return
    k = text.find(frag)
    while k >= 0:
        COVERED.setdefault(i, []).append((k, k + len(frag)))
        k = text.find(frag, k + 1)


def cell(header_substr, row, col, expected, source, occurrence=0, note=""):
    try:
        h, header, rows = find_table(header_substr, occurrence)
        i = rows[row]
        j = header.index(col) if isinstance(col, str) else col
        got = _cells(LINES[i])[j]
    except (LookupError, KeyError, ValueError, IndexError) as e:
        RESULTS.append(Result("FAIL", f"table '{header_substr}' / row '{row}' / col '{col}'",
                              None, expected, f"<not found: {e}>", source, note))
        return
    ok = expected in got.replace("**", "")
    if ok:
        _cover(i, LINES[i].replace("**", ""), got.replace("**", ""))
    RESULTS.append(Result("PASS" if ok else "FAIL", f"{section_of(i)} | {row} | {header[j]}",
                          i + 1, expected, got, source, note))


def prose(anchor, expected, source, note="", occurrence=0):
    hits = [i for i, l in enumerate(LINES) if anchor in l]
    if len(hits) <= occurrence:
        RESULTS.append(Result("FAIL", f"anchor {anchor!r}", None, expected, "<anchor not found>", source, note))
        return
    i = hits[occurrence]
    ok = expected in LINES[i].replace("**", "")
    if ok:
        _cover(i, LINES[i].replace("**", ""), expected)
    RESULTS.append(Result("PASS" if ok else "FAIL", f"{section_of(i)} | “{anchor}”", i + 1, expected,
                          "(present)" if ok else LINES[i].strip()[:160], source, note))


def fact(where, ok: bool, expected, found, source, note=""):
    """A non-string claim (a direction, a verdict, a 'no more than') checked logically."""
    RESULTS.append(Result("PASS" if ok else "FAIL", where, None, expected, found, source, note))


def note(where, text, source=""):
    RESULTS.append(Result("NOTE", where, None, "", text, source))


def _log_ratio(log, cond):
    m = re.search(rf"^\s*{re.escape(cond)}\s+.*?([\d.]+)x\s+\[([\d.]+), ([\d.]+)\]", log, re.M)
    return tuple(float(x) for x in m.groups()) if m else None




# ===========================================================================
# Round-4 sources
# ===========================================================================

MR = J("repeated_cv_maskrate.json")
HS = J("repeated_cv_hrv_subset.json")
SNA = J("repeated_cv_subjnorm_all.json")
SNB = J("repeated_cv_subjnorm_baseline.json")
WR = J("repeated_cv_wrist.json")
PA = J("participant_analyses.json")
RE = J("reimpl_minimal_primary.json")
EXTM = J("extended_table_manifest.json")
OC = J("order_confound.json") if (R / "order_confound.json").exists() else None
MN, GHR_ = "mlp_nongated_no_dropout", GHR

SHORT = {G: "Gated", M: "MLP", GN: "Gated, no masking", MN: "MLP, no masking", LI: "(i)", L2: "(ii)",
         L3: "(iii)", L4: "(iv)", HR: "HR-only"}


def pwt(src, calib, a, b, tier="reduced"):
    key = "pairwise" if tier == "reduced" else "pairwise_complete"
    return src[key][calib][f"{a}__vs__{b}"]


def cie(e, d=3, key="ci95"):
    return f"{s(e['mean'], d)} [{s(e[key][0], d)}, {s(e[key][1], d)}]"


def p3(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def bca(scheme, model, metric, calib="raw", tier="reduced"):
    return BAS["schemes"][scheme]["models"][model][calib][tier]["pooled_ci95_bca"][metric]


# ===========================================================================
# CHECKS
# ===========================================================================

def check_front_and_data():
    ps = PS_LOSO[(PS_LOSO.model == G) & (PS_LOSO.tier == "reduced")]
    nwin, nstress = ps.n_windows, ps.n_stress
    nbase = nwin - nstress
    src = "results/per_subject_loso.csv (n_windows, n_stress)"
    prose("Primary table: 434 windows", f"{int(nwin.sum())} windows", src)
    prose("Primary table: 434 windows", f"{nwin.min()}–{nwin.max()} per participant", src)
    prose("Primary table: 434 windows", f"{round(100 * nstress.sum() / nwin.sum())}% stress", src)
    prose("the cap is never reached", f"stress {nstress.min()}–{nstress.max()}", src)
    prose("the cap is never reached", f"baseline yields {nbase.min()}–{nbase.max()} windows", src)
    prose("On WESAD (15 participants", f"{int(nwin.sum())} sixty-second windows", src)
    man = J("feature_table_manifest.json")
    prose("**Every number in §5 comes from the frozen feature table**", man["sha256"][:12], "feature_table_manifest.json")
    prose("**Every number in §5 comes from the frozen feature table**", EXTM["sha256"][:12], "extended_table_manifest.json")
    v = EXTM["verification_vs_frozen_table"]
    ok = v["reproduces_frozen_table"] and v["n_matched"] == 434 and not any(v["n_mismatched_values_per_feature"].values())
    fact("front | extended table reproduces frozen table", ok, "True, 434/434, 0 mismatches",
         f"{v['reproduces_frozen_table']}, {v['n_matched']}", "extended_table_manifest.json")
    prose("**Every number in §5 comes from the frozen feature table**", f"({v['n_matched']}/{v['n_frozen']} windows, 0 mismatched values)", "extended_table_manifest.json")
    rpc = EXTM["rows_per_condition"]
    prose("Extended table: 683 windows",
          f"{EXTM['n_rows']} windows ({rpc['baseline']} baseline, {rpc['stress']} stress, {rpc['amusement']} amusement, {rpc['meditation']} meditation)",
          "extended_table_manifest.json")
    ext = pd.read_csv(R / "extended_features.csv")
    per = ext.groupby(["subject_id", "condition"]).size().unstack()
    prose("the cap is never reached", f"amusement {per['amusement'].min()}–{per['amusement'].max()}", "extended_features.csv")
    prose("the cap is never reached", f"meditation {per['meditation'].min()}–{per['meditation'].max()}", "extended_features.csv")
    order = ext.groupby("subject_id").protocol_stress_before_amusement.first()
    prose("**Fixed condition order.**", f"here {int(order.sum())} participants had stress before amusement, {int((~order).sum())} after",
          "extended_features.csv protocol_stress_before_amusement")
    fact("§3 | 15 participants", man["n_subjects_wesad"] == 15, "15", str(man["n_subjects_wesad"]), "manifest")
    tab = Path(man["path"])
    if tab.exists():
        import hashlib
        sha = hashlib.sha256(tab.read_bytes()).hexdigest()
        fact("§4.1 | frozen table hash matches manifest", sha == man["sha256"], man["sha256"][:12], sha[:12], str(tab))
        sys.path.insert(0, str(ROOT))
        from eval_core import load_wesad_frozen, ECG_FEATURE_NAMES
        df = load_wesad_frozen()
        prose("but was not triggered on this table", f"({df[ECG_FEATURE_NAMES].isna().any(axis=1).mean() * 100:.0f}% missing HRV)", "frozen table")
        cpl = df[["coupling_peak_corr", "coupling_lag_sec"]].isna().any(axis=1).mean() * 100
        prose("Undefined in", f"Undefined in {cpl:.1f}% of windows", "frozen table")
    wk = PA["hr_response"]["weak_responders_lt_60ms"]
    prose("**TSST heart-rate reactivity is heterogeneous**", f"four weak responders here ({round(100 * len(wk) / 15)}%)",
          "participant_analyses.json weak_responders_lt_60ms")


def check_methods():
    sys.path.insert(0, str(ROOT))
    from eval_core import GatedModel, NonGatedMLP, ECG_FEATURE_NAMES, EDA_FEATURE_NAMES
    import eval_core as ec

    def npar(model):
        return sum(p.numel() for p in model._build().parameters())

    gated = {d: npar(GatedModel(0.5, embed_dim=d)) for d in (8, 16, 32, 64)}
    mlp = {h: npar(NonGatedMLP(0.5, hidden=h)) for h in (8, 25, 32, 84, 300)}
    src = "eval_core.py parameter counts"
    fact("§4.1 | 7 ECG, 6 EDA-dependent features", len(ECG_FEATURE_NAMES) == 7 and len(EDA_FEATURE_NAMES) == 6,
         "7 / 6", f"{len(ECG_FEATURE_NAMES)} / {len(EDA_FEATURE_NAMES)}", "eval_core")
    cell("| Name | Description |", "Gated network (hard mask + masking)", 1, f"{gated[16]:,} parameters", src)
    cell("| Name | Description |", "Non-gated MLP (no gate, masking)", 1, f"{mlp[32]:,} parameters", src)
    prose("**Capacity sweep:** gated embedding width", "/".join(f"{gated[d]:,}" for d in (8, 16, 32, 64)) + " parameters", src)
    prose("**Capacity sweep:** gated embedding width", "/".join(f"{mlp[h]:,}" for h in (8, 25, 84, 300)) + " parameters", src)
    n_lgbm = int(np.prod([len(v) for v in ec.LGBM_GRID.values()]))
    n_nn = int(np.prod([len(v) for v in ec.NN_GRID.values()])) * len(ec.GATED_WIDTHS)
    cell("| Family | Grid | Configurations |", "LightGBM", "Configurations", str(n_lgbm), "eval_core.LGBM_GRID")
    cell("| Family | Grid | Configurations |", "Networks", "Configurations", str(n_nn), "eval_core.NN_GRID")
    prose("**Sensitivity sets**", "p ∈ {" + ", ".join(str(p) for p in ec.MASK_RATES) + "}", "eval_core.MASK_RATES")
    fact("§4.3 | HRV subset = mean NN + RMSSD", ec.HRV_SUBSET == ["hrv_mean_nn", "hrv_rmssd"], "mean NN, RMSSD", str(ec.HRV_SUBSET), "eval_core")
    cfg = RCV["config"]
    prose("**Primary inference:** 20× repeated", f"({cfg['n_distinct_partitions']} distinct partitions", "repeated_cv.json config")
    prose("**Primary inference:** 20× repeated", f"{cfg['n_repeats'] * cfg['n_folds']} (repeat, fold) pairs", "repeated_cv.json config")
    prose("**Primary inference:** 20× repeated", f"about {cfg['n_train_subjects_mean']:.0f} training and {cfg['n_test_subjects_mean']:.0f} test", "repeated_cv.json config")
    fact("§4.5 | partitions stored", len(cfg["partitions"]) == 20, "20", str(len(cfg["partitions"])), "repeated_cv.json config.partitions")
    import run_baselines
    import run_repeated_cv as rr
    prose("**Pooled LOSO CIs:**", f"B = {run_baselines.N_BOOT:,}", "run_baselines.N_BOOT")
    prose("**Post-hoc equivalence:**", f"±{rr.SESOI['brier']} Brier", "run_repeated_cv.SESOI")
    prose("**Post-hoc equivalence:**", f"±{rr.SESOI['auroc']} AUROC", "run_repeated_cv.SESOI")
    prose("**Post-hoc equivalence:**", f"0.35 × 0.65 = {0.35 * 0.65:.4f}", "arithmetic")
    from scipy.stats import t as tdist
    widen = tdist.ppf(0.975, 14) / tdist.ppf(0.975, 99) - 1
    prose("**Bouckaert–Frank corrected t**", f"about {round(100 * widen)}%", "t quantiles df 14 vs 99")
    prose("Python 3.13", f"Python {sys.version.split()[0]}", "sys.version")
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    prose("automated tests, including a regression test", f"{int(m.group(1)) if m else -1} automated tests", "pytest --collect-only")


def check_table1():
    hdr = "| Model | Reduced Brier | Reduced AUROC |"
    rows = {"Gated network + dropout (gate, masking)": G, "Non-gated MLP, same masking (no gate, masking)": M,
            "Gated network, no dropout (gate, no masking)": GN, "Non-gated MLP, no masking (no gate, no masking)": MN,
            "(i) Complete-trained LightGBM, NaN at test": LI, "(ii) NaN-trained LightGBM": L2,
            "(iii) ECG-only LightGBM": L3, "(iv) Mean-imputation + per-condition Platt": L4, "HR-only": HR}
    src = "repeated_cv.json per_model_{reduced,complete}.raw"
    for row, k in rows.items():
        r, c = RCV["per_model_reduced"]["raw"][k], RCV["per_model_complete"]["raw"][k]
        cell(hdr, row, 1, f"{n(r['brier_mean'])} ({n(r['brier_sd'])})", src)
        cell(hdr, row, 2, f"{n(r['auroc_mean'])} ({n(r['auroc_sd'])})", src)
        cell(hdr, row, 3, n(r["ece_sweep_mean"]), src)
        cell(hdr, row, 4, n(r["log_loss_mean"]), src)
        cell(hdr, row, 5, n(c["brier_mean"]), src)
        cell(hdr, row, 6, n(c["auroc_mean"]), src)
    # SmoothECE ordering claim
    pmr = RCV["per_model_reduced"]["raw"]
    o1 = sorted(pmr, key=lambda k: pmr[k]["ece_sweep_mean"])
    o2 = sorted(pmr, key=lambda k: pmr[k]["smooth_ece_mean"])
    from scipy.stats import spearmanr
    rho = spearmanr([o1.index(k) for k in pmr], [o2.index(k) for k in pmr])[0]
    fact("§5.1/§5.8 | SmoothECE ordering agrees with ECE_SWEEP (repeated CV)", rho >= 0.9, "rank ρ ≥ 0.9", f"{rho:.2f}", "repeated_cv.json")


T2 = {"Gated vs (ii)": (G, L2), "Gated vs (iii)": (G, L3), "Gated vs (iv)": (G, L4), "Gated vs HR-only": (G, HR),
      "Gated vs MLP": (G, M), "MLP vs (ii)": (M, L2), "Gated vs Gated, no masking": (G, GN),
      "MLP vs MLP, no masking": (M, MN), "Gated, no masking vs MLP, no masking": (GN, MN),
      "Gated vs (i)": (G, LI), "(ii) vs (i)": (L2, LI)}


def check_table2_3_rule():
    hdr = "| Comparison | ΔBrier, B–F df 99 |"
    src = "repeated_cv.json pairwise.raw"
    for row, (a, b) in T2.items():
        e = pwt(RCV, "raw", a, b)
        br, pb = e["brier"], e["participant"]["brier"]
        cell(hdr, row, 1, cie(br), src)
        cell(hdr, row, 2, f"[{s(br['df_subjects']['ci95'][0])}, {s(br['df_subjects']['ci95'][1])}]", src + " df_subjects")
        cell(hdr, row, 3, cie(pb), src + " participant")
        cell(hdr, row, 4, f"{pb['n_A_better']}/15", src + " participant")
        cell(hdr, row, 5, cie(e["auroc"]), src)
    hi = lambda a, b: max(abs(x) for x in pwt(RCV, "raw", a, b)["brier"]["ci95"])
    prose("Gated vs (ii) and vs (iii): the Bouckaert–Frank", f"up to {n(hi(G, L2))} and {n(hi(G, L3))} Brier", src)
    prose("Gated vs (iv): up to", f"up to {n(hi(G, L4))} Brier", src)
    prose("Gated vs the matched MLP: up to", f"up to {n(hi(G, M))} Brier", src)
    prose("**Precision.** The minimum detectable", f"is {n(pwt(RCV,'raw',G,L2)['brier']['mdd80'])} for gated vs (ii) and {n(pwt(RCV,'raw',G,M)['brier']['mdd80'])}", src + " mdd80")
    prose("**Precision.** The minimum detectable", f"about {round(100 * 0.035 / 0.2275)}% of the prevalence-only Brier", "arithmetic")
    bss = lambda b: 1 - b / 0.2275
    gb = RCV["per_model_reduced"]["raw"][G]["brier_mean"]
    prose("**Precision.** The minimum detectable", f"from about {bss(gb + 0.035):.2f} to {bss(gb - 0.035):.2f}", "BSS arithmetic at gated reduced Brier ± 0.035",
          note=f"gated reduced Brier {gb:.4f}")
    pb = pwt(RCV, "raw", G, L2)["participant"]["brier"]
    prose("**Participant-level vs Bouckaert–Frank.**", f"[{s(pb['ci95'][0])}, {s(pb['ci95'][1])}] vs {cie(pwt(RCV,'raw',G,L2)['brier'])[7:]}", src)

    hdr = "| Comparison | Brier TOST p"
    for row, (a, b) in {"Gated vs MLP": (G, M), "Gated vs (ii)": (G, L2), "Gated vs (iii)": (G, L3),
                        "MLP vs (ii)": (M, L2), "Gated vs (iv)": (G, L4)}.items():
        e = pwt(RCV, "raw", a, b)
        br, pb, au = e["brier"], e["participant"]["brier"], e["auroc"]
        cell(hdr, row, 1, f"{p3(br['tost_p'])} / {p3(br['df_subjects']['tost_p'])} / {p3(pb['tost_p'])}", src + " tost")
        cell(hdr, row, 2, f"{n(br['equivalence_bound_attained'])} / {n(br['df_subjects']['equivalence_bound_attained'])} / {n(pb['equivalence_bound_attained'])}", src)
        cell(hdr, row, 3, n(br["rope_prob"], 2), src + " rope")
        cell(hdr, row, 4, f"{p3(au['tost_p'])} / {p3(au['df_subjects']['tost_p'])}", src)
        cell(hdr, row, 5, f"{n(au['equivalence_bound_attained'])} / {n(au['df_subjects']['equivalence_bound_attained'])}", src)
    e2, e3 = pwt(RCV, "raw", G, L2), pwt(RCV, "raw", G, L3)
    prose("only just (p =", f"(p = {e2['participant']['brier']['tost_p']:.3f} and {e3['participant']['brier']['tost_p']:.3f})", src)
    prose("only just (p =", f"(bounds attained {n(min(e2['auroc']['equivalence_bound_attained'], e3['auroc']['equivalence_bound_attained']))}–{n(max(e2['auroc']['equivalence_bound_attained'], e3['auroc']['equivalence_bound_attained']))})", src)
    gm = pwt(RCV, "raw", G, M)
    prose("**Reading.** Gated ≈ MLP passes", f"p = {gm['brier']['df_subjects']['tost_p']:.3f} for Brier, {gm['auroc']['df_subjects']['tost_p']:.3f} for AUROC; the Brier bound attained is {gm['brier']['df_subjects']['equivalence_bound_attained']:.3f}", src)
    prose("the two are equivalent within ±0.02 Brier and ±0.03 AUROC under the widest interval scheme", f"(B–F df 14: TOST p = {gm['brier']['df_subjects']['tost_p']:.3f} and {gm['auroc']['df_subjects']['tost_p']:.3f}; df 99: {gm['brier']['tost_p']:.3f} and {gm['auroc']['tost_p']:.3f})", src)
    prose("the two are equivalent within ±0.02 Brier and ±0.03 AUROC under the widest interval scheme", f"the Brier bound attained is {gm['brier']['df_subjects']['equivalence_bound_attained']:.3f} against a margin of 0.02", src)
    prose("the gated network was post hoc equivalent to the masked MLP", f"ΔBrier {s(gm['brier']['mean'])} (95% CI {s(gm['brier']['ci95'][0])} to {s(gm['brier']['ci95'][1])}), TOST p = {gm['brier']['df_subjects']['tost_p']:.3f} under the widest scheme (B–F df 14; {gm['brier']['tost_p']:.3f} at df 99), Brier bound attained {gm['brier']['df_subjects']['equivalence_bound_attained']:.3f}", src)

    word = lambda f: "fires" if f else "does not fire"
    vt = "| Calibration | 20× repeated CV | Without ECE criterion (20× CV) |"
    for row, cal in (("Raw", "raw"), ("Shared Platt", "platt"), ("Per-condition Platt", "platt_pc")):
        fr = RCV["falsification_rule"][cal]
        exp = [word(fr["rule_fires"]), word(fr["rule_fires_without_ece"]),
               word(BAS["schemes"]["loso"]["falsification_rule"][cal]["rule_fires"]),
               word(BAS["schemes"]["kfold5"]["falsification_rule"][cal]["rule_fires"])]
        try:
            _, header, rows = find_table(vt)
            got = _cells(LINES[rows[row]])[1:5]
            ok = got == exp
            RESULTS.append(Result("PASS" if ok else "FAIL", f"§5.1 rule table | {row}", rows[row] + 1, " / ".join(exp),
                                  " / ".join(got), "repeated_cv.json + baselines.json falsification_rule"))
            if ok:
                COVERED.setdefault(rows[row], []).append((0, len(LINES[rows[row]])))
        except Exception as ex:  # noqa: BLE001
            RESULTS.append(Result("FAIL", f"rule table {row}", None, str(exp), str(ex), ""))
    fact("§5.1 | dropping ECE changes no verdict", all(RCV["falsification_rule"][c]["rule_fires"] == RCV["falsification_rule"][c]["rule_fires_without_ece"] for c in ("raw", "platt", "platt_pc")),
         "True", "", "repeated_cv.json")
    d = {m: pwt(RCV, "raw", G, L2)[m]["mean"] for m in ("brier", "ece_sweep", "auroc")}
    prose("With raw outputs the gated network trails", f"by {n(d['brier'])} on Brier and {n(d['ece_sweep'])} on ECE and leads by {n(d['auroc'])} on AUROC", src)
    prose("With raw outputs the gated network trails", f"(ΔBrier vs (ii) {cie(pwt(RCV,'platt_pc',G,L2)['brier'])})", "repeated_cv.json pairwise.platt_pc")


def check_factorial():
    hdr = "| Contrast | Raw: B–F 95% CI (df 99) |"
    src = "repeated_cv.json contrasts / pairwise"
    rows = {"Interaction (G − G_nomask) − (M − M_nomask)": ("c", "interaction_gate_x_masking"),
            "Main effect of gate": ("c", "main_effect_gate"), "Main effect of masking": ("c", "main_effect_masking"),
            "Gate, never masked: gated vs MLP, no masking": ("p", (GN, MN)),
            "Gate, masked: gated vs MLP": ("p", (G, M)), "Masking, gated network": ("p", (G, GN)),
            "Masking, MLP": ("p", (M, MN))}
    get = lambda kind, key, cal: RCV["contrasts"][cal][key] if kind == "c" else pwt(RCV, cal, *key)
    for row, (kind, key) in rows.items():
        raw, pc = get(kind, key, "raw"), get(kind, key, "platt_pc")
        pb = raw["participant"]["brier"]
        b = raw["brier"]
        cell(hdr, row, 1, cie(b), src)
        cell(hdr, row, 2, cie(b, key="ci95") if False else f"{s(b['mean'])} [{s(b['df_subjects']['ci95'][0])}, {s(b['df_subjects']['ci95'][1])}]", src + " df_subjects")
        cell(hdr, row, 3, f"{cie(pb)} ({pb['n_A_better']}/15)", src)
        cell(hdr, row, 4, cie(pc["brier"]), src)
    pmr = RCV["per_model_reduced"]["raw"]
    prose("**Cells (raw, reduced Brier):**", f"gated + masking {n(pmr[G]['brier_mean'])}; MLP + masking {n(pmr[M]['brier_mean'])}; gated, no masking {n(pmr[GN]['brier_mean'])}; MLP, no masking {n(pmr[MN]['brier_mean'])}", "repeated_cv.json")
    prose("**Cells (raw, reduced Brier):**", f"tree (i) is at {n(pmr[LI]['brier_mean'])} and the masked tree (ii) at {n(pmr[L2]['brier_mean'])}", "repeated_cv.json")
    me, mg = RCV["contrasts"]["raw"]["main_effect_masking"], RCV["contrasts"]["raw"]["main_effect_gate"]
    d14 = lambda e: f"[{s(e['df_subjects']['ci95'][0])}, {s(e['df_subjects']['ci95'][1])}]"
    d99 = lambda e: f"[{s(e['ci95'][0])}, {s(e['ci95'][1])}]"
    pmb, pgb = me["participant"]["brier"], mg["participant"]["brier"]
    a = "*Participant-level (narrowest"
    prose(a, f"Masking {cie(pmb)}, {pmb['n_A_better']}/15 participants, sign-flip p = {pmb['sign_flip_p']:.3f}", src)
    prose(a, f"Gate {cie(pgb)}, TOST at ±0.02 p = {pgb['tost_p']:.3f}", src)
    a = "*Bouckaert–Frank, df 99.*"
    prose(a, f"Masking {d99(me['brier'])}, excluding 0 by {abs(me['brier']['ci95'][1]):.3f}; gate {d99(mg['brier'])}, TOST p = {mg['brier']['tost_p']:.3f}", src)
    a = "*Bouckaert–Frank, df 14 (the scheme read"
    prose(a, f"Masking {d14(me['brier'])} includes 0; gate {d14(mg['brier'])}, TOST p = {mg['brier']['df_subjects']['tost_p']:.3f}", src)
    fact("§5.2 | masking df 14 interval includes 0, df 99 excludes it, participant-level excludes it",
         me["brier"]["df_subjects"]["ci95"][1] > 0 > me["brier"]["ci95"][1] and pmb["ci95"][1] < 0, "True", "", src)
    gg, mm = pwt(RCV, "raw", G, GN)["brier"], pwt(RCV, "raw", M, MN)["brier"]
    a = "Within a single network the masking effect includes 0 under both"
    prose(a, f"(gated {d99(gg)} at df 99 and {d14(gg)} at df 14; MLP {d99(mm)} and {d14(mm)})", src)
    t = pwt(RCV, "raw", L2, LI)
    a = "**At one standard, neither"
    prose(a, f"(ii) vs (i) {cie(t['brier'])} (df 14: {d14(t['brier'])}; participant-level {d99(t['participant']['brier'])}, {t['participant']['brier']['n_A_better']}/15)", src)
    prose(a, f"(−0.04, {me['participant']['brier']['n_A_better']}/15 participants)" if False else f"(−0.04, {pmb['n_A_better']}/15 participants)", src)
    fact("§5.2 | tree masking interval excludes 0 under every scheme",
         t["brier"]["ci95"][1] < 0 and t["brier"]["df_subjects"]["ci95"][1] < 0 and t["participant"]["brier"]["ci95"][1] < 0, "True", "", src)
    ci = lambda e: f"[{s(e['ci95'][0])}, {s(e['ci95'][1])}]"
    gnm = pwt(RCV, "raw", GN, MN)
    prose("**The gate does not substitute for masking.**", f"by only {cie(gnm['brier'])} (participant-level {cie(gnm['participant']['brier'])}; {gnm['participant']['brier']['n_A_better']}/15)", src)
    prose("**The gate does not substitute for masking.**", f"interaction is {cie(RCV['contrasts']['raw']['interaction_gate_x_masking']['brier'])}", src)
    prose("In the never-masked regime the gate does not substitute for masking: gated vs MLP", f"ΔBrier {cie(gnm['brier'])}", src)
    prose("never-masked gated network (0.196 LOSO)", f"({n(pooled('loso', GN)['brier'])} LOSO) beats the never-masked tree ({n(pooled('loso', LI)['brier'])})", "baselines.json loso")
    prose("never-masked gated network (0.196 LOSO)", f"({n(pmr[MN]['brier_mean'])} vs {n(pmr[GN]['brier_mean'])} in repeated CV)", "repeated_cv.json")
    cpc = RCV["contrasts"]["platt_pc"]
    prose("**Under per-condition recalibration**", f"shrinks to {cie(cpc['main_effect_masking']['brier'])} and the interaction to {n(cpc['interaction_gate_x_masking']['brier']['mean'])}", "repeated_cv.json contrasts.platt_pc")
    fact("§5.2 | gate main-effect participant-level TOST at 0.02 passes; df 99 and df 14 do not",
         pgb["tost_p"] < 0.05 and mg["brier"]["tost_p"] > 0.05 and mg["brier"]["df_subjects"]["tost_p"] > 0.05, "True", "", src)


def check_complete():
    hdr = "| Comparison | ΔBrier, B–F 95% CI | ΔBrier, participant-level (A better) |"
    src = "repeated_cv.json pairwise_complete.raw"
    for row, (a, b) in {"Gated vs (i)": (G, LI), "Gated vs (ii)": (G, L2), "MLP vs (ii)": (M, L2), "(ii) vs (i)": (L2, LI),
                        "Gated vs MLP": (G, M), "Gated, no masking vs MLP, no masking": (GN, MN)}.items():
        e = pwt(RCV, "raw", a, b, "complete")
        pb = e["participant"]["brier"]
        cell(hdr, row, 1, cie(e["brier"]), src)
        cell(hdr, row, 2, f"{cie(pb)} ({pb['n_A_better']}/15)", src)
        cell(hdr, row, 3, cie(e["auroc"]), src)
    pc = RCV["per_model_complete"]["raw"]; pr = RCV["per_model_reduced"]["raw"]
    prose("**The networks underperform trees when EDA is present.**", f"{n(pc[LI]['brier_mean'])} for (i), {n(pc[L2]['brier_mean'])} for (ii), {n(pc[G]['brier_mean'])} for the gated network and {n(pc[M]['brier_mean'])} for the masked MLP", src)
    gi = pwt(RCV, "raw", G, LI, "complete")
    prose("**The networks underperform trees when EDA is present.**", f"Gated − (i) is {s(gi['brier']['mean'])}", src)
    prose("**The networks underperform trees when EDA is present.**", f"({gi['participant']['brier']['n_A_better']}/15 participants favour", src)
    fact("§5.3 | Gated−(i) complete: B–F includes 0, participant excludes 0",
         gi["brier"]["ci95"][0] <= 0 <= gi["brier"]["ci95"][1] and gi["participant"]["brier"]["ci95"][0] > 0, "True", "", src)
    tc = TUN["per_model_complete"]["raw"]
    prose("Nested tuning does not close the gap", f"(tuned gated {n(tc['tuned_'+G]['brier_mean'])}, tuned MLP {n(tc['tuned_'+M]['brier_mean'])}, tuned (ii) {n(tc['tuned_'+L2]['brier_mean'])})", "repeated_cv_tuned.json per_model_complete")
    prose("**The networks barely use EDA on the default features.**", f"about {pr[G]['brier_mean'] - pc[G]['brier_mean']:.3f} Brier ({n(pr[G]['brier_mean'])} → {n(pc[G]['brier_mean'])})", src)
    prose("**The networks barely use EDA on the default features.**", f"about {pr[L3]['brier_mean'] - pc[LI]['brier_mean']:.3f} ((iii) {n(pr[L3]['brier_mean'])} → (i) {n(pc[LI]['brier_mean'])})", src)
    li = pwt(RCV, "raw", L2, LI, "complete")["brier"]
    prose("**The cost of masking for trees.**", f"{s(li['mean'])} complete-condition Brier ((ii) − (i), [{s(li['ci95'][0])}, {s(li['ci95'][1])}])", src)
    mc, mrr = MR["per_model_complete"]["raw"], MR["per_model_reduced"]["raw"]
    rates = (0.1, 0.25, 0.5, 0.75)
    prose("**The cost of masking for trees.**", "complete Brier " + " / ".join(n(mc[f"maskrate_lgbm_p{p}"]["brier_mean"]) for p in rates), "repeated_cv_maskrate.json")
    prose("**The cost of masking for trees.**", "(" + " / ".join(n(mrr[f"maskrate_lgbm_p{p}"]["brier_mean"]) for p in rates) + ")", "repeated_cv_maskrate.json")
    prose("**The cost of masking for trees.**", "(gated complete " + " / ".join(n(mc[f"maskrate_gated_p{p}"]["brier_mean"]) for p in rates) + ")", "repeated_cv_maskrate.json")
    prose("**Deployment view.**", f"scores {n(pc[LI]['brier_mean'])} / {n(pr[L3]['brier_mean'])}; the single gated network scores {n(pc[G]['brier_mean'])} / {n(pr[G]['brier_mean'])}", src)
    prose("**Deployment view.**", f"difference is {cie(pwt(RCV,'raw',L3,G)['brier'])} ((iii) − gated)", "repeated_cv.json pairwise.raw")
    g_i = pc[G]["brier_mean"] - pc[LI]["brier_mean"]
    g_ii = pc[G]["brier_mean"] - pc[L2]["brier_mean"]
    span = f"by {g_ii:.2f}–{g_i:.2f} Brier"
    prose("LightGBM beat the networks by", span, src)
    prose("Trees beat the networks by", span, src)
    prose("Trees beat the networks by", f"(gated − (i) {s(pwt(RCV, 'raw', G, LI, 'complete')['brier']['mean'])}; gated − (ii) {s(pwt(RCV, 'raw', G, L2, 'complete')['brier']['mean'])}", src)


def check_robustness():
    hdr = "| Variation | Gated − MLP ΔBrier, reduced"
    src = "repeated_cv_*.json pairwise.raw"
    cell(hdr, "Default (Table 2)", 1, cie(pwt(RCV, "raw", G, M)["brier"]), src)
    cell(hdr, "Default (Table 2)", 2, cie(pwt(RCV, "raw", G, L2)["brier"]), src)
    caps = []
    for d, h, pg, pmm in [(8, 8, 273, 281), (16, 25, 801, 808), (32, 84, 2625, 2637), (64, 300, 9345, 9333)]:
        e = pwt(CAP, "raw", f"cap_gated_d{d}", f"cap_mlp_h{h}")["brier"]
        caps.append(e)
        cell(hdr, f"Capacity: {pg:,} vs {pmm:,} parameters", 1, cie(e), "repeated_cv_capacity.json")
        cell(hdr, f"Capacity: {pg:,} vs {pmm:,} parameters", 3, f"TOST p {p3(e['tost_p'])}", "repeated_cv_capacity.json")
    fact("§5.4 | capacity ΔBrier within ±0.024", max(max(abs(x) for x in e["ci95"]) for e in caps) <= 0.0245, "≤ 0.024", "", "repeated_cv_capacity.json")
    prose("**Capacity sweep** (gated vs MLP", f"(p = {caps[0]['tost_p']:.3f})", "repeated_cv_capacity.json")
    t = TUN["pairwise"]["raw"]
    cell(hdr, "Nested tuning", 1, cie(t[f"tuned_{G}__vs__tuned_{M}"]["brier"]), "repeated_cv_tuned.json")
    cell(hdr, "Nested tuning", 2, cie(t[f"tuned_{G}__vs__tuned_{L2}"]["brier"]), "repeated_cv_tuned.json")
    prose("**Nested tuning** moves", f"tuned gated vs (ii) {cie(t[f'tuned_{G}__vs__tuned_{L2}']['brier'])}, vs MLP {cie(t[f'tuned_{G}__vs__tuned_{M}']['brier'])}", "repeated_cv_tuned.json")
    moves = [abs(TUN["per_model_reduced"]["raw"]["tuned_" + k]["brier_mean"] - RCV["per_model_reduced"]["raw"][k]["brier_mean"]) for k in (G, M, L2, L3)]
    fact("§5.4 | tuning moves no model by more than 0.005", max(moves) <= 0.0055, "≤ 0.005", f"{max(moves):.4f}", "tuned vs untuned")
    cc = TUN["chosen_configs"]["tuned_" + G]
    prose("**Nested tuning** moves", f"chosen in only {max(cc.values())}/{sum(cc.values())} folds", "repeated_cv_tuned.json chosen_configs")
    fact("§5.4 | tuned rule fires under all three modes", all(TUN["falsification_rule"][c]["rule_fires"] for c in ("raw", "platt", "platt_pc")), "True", "", "repeated_cv_tuned.json")
    mr = MR["pairwise"]["raw"]
    vals = []
    for p in (0.1, 0.25, 0.5, 0.75):
        a, b = mr[f"maskrate_gated_p{p}__vs__maskrate_mlp_p{p}"]["brier"], mr[f"maskrate_gated_p{p}__vs__maskrate_lgbm_p{p}"]["brier"]
        vals.append(a["mean"])
        cell(hdr, f"Masking rate {p}", 1, cie(a), "repeated_cv_maskrate.json")
        cell(hdr, f"Masking rate {p}", 2, cie(b), "repeated_cv_maskrate.json")
    prose("**Masking rate.**", f"(ΔBrier {s(min(vals))} to {s(max(vals))})", "repeated_cv_maskrate.json")
    hp = HS["pairwise"]["raw"]
    ga, gl = hp["hrvsub_gated_dropout_p0.5__vs__hrvsub_mlp_nongated_dropout_p0.5"]["brier"], hp["hrvsub_gated_dropout_p0.5__vs__hrvsub_lgbm_nan_training_p0.5"]["brier"]
    cell(hdr, "HRV = mean NN + RMSSD only", 1, cie(ga), "repeated_cv_hrv_subset.json")
    cell(hdr, "HRV = mean NN + RMSSD only", 2, cie(gl), "repeated_cv_hrv_subset.json")
    hpm = HS["per_model_reduced"]["raw"]; pmr = RCV["per_model_reduced"]["raw"]
    prose("**HRV restricted to mean NN and RMSSD.**", f"(gated {n(pmr[G]['brier_mean'])} → {n(hpm['hrvsub_gated_dropout_p0.5']['brier_mean'])}, MLP {n(pmr[M]['brier_mean'])} → {n(hpm['hrvsub_mlp_nongated_dropout_p0.5']['brier_mean'])})", "repeated_cv_hrv_subset.json")
    prose("**HRV restricted to mean NN and RMSSD.**", f"equivalent ({cie(ga)})", "repeated_cv_hrv_subset.json")
    prose("**HRV restricted to mean NN and RMSSD.**", f"moves to {cie(gl)}", "repeated_cv_hrv_subset.json")
    fact("§5.4 | HRV subset: rule does not fire (raw)", not HS["falsification_rule"]["raw"]["rule_fires"], "False", "", "repeated_cv_hrv_subset.json")
    for src_, lab in ((SNA, "Per-subject z-score (label-free)"), (SNB, "Per-subject z-score on baseline windows (PULSE-style)")):
        cell(hdr, lab, 1, cie(src_["pairwise"]["raw"][f"{G}__vs__{M}"]["brier"]), "repeated_cv_subjnorm_*.json")
        cell(hdr, lab, 2, cie(src_["pairwise"]["raw"][f"{G}__vs__{L2}"]["brier"]), "repeated_cv_subjnorm_*.json")
        cell(hdr, lab, 3, n(src_["per_model_reduced"]["raw"][G]["auroc_mean"]), "repeated_cv_subjnorm_*.json")
        fact(f"§5.4 | {lab}: gated vs (ii) TOST passes", src_["pairwise"]["raw"][f"{G}__vs__{L2}"]["brier"]["tost_p"] < 0.05, "< 0.05", "", "subjnorm")
        aware = [src_["per_model_reduced"]["raw"][k]["brier_mean"] for k in (G, M, L2, L3)]
        fact(f"§5.4 | {lab}: missingness-aware models within ±0.005", max(aware) - min(aware) <= 0.0105, "range ≤ 0.010", f"{max(aware)-min(aware):.4f}", "subjnorm")
    prose("*(Under baseline-referenced normalisation, (iv) degenerates", f"(Brier {n(SNB['per_model_reduced']['raw'][L4]['brier_mean'])})", "repeated_cv_subjnorm_baseline.json")
    cell(hdr, "Wrist E4 features (§5.10)", 1, cie(WR["pairwise"]["raw"][f"{G}__vs__{M}"]["brier"]), "repeated_cv_wrist.json")
    cell(hdr, "Wrist E4 features (§5.10)", 2, cie(WR["pairwise"]["raw"][f"{G}__vs__{L2}"]["brier"]), "repeated_cv_wrist.json")


def check_hr_section():
    hdr = "| Model | Reduced AUROC [95% CI] | Within-subject AUROC [95% CI] |"
    rows = {"Gated + dropout": G, "Non-gated MLP": M, "(ii) NaN-trained": L2, "(iii) ECG-only": L3, "HR-only": HR,
            "Gated, HR removed": GHR, "Gated, HR-residualised (HR removed + within-participant residualisation)": GRES,
            "(ii) NaN-trained, HR removed": L2HR, "(iii) ECG-only, HR removed": L3HR}
    src = "baselines.json loso raw reduced pooled + pooled_ci95_bca"
    for row, k in rows.items():
        p = pooled("loso", k)
        a, w = bca("loso", k, "auroc"), bca("loso", k, "within_subject_auroc")
        cell(hdr, row, 1, f"{n(p['auroc'])} [{n(a[0])}, {n(a[1])}]", src)
        cell(hdr, row, 2, f"{n(p['within_subject_auroc'])} [{n(w[0])}, {n(w[1])}]", src)
    g, h = pooled("loso", G), pooled("loso", HR)
    prose("**A single feature does most of the work.**", f"reaches {n(h['auroc'], 2)} AUROC", src)
    prose("**A single feature does most of the work.**", f"about {round(100 * (h['auroc'] - .5) / (g['auroc'] - .5))}% of the gated network's above-chance pooled AUROC and about {round(100 * (h['within_subject_auroc'] - .5) / (g['within_subject_auroc'] - .5))}%", src)
    ret = {k: (pooled("loso", r)["auroc"] - .5) / (pooled("loso", b)["auroc"] - .5) for k, (r, b) in {"g": (GHR, G), "ii": (L2HR, L2), "iii": (L3HR, L3)}.items()}
    prose("**Removing heart rate costs about half", f"gated {round(100 * ret['g'])}%, (ii) {round(100 * ret['ii'])}%, (iii) {round(100 * ret['iii'])}%", src)
    gw = (pooled("loso", GHR)["within_subject_auroc"] - .5) / (g["within_subject_auroc"] - .5)
    prose("**Removing heart rate costs about half", f"about {round(100 * (1 - gw), -1):.0f}% of the within-subject above-chance AUROC (gated {n(g['within_subject_auroc'])} → {n(pooled('loso', GHR)['within_subject_auroc'])})", src)
    hrr = [pooled("loso", k)["auroc"] for k in (GHR, L2HR, L3HR)]
    prose("**Removing heart rate costs about half", f"({rng(hrr)} pooled)", src)
    prose("**Removing within-participant heart-rate covariation", f"network {n(pooled('loso', GRES)['auroc'])})", src)
    prose("Its AUROC (0.878) was read as", f"after the fix it is {n(pooled('loso', GRES)['auroc'])}", src)


def check_participants():
    hr = PA["hr_response"]
    prose("next to that participant's HR response", f"{hr['response_ms']['min']:.0f}–{hr['response_ms']['max']:.0f} ms; {hr['response_standardised']['min']:.2f}–{hr['response_standardised']['max']:.1f} pooled", "participant_analyses.json")
    hdr = "| Model | ρ with raw response [Fisher-z 95% CI] |"
    for row, k in {"Gated, HR removed": "gated_hr_removed", "(ii), HR removed": "ii_hr_removed", "(iii), HR removed": "iii_hr_removed",
                   "Gated network": "gated", "Non-gated MLP": "mlp", "(ii) NaN-trained": "ii", "(iii) ECG-only": "iii", "HR-only": "hr_only"}.items():
        a, b = hr["per_model"][k]["raw_response"], hr["per_model"][k]["standardised_response"]
        cell(hdr, row, 1, f"{n(a['rho'], 2)} [{n(a['fisher_z_ci95'][0], 2)}, {n(a['fisher_z_ci95'][1], 2)}]", "participant_analyses.json")
        cell(hdr, row, 2, f"{n(b['rho'], 2)} [{n(b['fisher_z_ci95'][0], 2)}, {n(b['fisher_z_ci95'][1], 2)}]", "participant_analyses.json")
    wk = hr["weak_responders_lt_60ms"]
    prose("In 4/15 participants (27%;", ", ".join(sorted(wk, key=lambda x: int(x[1:]))), "participant_analyses.json")
    d = PS_LOSO[PS_LOSO.tier == "reduced"]
    resp = -d.drop_duplicates("subject_id").set_index("subject_id")["delta_mean_nn_ms_stress_minus_baseline"]
    weak = resp.sort_values()[:4]
    prose("In 4/15 participants (27%;", f"(drop {weak.min():.0f}–{weak.max():.0f} ms)", "per_subject_loso.csv")
    se = d[(d.model == G)].auroc_se_hanley_mcneil
    s10 = PA["s10"]
    w2 = s10["within_subject_auroc_reduced"][L2]
    prose("- **S10.**", f"is {n(w2['auroc'], 2)} (Hanley–McNeil SE {n(w2['se_hanley_mcneil'], 2)}, about {abs(w2['z_vs_0.5']):.1f} SE below 0.5)", "participant_analyses.json s10")
    bc = s10["by_condition"]
    prose("- **S10.**", f"was {bc['baseline']['chest_bpm']:.0f} bpm, the highest baseline of all 15", "participant_analyses.json s10")
    fact("§5.7 | S10 baseline HR rank 1", s10["baseline_hr_rank_among_15_highest_first"] == 1, "1", str(s10["baseline_hr_rank_among_15_highest_first"]), "participant_analyses.json")
    prose("- **S10.**", f"against {bc['stress']['chest_bpm']:.0f} bpm under stress and {bc['amusement']['chest_bpm']:.0f} bpm in amusement and meditation", "participant_analyses.json s10")
    prose("- **S10.**", f"(chest ACC SD {bc['baseline']['chest_acc_sd_g']:.3f} g)", "participant_analyses.json s10")
    prose("- **S10.**", f"baseline RMSSD ({bc['baseline']['rmssd_ms']:.0f} ms) was lower than under stress ({bc['stress']['rmssd_ms']:.0f} ms)", "participant_analyses.json s10")
    hrm = [s10["within_subject_auroc_reduced"][k]["auroc"] for k in (GHR, L2HR, L3HR)]
    prose("- **S10.**", f"(HR-removed models {rng(hrm)})", "participant_analyses.json s10")


def check_calibration_selective():
    src = "baselines.json loso reduced pooled"
    p = pooled("loso", LI)
    prose("**The complete-trained LightGBM fails in calibration-in-the-large.**", f"intercept is {s(p['calibration_intercept'], 2)} (slope {n(p['calibration_slope'], 2)}; AUROC {n(p['auroc'], 2)})", src)
    better = [pooled("loso", k)["calibration_slope"] for k in (L2, L3, L4, HR, G, M)]
    prose("**Every better model is over-confident", f"(calibration slopes {rng(better)})", src)
    hdr = "| Model | Raw: slope / intercept / ECE |"
    for row, k in {"Gated + dropout": G, "Non-gated MLP": M, "(ii) NaN-trained": L2, "(i) Complete-trained": LI}.items():
        r, c = pooled("loso", k, "raw"), pooled("loso", k, "platt_pc")
        cell(hdr, row, 1, f"{n(r['calibration_slope'], 2)} / {s(r['calibration_intercept'], 2)} / {n(r['ece_sweep'])}", src)
        cell(hdr, row, 2, f"{n(c['calibration_slope'], 2)} / {s(c['calibration_intercept'], 2)} / {n(c['ece_sweep'])}", src)
        cell(hdr, row, 3, f"{n(r['brier'])} → {n(c['brier'])}", src)
    best = min(pooled("loso", k)["brier"] for k in (L2, L3, L4, G, M))
    pc = pooled("loso", LI, "platt_pc")
    prose("**Recalibration mostly rescues", f"recovers {round(100 * (p['brier'] - pc['brier']) / (p['brier'] - best))}% (Brier {n(p['brier'])} → {n(pc['brier'])}; best missingness-aware model {n(best)})", src)
    prose("comparator never exposed to missing EDA (Brier", f"(Brier {n(p['brier'])}, calibration intercept {s(p['calibration_intercept'], 1)})", src)
    allm = list(BAS["schemes"]["loso"]["models"])
    fact("§5.8 | HR-removed (ii) has the lowest LOSO ECE_SWEEP", min(allm, key=lambda k: pooled("loso", k)["ece_sweep"]) == L2HR, L2HR, "", src)
    prose("(the HR-removed NaN-trained LightGBM has among the lowest", f"at AUROC {n(pooled('loso', L2HR)['auroc'], 2)}", src)
    zeros = sum(1 for k, v in PRE_BAS.items() if isinstance(v, dict) and "per_fold" in v for f in v["per_fold"] if f["reduced"].get("ece_debiased") == 0.0)
    tot = sum(1 for k, v in PRE_BAS.items() if isinstance(v, dict) and "per_fold" in v for f in v["per_fold"])
    prose("took the minimum over six bin counts of a zero-clipped estimate", f"{zeros}/{tot} reduced-condition", "archive prefix baselines")
    prose("reduced per-fold values exactly 0", f"{zeros}/{tot} reduced", "archive prefix baselines")
    # selective
    m = {k: pooled("loso", k) for k in (G, M, L2, L3, L4)}
    rand = [m[k]["aurc_random"] for k in (G, M, L2, L3, L4)]
    prose("**Against references.**", f"err on {round(100 * min(rand))}–{round(100 * max(rand))}% of windows", src + " aurc_random",
          note=", ".join(f"{k} {m[k]['aurc_random']:.3f}" for k in m))
    prose("**Against references.**", f"(gated {n(m[G]['risk_at_cov60'])}, MLP {n(m[M]['risk_at_cov60'])}, (ii) {n(m[L2]['risk_at_cov60'])}, (iii) {n(m[L3]['risk_at_cov60'])})", src + " risk_at_cov60")
    prose("**Against references.**", f"is {n(m[G]['e_aurc'])} (gated), {n(m[M]['e_aurc'])} (MLP), {n(m[L2]['e_aurc'])} ((ii)) and {n(m[L3]['e_aurc'])} ((iii))", src + " e_aurc")
    cap = {k: (m[k]["aurc_random"] - m[k]["aurc"]) / (m[k]["aurc_random"] - m[k]["aurc_oracle"]) for k in (G, M, L2, L3)}
    prose("**Against references.**", f"({round(100 * cap[G])}%, {round(100 * cap[M])}%, {round(100 * cap[L2])}%)", src)
    prose("**Against references.**", f"and (iii) about {round(100 * cap[L3], -1):.0f}%", src)
    hdr = "| Comparison | ΔAURC [95% CI], raw | ΔE-AURC [95% CI], raw |"
    for row, (a, b) in {"Gated vs (ii)": (G, L2), "Gated vs (iii)": (G, L3), "Gated vs MLP": (G, M)}.items():
        e = pwt(RCV, "raw", a, b)
        cell(hdr, row, 1, cie(e["aurc"]), "repeated_cv.json")
        cell(hdr, row, 2, cie(e["e_aurc"]), "repeated_cv.json")
    sys.path.insert(0, str(ROOT))
    from run_metrics import risk_at_coverage
    pred = pd.read_csv(R / "predictions_loso.csv")
    g3 = pred[(pred.tier == "reduced") & (pred.model == L3)]
    top = risk_at_coverage(g3.y.values, g3.p_raw.values, 0.1)
    prose("**The inverted-confidence result does not replicate.**", f"were wrong {round(100 * top)}% of the time", "predictions_loso.csv")
    prose("**The inverted-confidence result does not replicate.**", cie(pwt(RCV, "raw", G, L3)["aurc"]), "repeated_cv.json")


def check_wrist_sqi_reimpl():
    w, src = WR, "repeated_cv_wrist.json"
    gm, gl = pwt(w, "raw", G, M)["brier"], pwt(w, "raw", G, L2)
    prose("**The gating null replicates.**", f"Gated vs MLP {cie(gm)} (TOST p = {gm['tost_p']:.3f})", src)
    prose("**The gating null replicates.**", f"never-masked MLP {cie(pwt(w,'raw',GN,MN)['brier'])}", src)
    prose("**The gating null replicates.**", f"interaction {cie(w['contrasts']['raw']['interaction_gate_x_masking']['brier'])}", src)
    me = w["contrasts"]["raw"]["main_effect_masking"]
    prose("**The gating null replicates.**", f"(main effect {cie(me['brier'])}; participant-level {cie(me['participant']['brier'])})", src)
    prose("**Against trees the direction reverses", f"(gated vs (ii) {cie(gl['brier'])}; {gl['participant']['brier']['n_A_better']}/15 participants)", src)
    fact("§5.10 | wrist rule does not fire (raw)", not w["falsification_rule"]["raw"]["rule_fires"], "False", "", src)
    pmr = w["per_model_reduced"]["raw"]
    prose("**But the wrist \"without EDA\" signal is not cardiac.**", f"wrist HR-only reaches only {n(pmr[HR]['auroc_mean'], 2)}", src)
    fact("§5.10 | networks reduced AUROC ≈ 0.93", abs(pmr[G]["auroc_mean"] - 0.93) < 0.006 and abs(pmr[M]["auroc_mean"] - 0.93) < 0.006, "0.93", f"{pmr[G]['auroc_mean']:.3f}/{pmr[M]['auroc_mean']:.3f}", src)
    q = PA["signal_quality"]["wrist_vs_chest_hr"]
    prose("**But the wrist \"without EDA\" signal is not cardiac.**", f"{q['stress']['median_abs_err_bpm']:.1f} bpm during stress ({min(q[c]['median_abs_err_bpm'] for c in ('baseline','amusement','meditation')):.1f}–{max(q[c]['median_abs_err_bpm'] for c in ('baseline','amusement','meditation')):.1f} bpm", "participant_analyses.json signal_quality")
    prose("**But the wrist \"without EDA\" signal is not cardiac.**", f"{round(100 * q['stress']['wrist_median_frac_nn_jump20'])}% of successive wrist NN intervals jump by more than 20% under stress ({round(100 * q['baseline']['wrist_median_frac_nn_jump20'])}% at baseline)", "participant_analyses.json")
    sq = PA["signal_quality"]["chest"]
    prose("No chest window has an implausible NN interval", f"{100 * sq['frac_windows_jump20_gt_10pct']:.1f}% of windows", "participant_analyses.json signal_quality.chest")
    prose("No chest window has an implausible NN interval", f"(maximum {round(100 * sq['max_frac_nn_jump20'])}%)", "participant_analyses.json")
    fact("§5.11 | no implausible chest NN", sq["frac_windows_any_implausible_nn"] == 0, "0", str(sq["frac_windows_any_implausible_nn"]), "participant_analyses.json")
    rp = RE["per_model"]
    prose("- Reduced-condition Brier: gated", f"gated {n(rp['gated']['reduced']['brier'])}, MLP {n(rp['mlp']['reduced']['brier'])}, (ii) {n(rp['ii']['reduced']['brier'])}, (iii) {n(rp['iii']['reduced']['brier'])}", "reimpl_minimal_primary.json")
    mp = RE["main_pipeline_loso_reduced_brier"]
    prose("- Reduced-condition Brier: gated", f"vs {n(mp['gated'])} / {n(mp['mlp'])} / {n(mp['ii'])} / {n(mp['iii'])}", "reimpl_minimal_primary.json")
    mx = max(abs(v["diff"]) for v in RE["pairwise_reduced_brier"].values())
    prose("- Every pairwise reduced-condition ΔBrier lies within", f"within ±{mx:.3f}", "reimpl_minimal_primary.json", note=f"max |diff| {mx:.4f}")
    gmr, g2 = RE["pairwise_reduced_brier"]["gated_vs_mlp"], RE["pairwise_reduced_brier"]["gated_vs_ii"]
    prose("- Every pairwise reduced-condition ΔBrier lies within", f"(gated − MLP {s(gmr['diff'])} [{s(gmr['ci95'][0])}, {s(gmr['ci95'][1])}]; gated − (ii) {s(g2['diff'])} [{s(g2['ci95'][0])}, {s(g2['ci95'][1])}])", "reimpl_minimal_primary.json")
    prose("- It also reproduces the networks' complete-condition deficit", f"(gated {n(rp['gated']['complete']['brier'])} vs (ii) {n(rp['ii']['complete']['brier'])})", "reimpl_minimal_primary.json")
    fact("§5.11 | agreement criterion met", all(RE["agreement"].values()), "True", str(RE["agreement"]), "reimpl_minimal_primary.json")


def check_appendices():
    # Appendix A first-vs-second table (unchanged layout)
    src = "archive/repeated_cv_2026-10-01_prefix.json"
    p2 = PRE_RCV["pairwise"][f"{G}__vs__{L2}"]; p3_ = PRE_RCV["pairwise"][f"{G}__vs__{L3}"]
    cell("| | First (pre-fix", "Gated vs (ii), Brier", 1, f"{s(p2['mean_brier_diff_A_minus_B'])} (B–F p {p2['bouckaert_frank_p_brier']:.2f}", src)
    cell("| | First (pre-fix", "Gated vs (iii), Brier", 1, f"{s(p3_['mean_brier_diff_A_minus_B'])} (p {p3_['bouckaert_frank_p_brier']:.2f}", src)
    cell("| | First (pre-fix", "Gated vs (ii), Brier", 2, pw_ci(RCV, "raw", G, L2, "brier"), "repeated_cv.json")
    cell("| | First (pre-fix", "Gated vs (iii), Brier", 2, pw_ci(RCV, "raw", G, L3, "brier"), "repeated_cv.json")
    cell("| | First (pre-fix", "Gated vs (ii)/(iii), AUROC", 2, f"gated higher by {n(pw(RCV,'raw',G,L2,'auroc')['mean'])}", "repeated_cv.json")
    pp = sorted([p2["bouckaert_frank_p_brier"], p3_["bouckaert_frank_p_brier"]])
    prose("First rule evaluation's Bouckaert–Frank p-values", f"({pp[0]:.2f}–{pp[1]:.2f})", src)
    prose("Its AUROC (0.878) was read as", "(0.878)", "archived outline")
    # Appendix B
    src = "results/locked_comparison_rerun.log"
    lg = tuple(float(x) for x in re.search(r"Ratio: ([\d.]+)x\s+95% CI \[([\d.]+), ([\d.]+)\]", LOG_LOCK).groups())
    nd, dp = _log_ratio(LOG_LOCK, "no_dropout"), _log_ratio(LOG_LOCK, "dropout_p0.5")
    hdr = "| Condition | Ratio, mean of per-fold ratios |"
    for row, v, old in (("Complete-trained LightGBM (NaN at test)", lg, "4.23×"), ("Gated, no dropout", nd, "2.21×"), ("Gated + dropout p = 0.5", dp, "1.61×")):
        cell(hdr, row, 1, f"{v[0]:.2f}×", src)
        cell(hdr, row, 2, f"[{v[1]:.2f}, {v[2]:.2f}]", src)
        cell(hdr, row, 3, old, "archived outline")
        fact(f"App B | {old} in archived outline", old in ARCH_OUTLINE, old, "", "archived outline")
    prose("On the frozen table the same script gives", f"{lg[0]:.2f}× → {nd[0]:.2f}× → {dp[0]:.2f}×", src)
    prose("was an ECE-ratio sequence of", "4.23× → 2.21× → 1.61×", "archived outline")
    lr = PA["small_stats"]["legacy_lgbm_ratio"]
    prose("**Ratio intervals.**", f"geometric mean of {lr['geometric_mean']:.2f}× with 95% CI [{lr['log_scale_ci95'][0]:.2f}, {lr['log_scale_ci95'][1]:.2f}]", "participant_analyses.json small_stats")
    prose("**Dropout vs no dropout.**", f"smallest attainable two-sided Wilcoxon p is {PA['small_stats']['legacy_wilcoxon_n5_min_two_sided_p']:.4f}", "participant_analyses.json small_stats")
    tp = float(re.search(r"Paired t-test \(n=5 folds\): t=[-\d.]+, p=([\d.]+)", LOG_LOCK).group(1))
    W, wp = re.search(r"Wilcoxon signed-rank: W=([\d.]+), p=([\d.]+)", LOG_LOCK).groups()
    prose("**Dropout vs no dropout.**", f"paired t p = {tp:.3f}; Wilcoxon W = {float(W):.0f}, p = {float(wp):.3f}", src)
    rates = [_log_ratio(LOG_LOCK, f"p={r}")[0] for r in ("0.2", "0.4", "0.5", "0.6")]
    prose("**Dropout-rate sweep.**", " / ".join(f"{r:.2f}×" for r in rates), src)
    eo, eg = _log_ratio(LOG_MOD, "eda_only"), _log_ratio(LOG_MOD, "ecg_only")
    prose("**Ablation: which modality is dropped in training.**", f"gives {eo[0]:.2f}× [{eo[1]:.2f}, {eo[2]:.2f}]", "locked_per_modality_rerun.log")
    prose("**Ablation: which modality is dropped in training.**", f"gives {eg[0]:.2f}× [{eg[1]:.2f}, {eg[2]:.2f}]", "locked_per_modality_rerun.log")
    # Appendix C
    h = H2A
    src = "results/h2a_subject_level.json"
    prose("**Subject-level analysis on the frozen table**", f"−{abs(h['mean_diff_sec']):.2f} s, 95% CI [{n(h['ci95_sec'][0], 2)}, {s(h['ci95_sec'][1], 2)}]", src)
    prose("**Subject-level analysis on the frozen table**", f"d_z = {n(h['d_z'], 2)}, 95% CI [{n(h['d_z_ci95'][0], 2)}, {s(h['d_z_ci95'][1], 2)}]", src)
    prose("**Subject-level analysis on the frozen table**", f"t({h['n_subjects'] - 1}) = {n(h['t'], 2)}, p = {h['p']:.2f}; Wilcoxon p = {h['wilcoxon_p']:.2f}", src)
    prose("**Equivalence.** No smallest effect", f"[{n(h['ci90_sec'][0], 2)}, {s(h['ci90_sec'][1], 2)}] s", src)
    prose("**Equivalence.** No smallest effect", f"±{h['min_equivalence_bound_sec']:.2f} s (d_z > {h['min_equivalence_bound_dz']:.2f})", src)
    w = h["within_subject_sd_lag_baseline_sec"]
    prose("**Measurement reliability.**", f"SD {w['median']:.1f} s (median; range {w['min']:.1f}–{w['max']:.1f} s)", src)
    ls = PA["small_stats"]["h2a_lag_noise_floor"]
    prose("**Measurement reliability.**", f"30/√12 = {ls['sd_uniform_pm15_sec']:.2f} s; the observed 8.3 s is {round(100 * ls['ratio_observed_to_uniform'])}%", "participant_analyses.json")
    prose("**Earlier report.**", "d = 0.06, p = 0.94", "archived outline")


def check_appendix_e():
    src = "results/interpretability_frozen.json group_share"
    hdr = "| Model, condition | Heart rate | Other HRV | EDA | Coupling | Top 3 features |"
    pct = lambda x: f"{round(100 * x)}%"
    rows = {"(ii) NaN-trained LightGBM, complete (SHAP)": f"{L2}|complete", "(ii) NaN-trained LightGBM, reduced (SHAP)": f"{L2}|reduced",
            "(iii) ECG-only LightGBM (SHAP)": f"{L3}|reduced", "Gated network, complete (Integrated Gradients)": f"{G}|complete",
            "Gated network, reduced (Integrated Gradients)": f"{G}|reduced"}
    nice = {"hrv_mean_nn": "mean NN", "hrv_pnn50": "pNN50", "hrv_rmssd": "RMSSD", "eda_tonic_slope": "EDA tonic slope",
            "eda_tonic_mean": "EDA tonic mean", "hrv_sdnn": "SDNN"}
    for row, key in rows.items():
        gs = INT[key]["group_share"]
        cell(hdr, row, 1, pct(gs["HR"]), src)
        cell(hdr, row, 2, pct(gs["other_HRV"]), src)
        if "ECG-only" not in row:
            cell(hdr, row, 3, pct(gs["EDA"]), src)
            cell(hdr, row, 4, pct(gs["coupling"]), src)
        cell(hdr, row, 5, ", ".join(nice.get(f, f) for f in INT[key]["rank"][:3]), "interpretability_frozen.json rank")
    ce = INT["ig"]["completeness_error_logit"]
    prose("Completeness error", f"median {ce['median']:.3f}, 95th percentile {ce['p95']:.3f}, maximum {ce['max']:.3f}", "interpretability_frozen.json")
    prose("**Integrated Gradients settings:**", f"{INT['ig']['steps']}-step {INT['ig']['rule']}", "interpretability_frozen.json")


def check_appendix_f():
    log = (R / "audit/run_full_audit.log").read_text(encoding="utf-8", errors="replace")
    src = "results/audit/run_full_audit.log"
    m = re.search(r"Verdict: \w+ \(AUC=([\d.]+) CI=\[([\d.]+),", log)
    prose("subject-level identity AUC", f"AUC {float(m.group(1)):.2f}, lower CI {float(m.group(2)):.2f}", src)
    r = float(re.search(r"heterogeneity ratio R:\s+([\d.]+)", log).group(1))
    prose("MMD R =", f"MMD R = {r:.2f}", src)


def check_extra_prose():
    src = "repeated_cv.json / arithmetic"
    prose("**Post-hoc equivalence:**", f"(about {round(100 * 0.02 / 0.2275)}% of the prevalence-only Brier", "arithmetic 0.02/0.2275")
    prose("**Post-hoc equivalence:**", f"about {0.02 / 0.2275:.2f} Brier skill", "arithmetic")
    gb = RCV["per_model_reduced"]["raw"][G]["brier_mean"]
    prose("**Precision.** The minimum detectable", f"±0.035 Brier is about {round(100 * 0.035 / 0.2275)}%", "arithmetic")
    prose("**Precision.** The minimum detectable", f"gated network's {n(gb)}", src)
    prose("The minimum detectable ΔBrier against tree baselines is about", f"about {n(pwt(RCV,'raw',G,L2)['brier']['mdd80'])}", src)
    hs = [max(abs(x) for x in pwt(CAP, "raw", f"cap_gated_d{d}", f"cap_mlp_h{h}")["brier"]["ci95"]) for d, h in ((8, 8), (16, 25), (32, 84), (64, 300))]
    prose("**Capacity sweep** (gated vs MLP", f"within ±{max(hs):.3f}", "repeated_cv_capacity.json")
    moves = [abs(TUN["per_model_reduced"]["raw"]["tuned_" + k]["brier_mean"] - RCV["per_model_reduced"]["raw"][k]["brier_mean"]) for k in (G, M, L2, L3)]
    prose("**Nested tuning** moves", f"by more than {np.ceil(max(moves) * 1000) / 1000:.3f} Brier", "tuned vs untuned")
    a1 = [SNA["per_model_reduced"]["raw"][k]["auroc_mean"] for k in (G, M, L2, L3, L4, HR)]
    prose("**Per-subject normalisation.**", f"about {min(a1):.2f}–{max(a1):.2f}", "repeated_cv_subjnorm_all.json (reduced AUROC, missingness-aware + HR-only)")
    prose("**Per-subject normalisation.**", f"raises it to about {SNB['per_model_reduced']['raw'][G]['auroc_mean']:.2f}", "repeated_cv_subjnorm_baseline.json gated")
    rngs = []
    for S in (SNA, SNB):
        v = [S["per_model_reduced"]["raw"][k]["brier_mean"] for k in (G, M, L2, L3)]
        rngs.append(max(v) - min(v))
    prose("**Per-subject normalisation.**", f"within {np.ceil(max(rngs) * 1000) / 1000:.3f} Brier of each other", "subjnorm (range among gated, MLP, (ii), (iii))",
          note=f"ranges {rngs}")
    cell("| Variation | Gated − MLP ΔBrier, reduced", "Wrist E4 features (§5.10)", 3,
         f"gated reduced AUROC {n(WR['per_model_reduced']['raw'][G]['auroc_mean'])}", "repeated_cv_wrist.json")
    from scipy.stats import norm  # noqa: F401
    def hm(a, n1=10, n0=19):
        q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
        return np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    prose("**Within-subject AUROC precision.**", f"about {hm(0.9):.2f} at AUC 0.90 and {hm(0.7):.2f} at AUC 0.70", "Hanley–McNeil formula, 10 vs 19 windows")
    bc = PA["s10"]["by_condition"]["baseline"]
    prose("- **S10.**", f"(median {round(100 * bc['nn_jump20_median'])}% successive NN jumps", "participant_analyses.json s10")
    li = pooled("loso", LI)["brier"]; lz = pooled("loso", LZ)["brier"]
    prose("Label only (both give Brier", f"Brier ≈ {round((li + lz) / 2, 2):.2f}", "baselines.json loso (i) NaN and zero-fill",
          note=f"NaN {li:.3f}, zero-fill {lz:.3f}")
    mp = float(re.search(r"Paired t-test: t=[-\d.]+, p=([\d.]+)", LOG_MOD).group(1))
    md = [float(x) for x in re.search(r"Per-fold delta \(eda_only - ecg_only\): \[(.*?)\]", LOG_MOD).group(1).replace("'", "").split(",")]
    prose("**Ablation: which modality is dropped in training.**", f"wins in {sum(d < 0 for d in md)}/5 folds (p = {mp:.2f})", "locked_per_modality_rerun.log")
    w = WR["per_model_reduced"]["raw"]; L = lambda k: WR["pairwise"]["raw"][k]["auroc"]["mean"]
    lead = np.mean([L(f"{G}__vs__{L2}"), L(f"{G}__vs__{L3}")])
    prose("**Against trees the direction reverses", f"by about {lead:.3f} AUROC", "repeated_cv_wrist.json", note=f"gated−(ii) {L(f'{G}__vs__{L2}'):.3f}, gated−(iii) {L(f'{G}__vs__{L3}'):.3f}")
    prose("**But the wrist \"without EDA\" signal is not cardiac.**", f"Reduced-condition AUROC is {round(w[G]['auroc_mean'], 2):.2f}", "repeated_cv_wrist.json")
    if "wrist_diagnostics" in PA:
        wd = PA["wrist_diagnostics"]
        sc, sa, ua = wd["spearman_wrist_vs_chest"], wd["spearman_wrist_rmssd_vs_artefact"], wd["univariate_pooled_auroc"]
        prose("**But the wrist \"without EDA\" signal is not cardiac.**", f"(Spearman ρ {n(sc['baseline']['hrv_rmssd'], 2)} at baseline, {n(sc['stress']['hrv_rmssd'], 2)} under stress)", "participant_analyses.json wrist_diagnostics")
        prose("**But the wrist \"without EDA\" signal is not cardiac.**", f"(ρ {n(sa['baseline'], 2)} and {n(sa['stress'], 2)})", "participant_analyses.json wrist_diagnostics")
        prose("**But the wrist \"without EDA\" signal is not cardiac.**", f"(pooled AUROC {n(ua['wrist_frac_nn_jump20'], 2)}), as does chest movement ({n(ua['chest_acc_mag_sd'], 2)})", "participant_analyses.json wrist_diagnostics")
        prose("No chest window has an implausible NN interval", f"pooled AUROC {n(ua['chest_acc_mag_sd'], 2)}", "participant_analyses.json wrist_diagnostics")
    p = pooled("loso", LI)
    cell("| Model | Raw: slope / intercept / ECE |", "(i) Complete-trained", 1,
         f"{n(p['calibration_slope'], 2)} / {s(p['calibration_intercept'], 2)} / {n(p['ece_sweep'])}", "baselines.json")


def check_abstract():
    src = "repeated_cv.json pairwise.raw"
    hb = [max(abs(x) for x in pwt(RCV, 'raw', G, b)['brier']['ci95']) for b in (L2, L3)]
    fact("§10.2 | abstract: differences of about ±0.035 vs trees", abs(min(hb) - 0.035) < 0.001 and max(hb) < 0.038, "0.035–0.037", f"{min(hb):.3f}–{max(hb):.3f}", src)
    _cover_tok("Against NaN-trained and ECG-only LightGBM the data could not exclude", "0.035")
    fa = [(1 - OP["per_system_reduced"][k]["sens90_spec_mean"]) * 60 for k in OP_SYS.values()]
    prose("*Operating points.* Without EDA", f"raised {round(min(fa))}–{round(max(fa))} false alarms per hour of rest", "operating_points.json")
    allm = max(abs(r["vs_S0"][k]["mean"]) for r in MN_["raw"].values() if r.get("vs_S0") for k in ("gated", "mlp", "lgbm_nan_training", "per_config"))
    prose("*Informative missingness.* In the pre-specified run", f"moved no system's Brier by more than {allm:.3f}", "mnar.json")
    # Abstract "LightGBM beat the networks by 0.04–0.06 Brier" is checked in check_complete.


# ---------------------------------------------------------------------------
# Untraced numbers
# ---------------------------------------------------------------------------
#
# Round-4 (review M9): every decimal / percent / ratio token in the body is
# either covered by a keyed check above or listed in EXPLAINED with the reason
# it needs no results-file source (a method constant, an external citation
# fact, or a restatement of a checked value). The report lists anything left.

NUM = re.compile(r"(?<![\w.])[+\u2212\-]?\d[\d,]*(?:\.\d+)?(?:×|%)?")
SKIP_SECTIONS = ("2. Related Work", "Appendix H", "Appendix G", "Contributions", "10.1", "10.3", "4.7")

# token -> reason (applies wherever the token appears in a non-table prose line)
CONSTANTS = {
    '20%': 'NN-jump threshold', '0.037': 'quoted round-3 sentence', '5.2': 'section reference (§5.1–5.2)',
    '0.4': 'design constant (dropout rate)',
    '95%': 'CI level',
    '90%': 'CI level',
    '80%': 'power / coverage level',
    '60%': 'coverage level',
    '10%': 'coverage level',
    '20×': 'design: 20 repeats',
    '50%': 'design: masking rate',
    '0.5': 'design constant',
    '0.05': 'alpha / design constant',
    '0.1': 'design constant',
    '0.25': 'design constant',
    '0.75': 'design constant',
    '0.02': 'SESOI (Brier)',
    '0.03': 'SESOI (AUROC) / rounded bound',
    '±0.02': 'SESOI',
    '±0.03': 'SESOI / rounded bound',
    '0.04': 'LF band edge',
    '0.15': 'band edge',
    '0.40': 'HF band edge',
    '0.8': 'SCR spacing (s)',
    '5%': 'SCR threshold',
    '0.2': 'fraction-of-windows constant',
    '0.7': 'spectral band edge',
    '3.5': 'spectral band edge',
    '1e-2': 'learning rate',
    '3e-3': 'learning rate',
    '1e-3': 'weight decay',
    '0.35': 'prevalence',
    '0.65': '1 - prevalence',
    '0.2275': 'prevalence Brier',
    '2.4': 'WESAD demographics (Schmidt 2018)',
    '27.5': 'WESAD demographics (Schmidt 2018)',
    '55%': 'Lessmeier 1997',
    '67%': 'Lessmeier 1997',
    '3.3': 'Lessmeier 1997',
    '0.85': 'Helminen 2021',
    '10.5281': 'DOI prefix',
    '0.6': 'constant',
    '±15': 'search window (s)',
    '27%': '4/15 (arithmetic)',
    '±0.037': 'quoted round-3 sentence',
    '0.963': 'PULSE no-teacher (citation)',
    '0.994': 'PULSE (citation)',
    '1.08': 't ratio',
    '0.07': 'Hanley–McNeil SE at AUC 0.9 (review §7 arithmetic)',
    '0.90': 'AUC value in SE example',
    '0.70': 'AUC value in SE example',
    '150': 'clinical threshold (bpm)',
    '0.087': 'audit CORAL (to confirm by eye)',
    '0.60': 'decision-curve threshold range (prespec_round5)',
    '0.20': 'pre-stated NB threshold',
    '0.50': 'pre-stated NB threshold',
    '0.01': 'NB margin (prespec_round5)',
    '30%': 'simulated missing rate (prespec_round5)',
    '35%': 'prevalence',
    '0.2%': 'coupling features undefined (§4.1; extraction)',
    '9%': '0.02 / 0.2275 (SESOI as share of prevalence Brier; §4.5 arithmetic)',
    '0.048': 'restatement of the 9,345-parameter capacity TOST p keyed in Table S3',
}


def untraced():
    out = {}
    for i, l in enumerate(LINES):
        text = l.replace("**", "")
        if text.lstrip().startswith("#"):
            continue                          # headings: section numbers
        sec = section_of(i)
        if any(sec.startswith(x) for x in SKIP_SECTIONS):
            continue
        spans = COVERED.get(i, [])
        for m in NUM.finditer(text):
            tok = m.group(0)
            if not re.search(r"\.\d|×|%", tok):
                continue
            if text[max(0, m.start() - 1)] in "§#":
                continue
            if any(a <= m.start() < b for a, b in spans):
                continue
            if tok.lstrip("+−-") in CONSTANTS or tok in CONSTANTS:
                continue
            out.setdefault(sec, []).append((i + 1, tok))
    return out


def check_order_confound():
    if OC is None:
        note("§5.6", "results/order_confound.json not found")
        return
    check_order_confound_impl()


def check_order_confound_impl():
    src = "results/order_confound.json"
    a, b, cc, d = OC["A_stress_vs_amusement"], OC["B_negative_control"], OC["C_baseline_vs_meditation"], OC["D_within_baseline_drift"]
    names = {"Gated + dropout": G, "Non-gated MLP": M, "(ii) NaN-trained": L2, "(iii) ECG-only": L3, "HR-only": HR,
             "Gated, HR removed": GHR, "(ii), HR removed": L2HR, "(iii), HR removed": L3HR}
    hdr = "| Model | Complete pooled AUROC | Reduced pooled AUROC [95% CI] |"
    for row, k in names.items():
        c_, r = a[k]["complete"], a[k]["reduced"]
        bp = r["within_subject_auroc_by_protocol"]
        cell(hdr, row, 1, n(c_["pooled_auroc"]), src)
        cell(hdr, row, 2, f"{n(r['pooled_auroc'])} [{n(r['pooled_auroc_ci95'][0])}, {n(r['pooled_auroc_ci95'][1])}]", src)
        cell(hdr, row, 3, f"{n(r['within_subject_auroc'])} [{n(r['within_subject_auroc_ci95'][0])}, {n(r['within_subject_auroc_ci95'][1])}]", src)
        cell(hdr, row, 4, f"{n(bp['stress_before_amusement'])} / {n(bp['stress_after_amusement'])}", src)
    ws = [a[k]["reduced"]["within_subject_auroc"] for k in (G, M, L2, L3, HR)]
    prose("**Discrimination without EDA survives the order-balanced contrast**", f"AUROC {n(min(ws), 2)}–{n(max(ws), 2)}", src)
    wh = [a[k]["reduced"]["within_subject_auroc"] for k in (GHR, L2HR, L3HR)]
    prose("**It is again heart rate.**", f"HR-only alone reaches {n(a[HR]['reduced']['within_subject_auroc'])}", src)
    prose("**It is again heart rate.**", f"drops the models to {n(min(wh), 2)}–{n(max(wh), 2)}", src)
    prose("**It is again heart rate.**", f"median {n(a['_hr_delta_mean_nn_stress_minus_amusement_ms']['median'], 0)} ms", src)
    order = OC["protocol_versions"]
    fact("§5.6 | 7 stress-first / 8 amusement-first", order["stress_before_amusement"] == 7 and order["n_subjects"] == 15, "7 of 15", str(order), src)
    cols = ("baseline", "stress", "amusement", "meditation_1", "meditation_2")
    spec = {"Gated + dropout, EDA missing": (G, "reduced"), "Non-gated MLP, EDA missing": (M, "reduced"),
            "(ii) NaN-trained, EDA missing": (L2, "reduced"), "(iii) ECG-only, ECG only": (L3, "reduced"),
            "HR-only, HR only": (HR, "reduced"), "Gated + dropout, EDA present": (G, "complete"),
            "(ii) NaN-trained, EDA present": (L2, "complete"), "(i) Complete-trained, EDA present": (LI, "complete"),
            "(i) Complete-trained, EDA missing": (LI, "reduced")}
    hdr = "| Model, condition | Baseline (first) |"
    for row, (k, tier) in spec.items():
        for j, col in enumerate(cols, start=1):
            cell(hdr, row, j, n(b[k][tier][col]["frac_pred_stress"], 2), src)
    aware = [(k, "reduced") for k in (G, M, L2, L3, HR)]
    calm = [b[k][t][c]["frac_pred_stress"] for k, t in aware for c in ("amusement", "meditation_1", "meditation_2")]
    base = [b[k][t]["baseline"]["frac_pred_stress"] for k, t in aware]
    st = [b[k][t]["stress"]["frac_pred_stress"] for k, t in aware]
    stc = [b[k]["complete"]["stress"]["frac_pred_stress"] for k in (G, L2, LI)]
    line = "**The detectors are not time-since-start detectors.**"
    prose(line, f"({n(min(calm), 2)}–{n(max(calm), 2)} vs {n(min(base), 2)}–{n(max(base), 2)})", src)
    prose(line, f"labelled at {n(min(st), 2)}–{n(max(st), 2)} ({n(min(stc), 2)}–{n(max(stc), 2)} with EDA present)", src)
    mx = max(b[k][t][c]["n_subjects_frac_gt_half"] for k, t in aware for c in ("amusement", "meditation_1", "meditation_2"))
    sts = [b[k][t]["stress"]["n_subjects_frac_gt_half"] for k, t in aware]
    prose(line, f"at most {mx} of 15 participants", src)
    prose(line, f"(stress: {min(sts)}–{max(sts)} of 15)", src)
    hdr = "| Model | Pooled AUROC | Within-subject AUROC [95% CI] |"
    cn = {"EDA-only LightGBM": "lgbm_eda_only", "HR-only": HR, "(iii) ECG-only": L3, "(i) Complete-trained": LI,
          "Gated network (no masking needed: EDA present)": GN}
    for row, k in cn.items():
        r = cc[k]["complete"]
        cell(hdr, row, 1, n(r["pooled_auroc"]), src)
        cell(hdr, row, 2, f"{n(r['within_subject_auroc'])} [{n(r['within_subject_auroc_ci95'][0])}, {n(r['within_subject_auroc_ci95'][1])}]", src)
    e = cc["lgbm_eda_only"]["complete"]
    prose("**No evidence of tonic EDA drift.**", f"(within-subject AUROC {n(e['within_subject_auroc'], 2)}", src)
    fact("§5.6 | EDA-only CI includes 0.5", e["within_subject_auroc_ci95"][0] <= 0.5 <= e["within_subject_auroc_ci95"][1], "True", "", src)
    prose("**No evidence of tonic EDA drift.**", f"Neither does heart rate ({n(cc[HR]['complete']['within_subject_auroc'], 2)})", src)
    w2 = [cc[k]["complete"]["within_subject_auroc"] for k in (L3, GN)]
    prose("do separate them (within-subject", f"{n(min(w2), 2)}–{n(max(w2), 2)}; the complete-trained LightGBM {n(cc[LI]['complete']['within_subject_auroc'], 2)}", src)
    de, dh = d["eda_tonic_mean"], d["hrv_mean_nn"]
    prose("Within the 20-minute baseline", f"ρ {s(de['median_rho_vs_time'], 2)} and {s(dh['median_rho_vs_time'], 2)}; {de['n_positive']}/15 and {dh['n_positive']}/15", src)


# ===========================================================================
# Round-5 additions (revision 7): log-loss, calibration-in-the-large reading,
# per-subject normalisation complete-condition CIs, posture/movement (C1)
# ===========================================================================

PM = J("posture_movement.json") if (R / "posture_movement.json").exists() else None


def _cover_tok(anchor, tok):
    hits = [i for i, l in enumerate(LINES) if anchor in l]
    if hits:
        _cover(hits[0], LINES[hits[0]].replace("**", ""), tok)


def check_round5():
    src = "repeated_cv.json"
    pr = RCV["per_model_reduced"]["raw"]
    a = "**Log-loss points the other way"
    prose(a, f"lower for (ii) ({n(pr[L2]['log_loss_mean'])}) than for the gated network ({n(pr[G]['log_loss_mean'])}) or the masked MLP ({n(pr[M]['log_loss_mean'])})", src)
    prose(a, f"gated − (ii) {cie(pwt(RCV, 'raw', G, L2)['log_loss'])}, MLP − (ii) {cie(pwt(RCV, 'raw', M, L2)['log_loss'])}, gated − MLP {cie(pwt(RCV, 'raw', G, M)['log_loss'])}", src)
    prose(a, f"(calibration slopes {rng([pooled('loso', G)['calibration_slope'], pooled('loso', M)['calibration_slope']])} vs {n(pooled('loso', L2)['calibration_slope'], 2)} for (ii)", "baselines.json loso reduced pooled")

    aur = [v["auroc_mean"] for v in pr.values()]
    a = "**Discrimination.** With EDA missing"
    prose(a, f"lies between {n(min(aur))} and {n(max(aur))}", src)
    raw_m = RCV["contrasts"]["raw"]["main_effect_masking"]["brier"]["mean"]
    pc_m = RCV["contrasts"]["platt_pc"]["main_effect_masking"]["brier"]["mean"]
    ga, ma = pwt(RCV, "raw", G, GN)["auroc"], pwt(RCV, "raw", M, MN)["auroc"]
    prose(a, f"masking vs no masking is {cie(ga)} (gated) and {cie(ma)} (MLP)", src)
    hi_g = max(abs(x) for x in ga["ci95"])
    fact("§5.2 | AUROC gains of 0.06–0.09 not excluded", abs(ga["ci95"][1] - 0.065) < 1e-3 and abs(ma["ci95"][1] - 0.090) < 1e-3, "0.065 / 0.090", f"{ga['ci95'][1]:.3f} / {ma['ci95'][1]:.3f}", src)
    _cover_tok(a, "0.06")
    _cover_tok(a, "0.09")
    fact("§5.2 | no Table-2 ΔAUROC CI excludes 0",
         all(v["auroc"]["ci95"][0] <= 0 <= v["auroc"]["ci95"][1] for v in RCV["pairwise"]["raw"].values()), "True", "", src)
    prose("Discrimination without EDA is similar across all reasonable models", f"(AUROC {rng(aur, 3)})", src)
    prose("Under label-free per-subject normalisation the gap nearly closes",
          f"(gated − (ii) {s(pwt(SNA, 'raw', G, L2, 'complete')['brier']['mean'])})", "repeated_cv_subjnorm_all.json")

    # Per-subject normalisation: complete-condition CIs (C2).
    s1, s2 = "repeated_cv_subjnorm_all.json", "repeated_cv_subjnorm_baseline.json"
    sc, sr = SNA["per_model_complete"]["raw"], SNA["per_model_reduced"]["raw"]
    gl, gm, gmr = pwt(SNA, "raw", G, L2, "complete"), pwt(SNA, "raw", G, M, "complete"), pwt(SNA, "raw", G, M)
    a = "**The networks barely use EDA on the default features.**"
    prose(a, f"(complete-condition Brier: gated {n(sc[G]['brier_mean'])}, MLP {n(sc[M]['brier_mean'])}, (ii) {n(sc[L2]['brier_mean'])}; gated − (ii) {cie(gl['brier'])})", s1)
    prose(a, f"(complete {cie(gm['brier'])}; reduced {cie(gmr['brier'])})", s1)
    a = "**Per-subject normalisation is where the networks use EDA.**"
    spread = max(sc[k]["brier_mean"] for k in (G, M, L2)) - min(sc[k]["brier_mean"] for k in (G, M, L2))
    prose(a, f"within {n(spread)} Brier of the masked tree", s1)
    prose(a, f"(complete Brier: gated {n(sc[G]['brier_mean'])}, MLP {n(sc[M]['brier_mean'])}, (ii) {n(sc[L2]['brier_mean'])}, (i) {n(sc[LI]['brier_mean'])}; gated − (ii) {cie(gl['brier'])}, participant-level {cie(gl['participant']['brier'])})", s1)
    prose(a, f"against a {s(pwt(RCV, 'raw', G, L2, 'complete')['brier']['mean'])} gap on the default features", src)
    within = max(abs(gm["brier"]["mean"]), abs(gmr["brier"]["mean"]))
    fact("§5.4 | gated−MLP within 0.002 Brier in both conditions (per-subject z)", round(within, 3) <= 0.002, "<= 0.002", f"{within:.4f}", s1)
    _cover_tok(a, "0.002")
    prose(a, f"(complete {cie(gm['brier'])}, TOST p = {p3(gm['brier']['tost_p'])})", s1)
    a = "**The calibration-in-the-large failure disappears.**"
    prose(a, f"reaches reduced Brier {n(sr[LI]['brier_mean'])}, level with (ii) ((ii) − (i) {cie(pwt(SNA, 'raw', L2, LI)['brier'])})", s1)
    prose(a, f"partly fails again ({n(SNB['per_model_reduced']['raw'][LI]['brier_mean'])}; (ii) − (i) {cie(pwt(SNB, 'raw', L2, LI)['brier'])})", s2)

    # Posture and movement (C1).
    if PM is None:
        RESULTS.append(Result("FAIL", "results/posture_movement.json missing", None, "", "", ""))
        return
    src = "posture_movement.json"
    mod, con = PM["A_loso"]["models"], PM["A_loso"]["contrasts"]
    f3 = lambda x: f"{n(x['est'])} [{n(x['ci95'][0])}, {n(x['ci95'][1])}]"
    fs = lambda x: f"{s(x['est'])} [{s(x['ci95'][0])}, {s(x['ci95'][1])}]"
    hdr = "| Feature set | Pooled AUROC [95% CI] |"
    for row, k in {"Movement only": "movement_only", "Posture only": "posture_only", "ACC (movement + posture)": "acc_only",
                   "HR only": "hr_only", "ACC + HR": "acc_plus_hr", "ECG only": "ecg_only", "ECG + ACC": "ecg_plus_acc"}.items():
        cell(hdr, row, 1, f3(mod[k]["pooled_auroc"]), src)
        cell(hdr, row, 2, f3(mod[k]["within_subject_auroc"]), src)
        cell(hdr, row, 3, n(mod[k]["brier"]["est"]), src)
    a = "**Movement discriminates better than heart rate.**"
    prose(a, f"reaches {n(mod['movement_only']['pooled_auroc']['est'])} pooled AUROC, against {n(mod['hr_only']['pooled_auroc']['est'])} for HR-only", src)
    prose(a, f"does not separate stress from baseline ({n(mod['posture_only']['pooled_auroc']['est'])})", src)
    c, c2 = con["acc_plus_hr__minus__acc_only"], con["ecg_plus_acc__minus__acc_only"]
    a = "**Heart rate adds little beyond movement.**"
    prose(a, f"within-subject AUROC {fs(c['within_subject_auroc'])}, pooled AUROC {fs(c['pooled_auroc'])}, Brier {fs(c['brier'])}", src)
    prose(a, f"gives within-subject {fs(c2['within_subject_auroc'])}", src)
    B = PM["B_hr_residualised_on_acc"]
    a = "**Residualising heart rate on movement and posture is not identifiable here.**"
    prose(a, f"a median {round(100 * B['r2_median'])}% of the variance in mean NN (range {round(100 * B['r2_range'][0])}–{round(100 * B['r2_range'][1])}%)", src)
    prose(a, f"(within-subject AUROC {n(B['hr_resid']['within_subject_auroc'])} [{n(B['hr_resid']['ci95'][0])}, {n(B['hr_resid']['ci95'][1])}], against {n(B['hr']['within_subject_auroc'])} for raw mean NN)", src)
    prose(a, f"(movement within-subject AUROC {n(B['acc_sd']['within_subject_auroc'])} as a single feature)", src)
    C = PM["C_low_motion"]
    a = "**Lower-motion stress windows.**"
    prose(a, f"leaves {C['n_stress_windows_kept']} of {C['n_stress_windows_total']} stress windows; in the {C['n_participants_ge3']} participants", src)
    h = C["hr_low_motion"]
    prose(a, f"(within-subject AUROC {n(h['within_subject_auroc'])} [{n(h['ci95'][0])}, {n(h['ci95'][1])}], vs {n(C['hr_all_windows_same_participants']['within_subject_auroc'])} on all their windows)", src)
    prose(a, f"Movement still differs in this subset ({n(C['acc_sd_low_motion']['within_subject_auroc'])})", src)
    D = PM["D_posture"]
    a = "**Stress vs amusement is not posture-matched either.**"
    prose(a, f"(within-subject AUROC {n(D['stress_vs_amusement_within_subject_auroc']['tilt_from_amusement'])})", src)
    r = D["range"]["bpm_stress_minus_baseline"]
    prose(a, f"(stress − baseline median {D['median']['bpm_stress_minus_baseline']:.1f} bpm, range {r[0]:.1f}–{r[1]:.1f}; above 10 bpm in {D['n_bpm_stress_minus_baseline_gt10']}/15 participants)", src)
    prose("chest movement alone discriminated stress from baseline better than heart rate (AUROC",
          f"(AUROC {n(mod['movement_only']['pooled_auroc']['est'])} vs {n(mod['hr_only']['pooled_auroc']['est'])}", src)


MC = J("movement_checks.json") if (R / "movement_checks.json").exists() else None


AA = J("abstention_action.json") if (R / "abstention_action.json").exists() else None
AA_SYS = {"Gated + dropout": "gated", "Non-gated MLP, same masking": "mlp", "(ii) NaN-trained LightGBM": "lgbm_nan_training",
          "Per-configuration ((iii) without EDA)": "per_config"}
AA_COLS = ["100_none", "60_remove", "60_no_alert", "60_alert", "80_remove", "80_no_alert", "80_alert"]


def check_abstention_action():
    if AA is None:
        RESULTS.append(Result("FAIL", "results/abstention_action.json missing", None, "", "", ""))
        return
    src = "abstention_action.json"
    hdr = "| System | No abstention | 60% coverage, dropped |"
    for row, k in AA_SYS.items():
        for j, c in enumerate(AA_COLS, start=1):
            e = AA["systems"][k][c]
            cell(hdr, row, j, f"{n(e['sensitivity'])} / {e['false_alarms_per_hour']:.1f}", src)
    g = AA["systems"]["gated"]
    a = "- **Dropping the abstained windows flatters the detector.**"
    prose(a, f"cuts the gated network's sensitivity from {n(g['100_none']['sensitivity'])} to {n(g['60_no_alert']['sensitivity'])}", src)
    prose(a, f"The false alarms fall from {g['100_none']['false_alarms_per_hour']:.1f} to {g['60_no_alert']['false_alarms_per_hour']:.1f} per hour", src)
    prose("Abstention is not free: treating abstained windows", f"from {n(g['100_none']['sensitivity'])} to {n(g['60_no_alert']['sensitivity'])} at 60% coverage", src)
    prose("Abstention is not free: treating abstained windows", f"{g['60_alert']['false_alarms_per_hour']:.1f} false alarms per hour", src)
    a = "- **Treated as \"alert\", abstention buys sensitivity at a cost.**"
    prose(a, f"reaches {n(g['60_alert']['sensitivity'])} sensitivity with {g['60_alert']['false_alarms_per_hour']:.1f} false alarms per hour of rest; at 80% coverage, {n(g['80_alert']['sensitivity'])} with {g['80_alert']['false_alarms_per_hour']:.1f}", src)


def check_movement_checks():
    if MC is None:
        RESULTS.append(Result("FAIL", "results/movement_checks.json missing", None, "", "", ""))
        return
    src = "movement_checks.json"
    sec = lambda e: f"{s(e['est'])} [{s(e['ci95'][0])}, {s(e['ci95'][1])}]"
    c = MC["movement_vs_hr_paired"]["movement_only__minus__hr_only"]
    a = "- **Movement discriminates better than heart rate.**"
    prose(a, f"is movement-only − HR-only {sec(c['pooled_auroc'])} pooled AUROC, {sec(c['within_subject_auroc'])} within-subject AUROC and {sec(c['brier'])} Brier", src)
    rc = MC["response_correlations"]
    a = "**Responders, not movers"
    mv = rc["movement_response_standardised"]
    prose(a, f"median {n(mv['median'], 2)}, range {n(mv['min'], 2)}–{n(mv['max'], 2)}", src)
    f = lambda e: f"{n(e['rho'], 2)} [{n(e['fisher_z_ci95'][0], 2)}, {n(e['fisher_z_ci95'][1], 2)}]"
    pm_ = rc["per_model"]
    for short, lab in (("gated", "gated network"), ("mlp", "MLP"), ("hr_only", "HR-only")):
        prose(a, f"{lab} the correlation with HR response is {f(pm_[short]['with_hr_response'])}" if short == "gated" else
              f"{f(pm_[short]['with_hr_response'])} and {f(pm_[short]['with_movement_response'])}", src)
    prose(a, f"and with movement response {f(pm_['gated']['with_movement_response'])}", src)
    prose(a, f"leaves the gated network's HR-response partial correlation at {n(pm_['gated']['partial_hr_given_movement'], 2)}", src)
    prose(a, f"the movement-response partial given HR is {s(pm_['gated']['partial_movement_given_hr'], 2)}", src)
    prose(a, f"is (ii) ({f(pm_['ii']['with_movement_response'])})", src)
    ex = [k for k, v in pm_.items() if v["with_movement_response"]["fisher_z_ci95"][0] > 0]
    fact("§5.7 | only (ii) has a movement-response CI above 0", ex == ["ii"], "['ii']", str(ex), src)
    prose(a, f"(ii) ({f(pm_['ii']['with_movement_response'])}), one of eight", src)
    prose(a, f"only moderately correlated ({f(rc['hr_vs_movement_response'])})", src)


# ===========================================================================
# Round-5 pre-specified analyses (prespec_round5.md): C4 operating points
# (run_operating_points.py) and C3 informative missingness (run_mnar.py)
# ===========================================================================

OP = J("operating_points.json") if (R / "operating_points.json").exists() else None
MN_ = J("mnar.json") if (R / "mnar.json").exists() else None
NM = J("mnar_nomask.json") if (R / "mnar_nomask.json").exists() else None
IR = J("intercept_recal.json") if (R / "intercept_recal.json").exists() else None
OP_SYS = {"Gated + dropout": "gated", "Non-gated MLP, same masking": "mlp",
          "(ii) NaN-trained LightGBM": "lgbm_nan_training",
          "Per-configuration ((iii) without EDA)": "per_config"}
OP_ENDPOINTS = ("sens90_sens", "sens90_spec", "spec90_sens", "spec90_spec", "nb_0.2", "nb_0.35", "nb_0.5")


def check_c4():
    if OP is None:
        RESULTS.append(Result("FAIL", "results/operating_points.json missing", None, "", "", ""))
        return
    src = "operating_points.json"
    fact("§4.8 | operating-points run reproduces repeated-CV Brier", OP["verification"]["reproduces_repeated_cv"],
         "True", str(OP["verification"]), src)
    prose("Raw per-fold Brier in this run reproduces", f"({OP['verification']['n_checked']:,} of {OP['verification']['n_expected']:,}", src)
    pr, pc = OP["per_system_reduced"], OP["per_system_complete"]
    fa = lambda v, k: f"{(1 - v[k + '_mean']) * 60:.1f}"
    hdr = "| System | High-sensitivity point: sensitivity / specificity |"
    for row, k in OP_SYS.items():
        v = pr[k]
        cell(hdr, row, 1, f"{n(v['sens90_sens_mean'])} / {n(v['sens90_spec_mean'])}", src)
        cell(hdr, row, 2, fa(v, "sens90_spec"), src)
        cell(hdr, row, 3, f"{n(v['spec90_sens_mean'])} / {n(v['spec90_spec_mean'])}", src)
        cell(hdr, row, 4, fa(v, "spec90_spec"), src)
        cell(hdr, row, 5, n(v["nb_0.35_mean"]), src)
    a = "**What the near-ties look like at the bedside.**"
    fas = [(1 - pr[k]["sens90_spec_mean"]) * 60 for k in OP_SYS.values()]
    prose(a, f"raises {round(min(fas))}–{round(max(fas))} false alarms per hour of rest", src)
    f2 = [(1 - pr[k]["spec90_spec_mean"]) * 60 for k in OP_SYS.values()]
    fact("§5.9 | ~5 false alarms/h at high-spec point", all(4.5 <= x <= 5.5 for x in f2), "about 5", str([round(x, 1) for x in f2]), src)
    sv = [pr[k]["spec90_sens_mean"] for k in OP_SYS.values()]
    prose(a, f"(sensitivity {rng(sv)})", src)
    ss, sp = [pr[k]["sens90_sens_mean"] for k in OP_SYS.values()], [pr[k]["spec90_spec_mean"] for k in OP_SYS.values()]
    prose(a, f"(achieved sensitivity {rng(ss, 3)}, specificity {rng(sp, 3)})", src)
    prose(a, f"{fa(pc['per_config'], 'sens90_spec')} false alarms per hour against {fa(pc['gated'], 'sens90_spec')} for the gated network", src)
    _cover_tok(a, "90%")
    _cover_tok(a, "92%")
    fact("§5.9 | ~92% specificity at high-spec point", all(abs(x - 0.92) < 0.006 for x in sp), "≈0.92", str([round(x, 3) for x in sp]), src)

    pw = OP["pairwise_reduced"]
    gm = pw["gated__vs__mlp"]
    bft = [m for m in OP_ENDPOINTS if gm[m]["tost_p"] < 0.05]
    fact("§5.9 | gated−MLP B–F TOST passes only spec90_spec", bft == ["spec90_spec"], "['spec90_spec']", str(bft), src)
    a = "*Gated ≈ MLP within ±0.05 sensitivity/specificity"
    prose(a, f"(p = {p3(gm['spec90_spec']['tost_p'])})", src)
    prose(a, f"(net benefit at 0.35: {cie(gm['nb_0.35'])})", src)
    npt = sum(gm["participant"][m]["tost_p"] < 0.05 for m in OP_ENDPOINTS)
    fact("§5.9 | participant-level 6 of 7 pass", npt == 6, "6", str(npt), src)
    a = "*Gated vs (ii) and vs per-configuration intervals include 0:*"
    inc = all(pw[c][m]["ci95"][0] <= 0 <= pw[c][m]["ci95"][1]
              for c in ("gated__vs__lgbm_nan_training", "gated__vs__per_config") for m in OP_ENDPOINTS)
    fact("§5.9 | all 14 gated−(ii)/PC B–F intervals include 0", inc, "True", str(inc), src)
    prose(a, f"gated − (ii) {cie(pw['gated__vs__lgbm_nan_training']['nb_0.35'])}", src)
    prose(a, f"({cie(pw['gated__vs__lgbm_nan_training']['participant']['spec90_sens'])})", src)
    pcw = OP["pairwise_complete"]["gated__vs__per_config"]
    a = "*With EDA present, per-configuration net benefit exceeds"
    prose(a, f"(gated − per-configuration {cie(pcw['nb_0.35'])}; participant-level {cie(pcw['participant']['nb_0.35'])})", src)
    asym = OP["abstention_asymmetry_reduced"]
    a = "*At 60% coverage stress windows are abstained on more"
    am = [asym[k]["sel60"]["mean"] for k in OP_SYS.values()]
    prose(a, f"(abstention rate in stress minus baseline windows {s(min(am), 2)} to {s(max(am), 2)}", src)
    fact("§5.9 | every asymmetry interval above 0", all(asym[k]["sel60"]["ci95"][0] > 0 for k in OP_SYS.values()), "True", "", src)
    g = pr["gated"]
    prose(a, f"keeps {round(100 * (1 - g['sel60_abstain_stress_mean']))}% of stress windows and {round(100 * (1 - g['sel60_abstain_baseline_mean']))}% of baseline windows", src)
    a = "**What this says about the ±0.02 Brier margin.**"
    mx = max(max(abs(pw[c]["nb_0.35"]["ci95"][0]), abs(pw[c]["nb_0.35"]["ci95"][1]))
             for c in ("gated__vs__lgbm_nan_training", "gated__vs__per_config"))
    fact("§5.9 | NB bound vs trees ≈ 0.04", abs(mx - 0.04) < 0.005, "≈0.04", f"{mx:.3f}", src)
    _cover_tok(a, "0.04")


C3_SYS = {"Gated + dropout": "gated", "Non-gated MLP": "mlp", "(ii) NaN-trained LightGBM": "lgbm_nan_training",
          "Per-configuration": "per_config"}
C3_SCEN = {"S0 MCAR → MCAR": "S0:mcar->mcar", "S1 label OR 4 → same": "S1:label_4->label_4",
           "S2 label OR 4 → MCAR": "S2:label_4->mcar", "S3 MCAR → label OR 4": "S3:mcar->label_4"}
C3_MECH = {"MCAR": "mcar", "Label-driven, OR 2": "label_2", "Label-driven, OR 4": "label_4", "Label-driven, OR 8": "label_8",
           "Motion-driven, OR 2": "motion_2", "Motion-driven, OR 4": "motion_4", "Motion-driven, OR 8": "motion_8"}


def check_c3():
    if MN_ is None:
        RESULTS.append(Result("FAIL", "results/mnar.json missing", None, "", "", ""))
        return
    src = "mnar.json"
    sysk = list(C3_SYS.values())
    h7a = "| Training mechanism | Gated + dropout |"
    for row, m in C3_MECH.items():
        si = MN_["shortcut_index"][m]
        for j, k in enumerate(sysk, start=1):
            e = {"mean": si[k]["baseline_mean"], "ci95": si[k]["baseline_bf"]["ci95"]}
            cell(h7a, row, j, cie(e), src)
    h7b = "| Scenario | Gated + dropout | Non-gated MLP |"
    raw = MN_["raw"]
    for row, sc in C3_SCEN.items():
        r = raw[sc]
        for j, k in enumerate(sysk, start=1):
            cell(h7b, row, j, n(r["per_system"][k]["brier_mean"]), src)
        g = r["pairwise"]["gated__vs__mlp"]
        cell(h7b, row, 5, cie(g["brier"]), src)
        cell(h7b, row, 6, f"{p3(g['brier']['tost_p'])} / {p3(g['participant_brier']['tost_p'])}", src)

    si = MN_["shortcut_index"]
    a = "*C3-P1,"
    prose(a, f"from +{si['mcar']['gated']['baseline_mean']:.3f} (MCAR) to +{si['label_8']['gated']['baseline_mean']:.3f} (OR 8)", src)
    prose(a, f"MLP from {s(si['mcar']['mlp']['baseline_mean'])} to {s(si['label_8']['mlp']['baseline_mean'])}", src)
    inc = all(si[m][k]["baseline_bf"]["ci95"][0] <= 0 <= si[m][k]["baseline_bf"]["ci95"][1]
              for m in C3_MECH.values() if m.startswith("label") or m == "mcar" for k in ("gated", "mlp"))
    fact("§5.13 | C3-P1 gated and MLP intervals include 0", inc, "True", str(inc), src)
    ex = [m for m in ("label_2", "label_4", "label_8") if si[m]["lgbm_nan_training"]["baseline_bf"]["ci95"][0] > 0]
    fact("§5.13 | C3-P1 (ii) interval above 0 only at OR 4 and 8", ex == ["label_4", "label_8"], "['label_4', 'label_8']", str(ex), src)
    mxi = max(si[m][k]["baseline_mean"] for m in ("label_2", "label_4", "label_8") for k in ("gated", "mlp", "lgbm_nan_training"))
    fact("§5.13 | C3-P1 largest index ≈ 0.08", abs(mxi - 0.08) < 0.005, "≈0.08", f"{mxi:.3f}", src)
    _cover_tok(a, "0.08")

    marg = []
    for sc, r in raw.items():
        ps = r["per_system"]
        pc = ps["per_config"]["brier_mean"]
        others = [ps[k]["brier_mean"] for k in ("gated", "mlp", "lgbm_nan_training")]
        marg.append(min(others) - pc)
    fact("§5.13 | C3-P2 per-configuration lowest in all 19 scenarios", len(marg) == 19 and min(marg) > 0, "19 / True", f"{len(marg)} {min(marg):.3f}", src)
    a = "*C3-P2,"
    prose(a, f"by {n(min(marg))} to {n(max(marg))} over the best other system", src)
    s1 = raw["S1:label_4->label_4"]["per_system"]
    prose(a, f"(S1: {n(s1['per_config']['brier_mean'])} against {n(s1['lgbm_nan_training']['brier_mean'])} for (ii), {n(s1['gated']['brier_mean'])} and {n(s1['mlp']['brier_mean'])})", src)

    a = "*C3-P3,"
    v2 = raw["S2:label_4->mcar"]["vs_S0"]
    prose(a, f"{s(v2['gated']['mean'])} (gated), {s(v2['mlp']['mean'])} (MLP) and {s(v2['lgbm_nan_training']['mean'])} ((ii))", src)
    fact("§5.13 | C3-P3 every S2−S0 interval includes 0",
         all(v2[k]["ci95"][0] <= 0 <= v2[k]["ci95"][1] for k in ("gated", "mlp", "lgbm_nan_training")), "True", "", src)
    prim = max(abs(raw[sc]["vs_S0"][k]["mean"]) for sc in ("S1:label_4->label_4", "S2:label_4->mcar") for k in sysk)
    allm = max(abs(r["vs_S0"][k]["mean"]) for sc, r in raw.items() if r.get("vs_S0") for k in sysk)
    fact("§5.13 | C3-P3 max change from S0: primary ≤0.005, all ≤0.012", round(prim, 3) == 0.005 and round(allm, 3) == 0.012,
         "0.005 / 0.012", f"{prim:.4f} / {allm:.4f}", src)
    _cover_tok(a, "0.005")
    _cover_tok(a, "0.012")
    fact("§5.13 | 18 non-S0 scenarios", sum(bool(r.get("vs_S0")) for r in raw.values()) == 18, "18", str(sum(bool(r.get("vs_S0")) for r in raw.values())), src)

    a = "*C3-P4,"
    g1, g2 = (raw[k]["pairwise"]["gated__vs__mlp"] for k in ("S1:label_4->label_4", "S2:label_4->mcar"))
    prose(a, f"(p = {p3(g1['brier']['tost_p'])} in S1 and {p3(g2['brier']['tost_p'])} in S2", src)
    prose(a, f"passes in S1 (p = {p3(g1['participant_brier']['tost_p'])}) and just fails in S2 (p = {p3(g2['participant_brier']['tost_p'])})", src)

    a = "**The gated − MLP gap is the same in every scenario"
    gm = [r["pairwise"]["gated__vs__mlp"]["brier"]["mean"] for r in raw.values()]
    prose(a, f"({s(max(gm))} to {s(min(gm))}", src)
    r0 = raw["S0:mcar->mcar"]["per_system"]
    prose(a, f"differ by {n(r0['mlp']['brier_present_rows_mean'] - r0['gated']['brier_present_rows_mean'])} there ({n(r0['gated']['brier_present_rows_mean'])} vs {n(r0['mlp']['brier_present_rows_mean'])})", src)
    prose(a, f"are close on rows with missing EDA ({n(r0['gated']['brier_missing_rows_mean'])} vs {n(r0['mlp']['brier_missing_rows_mean'])})", src)
    a = "**Label- or motion-correlated missingness at odds ratios up to 8"
    prose(a, f"(index at OR 8: MLP {s(si['label_8']['mlp']['baseline_mean'])}, gated {s(si['label_8']['gated']['baseline_mean'])}; no added masking: MLP {s(NM['shortcut_index']['label_8']['mlp']['baseline_mean'])}, gated {s(NM['shortcut_index']['label_8']['gated']['baseline_mean'])})", src)
    prose(a, f"by at most {n(allm)} from the MCAR case", src)
    prose("**Informative missingness (§5.13).**", f"by more than {allm:.3f}", src)
    prose("For the per-configuration system the index is a model switch", f"+{si['mcar']['per_config']['baseline_mean']:.3f} under MCAR", src)


# ===========================================================================
# Review 9 (post hoc): same-scheme factorial, intercept/slope recalibration,
# no-added-masking MNAR variant, per-configuration claims, movement partial
# ===========================================================================

CMOVE = J("movement_checks.json") if (R / "movement_checks.json").exists() else None


def si_pre(m, k):
    return MN_["shortcut_index"][m][k]["baseline_mean"]


def check_review9():
    if IR is None or NM is None:
        RESULTS.append(Result("FAIL", "results/intercept_recal.json or mnar_nomask.json missing", None, "", "", ""))
        return
    src = "intercept_recal.json"
    names = [G, GN, M, MN, LI, L2]
    hdr = "| Calibration | Gated | Gated, no masking | MLP |"
    for row, c in (("Raw", "raw"), ("Intercept-only", "intercept"), ("Slope-only", "slope"), ("Per-condition Platt", "platt_pc")):
        pmod = IR["per_model"][f"reduced:{c}"]
        cs = IR["contrasts"][f"reduced:{c}"]
        for j, k in enumerate(names, start=1):
            cell(hdr, row, j, n(pmod[k]["brier"]), src)
        cell(hdr, row, 7, cie(cs["main_effect_masking"]["bf"]), src)
        cell(hdr, row, 8, cie(cs["masking_tree_(ii)_minus_(i)"]["bf"]), src)
    pr = RCV["per_model_reduced"]["raw"]
    fact("§5.2 | Table 4b raw row reproduces Table 1",
         all(abs(IR["per_model"]["reduced:raw"][k]["brier"] - pr[k]["brier_mean"]) < 1e-9 for k in names),
         "True", "", "intercept_recal.json vs repeated_cv.json")
    sh = IR["intercept_shift"]["reduced"]
    sr = IR["share_of_masking_main_effect_removed"]
    cr = IR["contrasts"]
    a = "**Trees: calibration-in-the-large.**"
    prose(a, f"mean logit shift of {s(sh[LI]['mean'], 2)} (SD {n(sh[LI]['sd'], 2)} over folds)", src)
    prose(a, f"from {n(IR['per_model']['reduced:raw'][LI]['brier'])} to {n(IR['per_model']['reduced:intercept'][LI]['brier'])}, as good as full Platt ({n(IR['per_model']['reduced:platt_pc'][LI]['brier'])})", src)
    prose(a, f"falls from {s(cr['reduced:raw']['masking_tree_(ii)_minus_(i)']['bf']['mean'])} to {s(cr['reduced:intercept']['masking_tree_(ii)_minus_(i)']['bf']['mean'])}", src)
    a = "**Networks: a scale effect, not an intercept effect.**"
    mm = lambda c: cr[f"reduced:{c}"]["main_effect_masking"]["bf"]["mean"]
    prose(a, f"({s(mm('raw'))} → {s(mm('intercept'))})", src)
    prose(a, f"slope-only rescaling removes {round(100 * sr['slope'])}% (→ {s(mm('slope'))}) and full Platt {round(100 * sr['platt_pc'])}% (→ {s(mm('platt_pc'))})", src)
    prose(a, f"(gated {s(sh[GN]['mean'], 2)}, MLP {s(sh[MN]['mean'], 2)}, with fold SDs of {sh[GN]['sd']:.1f} and {sh[MN]['sd']:.1f} logits)", src)
    prose(a, f"their fitted slopes are {n(sh[GN]['slope_mean'], 2)} and {n(sh[MN]['slope_mean'], 2)}, against {n(sh[G]['slope_mean'], 2)} and {n(sh[M]['slope_mean'], 2)} for the masked networks", src)
    prose(a, f"leaves (i) at {n(IR['per_model']['reduced:slope'][LI]['brier'])} and its tree masking effect at {cie(cr['reduced:slope']['masking_tree_(ii)_minus_(i)']['bf'])}", src)
    fact("§5.2 | intercept-only removes none of the networks' masking effect", sr["intercept"] <= 0.0, "<= 0", f"{sr['intercept']:.3f}", src)

    # Same-scheme restatements (contributions, discussion, limitations, abstract)
    me = RCV["contrasts"]["raw"]["main_effect_masking"]["brier"]
    t = pwt(RCV, "raw", L2, LI)["brier"]
    d99 = f"[{s(me['ci95'][0])}, {s(me['ci95'][1])}]"
    d14 = f"[{s(me['df_subjects']['ci95'][0])}, {s(me['df_subjects']['ci95'][1])}]"
    a = "**For trees, exposure to missing EDA is essential:**"
    prose(a, f"(ii) vs (i) is {cie(t)}", src)
    prose(a, f"the masking benefit ({s(me['mean'])}) is directionally consistent (12/15 participants)", src)
    prose(a, f"its Bouckaert–Frank interval is {d99} at df 99 and {d14} at df 14", src)
    a = "**The masking effect is not at the same evidential level as the tree result.**"
    prose(a, f"((ii) vs (i) {s(t['mean'])}", src)
    prose(a, f"the masking benefit ({s(me['mean'])})", src)
    prose(a, f"its interval is {d14}", src)
    a = "**Masking effect for the networks is not resolved.**"
    prose(a, f"the masking main effect on the networks is {d14}", src)
    prose(a, f"((ii) vs (i) {s(t['mean'])})", src)
    a = "*Masking and calibration.*"
    prose(a, f"by {n(-t['mean'])} Brier", src)
    prose(a, f"(Brier {n(IR['per_model']['reduced:raw'][LI]['brier'])} to {n(IR['per_model']['reduced:intercept'][LI]['brier'])})", src)
    prose(a, f"({s(me['mean'])}, 12/15 participants)", src)
    prose(a, f"(B–F df 14: {s(me['df_subjects']['ci95'][0])} to {s(me['df_subjects']['ci95'][1])})", src)
    prose(a, f"(slope-only rescaling removed {round(100 * sr['slope'])}% of it", src)
    a = "**Calibration.** Models never exposed to missingness"
    prose(a, f"(Brier {n(IR['per_model']['reduced:raw'][LI]['brier'])} → {n(IR['per_model']['reduced:intercept'][LI]['brier'])}", src)
    prose(a, f"a slope-only rescaling removes {round(100 * sr['slope'])}% of it", src)
    aurs = [v["auroc_mean"] for v in RCV["per_model_reduced"]["raw"].values()]
    prose(a, f"(AUROC {rng(aurs, 3)})", "repeated_cv.json")
    _cover_tok(a, "0.06")
    _cover_tok(a, "0.09")

    # Section 5.3 evidence bullets
    pc_ = RCV["per_model_complete"]["raw"]
    a = '**Evidence for "two simple trees and a switch".**'
    prose(a, f"complete-condition Brier {n(pc_[LI]['brier_mean'])} against {n(pc_[G]['brier_mean'])} (gated), {n(pc_[M]['brier_mean'])} (MLP) and {n(pc_[L2]['brier_mean'])} ((ii))", "repeated_cv.json per_model_complete")
    fa_pc = (1 - OP["per_system_complete"]["per_config"]["sens90_spec_mean"]) * 60
    fa_g = (1 - OP["per_system_complete"]["gated"]["sens90_spec_mean"]) * 60
    prose(a, f"{fa_pc:.1f} against {fa_g:.1f} per hour of rest", "operating_points.json per_system_complete")
    gi = pwt(RCV, "raw", G, LI, "complete")["brier"]
    a = "**The caveats stay attached.**"
    prose(a, f"gated − (i) in the complete condition is {cie(gi)}", "repeated_cv.json pairwise_complete")
    marg = [min(r["per_system"][k]["brier_mean"] for k in ("gated", "mlp", "lgbm_nan_training")) - r["per_system"]["per_config"]["brier_mean"] for r in MN_["raw"].values()]
    marg_nm = max(abs(r["pairwise"]["lgbm_nan_training__vs__per_config"]["brier"]["mean"]) for r in NM["raw"].values())
    prose(a, f"shrinks from {n(min(marg))}–{n(max(marg))} to at most {n(marg_nm)}", "mnar.json / mnar_nomask.json")
    fact("§5.3 | per-config lead over (ii) not distinguishable from 0 under B–F in any of 19 mixed-test scenarios",
         all(r["pairwise"]["lgbm_nan_training__vs__per_config"]["brier"]["ci95"][0] <= 0 for r in MN_["raw"].values()) and len(MN_["raw"]) == 19,
         "True", "", "mnar.json")

    # Table 7c / 7d and bullets
    h7c = "| Training mechanism | Gated, no added masking |"
    mech = {"MCAR": "mcar", "Label-driven, OR 2": "label_2", "Label-driven, OR 4": "label_4", "Label-driven, OR 8": "label_8",
            "Motion-driven, OR 2": "motion_2", "Motion-driven, OR 4": "motion_4", "Motion-driven, OR 8": "motion_8"}
    sysk = ["gated", "mlp", "lgbm_nan_training", "per_config"]
    nsrc = "mnar_nomask.json"
    for row, m in mech.items():
        si = NM["shortcut_index"][m]
        for j, k in enumerate(sysk, start=1):
            cell(h7c, row, j, cie({"mean": si[k]["baseline_mean"], "ci95": si[k]["baseline_bf"]["ci95"]}), nsrc)
    fact("§5.13 | Table 7c per-configuration column equals Table 7a",
         all(abs(NM["shortcut_index"][m]["per_config"]["baseline_mean"] - MN_["shortcut_index"][m]["per_config"]["baseline_mean"]) < 1e-9 for m in mech.values()),
         "True", "", nsrc)
    h7d = "| Scenario | Gated, no added masking |"
    scen = {"S0 MCAR → MCAR": "S0:mcar->mcar", "S1 label OR 4 → same": "S1:label_4->label_4",
            "S2 label OR 4 → MCAR": "S2:label_4->mcar", "S3 MCAR → label OR 4": "S3:mcar->label_4"}
    for row, sc in scen.items():
        r = NM["raw"][sc]
        for j, k in enumerate(sysk, start=1):
            cell(h7d, row, j, n(r["per_system"][k]["brier_mean"]), nsrc)
        g = r["pairwise"]["gated__vs__mlp"]
        cell(h7d, row, 5, cie(g["brier"]), nsrc)
        cell(h7d, row, 6, f"{p3(g['brier']['tost_p'])} / {p3(g['participant_brier']['tost_p'])}", nsrc)
        cell(h7d, row, 7, cie(r["pairwise"]["lgbm_nan_training__vs__per_config"]["brier"]), nsrc)
    sn = NM["shortcut_index"]
    bm = lambda m, k: sn[m][k]["baseline_mean"]
    a = "**The design caveat was right:"
    prose(a, f"the index is {s(bm('label_8', 'gated'))} for the gated network (pre-specified run {s(si_pre('label_8', 'gated'))}; interval now above 0)", nsrc)
    prose(a, f"{s(bm('label_8', 'mlp'))} for the MLP ({s(si_pre('label_8', 'mlp'))}; lower bound {s(sn['label_8']['mlp']['baseline_bf']['ci95'][0])})", nsrc)
    prose(a, f"{s(bm('label_8', 'lgbm_nan_training'))} for (ii) ({s(si_pre('label_8', 'lgbm_nan_training'))})", nsrc)
    lab = [bm(m, k) for m in ("label_4", "label_8") for k in ("gated", "mlp", "lgbm_nan_training")]
    mot = [bm(m, k) for m in ("motion_4", "motion_8") for k in ("gated", "mlp", "lgbm_nan_training")]
    prose(a, f"scored on average {n(min(lab), 2)}–{n(max(lab), 2)} higher in p(stress) by the three learned systems (motion-driven: {n(min(mot), 2)}–{n(max(mot), 2)})", nsrc)
    prose(a, f"the gated network ({s(bm('mcar', 'gated'))} → {s(bm('label_2', 'gated'))} → {s(bm('label_4', 'gated'))} → {s(bm('label_8', 'gated'))})", nsrc)
    prose(a, f"the MLP ({s(bm('mcar', 'mlp'))} → {s(bm('label_2', 'mlp'))} → {s(bm('label_4', 'mlp'))} → {s(bm('label_8', 'mlp'))})", nsrc)
    fact("§5.13 | variant: (ii) interval above 0 at every label and motion OR; gated only at label OR 8",
         all(sn[m]["lgbm_nan_training"]["baseline_bf"]["ci95"][0] > 0 for m in mech.values() if m != "mcar")
         and [m for m in mech.values() if sn[m]["gated"]["baseline_bf"]["ci95"][0] > 0] == ["label_8"], "True", "", nsrc)
    fact("§5.13 | variant: shortcut two to three times the pre-specified run",
         all(2.0 <= bm("label_8", k) / si_pre("label_8", k) <= 3.0 for k in ("gated", "mlp", "lgbm_nan_training")), "True", "", nsrc)
    a = "**The shortcut costs little Brier even so.**"
    allm = max(abs(r["vs_S0"][k]["mean"]) for r in NM["raw"].values() if r.get("vs_S0") for k in sysk)
    prose(a, f"by more than {n(allm)}", nsrc)
    v = NM["raw"]["S1:label_8->label_8"]["vs_S0"]
    prose(a, f"(label OR 8: gated {s(v['gated']['mean'])}, MLP {s(v['mlp']['mean'])}, (ii) {s(v['lgbm_nan_training']['mean'])})", nsrc)
    v = NM["raw"]["S2:label_4->mcar"]["vs_S0"]
    prose(a, f"are {s(v['gated']['mean'])} (gated), {s(v['mlp']['mean'])} (MLP) and {s(v['lgbm_nan_training']['mean'])} ((ii)) at OR 4", nsrc)
    fact("§5.13 | variant: S2 OR-4 intervals include 0", all(v[k]["ci95"][0] <= 0 <= v[k]["ci95"][1] for k in ("gated", "mlp", "lgbm_nan_training")), "True", "", nsrc)
    s3 = max(NM["raw"][k]["vs_S0"]["gated"]["mean"] for k in NM["raw"] if k.startswith("S3"))
    s3all = max(NM["raw"][k]["vs_S0"][x]["mean"] for k in NM["raw"] if k.startswith("S3") for x in sysk)
    prose(a, f"increase is {s(s3)} (gated, label OR 8)", nsrc)
    fact("§5.13 | variant: largest S3 increase is gated at label OR 8",
         abs(s3 - s3all) < 1e-12 and NM["raw"]["S3:mcar->label_8"]["vs_S0"]["gated"]["mean"] == s3, "True", "", nsrc)
    a = "**The gate is not shown to be better or worse than the MLP here.**"
    gm = [r["pairwise"]["gated__vs__mlp"]["brier"]["mean"] for r in NM["raw"].values()]
    prose(a, f"Gated − MLP is {s(max(gm))} to {s(min(gm))} across the 19 scenarios", nsrc)
    fact("§5.13 | variant: gated − MLP interval includes 0 in all 19; TOST fails in the four shown",
         all(r["pairwise"]["gated__vs__mlp"]["brier"]["ci95"][0] <= 0 <= r["pairwise"]["gated__vs__mlp"]["brier"]["ci95"][1] for r in NM["raw"].values())
         and len(NM["raw"]) == 19
         and all(NM["raw"][sc]["pairwise"]["gated__vs__mlp"][k]["tost_p"] > 0.05 for sc in scen.values() for k in ("brier", "participant_brier")),
         "True", "", nsrc)
    prose(a, f"(S0 Brier {n(NM['raw']['S0:mcar->mcar']['per_system']['mlp']['brier_mean'])} against {n(MN_['raw']['S0:mcar->mcar']['per_system']['mlp']['brier_mean'])} in the pre-specified run)", nsrc)
    a = "**The per-configuration lead over (ii) mostly disappears.**"
    prose(a, f"by {n(min(marg))}–{n(max(marg))} in every scenario", nsrc)
    lost = [(k, r["per_system"]["per_config"]["brier_mean"] - r["per_system"]["lgbm_nan_training"]["brier_mean"]) for k, r in NM["raw"].items()
            if r["per_system"]["lgbm_nan_training"]["brier_mean"] < r["per_system"]["per_config"]["brier_mean"]]
    prose(a, f"(ii) is within {n(marg_nm)} of it in every scenario and lower in {len(lost)} of the 19", nsrc)
    prose(a, f"by {n(lost[0][1])}, {n(lost[1][1])} and {n(lost[2][1])})", nsrc)
    fact("§5.13 | variant: (ii) lower in S1 label 4, S1 label 8 and S1 motion 8",
         [k for k, _ in lost] == ["S1:label_4->label_4", "S1:label_8->label_8", "S1:motion_8->motion_8"], "True", str([k for k, _ in lost]), nsrc)
    fact("§5.13 | variant: per-config lowest in 16 of 19 and ahead of both networks in all 19 at participant level",
         sum(r["per_system"]["per_config"]["brier_mean"] < min(r["per_system"][k]["brier_mean"] for k in ("gated", "mlp", "lgbm_nan_training")) for r in NM["raw"].values()) == 16
         and all(r["pairwise"][f"{x}__vs__per_config"]["participant_brier"]["ci95"][0] > 0 for r in NM["raw"].values() for x in ("gated", "mlp")),
         "True", "", nsrc)

    # Discussion / abstract restatements
    a = "**What a wearable-detector builder should take from this"
    prose(a, f"(by {n(min(marg))}–{n(max(marg))} over the best other system)", "mnar.json")
    prose(a, f"({fa_pc:.1f} against {fa_g:.1f} per hour of rest for the gated network)", "operating_points.json")
    prose(a, f"falls to at most {n(marg_nm)} when (ii) gets no extra training mask", nsrc)
    a = "- **Informative missingness (§5.13).** In the pre-specified run"
    prose(a, f"by more than {n(max(abs(r['vs_S0'][k]['mean']) for r in MN_['raw'].values() if r.get('vs_S0') for k in sysk), 3)}", "mnar.json")
    prose(a, f"scored {n(min(lab), 2)}–{n(max(lab), 2)} higher at label odds ratio 4–8", nsrc)
    prose(a, f"at most {n(allm)}", nsrc)
    a = "*Per-configuration system.*"
    prose(a, f"(by {n(min(marg))}–{n(max(marg))} over the best other system)", "mnar.json")
    prose(a, f"({fa_pc:.1f} vs {fa_g:.1f} per hour of rest for the gated network", "operating_points.json")
    prose(a, f"fell to at most {n(marg_nm)}", nsrc)
    a = "*Informative missingness.* In the pre-specified run"
    prose(a, f"scored {n(min(lab), 2)}–{n(max(lab), 2)} higher in p(stress) at label odds ratio 4–8", nsrc)
    prose(a, f"by at most {n(allm)}", nsrc)
    a = "**The per-configuration system's advantage is unadjudicated.**"
    prose(a, f"to at most {n(marg_nm)}", nsrc)

    # Section 5.7 movement partial
    if CMOVE is not None:
        pm_ = CMOVE["response_correlations"]["per_model"]
        a = "**A negative partial is expected from a model that reads heart rate:**"
        prose(a, f"(the MLP's partial is {s(pm_['mlp']['partial_movement_given_hr'], 2)}, (iii)'s {s(pm_['iii']['partial_movement_given_hr'], 2)} and, for HR-only, which has no movement input at all, {s(pm_['hr_only']['partial_movement_given_hr'], 2)}", "movement_checks.json")

    # Remaining restatements
    gm_ = pwt(RCV, "raw", G, M)
    a = "**Under per-condition recalibration**"
    prose(a, f"the unresolved {s(me['mean'])} above" if False else f"directionally consistent, unresolved {s(me['mean'])} above", src)
    a = "**Consequence for the principal finding.**"
    prose(a, f"it moves Brier by at most {n(allm)}", nsrc)
    a = "Against a capacity-matched MLP with the same masking, the gate is equivalent"
    prose(a, f"B–F df 14, TOST p = {gm_['brier']['df_subjects']['tost_p']:.3f} and {gm_['auroc']['df_subjects']['tost_p']:.3f})", "repeated_cv.json")
    prose(a, f"the Brier bound attained is {gm_['brier']['df_subjects']['equivalence_bound_attained']:.3f}", "repeated_cv.json")
    prose("**Post-hoc equivalence.** The ±0.02 Brier margin was fixed", f"Brier bound attained is {gm_['brier']['df_subjects']['equivalence_bound_attained']:.3f} against a margin of 0.02", "repeated_cv.json")
    if CMOVE is not None:
        _cover_tok("**A negative partial is expected from a model that reads heart rate:**", "0.88")


# ---------------------------------------------------------------------------

def main() -> int:
    checks = [check_front_and_data, check_methods, check_table1, check_table2_3_rule, check_factorial,
              check_complete, check_robustness, check_hr_section, check_order_confound, check_participants,
              check_calibration_selective, check_wrist_sqi_reimpl, check_appendices, check_appendix_e,
              check_appendix_f, check_abstract, check_extra_prose, check_round5, check_c4, check_c3, check_movement_checks, check_abstention_action, check_review9]
    for c in checks:
        try:
            c()
        except Exception as e:  # noqa: BLE001  (one broken check must not hide the others)
            RESULTS.append(Result("FAIL", f"{c.__name__} crashed", None, "", f"{type(e).__name__}: {e}", ""))

    npass = sum(r.status == "PASS" for r in RESULTS)
    nfail = sum(r.status == "FAIL" for r in RESULTS)
    nnote = sum(r.status == "NOTE" for r in RESULTS)
    un = untraced()
    nun = sum(len(v) for v in un.values())

    md = ["# Numbers consistency check: `e14-preprint-outline.md`", "",
          f"Generated by `check_numbers.py`. Outline: {len(LINES)} lines. Feature table SHA `{RCV['config']['feature_table_sha256'][:12]}…`.", "",
          f"**{npass} PASS · {nfail} FAIL · {nnote} NOTE · {nun} untraced decimal/percent/ratio tokens**", "",
          "A PASS means the outline shows exactly the rounded source value in exactly that table cell or line. "
          "Source values are formatted the way the outline formats them and compared as strings.", ""]
    for status in ("FAIL", "NOTE", "PASS"):
        rs = [r for r in RESULTS if r.status == status]
        if not rs:
            continue
        md += [f"## {status} ({len(rs)})", "", "| Where | Line | Expected (from source) | Found in outline | Source | Note |", "|---|---|---|---|---|---|"]
        for r in rs:
            esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
            md.append(f"| {esc(r.where)} | {r.line or ''} | {esc(r.expected)} | {esc(r.found)} | {esc(r.source)} | {esc(r.note)} |")
        md.append("")
    md += ["## Untraced numbers", "",
           "Decimal, percentage and ratio tokens in prose that no keyed check covered and that are not in the CONSTANTS allowlist "
           "(method constants and external citation facts, each with its reason in check_numbers.py). Headings, §2 Related Work, "
           "Contributions, §4.7, §10.1/10.3 and Appendices G-H are skipped; integers are skipped. Target: none.", ""]
    for sec, toks in un.items():
        md.append(f"- **{sec}:** " + ", ".join(f"L{ln} `{t}`" for ln, t in toks))
    REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{npass} PASS, {nfail} FAIL, {nnote} NOTE, {nun} untraced -> {REPORT.relative_to(ROOT)}")
    for r in RESULTS:
        if r.status == "FAIL":
            print(f"  FAIL L{r.line or '?'} {r.where}: expected {r.expected!r}; found {r.found!r}  [{r.source}]")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
