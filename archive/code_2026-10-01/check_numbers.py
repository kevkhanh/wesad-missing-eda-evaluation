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
    k = text.find(frag)
    if k >= 0:
        COVERED.setdefault(i, []).append((k, k + len(frag)))


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
        _cover(i, LINES[i], got)
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


# ===========================================================================
# CHECKS
# ===========================================================================

def check_data_section():
    ps = PS_LOSO[(PS_LOSO.model == G) & (PS_LOSO.tier == "reduced")]
    nwin, nstress = ps.n_windows, ps.n_stress
    nbase = nwin - nstress
    src = "results/per_subject_loso.csv (n_windows, n_stress; gated, reduced)"
    prose("Result: 434 windows", f"{int(nwin.sum())} windows", src)
    prose("Result: 434 windows", f"{nwin.min()}–{nwin.max()} per participant", src)
    prose("Result: 434 windows", f"{round(100 * nstress.sum() / nwin.sum())}% stress", src)
    prose("the cap is never reached", f"stress {nstress.min()}–{nstress.max()}", src)
    prose("the cap is never reached", f"baseline yields {nbase.min()}–{nbase.max()} windows", src)
    prose("Background.** Wearable", "", "")  # placeholder no-op anchor for abstract presence
    RESULTS.pop()
    prose("On WESAD (15 participants", f"{int(nwin.sum())} sixty-second windows", src)
    man = J("feature_table_manifest.json")
    prose("Every number in §5 comes from the frozen feature table", man["sha256"][:12], "results/feature_table_manifest.json")
    fact("§3 | 15 participants", man["n_subjects_wesad"] == 15 and ps.subject_id.nunique() == 15,
         "15", str(man["n_subjects_wesad"]), "feature_table_manifest.json")

    # Feature-table missingness claims (§4.1)
    tab = Path(man["path"])
    if tab.exists():
        import hashlib
        sha = hashlib.sha256(tab.read_bytes()).hexdigest()
        fact("§4.1 | frozen table hash matches manifest", sha == man["sha256"], man["sha256"][:12], sha[:12], str(tab))
        sys.path.insert(0, str(ROOT))
        from eval_core import load_wesad_frozen, ECG_FEATURE_NAMES
        df = load_wesad_frozen()
        miss_hrv = float(df[ECG_FEATURE_NAMES].isna().any(axis=1).mean() * 100)
        miss_cpl = float(df[["coupling_peak_corr", "coupling_lag_sec"]].isna().any(axis=1).mean() * 100)
        prose("It was not triggered on this table", f"{miss_hrv:.0f}% missing HRV", "frozen table: ECG NaN rate")
        prose("Undefined in", f"Undefined in {miss_cpl:.1f}% of windows", "frozen table: coupling NaN rate")
    else:
        note("§4.1", f"frozen table not found at {tab}; HRV/coupling missingness not checked")


def check_methods():
    sys.path.insert(0, str(ROOT))
    import torch  # noqa: F401
    from eval_core import GatedModel, NonGatedMLP, ECG_FEATURE_NAMES, EDA_FEATURE_NAMES
    import eval_core as ec

    def npar(model):
        return sum(p.numel() for p in model._build().parameters())

    gated = {d: npar(GatedModel(0.5, embed_dim=d)) for d in (8, 16, 32, 64)}
    mlp = {h: npar(NonGatedMLP(0.5, hidden=h)) for h in (8, 25, 32, 84, 300)}
    src = "eval_core.py: parameter count of built networks"
    fact("§4.1/§4.2 | feature counts 7 ECG, 6 EDA-dependent",
         len(ECG_FEATURE_NAMES) == 7 and len(EDA_FEATURE_NAMES) == 6,
         "7 / 6", f"{len(ECG_FEATURE_NAMES)} / {len(EDA_FEATURE_NAMES)}", "eval_core feature lists")
    cell("| Name | Description |", "Gated network", 1, f"{gated[16]:,} parameters", src)
    cell("| Name | Description |", "Non-gated MLP", 1, f"{mlp[32]:,} parameters", src)
    prose("**Capacity sweep:** gated embedding width",
          "/".join(f"{gated[d]:,}" for d in (8, 16, 32, 64)) + " parameters", src)
    prose("**Capacity sweep:** gated embedding width",
          "/".join(f"{mlp[h]:,}" for h in (8, 25, 84, 300)) + " parameters", src)
    n_lgbm = int(np.prod([len(v) for v in ec.LGBM_GRID.values()]))
    n_nn = int(np.prod([len(v) for v in ec.NN_GRID.values()])) * len(ec.GATED_WIDTHS)
    cell("| Family | Grid | Configurations |", "LightGBM", "Configurations", str(n_lgbm), "eval_core.LGBM_GRID")
    cell("| Family | Grid | Configurations |", "Networks", "Configurations", str(n_nn), "eval_core.NN_GRID × widths")
    prose("Nested tuning covers small grids only", f"{n_lgbm} LightGBM and {n_nn} network", "eval_core grids")
    prose("Nested tuning covers small grids only",
          f"{min(gated.values()):,}–{max(gated[64], mlp[300]):,} parameters", src)
    if mlp[32] != gated[16]:
        note("§4.3 | default MLP size",
             f"Default MLP has {mlp[32]:,} parameters vs the gated network's {gated[16]:,} "
             f"(28% more). The abstract and §2 call it 'capacity-matched'; strictly only the "
             f"capacity-sweep pairs are matched (16→{gated[16]} vs 25→{mlp[25]}).", src)

    cfg = RCV["config"]
    prose("**Primary inference:** 20× repeated", f"{cfg['n_distinct_partitions']} distinct partitions", "repeated_cv.json config")
    prose("**Primary inference:** 20× repeated", f"There are {cfg['n_repeats'] * cfg['n_folds']} (repeat, fold) pairs", "repeated_cv.json config")
    prose("**Primary inference:** 20× repeated",
          f"about {cfg['n_train_subjects_mean']:.0f} training and {cfg['n_test_subjects_mean']:.0f} test", "repeated_cv.json config")
    import run_baselines
    prose("Pooled estimates carry 95% CIs", f"({run_baselines.N_BOOT} resamples)", "run_baselines.N_BOOT")
    prose("Python 3.13", f"Python {sys.version.split()[0]}", "sys.version of the interpreter running this check")

    import subprocess
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    ntests = int(m.group(1)) if m else -1
    prose("automated tests, including a regression test", f"{ntests} automated tests", "pytest --collect-only tests/")


def check_table1():
    hdr = "| Model | Brier | AUROC | ECE_SWEEP* | Log-loss |"
    rows = {"Gated network + dropout": G, "Non-gated MLP, same masking": M,
            "(ii) NaN-trained LightGBM": L2, "(iii) ECG-only LightGBM": L3,
            "(iv) Mean-imputation + per-condition Platt": L4, "HR-only": HR,
            "Gated network, no dropout": GN}
    src = "repeated_cv.json per_model_reduced.raw.<model>"
    for row, k in rows.items():
        cell(hdr, row, "Brier", f"{n(pm(RCV,'raw',k,'brier_mean'))} ({n(pm(RCV,'raw',k,'brier_sd'))})", src)
        cell(hdr, row, "AUROC", f"{n(pm(RCV,'raw',k,'auroc_mean'))} ({n(pm(RCV,'raw',k,'auroc_sd'))})", src)
        cell(hdr, row, "ECE_SWEEP*", n(pm(RCV, 'raw', k, 'ece_sweep_mean')), src)
        cell(hdr, row, "Log-loss", n(pm(RCV, 'raw', k, 'log_loss_mean')), src)


TABLE2 = {"Gated vs (ii)": (G, L2), "Gated vs (iii)": (G, L3), "Gated vs (iv)": (G, L4),
          "Gated vs HR-only": (G, HR), "Gated vs no dropout": (G, GN),
          "Gated vs non-gated MLP": (G, M), "MLP vs (ii)": (M, L2)}


def check_table2_and_rule():
    hdr = "| Comparison | ΔBrier [95% CI] | ΔAUROC [95% CI] | Participants where A is better |"
    src = "repeated_cv.json pairwise.raw.<A>__vs__<B>"
    for row, (a, b) in TABLE2.items():
        cell(hdr, row, 1, pw_ci(RCV, "raw", a, b, "brier"), src)
        cell(hdr, row, 2, pw_ci(RCV, "raw", a, b, "auroc"), src)
        sf = pw(RCV, "raw", a, b, "subject_sign_flip_brier")
        cell(hdr, row, 3, f"{sf['n_subjects_A_better']}/{sf['n_subjects']}", src + ".subject_sign_flip_brier")

    # Rule bullets (raw)
    d = {m: pw(RCV, "raw", G, L2, m)["mean"] for m in ("brier", "ece_sweep", "auroc")}
    d3 = {m: pw(RCV, "raw", G, L3, m)["mean"] for m in ("brier", "ece_sweep", "auroc")}
    prose("**Raw probabilities.** The gated network trails", f"by {n(d['brier'])} on Brier", src)
    prose("**Raw probabilities.** The gated network trails", f"{n(d['ece_sweep'])} on ECE", src)
    prose("**Raw probabilities.** The gated network trails", f"leads by {n(d['auroc'])} on AUROC", src)
    fact("§5.1 | raw: same gaps vs (iii)",
         all(round(d[m], 3) == round(d3[m], 3) for m in d), "gaps vs (iii) equal those vs (ii) at 3 dp",
         ", ".join(f"{m} {d3[m]:+.4f}" for m in d3), src)
    # Platt bullet
    fr = RCV["falsification_rule"]["platt"]
    fact("§5.1 | shared Platt: gated beats (iii) on all three",
         fr[L3]["gated_beats_on_all_three"], "True", str(fr[L3]["gated_beats_on"]), "repeated_cv.json falsification_rule.platt")
    fact("§5.1 | shared Platt: gated trails (ii) on Brier and ECE",
         (not fr[L2]["gated_beats_on"]["brier"]) and (not fr[L2]["gated_beats_on"]["ece_sweep"]),
         "brier False, ece False", str(fr[L2]["gated_beats_on"]), "repeated_cv.json falsification_rule.platt")
    prose("**Per-condition Platt.** Under 20× repeated CV", pw_ci(RCV, "platt_pc", G, L2, "brier"),
          "repeated_cv.json pairwise.platt_pc")
    # Verdict table
    vt = "| Calibration | 20× repeated CV | LOSO pooled | Single 5-fold |"
    word = lambda f: "fires" if f else "does not fire"
    for row, cal in (("Raw", "raw"), ("Shared Platt", "platt"), ("Per-condition Platt", "platt_pc")):
        cell(vt, row, 1, word(RCV["falsification_rule"][cal]["rule_fires"]), f"repeated_cv.json falsification_rule.{cal}.rule_fires")
        for col, sch in ((2, "loso"), (3, "kfold5")):
            f = BAS["schemes"][sch]["falsification_rule"][cal]
            fires = f.get("rule_fires", not all(v["gated_beats_on_all_three"] for k, v in f.items() if isinstance(v, dict)))
            got_word = word(fires)
            # "fires" is a substring of "does not fire"? no: 'fires' vs 'fire' -> compare exactly
            try:
                _, header, rows = find_table(vt)
                actual = _cells(LINES[rows[row]])[col]
                ok = actual == got_word
                RESULTS.append(Result("PASS" if ok else "FAIL", f"§5.1 verdict | {row} | {header[col]}",
                                      rows[row] + 1, got_word, actual, f"baselines.json schemes.{sch}.falsification_rule.{cal}"))
                if ok:
                    _cover(rows[row], LINES[rows[row]], actual)
            except Exception as e:  # noqa: BLE001
                RESULTS.append(Result("FAIL", f"verdict {row}/{sch}", None, got_word, str(e), "baselines.json"))
    # §4.7 claim: Table 2 CIs include zero under every calibration mode
    bad = []
    for cal in ("raw", "platt", "platt_pc"):
        for row, (a, b) in TABLE2.items():
            for m in ("brier", "auroc"):
                lo, hi = pw(RCV, cal, a, b, m)["ci95"]
                if not (lo <= 0 <= hi):
                    bad.append(f"{row} {m} {cal} [{lo:+.3f}, {hi:+.3f}]")
    fact("§4.7 | 'CIs of Table 2 include zero under every calibration mode'", not bad,
         "all Brier/AUROC CIs contain 0 (raw, shared Platt, per-condition Platt)",
         "; ".join(bad) or "all contain 0", "repeated_cv.json pairwise.{raw,platt,platt_pc}")
    # ±0.035 statement
    his = [abs(x) for a, b in ((G, L2), (G, L3)) for x in pw(RCV, "raw", a, b, "brier")["ci95"]]
    prose("**The defensible statement:**", f"±{max(his):.3f} Brier", "repeated_cv.json pairwise.raw gated vs (ii),(iii)")


def check_table3():
    hdr = "| Model | Complete Brier | Reduced Brier [CI] |"
    rows = {"(i) Complete-trained, NaN at test": LI, "(ii) NaN-trained": L2, "(iii) ECG-only": L3,
            "(iv) Mean-imputation + per-condition Platt": L4, "HR-only": HR, "Gated + dropout": G,
            "Gated, no dropout": GN, "Non-gated MLP": M}
    src = "baselines.json schemes.loso.models.<model>.raw"
    for row, k in rows.items():
        p = pooled("loso", k)
        cell(hdr, row, 1, n(pooled("loso", k, tier="complete")["brier"]), src + ".complete.pooled")
        cell(hdr, row, 2, pooled_ci("loso", k, "brier"), src + ".reduced")
        cell(hdr, row, 3, pooled_ci("loso", k, "auroc"), src + ".reduced")
        cell(hdr, row, 4, n(p["within_subject_auroc"]), src + ".reduced.pooled")
        cell(hdr, row, 5, pooled_ci("loso", k, "ece_sweep"), src + ".reduced")
        cell(hdr, row, 6, f"{n(p['calibration_slope'], 2)} / {s(p['calibration_intercept'], 2)}", src + ".reduced.pooled")
    # Deployment view
    src2 = "baselines.json loso pooled brier"
    prose("**Deployment view.**", f"scores {n(pooled('loso', LI, tier='complete')['brier'])} when EDA is present", src2)
    prose("**Deployment view.**", f"{n(pooled('loso', L3)['brier'])} when it is missing", src2)
    prose("**Deployment view.**", f"network scores {n(pooled('loso', G, tier='complete')['brier'])} and {n(pooled('loso', G)['brier'])}", src2)
    ok = (pooled("loso", LI, tier="complete")["brier"] <= pooled("loso", G, tier="complete")["brier"]
          and pooled("loso", L3)["brier"] <= pooled("loso", G)["brier"])
    fact("§5.1 | per-configuration system matches or beats gated in both conditions", ok, "True", str(ok), src2)
    # "same picture" in 5-fold: CIs overlap
    for k in (L2, L3, M):
        lo1, hi1 = pci("kfold5", G, "brier"); lo2, hi2 = pci("kfold5", k, "brier")
        fact(f"§5.1 | 5-fold CIs overlap: gated vs {k}", lo1 <= hi2 and lo2 <= hi1, "overlap",
             f"[{lo1:.3f},{hi1:.3f}] vs [{lo2:.3f},{hi2:.3f}]", "baselines.json kfold5 pooled_ci95")


def check_section_5_2():
    hdr = "| Gated width (parameters) | MLP width (parameters) |"
    src = "repeated_cv_capacity.json pairwise.raw"
    for row, (dd, hh) in (("8 (273)", (8, 8)), ("16 (801)", (16, 25)), ("32 (2,625)", (32, 84)), ("64 (9,345)", (64, 300))):
        a, b = f"cap_gated_d{dd}", f"cap_mlp_h{hh}"
        cell(hdr, row, 2, pw_ci(CAP, "raw", a, b, "brier"), src)
        cell(hdr, row, 3, pw_ci(CAP, "raw", a, b, "auroc"), src)
    # cap_gated_d16 reproduces default gated
    fact("§5.2 | cap_gated_d16 == default gated (Brier)",
         abs(pm(CAP, "raw", "cap_gated_d16", "brier_mean") - pm(RCV, "raw", G, "brier_mean")) < 1e-9,
         n(pm(RCV, "raw", G, "brier_mean"), 4), n(pm(CAP, "raw", "cap_gated_d16", "brier_mean"), 4), src)

    hdr = "| Tuned model | Brier | AUROC | ECE_SWEEP |"
    tmap = {"Gated network + dropout": G, "Non-gated MLP": M, "(ii) NaN-trained LightGBM": L2, "(iii) ECG-only LightGBM": L3}
    src = "repeated_cv_tuned.json per_model_reduced.raw.tuned_<model>"
    for row, k in tmap.items():
        for col, met in (("Brier", "brier_mean"), ("AUROC", "auroc_mean"), ("ECE_SWEEP", "ece_sweep_mean")):
            cell(hdr, row, col, n(pm(TUN, "raw", "tuned_" + k, met)), src)
    hdr = "| Comparison (tuned) | ΔBrier [95% CI] | ΔAUROC [95% CI] |"
    src = "repeated_cv_tuned.json pairwise.raw"
    for row, (a, b) in {"Gated vs (ii)": (G, L2), "Gated vs (iii)": (G, L3), "Gated vs MLP": (G, M), "MLP vs (ii)": (M, L2)}.items():
        cell(hdr, row, 1, pw_ci(TUN, "raw", "tuned_" + a, "tuned_" + b, "brier"), src)
        cell(hdr, row, 2, pw_ci(TUN, "raw", "tuned_" + a, "tuned_" + b, "auroc"), src)
    fact("§5.2 | rule fires under tuning (raw and Platt)",
         TUN["falsification_rule"]["raw"]["rule_fires"] and TUN["falsification_rule"]["platt"]["rule_fires"],
         "True, True", f"{TUN['falsification_rule']['raw']['rule_fires']}, {TUN['falsification_rule']['platt']['rule_fires']}",
         "repeated_cv_tuned.json falsification_rule")
    moves = {k: abs(pm(TUN, "raw", "tuned_" + k, "brier_mean") - pm(RCV, "raw", k, "brier_mean")) for k in tmap.values()}
    fact("§5.2/§7 | 'tuning moves no model by more than 0.005 Brier'", max(moves.values()) <= 0.005 + 5e-4,
         "max |Δ| ≤ 0.005 (at 3 dp)", ", ".join(f"{k} {v:.4f}" for k, v in moves.items()),
         "repeated_cv_tuned vs repeated_cv per_model_reduced.raw brier_mean")
    # Abstract: tuned gated vs MLP
    prose("similar at all four sizes and after nested tuning",
          f"{s(pw(TUN,'raw','tuned_'+G,'tuned_'+M,'brier')['mean'])}, {s(pw(TUN,'raw','tuned_'+G,'tuned_'+M,'brier')['ci95'][0])} to {s(pw(TUN,'raw','tuned_'+G,'tuned_'+M,'brier')['ci95'][1])}",
          src)
    # Chosen configs
    cc = TUN["chosen_configs"]
    g_top, g_cnt = max(cc["tuned_" + G].items(), key=lambda kv: kv[1])
    tot = sum(cc["tuned_" + G].values())
    g = json.loads(g_top)
    prose("**Configurations chosen.**", f"embedding {g['embed_dim']}, {g['epochs']} epochs", "repeated_cv_tuned.json chosen_configs")
    prose("**Configurations chosen.**", f"({g_cnt}/{tot} folds)", "repeated_cv_tuned.json chosen_configs")
    fact("§5.2 | gated modal config lr 3e-3, wd 1e-3", g["lr"] == 0.003 and g["weight_decay"] == 0.001,
         "lr 0.003, wd 0.001", str(g), "chosen_configs")
    m_counts = {}
    for k, v in cc["tuned_" + M].items():
        h = json.loads(k)["hidden"]; m_counts[h] = m_counts.get(h, 0) + v
    m_wd = sum(v for k, v in cc["tuned_" + M].items() if json.loads(k)["weight_decay"] == 0.001)
    fact("§5.2 | MLP: largest width (84) most chosen, with weight decay 1e-3",
         max(m_counts, key=m_counts.get) == 84 and m_wd > tot / 2,
         "modal hidden = 84; wd 1e-3 in majority", f"hidden counts {m_counts}; wd=1e-3 in {m_wd}/{tot}", "chosen_configs")
    for k in (L2, L3):
        top = json.loads(max(cc["tuned_" + k].items(), key=lambda kv: kv[1])[0])
        fact(f"§5.2 | {k} modal config: 50 trees, lr 0.05",
             top["n_estimators"] == 50 and top["learning_rate"] == 0.05, "50 trees, lr 0.05", str(top), "chosen_configs")

    # Training-with-missingness bullet (LOSO pooled)
    src = "baselines.json loso reduced pooled brier"
    prose("**Training with missingness is the active ingredient.**", f"Brier {n(pooled('loso', LI)['brier'])}", src)
    prose("**Training with missingness is the active ingredient.**", f"without dropout to {n(pooled('loso', GN)['brier'])}", src)
    aware = [pooled("loso", k)["brier"] for k in (L2, L3, L4, G, M)]
    prose("**Training with missingness is the active ingredient.**", f"lands at {rng(aware, 3)}", src)
    # Dropout bullet
    prose("**Dropout within the gated network.**", pw_ci(RCV, "raw", G, GN, "brier"), "repeated_cv.json pairwise.raw gated vs no-dropout")
    sf = pw(RCV, "raw", G, GN, "subject_sign_flip_brier")
    prose("**Dropout within the gated network.**", f"{sf['n_subjects_A_better']} of {sf['n_subjects']} participants", "repeated_cv.json sign flip")


def check_section_5_3():
    hdr = "| Model | Reduced AUROC [95% CI] | Within-subject AUROC |"
    rows = {"Gated + dropout": G, "HR-only": HR, "Gated, HR removed": GHR,
            "Gated, HR-residualized (HR removed + within-participant residualization)": GRES,
            "(ii) NaN-trained, HR removed": L2HR, "(iii) ECG-only, HR removed": L3HR}
    src = "baselines.json loso raw reduced"
    for row, k in rows.items():
        cell(hdr, row, 1, pooled_ci("loso", k, "auroc"), src)
        cell(hdr, row, 2, n(pooled("loso", k)["within_subject_auroc"]), src)
    prose("**A single feature does most of the work.**", f"reaches {n(pooled('loso', HR)['auroc'], 2)} AUROC", src)
    hrr = [pooled("loso", k)["auroc"] for k in (GHR, L2HR, L3HR)]
    prose("**Removing heart rate costs about half", f"({n(pooled('loso', G)['auroc'], 2)} → {rng(hrr)})", src)
    prose("Removing heart rate reduced AUROC from about", f"about {n(pooled('loso', G)['auroc'], 2)} to {rng(hrr)}", src)
    prose("**Removing within-participant heart-rate covariation", f"({n(pooled('loso', GRES)['auroc'], 2)})", src)
    prose("Its AUROC (0.878) was read as", f"after the fix it is {n(pooled('loso', GRES)['auroc'])}", src)
    # half of above-chance
    frac = [(pooled("loso", G)["auroc"] - a) / (pooled("loso", G)["auroc"] - 0.5) for a in hrr]
    fact("§5.3 | 'about half the above-chance AUROC'", 0.4 <= np.mean(frac) <= 0.6,
         "fraction lost ≈ 0.5", ", ".join(f"{f:.2f}" for f in frac), src)


def _spearman(ps: pd.DataFrame, model: str) -> float:
    d = ps[(ps.tier == "reduced")]
    hr = d.drop_duplicates("subject_id").set_index("subject_id")["delta_mean_nn_ms_stress_minus_baseline"]
    g = d[d.model == model].set_index("subject_id").reindex(hr.index)
    return float(stats.spearmanr(-hr, g["auroc_raw"]).statistic)


def check_section_5_4():
    src = "per_subject_loso.csv (recomputed as in make_figures.forest)"
    d = PS_LOSO[PS_LOSO.tier == "reduced"]
    hr = -d.drop_duplicates("subject_id").set_index("subject_id")["delta_mean_nn_ms_stress_minus_baseline"]
    prose("next to that participant's heart-rate response", f"{hr.min():.0f}–{hr.max():.0f} ms", src)
    rho = {"Gated network": _spearman(PS_LOSO, G), "NaN-trained LightGBM": _spearman(PS_LOSO, L2), "HR-only": _spearman(PS_LOSO, HR)}
    for row, v in rho.items():
        cell("| Model | ρ |", row, 1, n(v, 2), src)
    prose("Per-subject discrimination tracks", f"ρ = {rng(rho.values())}", src)
    prose("Per-participant AUROC tracked each", f"ρ {rng(rho.values())}", src)
    k5_models = [m for m in PS_K5.model.unique()]
    k5 = {m: _spearman(PS_K5, m) for m in k5_models}
    with_hr = {m: v for m, v in k5.items() if "hr_removed" not in m and "hr_resid" not in m}
    prose("The 5-fold check gives", f"ρ = {rng(with_hr.values())}", "per_subject_kfold5.csv, models with heart rate",
          note="Range taken over models that use heart rate. HR-removed models give: "
               + ", ".join(f"{m} {v:+.2f}" for m, v in k5.items() if m not in with_hr))
    strong = hr[hr >= 239].index
    prose("The 7 participants with a drop of", f"The {len(strong)} participants", src)
    fact("§5.4 | 239 ms is the 7th-largest response", sorted(hr.values)[-7] >= 239 > sorted(hr.values)[-8],
         "7th largest ≥ 239 > 8th", f"{sorted(hr.values)[-7]:.1f} / {sorted(hr.values)[-8]:.1f}", src)
    hr_models = [G, M, L2, L3, L4, HR, GN, LI, LZ]
    mins = {m: d[(d.model == m) & d.subject_id.isin(strong)].auroc_raw.min() for m in hr_models}
    fact("§5.4 | strong responders ≥ 0.90 in every model with HR", min(mins.values()) >= 0.895,
         "min ≥ 0.90", ", ".join(f"{m} {v:.2f}" for m, v in mins.items()), src)
    netw = [mins[m] for m in (G, M, HR)]
    prose("score at least 0.90 in every model", f"({rng(netw)} for the networks and HR-only)", src)
    nohr = {m: d[(d.model == m) & d.subject_id.isin(strong)].auroc_raw.min() for m in (GHR, GRES, L2HR, L3HR)}
    prose("With heart rate removed, the worst of those", f"falls to {rng(nohr.values())}", src,
          note=", ".join(f"{m} {v:.2f}" for m, v in nohr.items()))
    weak = hr.sort_values().index[:4]
    prose("In the 4 weakest responders", "(" + ", ".join(x.replace("WESAD_", "") for x in weak), src)
    prose("In the 4 weakest responders", f"{hr[weak].min():.0f}–{hr[weak].max():.0f} ms", src)
    s10 = d[(d.model == L2) & (d.subject_id == "WESAD_S10")].auroc_raw.iloc[0]
    prose("In the 4 weakest responders", f"(AUROC {n(s10, 2)})", src)


def check_section_5_5():
    src = "baselines.json loso reduced pooled"
    p = pooled("loso", LI)
    prose("**The complete-trained LightGBM fails in calibration-in-the-large.**", f"intercept is {s(p['calibration_intercept'], 2)}", src)
    prose("**The complete-trained LightGBM fails in calibration-in-the-large.**", f"({n(p['calibration_slope'], 2)})", src)
    prose("**The complete-trained LightGBM fails in calibration-in-the-large.**", f"(AUROC {n(p['auroc'], 2)})", src)
    better = [pooled("loso", k)["calibration_slope"] for k in (L2, L3, L4, HR, G, M)]
    prose("**Every better model is over-confident", f"slopes are {rng(better)}", src)
    prose("All models were over-confident without EDA", f"slope {rng(better)}", src)
    shared = [pooled(sch, k, "platt")["calibration_slope"] for sch in ("loso", "kfold5") for k in (G, M)]
    prose("**The shared calibrator over-corrects the networks.**", f"({rng(shared)} across schemes)", "baselines.json {loso,kfold5} platt reduced slope, networks")
    hdr = "| Model | Raw: slope / intercept / ECE |"
    for row, k in {"Gated + dropout": G, "Non-gated MLP": M, "(ii) NaN-trained": L2, "(i) Complete-trained": LI}.items():
        r, c = pooled("loso", k, "raw"), pooled("loso", k, "platt_pc")
        cell(hdr, row, 1, f"{n(r['calibration_slope'], 2)} / {s(r['calibration_intercept'], 2)} / {n(r['ece_sweep'])}", src + " raw")
        cell(hdr, row, 2, f"{n(c['calibration_slope'], 2)} / {s(c['calibration_intercept'], 2)} / {n(c['ece_sweep'])}", src + " platt_pc")
        cell(hdr, row, 3, f"{n(r['brier'])} → {n(c['brier'])}", src)
    prose("**Recalibration alone mostly rescues", f"Brier {n(pooled('loso', LI)['brier'])} → {n(pooled('loso', LI, 'platt_pc')['brier'])}", src)
    prose("alone reduced the complete-trained LightGBM's Brier", f"from {n(pooled('loso', LI)['brier'])} to {n(pooled('loso', LI, 'platt_pc')['brier'])}", src)
    prose("That model under-predicted stress when EDA was absent", f"intercept {s(p['calibration_intercept'], 1)}", src)
    # ECE lesson (current numbers)
    allm = list(BAS["schemes"]["loso"]["models"])
    lowest = min(allm, key=lambda k: pooled("loso", k)["ece_sweep"])
    q = pooled("loso", L2HR)
    fact("§5.5 | HR-removed NaN-trained LightGBM has the lowest LOSO ECE_SWEEP", lowest == L2HR, L2HR, lowest, src)
    prose("under LOSO the HR-removed NaN-trained LightGBM", f"({n(q['ece_sweep'])}) at AUROC {n(q['auroc'], 2)}, with slope {n(q['calibration_slope'], 2)}", src)
    # ECE lesson (pre-fix numbers, archived)
    zeros = sum(1 for k, v in PRE_BAS.items() if isinstance(v, dict) and "per_fold" in v
                for f in v["per_fold"] if f["reduced"].get("ece_debiased") == 0.0)
    tot = sum(1 for k, v in PRE_BAS.items() if isinstance(v, dict) and "per_fold" in v for f in v["per_fold"])
    prose("reduced per-fold values exactly 0", f"{zeros}/{tot} reduced", "archive/baselines_2026-10-01_prefix.json")
    ew = {k: np.mean([f["reduced"]["ece_equal_width"] for f in v["per_fold"]])
          for k, v in PRE_BAS.items() if isinstance(v, dict) and "per_fold" in v}
    others = [v for k, v in ew.items() if k not in (GHR,)]
    note("§5.5 | pre-fix equal-width ECE (archive)",
         f"HR-removed gated {ew.get(GHR, float('nan')):.3f}; others " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(ew.items(), key=lambda kv: -kv[1])),
         "archive/baselines_2026-10-01_prefix.json (mean of per-fold reduced ece_equal_width)")
    prose("by conventional equal-width ECE", f"second-worst ({n(ew[GHR])}; only the complete-trained LightGBM was worse at {n(ew[LI])}", "archive/baselines_2026-10-01_prefix.json")


def check_section_5_6():
    src = "baselines.json loso raw reduced pooled aurc"
    a = {k: pooled("loso", k)["aurc"] for k in (G, M, L2, L3)}
    prose("**The pooled curves suggested a difference", f"(gated {n(a[G])}, MLP {n(a[M])})", src)
    prose("**The pooled curves suggested a difference", f"(ii) ({n(a[L2])}) and (iii) ({n(a[L3])})", src)
    hdr = "| Comparison | ΔAURC [95% CI], raw | ΔAURC, per-condition Platt |"
    for row, (x, y) in {"Gated vs (ii)": (G, L2), "Gated vs (iii)": (G, L3), "Gated vs MLP": (G, M)}.items():
        cell(hdr, row, 1, pw_ci(RCV, "raw", x, y, "aurc"), "repeated_cv.json pairwise.raw aurc")
        cell(hdr, row, 2, pw_ci(RCV, "platt_pc", x, y, "aurc"), "repeated_cv.json pairwise.platt_pc aurc")
    # risk at full and 60% coverage (pooled LOSO predictions)
    sys.path.insert(0, str(ROOT))
    from run_metrics import risk_at_coverage
    pred = pd.read_csv(R / "predictions_loso.csv")
    red = pred[pred.tier == "reduced"] if "tier" in pred else pred
    rows = []
    for k in (G, M, L2, L3, L4):
        g = red[red.model == k]
        y, pr = g["label"].values if "label" in g else g["y"].values, g["p_raw"].values
        rows.append((k, risk_at_coverage(y, pr, 1.0), risk_at_coverage(y, pr, 0.6), risk_at_coverage(y, pr, 0.1)))
    full = [r[1] for r in rows]; sixty = [r[2] for r in rows]
    fact("§5.6 | 'about 0.20 at full coverage to about 0.10 at 60%'",
         all(0.15 <= v <= 0.25 for v in full) and all(0.05 <= v <= 0.15 for v in sixty),
         "full ∈ [0.15,0.25], 60% ∈ [0.05,0.15]",
         "; ".join(f"{k}: {f:.3f} → {sx:.3f}" for k, f, sx, _ in rows), "predictions_loso.csv via run_metrics.risk_at_coverage")
    r3 = [r for r in rows if r[0] == L3][0]
    prose("most confident 10% of windows were wrong", f"wrong {round(100 * r3[3])}% of the time", "predictions_loso.csv via run_metrics.risk_at_coverage")
    halves = all(sx <= 0.6 * f for _, f, sx, _ in rows)
    fact("§6 | 'abstention halves the error … at 60% coverage'", halves, "60%-coverage risk ≤ ~half of full",
         "; ".join(f"{k}: ratio {sx / f:.2f}" for k, f, sx, _ in rows), "predictions_loso.csv")


def _log_ratio(log, cond):
    m = re.search(rf"^\s*{re.escape(cond)}\s+.*?([\d.]+)x\s+\[([\d.]+), ([\d.]+)\]", log, re.M)
    return tuple(float(x) for x in m.groups()) if m else None


def check_appendix_b_and_5_7():
    src = "results/locked_comparison_rerun.log"
    lg = re.search(r"Ratio: ([\d.]+)x\s+95% CI \[([\d.]+), ([\d.]+)\]", LOG_LOCK)
    lgbm = tuple(float(x) for x in lg.groups())
    nd = _log_ratio(LOG_LOCK, "no_dropout")
    dp = _log_ratio(LOG_LOCK, "dropout_p0.5")
    hdr = "| Condition | Ratio, mean of per-fold ratios | 95% CI |"
    for row, v in (("Complete-trained LightGBM (NaN at test)", lgbm), ("Gated, no dropout", nd), ("Gated + dropout p = 0.5", dp)):
        cell(hdr, row, 1, f"{v[0]:.2f}×", src)
        cell(hdr, row, 2, f"[{v[1]:.2f}, {v[2]:.2f}]", src)
    prose("On the frozen table the same script gives", f"{lgbm[0]:.2f}× → {nd[0]:.2f}× → {dp[0]:.2f}×", src)
    for row, old in (("Complete-trained LightGBM (NaN at test)", "4.23×"), ("Gated, no dropout", "2.21×"), ("Gated + dropout p = 0.5", "1.61×")):
        cell(hdr, row, 3, old, "archive/e14-preprint-outline_2026-10-01_pre-rerun.md")
        fact(f"App B | earlier-draft {old} present in archived outline", old in ARCH_OUTLINE, old, str(old in ARCH_OUTLINE), "archived outline")
    prose("was an ECE-ratio sequence of", "4.23× → 2.21× → 1.61×", "archive/e14-preprint-outline_2026-10-01_pre-rerun.md")
    # paired test
    tp = float(re.search(r"Paired t-test \(n=5 folds\): t=[-\d.]+, p=([\d.]+)", LOG_LOCK).group(1))
    W, wp = re.search(r"Wilcoxon signed-rank: W=([\d.]+), p=([\d.]+)", LOG_LOCK).groups()
    deltas = [float(x) for x in re.search(r"Per-fold delta \(dropout - no_dropout\): \[(.*?)\]", LOG_LOCK).group(1).replace("'", "").split(",")]
    prose("**Dropout vs no dropout.**", f"{sum(d < 0 for d in deltas)} of 5 folds", src)
    prose("**Dropout vs no dropout.**", f"paired t p = {tp:.3f}", src)
    prose("**Dropout vs no dropout.**", f"W = {float(W):.0f}, p = {float(wp):.3f}", src)
    rates = [_log_ratio(LOG_LOCK, f"p={r}")[0] for r in ("0.2", "0.4", "0.5", "0.6")]
    prose("**Dropout-rate sweep.**", " / ".join(f"{r:.2f}×" for r in rates), src)
    src = "results/locked_per_modality_rerun.log"
    eo, eg = _log_ratio(LOG_MOD, "eda_only"), _log_ratio(LOG_MOD, "ecg_only")
    prose("**Ablation: which modality is dropped in training.**", f"gives {eo[0]:.2f}× [{eo[1]:.2f}, {eo[2]:.2f}]", src)
    prose("**Ablation: which modality is dropped in training.**", f"gives {eg[0]:.2f}× [{eg[1]:.2f}, {eg[2]:.2f}]", src)
    md = [float(x) for x in re.search(r"Per-fold delta \(eda_only - ecg_only\): \[(.*?)\]", LOG_MOD).group(1).replace("'", "").split(",")]
    mp = float(re.search(r"Paired t-test: t=[-\d.]+, p=([\d.]+)", LOG_MOD).group(1))
    prose("**Ablation: which modality is dropped in training.**", f"in {sum(d < 0 for d in md)}/5 folds (p = {mp:.2f})", src)


def check_appendix_a():
    src = "archive/repeated_cv_2026-10-01_prefix.json pairwise"
    p2 = PRE_RCV["pairwise"][f"{G}__vs__{L2}"]; p3 = PRE_RCV["pairwise"][f"{G}__vs__{L3}"]
    cell("| | First (pre-fix", "Gated vs (ii), Brier", 1, f"{s(p2['mean_brier_diff_A_minus_B'])} (B–F p {p2['bouckaert_frank_p_brier']:.2f}", src)
    cell("| | First (pre-fix", "Gated vs (iii), Brier", 1, f"{s(p3['mean_brier_diff_A_minus_B'])} (p {p3['bouckaert_frank_p_brier']:.2f}", src)
    cell("| | First (pre-fix", "Gated vs (ii), Brier", 2, pw_ci(RCV, "raw", G, L2, "brier"), "repeated_cv.json")
    cell("| | First (pre-fix", "Gated vs (iii), Brier", 2, pw_ci(RCV, "raw", G, L3, "brier"), "repeated_cv.json")
    cell("| | First (pre-fix", "Gated vs (ii)/(iii), AUROC", 2, f"gated higher by {n(pw(RCV,'raw',G,L2,'auroc')['mean'])}", "repeated_cv.json")
    ps = [v["bouckaert_frank_p_brier"] for v in PRE_RCV["pairwise"].values()]
    pp = sorted([p2["bouckaert_frank_p_brier"], p3["bouckaert_frank_p_brier"]])
    prose("First rule evaluation's Bouckaert–Frank p-values", f"({pp[0]:.2f}–{pp[1]:.2f})", src,
          note="Range is over gated vs (ii),(iii). All pre-fix pairwise p: " + ", ".join(f"{x:.2f}" for x in ps))
    au = {k: PRE_RCV["per_model_reduced"][k]["auroc_mean"] for k in (G, L2, L3)}
    fact("App A | pre-fix: gated AUROC lower than (ii) and (iii)", au[G] < au[L2] and au[G] < au[L3],
         "gated lower", ", ".join(f"{k} {v:.3f}" for k, v in au.items()), "archive repeated_cv prefix per_model_reduced")
    prose("Its AUROC (0.878) was read as", "(0.878)", "archive/e14-preprint-outline_2026-10-01_pre-rerun.md (hr-resid row: 0.878)")
    fact("App A #6 | 0.878 appears in the archived outline's hr-resid row",
         re.search(r"HR-residualized HRV \|[^\n]*0\.878", ARCH_OUTLINE) is not None, "present", "", "archived outline")


def check_appendix_c():
    src = "results/h2a_subject_level.json"
    h = H2A
    prose("Per-participant mean `coupling_lag_sec`", f"mean {n(h['mean_diff_sec'], 2)} s, 95% CI [{n(h['ci95_sec'][0], 2)}, {s(h['ci95_sec'][1], 2)}]", src)
    prose("d_z = −0.39, 95% CI", f"d_z = {n(h['d_z'], 2)}, 95% CI [{n(h['d_z_ci95'][0], 2)}, {s(h['d_z_ci95'][1], 2)}]", src)
    prose("d_z = −0.39, 95% CI", f"t({h['n_subjects'] - 1}) = {n(h['t'], 2)}, p = {h['p']:.2f}; Wilcoxon p = {h['wilcoxon_p']:.2f}", src)
    prose("participants show a longer lag under stress", f"{h['n_subjects_lag_longer_under_stress']} of {h['n_subjects']}", src)
    prose("**Equivalence.** No smallest effect", f"[{n(h['ci90_sec'][0], 2)}, {s(h['ci90_sec'][1], 2)}] s", src)
    prose("**Equivalence.** No smallest effect", f"±{h['min_equivalence_bound_sec']:.2f} s (d_z > {h['min_equivalence_bound_dz']:.2f})", src)
    w, ld = h["within_subject_sd_lag_baseline_sec"], h["lag_distribution_sec"]
    prose("**Measurement reliability.**", f"SD {w['median']:.1f} s (median across participants; range {w['min']:.1f}–{w['max']:.1f} s)", src)
    prose("**Measurement reliability.**", f"±{h['search_range_sec']:.0f} s search range", src)
    prose("**Measurement reliability.**", f"median {n(ld['median'], 2)} s, IQR {n(ld['q25'], 2)} to {s(ld['q75'], 2)} s", src)
    prose("Window-level, n = 445 (old table)", "d = 0.06, p = 0.94", "archive/e14-preprint-outline_2026-10-01_pre-rerun.md")
    fact("App C | earlier window-level result present in archived outline", "Cohen's d = 0.06, p = 0.94" in ARCH_OUTLINE,
         "present", str("Cohen's d = 0.06, p = 0.94" in ARCH_OUTLINE), "archived outline")


def check_appendix_e():
    src = "results/interpretability_frozen.json group_share"
    hdr = "| Model, condition | Heart rate | Other HRV | EDA | Coupling | Top 3 features |"
    pct = lambda x: f"{round(100 * x)}%"
    rows = {"(ii) NaN-trained LightGBM, complete (SHAP)": f"{L2}|complete",
            "(ii) NaN-trained LightGBM, reduced (SHAP)": f"{L2}|reduced",
            "(iii) ECG-only LightGBM (SHAP)": f"{L3}|reduced",
            "Gated network, complete (Integrated Gradients)": f"{G}|complete",
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
        top3 = ", ".join(nice.get(f, f) for f in INT[key]["rank"][:3])
        cell(hdr, row, 5, top3, "interpretability_frozen.json rank[:3]")
    ce = INT["ig"]["completeness_error_logit"]
    prose("Completeness error", f"median {ce['median']:.3f}, 95th percentile {ce['p95']:.3f}, maximum {ce['max']:.3f}", "interpretability_frozen.json ig")
    prose("**Integrated Gradients settings:**", f"{INT['ig']['steps']}-step {INT['ig']['rule']}", "interpretability_frozen.json ig")
    tops = {k.split("|")[0] + "|" + k.split("|")[1]: v["rank"][0] for k, v in INT.items() if "|" in k}
    fact("App E | heart rate is the top feature everywhere", set(tops.values()) == {"hrv_mean_nn"}, "hrv_mean_nn", str(tops), "interpretability_frozen.json")


def check_appendix_f():
    log = (R / "audit/run_full_audit.log").read_text(encoding="utf-8", errors="replace")
    src = "results/audit/run_full_audit.log"
    m = re.search(r"Verdict: \w+ \(AUC=([\d.]+) CI=\[([\d.]+),", log)
    prose("subject-level identity AUC", f"AUC {float(m.group(1)):.2f}, lower CI {float(m.group(2)):.2f}", src)
    r = float(re.search(r"heterogeneity ratio R:\s+([\d.]+)", log).group(1))
    prose("MMD R =", f"MMD R = {r:.2f}", src)
    note("App F | CORAL drop 0.087", "not located by pattern in the audit log; confirm by eye (step 8 section)", src)


def check_abstract_and_contrib():
    src = "repeated_cv.json pairwise.raw"
    prose("The gated network was indistinguishable from NaN-trained LightGBM",
          f"ΔBrier {s(pw(RCV,'raw',G,L2,'brier')['mean'])}, 95% CI {s(pw(RCV,'raw',G,L2,'brier')['ci95'][0])} to {s(pw(RCV,'raw',G,L2,'brier')['ci95'][1])}", src)
    prose("The gated network was indistinguishable from NaN-trained LightGBM",
          f"({s(pw(RCV,'raw',G,L3,'brier')['mean'])}, {s(pw(RCV,'raw',G,L3,'brier')['ci95'][0])} to {s(pw(RCV,'raw',G,L3,'brier')['ci95'][1])})", src)
    prose("The gated network was indistinguishable from NaN-trained LightGBM",
          f"({s(pw(RCV,'raw',G,M,'brier')['mean'])}, {s(pw(RCV,'raw',G,M,'brier')['ci95'][0])} to {s(pw(RCV,'raw',G,M,'brier')['ci95'][1])}", src)
    prose("That model under-predicted stress when EDA was absent", f"(Brier {n(pooled('loso', LI)['brier'])}", "baselines.json loso")
    prose("Every between-model CI in Table 2 spans roughly", "±0.03 Brier", "repeated_cv.json (see note)",
          note="Half-widths of Table 2 ΔBrier CIs: " + ", ".join(
              f"{r} {(pw(RCV,'raw',a,b,'brier')['ci95'][1]-pw(RCV,'raw',a,b,'brier')['ci95'][0])/2:.3f}" for r, (a, b) in TABLE2.items()))
    prose("the narrowest intervals in the study", pw_ci(RCV, "raw", G, M, "brier"), src)
    prose("the narrowest intervals in the study", pw_ci(RCV, "raw", G, M, "auroc"), src)
    # 'narrowest' — check gated-vs-MLP Brier CI is the narrowest in repeated_cv pairwise raw
    widths = {k: v["brier"]["ci95"][1] - v["brier"]["ci95"][0] for k, v in RCV["pairwise"]["raw"].items()}
    fact("§5.2 | gated vs MLP has the narrowest ΔBrier CI (untuned repeated CV)",
         min(widths, key=widths.get) == f"{G}__vs__{M}", f"{G}__vs__{M}", min(widths, key=widths.get), src,
         note="Capacity-sweep CIs at 32/64 widths are narrower (±0.016–0.018); 'in the study' is true only for the untuned comparison set.")
    capw = {k: (v["brier"]["ci95"][1] - v["brier"]["ci95"][0]) / 2 for k, v in CAP["pairwise"]["raw"].items()}
    note("§5.2 | capacity-sweep ΔBrier CI half-widths", ", ".join(f"{k} {v:.3f}" for k, v in capw.items()), "repeated_cv_capacity.json")


# ---------------------------------------------------------------------------
# Untraced numbers
# ---------------------------------------------------------------------------

NUM = re.compile(r"(?<![\w.])[+\u2212\-]?\d[\d,]*(?:\.\d+)?(?:×|%)?")
SKIP_SECTIONS = ("2. Related Work", "Appendix H", "Contributions")


def untraced():
    out = {}
    for i, l in enumerate(LINES):
        text = l.replace("**", "")
        sec = section_of(i)
        if any(sec.startswith(x) for x in SKIP_SECTIONS):
            continue
        spans = COVERED.get(i, [])
        for m in NUM.finditer(text):
            tok = m.group(0)
            if not re.search(r"\.\d|×|%", tok):
                continue                      # integers (counts, years, section numbers) are skipped
            if text[max(0, m.start() - 1)] in "§#" or re.match(r"\d+\.\d+ ", text[m.start():m.start() + 6] or "") and sec == tok:
                continue
            if any(a <= m.start() < b for a, b in spans):
                continue
            out.setdefault(sec, []).append((i + 1, tok))
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    checks = [check_data_section, check_methods, check_table1, check_table2_and_rule, check_table3,
              check_section_5_2, check_section_5_3, check_section_5_4, check_section_5_5, check_section_5_6,
              check_appendix_b_and_5_7, check_appendix_a, check_appendix_c, check_appendix_e, check_appendix_f,
              check_abstract_and_contrib]
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
           "Decimal, percentage and ratio tokens that no keyed check covered (§2 Related Work, Contributions and Appendix H are skipped; integers are skipped). "
           "Most are restatements of a checked value elsewhere, external facts (WESAD demographics, citations), or descriptive prose. "
           "Each should be either covered by a new check or confirmed by eye.", ""]
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
