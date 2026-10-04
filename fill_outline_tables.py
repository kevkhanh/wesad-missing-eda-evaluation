"""
Fill the ⟦TABLE…⟧ markers in e14-preprint-outline.md from the results files
(round 4). Every table cell is formatted from a results key, never typed by
hand, and check_numbers.py re-verifies the same cells afterwards.

Usage: python fill_outline_tables.py   (idempotent: only replaces markers
that are still present)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
OUTLINE = ROOT / "e14-preprint-outline.md"
MINUS = "−"


def J(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def n(x, d=3):
    return f"{x:.{d}f}".replace("-", MINUS)


def s(x, d=3):
    return f"{x:+.{d}f}".replace("-", MINUS)


def ci(e, d=3, key="ci95"):
    return f"{s(e['mean'], d)} [{s(e[key][0], d)}, {s(e[key][1], d)}]"


def ci_only(lo_hi, d=3):
    return f"[{s(lo_hi[0], d)}, {s(lo_hi[1], d)}]"


def p3(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


G, GN = "gated_dropout_p0.5", "gated_no_dropout"
M, MN = "mlp_nongated_dropout_p0.5", "mlp_nongated_no_dropout"
L1, L2, L3, L4, HR = ("lgbm_nan_at_test", "lgbm_nan_training_p0.5", "lgbm_ecg_only_per_config",
                      "lgbm_mean_impute_recal", "hr_only")
LABEL = {G: "Gated network + dropout (gate, masking)", M: "Non-gated MLP, same masking (no gate, masking)",
         GN: "Gated network, no dropout (gate, no masking)", MN: "Non-gated MLP, no masking (no gate, no masking)",
         L1: "(i) Complete-trained LightGBM, NaN at test", L2: "(ii) NaN-trained LightGBM",
         L3: "(iii) ECG-only LightGBM", L4: "(iv) Mean-imputation + per-condition Platt", HR: "HR-only"}
SHORT = {G: "Gated", M: "MLP", GN: "Gated, no masking", MN: "MLP, no masking", L1: "(i)", L2: "(ii)",
         L3: "(iii)", L4: "(iv)", HR: "HR-only"}

RCV = J("repeated_cv.json")


def table1():
    rows = []
    for m in (G, M, GN, MN, L1, L2, L3, L4, HR):
        r, c = RCV["per_model_reduced"]["raw"][m], RCV["per_model_complete"]["raw"][m]
        rows.append(f"| {LABEL[m]} | {n(r['brier_mean'])} ({n(r['brier_sd'])}) | {n(r['auroc_mean'])} "
                    f"({n(r['auroc_sd'])}) | {n(r['ece_sweep_mean'])} | {n(r['log_loss_mean'])} | "
                    f"{n(c['brier_mean'])} | {n(c['auroc_mean'])} |")
    return "\n".join(rows)


T2 = [(G, L2), (G, L3), (G, L4), (G, HR), (G, M), (M, L2), (G, GN), (M, MN), (GN, MN), (G, L1), (L2, L1)]


def _pw(a, b, tier="reduced", calib="raw", src=RCV):
    key = "pairwise" if tier == "reduced" else "pairwise_complete"
    return src[key][calib][f"{a}__vs__{b}"]


def table2():
    rows = []
    for a, b in T2:
        e = _pw(a, b)
        br, pb = e["brier"], e["participant"]["brier"]
        rows.append(f"| {SHORT[a]} vs {SHORT[b]} | {ci(br)} | {ci_only(br['df_subjects']['ci95'])} | "
                    f"{ci(pb)} | {pb['n_A_better']}/15 | {ci(e['auroc'])} |")
    return "\n".join(rows)


def table3eq():
    rows = []
    for a, b in [(G, M), (G, L2), (G, L3), (M, L2), (G, L4)]:
        e = _pw(a, b)
        br, pb, au = e["brier"], e["participant"]["brier"], e["auroc"]
        rows.append(f"| {SHORT[a]} vs {SHORT[b]} | {p3(br['tost_p'])} / {p3(br['df_subjects']['tost_p'])} / "
                    f"{p3(pb['tost_p'])} | {n(br['equivalence_bound_attained'])} / "
                    f"{n(pb['equivalence_bound_attained'])} | {n(br['rope_prob'], 2)} | {p3(au['tost_p'])} | "
                    f"{n(au['equivalence_bound_attained'])} |")
    return "\n".join(rows)


def ruletable():
    bas = J("baselines.json")["schemes"]
    f = lambda b: "fires" if b else "does not fire"
    rows = []
    for calib, lab in (("raw", "Raw"), ("platt", "Shared Platt"), ("platt_pc", "Per-condition Platt")):
        r = RCV["falsification_rule"][calib]
        rows.append(f"| {lab} | {f(r['rule_fires'])} | {f(r['rule_fires_without_ece'])} | "
                    f"{f(bas['loso']['falsification_rule'][calib]['rule_fires'])} | "
                    f"{f(bas['kfold5']['falsification_rule'][calib]['rule_fires'])} |")
    return "\n  ".join(rows)


def table5():
    rows = []
    items = [("Interaction (G − G_nomask) − (M − M_nomask)", ("c", "interaction_gate_x_masking")),
             ("Main effect of gate", ("c", "main_effect_gate")),
             ("Main effect of masking", ("c", "main_effect_masking")),
             ("Gate, never masked: gated vs MLP, no masking", ("p", (GN, MN))),
             ("Gate, masked: gated vs MLP", ("p", (G, M))),
             ("Masking, gated network", ("p", (G, GN))),
             ("Masking, MLP", ("p", (M, MN)))]
    for lab, (kind, key) in items:
        if kind == "c":
            raw, pc = RCV["contrasts"]["raw"][key], RCV["contrasts"]["platt_pc"][key]
        else:
            raw, pc = _pw(*key), _pw(*key, calib="platt_pc")
        pb = raw["participant"]["brier"]
        rows.append(f"| {lab} | {ci(raw['brier'])} | {ci(pb)} ({pb['n_A_better']}/15) | {ci(pc['brier'])} |")
    return "\n".join(rows)


def table6():
    rows = []
    for a, b in [(G, L1), (G, L2), (M, L2), (L2, L1), (G, M), (GN, MN)]:
        e = _pw(a, b, tier="complete")
        pb = e["participant"]["brier"]
        rows.append(f"| {SHORT[a]} vs {SHORT[b]} | {ci(e['brier'])} | {ci(pb)} ({pb['n_A_better']}/15) | "
                    f"{ci(e['auroc'])} |")
    return "\n".join(rows)


def table7():
    rows = [f"| Default (Table 2) | {ci(_pw(G, M)['brier'])} | {ci(_pw(G, L2)['brier'])} | |"]
    cap = J("repeated_cv_capacity.json")
    for d, h, pg, pm in [(8, 8, 273, 281), (16, 25, 801, 808), (32, 84, 2625, 2637), (64, 300, 9345, 9333)]:
        e = cap["pairwise"]["raw"][f"cap_gated_d{d}__vs__cap_mlp_h{h}"]["brier"]
        rows.append(f"| Capacity: {pg:,} vs {pm:,} parameters | {ci(e)} | — | TOST p {p3(e['tost_p'])} |")
    tun = J("repeated_cv_tuned.json")["pairwise"]["raw"]
    rows.append(f"| Nested tuning | {ci(tun['tuned_gated_dropout_p0.5__vs__tuned_mlp_nongated_dropout_p0.5']['brier'])} | "
                f"{ci(tun['tuned_gated_dropout_p0.5__vs__tuned_lgbm_nan_training_p0.5']['brier'])} | |")
    mr = J("repeated_cv_maskrate.json")["pairwise"]["raw"]
    for p in (0.1, 0.25, 0.5, 0.75):
        rows.append(f"| Masking rate {p} | {ci(mr[f'maskrate_gated_p{p}__vs__maskrate_mlp_p{p}']['brier'])} | "
                    f"{ci(mr[f'maskrate_gated_p{p}__vs__maskrate_lgbm_p{p}']['brier'])} | |")
    hs = J("repeated_cv_hrv_subset.json")["pairwise"]["raw"]
    rows.append(f"| HRV = mean NN + RMSSD only | "
                f"{ci(hs['hrvsub_gated_dropout_p0.5__vs__hrvsub_mlp_nongated_dropout_p0.5']['brier'])} | "
                f"{ci(hs['hrvsub_gated_dropout_p0.5__vs__hrvsub_lgbm_nan_training_p0.5']['brier'])} | rule does not fire (raw) |")
    for tr, lab in (("subjnorm_all", "Per-subject z-score (label-free)"),
                    ("subjnorm_baseline", "Per-subject z-score on baseline windows (PULSE-style)")):
        r = J(f"repeated_cv_{tr}.json")
        rows.append(f"| {lab} | {ci(r['pairwise']['raw'][f'{G}__vs__{M}']['brier'])} | "
                    f"{ci(r['pairwise']['raw'][f'{G}__vs__{L2}']['brier'])} | gated reduced AUROC "
                    f"{n(r['per_model_reduced']['raw'][G]['auroc_mean'])} |")
    if (R / "repeated_cv_wrist.json").exists():
        w = J("repeated_cv_wrist.json")
        rows.append(f"| Wrist E4 features (§5.10) | {ci(w['pairwise']['raw'][f'{G}__vs__{M}']['brier'])} | "
                    f"{ci(w['pairwise']['raw'][f'{G}__vs__{L2}']['brier'])} | gated reduced AUROC "
                    f"{n(w['per_model_reduced']['raw'][G]['auroc_mean'])} |")
    return "\n".join(rows)


def table8():
    loso = J("baselines.json")["schemes"]["loso"]["models"]
    rows = []
    for m, lab in [(G, "Gated + dropout"), (M, "Non-gated MLP"), (L2, "(ii) NaN-trained"),
                   (L3, "(iii) ECG-only"), (HR, "HR-only"), ("gated_dropout_hr_removed", "Gated, HR removed"),
                   ("gated_dropout_hr_resid", "Gated, HR-residualised (HR removed + within-participant residualisation)"),
                   ("lgbm_nan_training_hr_removed", "(ii) NaN-trained, HR removed"),
                   ("lgbm_ecg_only_hr_removed", "(iii) ECG-only, HR removed")]:
        e = loso[m]["raw"]["reduced"]
        p, b = e["pooled"], e["pooled_ci95_bca"]
        rows.append(f"| {lab} | {n(p['auroc'])} [{n(b['auroc'][0])}, {n(b['auroc'][1])}] | "
                    f"{n(p['within_subject_auroc'])} [{n(b['within_subject_auroc'][0])}, "
                    f"{n(b['within_subject_auroc'][1])}] |")
    return "\n".join(rows)


def caltable():
    loso = J("baselines.json")["schemes"]["loso"]["models"]
    rows = []
    for m, lab in [(G, "Gated + dropout"), (M, "Non-gated MLP"), (L2, "(ii) NaN-trained"),
                   (L1, "**(i) Complete-trained**")]:
        r, pc = loso[m]["raw"]["reduced"]["pooled"], loso[m]["platt_pc"]["reduced"]["pooled"]
        rows.append(f"| {lab} | {n(r['calibration_slope'], 2)} / {s(r['calibration_intercept'], 2)} / "
                    f"{n(r['ece_sweep'])} | {n(pc['calibration_slope'], 2)} / {s(pc['calibration_intercept'], 2)} / "
                    f"{n(pc['ece_sweep'])} | {n(r['brier'])} → {n(pc['brier'])} |")
    return "\n  ".join(rows)


def aurctable():
    rows = []
    for a, b in [(G, L2), (G, L3), (G, M)]:
        e = _pw(a, b)
        rows.append(f"| {SHORT[a]} vs {SHORT[b]} | {ci(e['aurc'])} | {ci(e['e_aurc'])} |")
    return "\n  ".join(rows)


def rhotable():
    hr = J("participant_analyses.json")["hr_response"]["per_model"]
    rows = []
    for k, lab in [("gated_hr_removed", "Gated, HR removed"), ("ii_hr_removed", "(ii), HR removed"),
                   ("iii_hr_removed", "(iii), HR removed"), ("gated", "Gated network"),
                   ("mlp", "Non-gated MLP"), ("ii", "(ii) NaN-trained"), ("iii", "(iii) ECG-only"),
                   ("hr_only", "HR-only")]:
        a, b = hr[k]["raw_response"], hr[k]["standardised_response"]
        rows.append(f"| {lab} | {n(a['rho'], 2)} [{n(a['fisher_z_ci95'][0], 2)}, {n(a['fisher_z_ci95'][1], 2)}] | "
                    f"{n(b['rho'], 2)} [{n(b['fisher_z_ci95'][0], 2)}, {n(b['fisher_z_ci95'][1], 2)}] |")
    return "\n  ".join(rows)


OCN = {"gated_dropout_p0.5": "Gated + dropout", "mlp_nongated_dropout_p0.5": "Non-gated MLP",
       "lgbm_nan_training_p0.5": "(ii) NaN-trained", "lgbm_ecg_only_per_config": "(iii) ECG-only",
       "hr_only": "HR-only", "gated_dropout_hr_removed": "Gated, HR removed",
       "lgbm_nan_training_hr_removed": "(ii), HR removed", "lgbm_ecg_only_hr_removed": "(iii), HR removed",
       "lgbm_nan_at_test": "(i) Complete-trained", "lgbm_eda_only": "EDA-only LightGBM",
       "gated_no_dropout": "Gated network (no masking needed: EDA present)"}


def table_a():
    a = J("order_confound.json")["A_stress_vs_amusement"]
    rows = []
    for m in ["gated_dropout_p0.5", "mlp_nongated_dropout_p0.5", "lgbm_nan_training_p0.5", "lgbm_ecg_only_per_config",
              "hr_only", "gated_dropout_hr_removed", "lgbm_nan_training_hr_removed", "lgbm_ecg_only_hr_removed"]:
        c, r = a[m]["complete"], a[m]["reduced"]
        bp = r["within_subject_auroc_by_protocol"]
        rows.append(f"| {OCN[m]} | {n(c['pooled_auroc'])} | {n(r['pooled_auroc'])} [{n(r['pooled_auroc_ci95'][0])}, "
                    f"{n(r['pooled_auroc_ci95'][1])}] | {n(r['within_subject_auroc'])} [{n(r['within_subject_auroc_ci95'][0])}, "
                    f"{n(r['within_subject_auroc_ci95'][1])}] | {n(bp['stress_before_amusement'])} / {n(bp['stress_after_amusement'])} |")
    return chr(10).join(rows)


def table_b():
    b = J("order_confound.json")["B_negative_control"]
    rows = []
    spec = [("gated_dropout_p0.5", "reduced", "EDA missing"), ("mlp_nongated_dropout_p0.5", "reduced", "EDA missing"),
            ("lgbm_nan_training_p0.5", "reduced", "EDA missing"), ("lgbm_ecg_only_per_config", "reduced", "ECG only"),
            ("hr_only", "reduced", "HR only"), ("gated_dropout_p0.5", "complete", "EDA present"),
            ("lgbm_nan_training_p0.5", "complete", "EDA present"), ("lgbm_nan_at_test", "complete", "EDA present"),
            ("lgbm_nan_at_test", "reduced", "EDA missing")]
    for m, tier, lab in spec:
        r = b[m][tier]
        rows.append(f"| {OCN[m]}, {lab} | " + " | ".join(n(r[c]["frac_pred_stress"], 2) for c in
                    ("baseline", "stress", "amusement", "meditation_1", "meditation_2")) + " |")
    return chr(10).join(rows)


def table_c():
    c = J("order_confound.json")["C_baseline_vs_meditation"]
    rows = []
    for m in ["lgbm_eda_only", "hr_only", "lgbm_ecg_only_per_config", "lgbm_nan_at_test", "gated_no_dropout"]:
        r = c[m]["complete"]
        rows.append(f"| {OCN[m]} | {n(r['pooled_auroc'])} | {n(r['within_subject_auroc'])} "
                    f"[{n(r['within_subject_auroc_ci95'][0])}, {n(r['within_subject_auroc_ci95'][1])}] |")
    return chr(10).join(rows)


FILLERS = {"⟦TABLE1⟧": table1, "⟦TABLE2⟧": table2, "⟦TABLE3EQ⟧": table3eq, "⟦RULETABLE⟧": ruletable,
           "⟦TABLE5⟧": table5, "⟦TABLE6⟧": table6, "⟦TABLE7⟧": table7, "⟦TABLE8⟧": table8,
           "⟦CALTABLE⟧": caltable, "⟦AURCTABLE⟧": aurctable, "⟦RHOTABLE⟧": rhotable,
           "⟦TABLE_A⟧": table_a, "⟦TABLE_B⟧": table_b, "⟦TABLE_C⟧": table_c}


def main():
    text = OUTLINE.read_text(encoding="utf-8")
    for marker, fn in FILLERS.items():
        if marker in text:
            try:
                text = text.replace(marker, fn())
                print("filled", marker)
            except (FileNotFoundError, KeyError) as e:
                print("skipped", marker, type(e).__name__, e)
    OUTLINE.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
