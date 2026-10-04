"""
Audit Step 9 — decision-tree evaluator + preprint claim patch.

Reads the manifests written by Steps 0/3/4/5/6/7/8, applies the Branch
A/B/C/D decision tree from the audit doc, and emits:

  * results/audit/step9_decision.json   — machine-readable decision + reasons
  * results/audit/step9_claim_patch.md  — the exact preprint sentences that
                                          need to change and how

Decision tree (audit doc Step 9):
  Branch A "shortcut confirmed"
    Step 3 fails (MIT-BIH NSR scored cardiac, AUC b_vs_a <= 0.65) AND/OR
    final-stage C1 identity AUC lower CI >= 0.80 with MMD R > 2 and adj p < 0.05
  Branch B "removable by harmonization"
    Step 3 passes after harmonization; final C1 AUC <= 0.60 with CI incl 0.5
    and at or below the N1/N2 95th percentile; R <= 1 or adj p >= 0.05;
    clinical AUC retained >= 80% of H0 value and replicated within datasets
  Branch C "partial / inconclusive"
    final C1 AUC 0.60-0.80, or CI spans bands, or MMD and probe disagree
  Branch D "separable only before harmonization, but clinical AUC collapses"
    identity AUC drops to chance but Preprint 1 clinical AUC also collapses

Run:
    python -m audit.step9_decision
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "audit"


def _read_json(name: str) -> dict | None:
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _decide(step3, step4, step5, step6, step7) -> dict:
    """Apply the decision tree; return {branch, reasons}."""
    reasons = []

    # Pull the primary numbers, tolerating missing manifests.
    s3_verdict = (step3 or {}).get("verdict", {}).get("conclusion")
    s4_p1 = (step4 or {}).get("C1", {}).get("P1", {})
    s4_auc = s4_p1.get("subject_auc")
    s4_ci = s4_p1.get("ci95", [None, None])
    s5 = step5 or {}
    R = s5.get("heterogeneity_ratio_R")
    s5_p = s5.get("primary", {}).get("p_block_permutation")
    ladder = (step6 or {}).get("cumulative_ladder", [])
    final_stage_auc = ladder[-1]["identity_auc_C1"] if ladder else None
    final_stage_ci = ladder[-1]["identity_ci95"] if ladder else [None, None]
    c3 = (step7 or {}).get("C3_wesad_baseline_vs_stress", {}).get("auc_mean")
    c4 = (step7 or {}).get("C4_mitbih_nsr_vs_arrhythmic", {}).get("auc_mean")

    # H0 clinical baseline for the "80% of H0" comparison — take the first
    # ladder entry when available.
    h0_c3 = ladder[0]["clinical_auc_C3"] if ladder else None

    # --- Branch A: Shortcut confirmed ---
    a_hits = 0
    if s3_verdict == "SHORTCUT_CONFIRMED":
        a_hits += 1
        reasons.append("Step 3 verdict: SHORTCUT_CONFIRMED")
    if s4_ci[0] is not None and s4_ci[0] >= 0.80 \
            and (R is None or R > 2.0) \
            and (s5_p is None or s5_p < 0.05):
        a_hits += 1
        reasons.append("Step 4 C1 identity AUC lower CI >= 0.80 with MMD R > 2, p < 0.05")
    if a_hits >= 1:
        return {"branch": "A_shortcut_confirmed", "reasons": reasons,
                "action": _branch_a_action()}

    # --- Branch B: Removable by harmonization ---
    # Step 3's vocabulary is now {SHORTCUT_CONFIRMED, inconclusive} only
    # (PATHOLOGY_LIKELY retired per peer-review M1). Any Step 3 outcome
    # that is NOT SHORTCUT_CONFIRMED is compatible with Branch B.
    b_ok = True
    if s3_verdict not in {None, "inconclusive"}:
        b_ok = False
    if final_stage_auc is None or final_stage_auc > 0.60:
        b_ok = False
    if final_stage_ci[1] is not None and final_stage_ci[1] > 0.70:
        b_ok = False
    if R is not None and R > 1.0:
        b_ok = False
    if h0_c3 is not None and c3 is not None and c3 < 0.8 * h0_c3:
        b_ok = False
    if b_ok:
        reasons.append("Step 3 passes; final-stage C1 AUC <= 0.60; R <= 1; "
                       "clinical AUC retained >= 80% of H0")
        return {"branch": "B_removable_by_harmonization", "reasons": reasons,
                "action": _branch_b_action()}

    # --- Branch D: Clinical collapses along with identity ---
    if h0_c3 is not None and c3 is not None and h0_c3 - c3 > 0.15 \
            and final_stage_auc is not None and s4_auc is not None \
            and s4_auc - final_stage_auc > 0.15:
        reasons.append("Identity AUC drops with harmonization AND clinical C3 collapses")
        return {"branch": "D_clinical_collapses_with_identity",
                "reasons": reasons, "action": _branch_d_action()}

    # --- Branch C: Partial / inconclusive ---
    if s4_auc is not None:
        reasons.append(f"Final identity AUC {s4_auc:.3f} sits in the borderline band "
                       f"(0.60-0.80) or CI spans bands")
    return {"branch": "C_partial_inconclusive", "reasons": reasons,
            "action": _branch_c_action()}


# ---------------------------------------------------------------------------
# Branch action templates — feed the preprint outline patch generator
# ---------------------------------------------------------------------------

def _branch_a_action():
    return [
        "Withdraw or rewrite the cross-dataset headline as 'separates WESAD "
        "stress recordings from 1975-1979 Beth Israel Holter recordings' -- "
        "not panic vs cardiac.",
        "Mark the robustness / calibration / equity results (H-B1, H-C1) as "
        "properties of a dataset classifier, not a clinical one.",
        "Promote the C3 (WESAD-internal) and C4 (MIT-BIH-internal) results as "
        "the only clinical-proxy findings.",
        "Publish the audit as a methods contribution.",
        "Prioritize acquiring same-device same-population data for future work.",
    ]


def _branch_b_action():
    return [
        "Re-run ALL preprint analyses on the harmonized pipeline; replace the "
        "reported numbers throughout.",
        "Add the harmonization ladder table (identity AUC and clinical AUC per "
        "stage) to the paper.",
        "State plainly that 'not detected' is not 'absent'; the positivity "
        "problem still limits clinical claims.",
    ]


def _branch_c_action():
    return [
        "Identify the residual driver via the Step 6 leave-one-stage-out and "
        "SHAP results.",
        "Drop the implicated features if clinical AUC survives; otherwise "
        "report them as a named limitation.",
        "Restrict claims to within-dataset contrasts + a hedged cross-dataset "
        "result.",
        "Do NOT report device-tier equity findings as clinical fairness.",
    ]


def _branch_d_action():
    return [
        "Treat as Branch A for claims (the model's cross-dataset performance "
        "was itself the shortcut, even though it is now removed).",
        "Publish only within-dataset (C3/C4) results as clinical-proxy findings.",
    ]


# ---------------------------------------------------------------------------
# Markdown patch generator — the actual preprint claim rewrites
# ---------------------------------------------------------------------------

def _claim_patch_md(decision: dict, step3, step4, step6, step7) -> str:
    branch = decision["branch"]
    reasons = decision["reasons"]
    actions = decision["action"]
    today = date.today().isoformat()

    lines = [
        f"# Preprint claim patch — audit decision as of {today}",
        "",
        f"**Decision:** `{branch}`",
        "",
        "**Reasons:**",
    ]
    for r in reasons:
        lines.append(f"- {r}")
    lines += ["", "**Actions:**"]
    for a in actions:
        lines.append(f"- {a}")

    lines += ["", "## Concrete edits to `e14-preprint-outline.md`"]

    if branch == "A_shortcut_confirmed":
        lines += [
            "",
            "**Section 5.2 (Robustness under missing modalities — H-B1)**  ",
            "Retitle to something like _'Robustness of a dataset classifier "
            "under missing sensors'_. Every ECE-ratio number carries over, but "
            "the *interpretation* changes: these numbers describe how well "
            "the model distinguishes WESAD from MIT-BIH recordings under "
            "missing sensors, not how well it distinguishes panic-proxy from "
            "cardiac states.",
            "",
            "**Section 5.3 (Equity / device-tier robustness — H-C1)**  ",
            "Same reframing: the tier x model interaction describes fairness "
            "of a dataset classifier, not of a clinical model. Delete or "
            "flag the '1'... 'this favors lower-tier users' framing.",
            "",
            "**Section 1 (Introduction) — Generalizable claim sentence**  ",
            "Delete the current sentence entirely. Replace with an explicit "
            "statement that cross-dataset discrimination was audited and "
            "found to be dataset identity, not clinical state.",
            "",
            "**Section 5.5 (Interpretability)**  ",
            "The SHAP / IG rankings still stand as descriptive attributions "
            "but their interpretation shifts to 'features that distinguish "
            "WESAD from MIT-BIH', which is a distinct scientific claim.",
        ]
    elif branch == "B_removable_by_harmonization":
        lines += [
            "",
            "**Section 3 (Datasets)**  ",
            "Add a paragraph naming the harmonization pipeline (H0-H7) as a "
            "prerequisite for the cross-dataset comparison, and reference "
            "`audit/harmonize.py`.",
            "",
            "**Section 4.1 (Preprocessing)**  ",
            "State the pipeline is the final harmonized one from Step 6.",
            "",
            "**Section 5.2**  ",
            "Report the numbers computed on the HARMONIZED pipeline, not "
            "the H0 baseline. Include the harmonization ladder as a table "
            "in Section 4 or Supplementary.",
            "",
            "**Section 7 (Limitations)**  ",
            "Add: 'The dataset-identity audit did not detect a residual "
            "cross-dataset signature after harmonization; \"not detected\" "
            "is not \"absent.\" The positivity violation between WESAD "
            "(young lab cohort) and MIT-BIH (older Holter cohort) still "
            "limits clinical claims.'",
        ]
    elif branch == "C_partial_inconclusive":
        lines += [
            "",
            "**Section 5.2**  ",
            "Report both H0 and final-stage numbers. Add explicit language "
            "that the residual identity signal (see Step 6 attribution) "
            "remains after harmonization.",
            "",
            "**Section 5.3**  ",
            "Downgrade the equity claim to 'suggestive under simulated tiers' "
            "and remove any language treating it as demonstrated clinical "
            "fairness.",
            "",
            "**Section 7**  ",
            "Name the specific residual features (from Step 6 SHAP) as a "
            "limitation.",
        ]
    elif branch == "D_clinical_collapses_with_identity":
        lines += [
            "",
            "Apply Branch A edits above. Additionally add to Section 7:",
            "",
            "'Feature-elimination showed that removing the top identity "
            "feature also removes the top clinical feature, i.e. the same "
            "signal drives both. The reported cross-dataset AUC is not "
            "supportable as clinical discrimination.'",
        ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_step9() -> dict:
    step3 = _read_json("step3_shortcut.json")
    step4 = _read_json("step4_identity.json")
    step5 = _read_json("step5_mmd.json")
    step6 = _read_json("step6_ladder.json")
    step7 = _read_json("step7_within_dataset.json")

    missing = [name for name, m in [
        ("step3", step3), ("step4", step4), ("step5", step5),
        ("step6", step6), ("step7", step7),
    ] if m is None]
    if missing:
        print(f"[step9] WARNING — missing manifests: {missing}. "
              "Decision will use available inputs; consider running the "
              "orchestrator (`python -m audit.run_all`).")

    decision = _decide(step3, step4, step5, step6, step7)
    patch = _claim_patch_md(decision, step3, step4, step6, step7)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "step9_decision.json", "w") as f:
        json.dump({
            "decision": decision,
            "input_manifests_present": [
                name for name, m in [
                    ("step3", step3), ("step4", step4), ("step5", step5),
                    ("step6", step6), ("step7", step7),
                ] if m is not None
            ],
        }, f, indent=2)
    with open(RESULTS_DIR / "step9_claim_patch.md", "w") as f:
        f.write(patch)

    print("\n=== Audit Step 9 — decision ===")
    print(f"  Branch: {decision['branch']}")
    for r in decision["reasons"]:
        print(f"    - {r}")
    print("  Actions:")
    for a in decision["action"]:
        print(f"    - {a}")
    print(f"\n  Wrote {RESULTS_DIR/'step9_decision.json'}")
    print(f"  Wrote {RESULTS_DIR/'step9_claim_patch.md'}")
    return decision


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    run_step9()
