# Preprint claim patch — audit decision as of 2026-10-02

**Decision:** `C_partial_inconclusive`

**Reasons:**
- Final identity AUC 0.000 sits in the borderline band (0.60-0.80) or CI spans bands

**Actions:**
- Identify the residual driver via the Step 6 leave-one-stage-out and SHAP results.
- Drop the implicated features if clinical AUC survives; otherwise report them as a named limitation.
- Restrict claims to within-dataset contrasts + a hedged cross-dataset result.
- Do NOT report device-tier equity findings as clinical fairness.

## Concrete edits to `e14-preprint-outline.md`

**Section 5.2**  
Report both H0 and final-stage numbers. Add explicit language that the residual identity signal (see Step 6 attribution) remains after harmonization.

**Section 5.3**  
Downgrade the equity claim to 'suggestive under simulated tiers' and remove any language treating it as demonstrated clinical fairness.

**Section 7**  
Name the specific residual features (from Step 6 SHAP) as a limitation.
