"""
Audit orchestrator — Steps 0 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9.

Each step writes its own manifest under results/audit/. Step 9 reads them
and emits the decision + preprint claim patch.

Run:
    python -m audit.run_all              # full run (slow: MMD + ladder)
    python -m audit.run_all --fast       # smoke pass: reduced permutations,
                                          # subset of stages
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit.step0_freeze import run_step0
from audit.step3_shortcut import run_step3
from audit.step4_identity_probe import run_step4
from audit.step5_mmd import run_step5
from audit.step6_harmonization import run_step6
from audit.step7_within_dataset import run_step7
from audit.step8_alignment import run_step8
from audit.step9_decision import run_step9


def _hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_all(fast: bool = False, ladder_stages=None) -> dict:
    out = {}
    _hdr("STEP 0 — freeze + Driver #1")
    out["step0"] = run_step0()
    _hdr("STEP 3 — production-model shortcut test")
    out["step3"] = run_step3()
    _hdr("STEP 4 — dataset-identity probe (P1-P3)")
    out["step4"] = run_step4(fast=fast)
    _hdr("STEP 5 — MMD block-permutation test")
    out["step5"] = run_step5(fast=fast)
    _hdr("STEP 6 — harmonization ladder")
    stages = ladder_stages or (["H0", "H1", "H2", "H6"] if fast else None)
    out["step6"] = run_step6(stages=stages, leave_one_out=not fast, fast=fast)
    _hdr("STEP 7 — within-dataset clinical re-validation")
    out["step7"] = run_step7(fast=fast)
    _hdr("STEP 8 — conditional CORAL alignment (diagnostic)")
    try:
        out["step8"] = run_step8()
    except Exception as e:
        print(f"[step8] skipped due to: {e!r}")
        out["step8"] = None
    _hdr("STEP 9 — decision + preprint patch")
    out["step9"] = run_step9()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--stages", nargs="+", default=None,
                    help="Subset of H0..H7 to apply cumulatively in Step 6.")
    args = ap.parse_args()
    run_all(fast=args.fast, ladder_stages=args.stages)
