"""Assert the new current coupling leaves the pre-R8 sets at scale 1.0.

R7-A trained 47 epochs against priors computed at the template current. If
case_current_scale started honouring doe_data_240's NOMINAL PeakCurrent -- which the
section-24 defect made meaningless (labelled 224 A, solved at 650.5 A) -- the cached
priors would mismatch, silently recompute at a current the case never saw, and the
resumed epochs 48-50 would be learning from different features than epochs 1-47.
"""
import json
import sys

sys.path.insert(0, "/workspace/app")
from eval.doe_dataset import load_doe_cases          # noqa: E402
from fem_warmstart.prior import case_current_scale   # noqa: E402

CHECKS = [
    ("backup/doe_data_240", [0, 4, 239], 1.0, "R7-A training set: must stay at template current"),
    ("backup/doe_data_v2", [0, 239], 1.0, "v2 re-labelled legacy: wired=0 -> template current"),
    ("backup/doe_data_v2", [240, 280], 0.5, "v2 pilot 325.3 A -> half"),
    ("backup/doe_data_v2", [279, 319], 0.25, "v2 pilot 162.6 A -> quarter"),
]

bad = []
for data_dir, cases, expect, why in CHECKS:
    man = json.loads(open(f"{data_dir}/doe_manifest.json", encoding="utf-8").read())
    for ci in cases:
        rec = load_doe_cases(man, data_dir, case_indices=[ci]).records[0]
        got = case_current_scale(rec)
        ok = abs(got - expect) < 1e-6
        if not ok:
            bad.append((data_dir, ci, expect, got))
        print(f"{data_dir:24s} case {ci:4d}: scale {got:.6f} (expect {expect:.4f}) "
              f"{'OK' if ok else 'MISMATCH'}   [{why}]")

print("PRIOR_SCALE_CHECK_" + ("PASS" if not bad else "FAIL"))
sys.exit(0 if not bad else 1)
