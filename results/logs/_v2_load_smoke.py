"""Load smoke for the repaired v2 manifest: resolving a path is not the same as reading it.

Picks boundary cases across both halves of the 320-case set (legacy re-labelled 240,
which physically span doe_data_240 and doe_data_120, plus the ingested pilot 80) and
actually parses each one, reporting mesh size, sample count and the recorded excitation.
"""
import json
import sys

sys.path.insert(0, "/workspace/app")
from eval.doe_dataset import load_doe_cases  # noqa: E402

DATA_DIR = "backup/doe_data_v2"
CASES = [0, 39, 40, 119, 120, 239, 240, 279, 280, 319]

man = json.loads(open(f"{DATA_DIR}/doe_manifest.json", encoding="utf-8").read())
bad = []
for ci in CASES:
    try:
        rec = load_doe_cases(man, DATA_DIR, case_indices=[ci]).records[0]
        ipk = float(rec.condition.get("PeakCurrent", float("nan")))
        wired = rec.condition.get("excitation_wired")
        print(f"case {ci:4d}: nodes {len(rec.samples[0].node_x_mm):6d}  tris {len(rec.mesh.tri):6d}  "
              f"samples {len(rec.samples):3d}  Ipk {ipk:8.2f} A  wired={wired}")
    except Exception as exc:                                    # noqa: BLE001
        bad.append((ci, repr(exc)))
        print(f"case {ci:4d}: LOAD FAILED -- {exc!r}")

print("V2_LOAD_SMOKE_" + ("PASS" if not bad else "FAIL"))
sys.exit(0 if not bad else 1)
