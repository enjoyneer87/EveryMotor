"""Can the v2 (320-case) manifest actually be loaded from inside the training container?

doe_build_v2.py stored HOST-absolute paths (D:\\KDH\\NvidiaNemo\\backup\\doe_data_240\\...)
for the 240 re-labelled cases. The container sees only /workspace/app, and
resolve_h5_path's fallback looks under <data_dir>/case_XXXX/postproc/ -- but
backup/doe_data_v2 holds case dirs 0240..0319 only. This probe reports, per case,
whether every h5 resolves, so a silently-short 320-case run cannot happen.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, "/workspace/app")
from eval.doe_dataset import resolve_h5_path  # noqa: E402

data_dir = Path("backup/doe_data_v2")
man = json.loads((data_dir / "doe_manifest.json").read_text(encoding="utf-8"))

missing_legacy, missing_new, ok_legacy, ok_new = [], [], 0, 0
for c in man["cases"]:
    idx = int(c["index"])
    bad = [p for p in c["h5_paths"] if resolve_h5_path(p, data_dir, idx) is None]
    if idx < 240:
        if bad:
            missing_legacy.append(idx)
        else:
            ok_legacy += 1
    elif bad:
        missing_new.append(idx)
    else:
        ok_new += 1

print(f"legacy (0-239): resolved {ok_legacy}/240, unresolved {len(missing_legacy)}")
print(f"pilot  (240-319): resolved {ok_new}/80, unresolved {len(missing_new)}")
if missing_legacy:
    print(f"  first unresolved legacy: {missing_legacy[:5]}")
    print(f"  recorded path example: {man['cases'][missing_legacy[0]]['h5_paths'][0]}")
if missing_new:
    print(f"  first unresolved pilot: {missing_new[:5]}")
print("V2_PATH_GATE_" + ("PASS" if not (missing_legacy or missing_new) else "FAIL"))
