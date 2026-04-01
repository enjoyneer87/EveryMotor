import json
from pathlib import Path
import sys

sys.path.insert(0, "/workspace/host_data")
from doe_data_utils import parse_h5_timeseries

data_dir = Path("/workspace/host_data/doe_data")
manifest = json.load(open(data_dir / "doe_manifest.json"))
case = manifest["cases"][5]

records = []
for h5p in case.get("h5_paths") or []:
    h5_basename = h5p.replace("\\", "/").split("/")[-1]
    candidates = [
        Path(h5p),
        data_dir / f"case_{case['index']:04d}" / "postproc" / h5_basename,
        data_dir / h5_basename,
    ]
    h5_file = None
    for c in candidates:
        if c.exists():
            h5_file = c
            break
    if h5_file is None:
        continue
    try:
        records.extend(parse_h5_timeseries(h5_file))
    except Exception:
        pass

print("steps", len(records))
print("unique_n", sorted({r["_n"] for r in records}))
print("first_n", records[0]["_n"] if records else None)
