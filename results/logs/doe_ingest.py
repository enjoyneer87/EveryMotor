"""R4(a) DOE ingest -- copy generated cases into backup/doe_data with renumbering + manifest merge.

Reads SRC/doe_manifest.json (cases index 0..N-1), copies each case's postproc/*.h5 into
DST/case_{OFFSET+i:04d}/postproc/, writes meta.json, and merges entries into
DST/doe_manifest.json (n_cases += N). Idempotent-ish: overwrites target case dirs.

Env knobs: DOE_SRC, DOE_OFFSET (40), DOE_DST (D:/KDH/NvidiaNemo/backup/doe_data), DOE_DRY (0/1)
Run with any python that has numpy (host or container). No Motor-CAD needed.
"""
import json
import os
import shutil
import pathlib

SRC = pathlib.Path(os.environ.get("DOE_SRC", r"D:/KDH/Sim_4SolverX/DOE_Ext"))
OFFSET = int(os.environ.get("DOE_OFFSET", "40"))
DST = pathlib.Path(os.environ.get("DOE_DST", r"D:/KDH/NvidiaNemo/backup/doe_data"))
DRY = os.environ.get("DOE_DRY", "0") == "1"

src_manifest = json.loads((SRC / "doe_manifest.json").read_text(encoding="utf-8"))
src_cases = src_manifest.get("cases", [])
print(f"[ingest] SRC={SRC} cases={len(src_cases)} OFFSET={OFFSET} DST={DST} DRY={DRY}", flush=True)

dst_manifest_path = DST / "doe_manifest.json"
dst_manifest = json.loads(dst_manifest_path.read_text(encoding="utf-8"))
existing = {int(c["index"]) for c in dst_manifest.get("cases", [])}

new_entries = []
for c in src_cases:
    si = int(c["index"])
    di = OFFSET + si
    if di in existing:
        print(f"[ingest] SKIP case {di} already in DST manifest", flush=True)
        continue
    src_case = SRC / f"case_{si:04d}"
    dst_case = DST / f"case_{di:04d}"
    src_pp = src_case / "postproc"
    dst_pp = dst_case / "postproc"
    h5s = sorted(src_pp.glob("*.h5")) if src_pp.is_dir() else []
    if not h5s:
        print(f"[ingest] WARN case {si}: no h5 in {src_pp} -- skipping", flush=True)
        continue
    if not DRY:
        dst_pp.mkdir(parents=True, exist_ok=True)
        for h in h5s:
            shutil.copy2(h, dst_pp / h.name)
    dst_h5_paths = [str((dst_pp / h.name)).replace("/", "\\") for h in h5s]
    meta = {
        "index": di,
        "geometry": c.get("geometry", {}),
        "electrical": c.get("electrical", {}),
        "h5_paths": dst_h5_paths,
        "solve_time_s": c.get("solve_time_s"),
    }
    if not DRY:
        (dst_case / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    new_entries.append(meta)
    print(f"[ingest] case {si} -> {di}: {len(h5s)} h5 copied", flush=True)

if not DRY and new_entries:
    dst_manifest["cases"] = dst_manifest.get("cases", []) + new_entries
    dst_manifest["n_cases"] = len(dst_manifest["cases"])
    # backup the old manifest once
    bak = dst_manifest_path.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(dst_manifest_path, bak)
    dst_manifest_path.write_text(json.dumps(dst_manifest, indent=2), encoding="utf-8")
    print(f"[ingest] merged {len(new_entries)} cases -> n_cases={dst_manifest['n_cases']}", flush=True)
else:
    print(f"[ingest] {'DRY-run, ' if DRY else ''}new={len(new_entries)}", flush=True)
print("[ingest] DONE", flush=True)
