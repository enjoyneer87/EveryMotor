"""Extract an explicit allowlist from an existing verified archive, never a solver.

No extractall(), symbolic links, environment files, or arbitrary path traversal.
The new output directory must not already exist.
"""
from pathlib import Path
import argparse
import hashlib
import shutil
import tarfile
from field_reader import read_json, sha, write_json

ALLOW = {"PARAMETERS.json", "MODEL.json", "INPUT_CONTRACT.json", "case.sif", "circuits.dat", "circuits.dat.names",
         "circuits.definitions", "solver.log", "mesh/mesh.nodes", "mesh/mesh.elements", "mesh/mesh.boundary", "mesh/mesh.header",
         "mesh/native.result", "mesh/field_t0001.vtu", "mesh/field_t0256.vtu", "mesh/field_t0512.vtu"}


def extract(archive, out, expected_archive_sha256, expected_files):
    archive, out = Path(archive), Path(out)
    actual = sha(archive)
    assert actual == expected_archive_sha256, "Archive hash mismatch; do not substitute"
    out.mkdir(parents=True, exist_ok=False)
    matches = []
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            name = m.name.replace("\\", "/")
            if "/raw/" not in name:
                continue
            rel = name.split("/raw/", 1)[1]
            if rel not in ALLOW:
                continue
            assert m.isfile() and not m.issym() and not m.islnk()
            dest = out / rel
            assert dest.resolve().is_relative_to(out.resolve())
            dest.parent.mkdir(parents=True, exist_ok=True)
            assert not dest.exists(), "Duplicate archive member"
            with tf.extractfile(m) as src, dest.open("xb") as dst:
                shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            digest = sha(dest)
            want = expected_files[rel]["sha256"]
            assert digest == want, f"Raw member mismatch: {rel}"
            matches.append(dict(path=rel, sha256=digest, bytes=dest.stat().st_size))
    assert {x["path"] for x in matches} == ALLOW
    receipt = dict(archive_sha256=actual, archive_bytes=archive.stat().st_size, files=matches, status="PASS", new_solver_calls=0)
    write_json(out.parent / "selective_extract_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--archive",required=True);p.add_argument("--out",required=True)
    p.add_argument("--archive-sha256",required=True);p.add_argument("--expected",required=True)
    a=p.parse_args();expected=read_json(a.expected)
    result=extract(a.archive,a.out,a.archive_sha256,{r['path']:r for r in expected['archive_files']})
    print({"status":result['status'],"files":len(result['files'])})
