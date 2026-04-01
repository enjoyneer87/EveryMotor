import json

from runtime_paths import get_runtime_paths

paths = get_runtime_paths()
manifest_path = paths["doe_data"] / "doe_manifest.json"

with open(manifest_path, encoding="utf-8") as f:
    m = json.load(f)

print(f"Manifest: {manifest_path}")
print(f"Cases: {m['n_cases']}, Failed: {len(m.get('failed',[]))}")
c = m["cases"][0]
print(f"Keys: {list(c.keys())}")
print(f"H5: {c.get('h5_paths',['?'])}")
print(f"Geometry: {c.get('geometry',{})}")
print(f"Electrical: {c.get('electrical',{})}")
