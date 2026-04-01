import json
with open("/workspace/host_data/doe_data/doe_manifest.json") as f:
    m = json.load(f)
print(f"Cases: {m['n_cases']}, Failed: {len(m.get('failed',[]))}")
c = m["cases"][0]
print(f"Keys: {list(c.keys())}")
print(f"H5: {c.get('h5_paths',['?'])}")
print(f"Geometry: {c.get('geometry',{})}")
print(f"Electrical: {c.get('electrical',{})}")
