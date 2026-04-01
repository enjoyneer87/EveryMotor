#!/usr/bin/env python3
"""Check DOE data loading quality."""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/host_data")
from doe_data_utils import load_doe_data, build_gino_sample

# 1. Load data
records, conditions = load_doe_data("/workspace/host_data/doe_data", max_steps_per_case=None)
print(f"\n=== DATA SUMMARY ===")
print(f"Total records: {len(records)}")
print(f"Total conditions: {len(conditions)}")

if not records:
    print("ERROR: No records loaded!")
    sys.exit(1)

# 2. Check record content
rec0 = records[0]
print(f"\n=== SAMPLE RECORD (idx=0) ===")
print(f"Keys: {list(rec0.keys())}")
print(f"Nodes (_n): {rec0['_n']}")
print(f"pos_x range: [{rec0['pos_x'].min():.3f}, {rec0['pos_x'].max():.3f}]")
print(f"pos_y range: [{rec0['pos_y'].min():.3f}, {rec0['pos_y'].max():.3f}]")
print(f"bx range: [{rec0['bx'].min():.6f}, {rec0['bx'].max():.6f}]")
print(f"by range: [{rec0['by'].min():.6f}, {rec0['by'].max():.6f}]")
print(f"a range: [{rec0['a'].min():.6f}, {rec0['a'].max():.6f}]")
print(f"j range: [{rec0['j'].min():.6f}, {rec0['j'].max():.6f}]")

# 3. Check conditions
cond0 = conditions[0]
print(f"\n=== SAMPLE CONDITION ===")
for k, v in cond0.items():
    print(f"  {k}: {v}")

# 4. Check for NaN/Inf in raw data
nan_count = 0
inf_count = 0
zero_bx_count = 0
for i, rec in enumerate(records):
    for key in ['bx', 'by', 'a', 'j']:
        arr = rec[key]
        if np.isnan(arr).any():
            nan_count += 1
            if nan_count <= 3:
                print(f"  NaN found in rec[{i}][{key}]")
        if np.isinf(arr).any():
            inf_count += 1
    if np.all(rec['bx'] == 0) and np.all(rec['by'] == 0):
        zero_bx_count += 1

print(f"\n=== DATA QUALITY ===")
print(f"Records with NaN: {nan_count}")
print(f"Records with Inf: {inf_count}")
print(f"Records with all-zero Bx,By: {zero_bx_count}")

# 5. Build a GINO sample and check
from doe_data_utils import scatter_elem_to_node
s = build_gino_sample(records[0], conditions[0], latent_res=32)
if s:
    print(f"\n=== GINO SAMPLE ===")
    print(f"input_geom: {s['input_geom'].shape}, range [{s['input_geom'].min():.4f}, {s['input_geom'].max():.4f}]")
    print(f"features: {s['features'].shape}")
    for i in range(s['features'].shape[1]):
        col = s['features'][:, i]
        print(f"  feat[{i}]: min={col.min():.6f}, max={col.max():.6f}, std={col.std():.6f}")
    print(f"target: {s['target'].shape}, range [{s['target'].min():.6f}, {s['target'].max():.6f}]")
    print(f"  Bx std: {s['target'][:,0].std():.6f}")
    print(f"  By std: {s['target'][:,1].std():.6f}")
    print(f"latent_queries: {s['latent_queries'].shape}")
    
    # Check NaN
    for k in ['input_geom', 'features', 'target', 'latent_queries']:
        has_nan = np.isnan(s[k]).any()
        has_inf = np.isinf(s[k]).any()
        if has_nan or has_inf:
            print(f"  WARNING: {k} has NaN={has_nan}, Inf={has_inf}")
else:
    print("ERROR: build_gino_sample returned None!")

# 6. Check normalization
print(f"\n=== NORMALIZATION CHECK (first 5 samples) ===")
samples = []
for i in range(min(5, len(records))):
    s = build_gino_sample(records[i], conditions[i], latent_res=32)
    if s:
        samples.append(s)

all_feat = np.concatenate([s['features'] for s in samples], axis=0)
all_tgt = np.concatenate([s['target'] for s in samples], axis=0)
feat_mean = all_feat.mean(axis=0)
feat_std = all_feat.std(axis=0)
tgt_mean = all_tgt.mean(axis=0)
tgt_std = all_tgt.std(axis=0)

print(f"Feature means: {feat_mean}")
print(f"Feature stds:  {feat_std}")
print(f"Target means:  {tgt_mean}")
print(f"Target stds:   {tgt_std}")
print(f"Any feat_std == 0? {(feat_std < 1e-6).any()} -> which: {np.where(feat_std < 1e-6)[0]}")
