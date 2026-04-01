"""Check stats of converted FSG motor data."""
import numpy as np
from pathlib import Path
import random

base = Path("/workspace/multiscale-pde-operators/datasets/motor/BM2")
files = sorted(base.glob("*Bnorm_matinfo.txt"))
print(f"Total files: {len(files)}")

random.seed(42)
sample_files = random.sample(files, min(20, len(files)))
all_bn, all_mat = [], []
for f in sample_files:
    data = np.loadtxt(f, skiprows=9)
    all_bn.append(data[:, 2])
    all_mat.append(data[:, 3])

bn = np.concatenate(all_bn)
mt = np.concatenate(all_mat)
print(f"Bnorm: mean={bn.mean():.6f}, std={bn.std():.6f}, min={bn.min():.6f}, max={bn.max():.6f}")
print(f"Material unique: {sorted(np.unique(mt).astype(int).tolist())}")
print(f"Material: mean={mt.mean():.6f}, std={mt.std():.6f}")

print("\n--- Sample file head (first 14 lines) ---")
with open(files[0]) as fp:
    for i, line in enumerate(fp):
        if i < 14:
            print(line.rstrip())
        else:
            break
