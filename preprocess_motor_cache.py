"""Pre-process FSG Motor data: read txt -> Delaunay + FaceToEdge -> save as .pt
This eliminates the need to recompute Delaunay triangulation every epoch.
"""
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch_geometric as tg
from torch_geometric.transforms import Compose, Delaunay, FaceToEdge

BASE_DIR = Path("/workspace/multiscale-pde-operators/datasets/motor/BM2")
CACHE_DIR = Path("/workspace/multiscale-pde-operators/datasets/motor_cached")
ROTOR_POS = "pos1_"
TEST_RATIO = 0.1
VAL_RATIO = 0.1
RANDOM_SEED = 42

def read_and_transform(f: Path):
    """Read a single Bnorm_matinfo.txt file and apply Delaunay + FaceToEdge."""
    txt_start_idx = 9
    transform = Compose([Delaunay(), FaceToEdge()])

    with open(f) as source:
        xs, ys, norms, materials = [], [], [], []
        for line in source.readlines()[txt_start_idx:]:
            values = line.split()
            x, y, bnorm, material = values
            xs.append(float(x))
            ys.append(float(y))
            norms.append(float(bnorm))
            materials.append(float(material))

    xs = np.array(xs)
    ys = np.array(ys)
    norms = np.array(norms)
    materials = np.array(materials)

    data = tg.data.Data(
        pos=torch.tensor(np.stack([xs, ys], axis=1), dtype=torch.float32),
        x=torch.tensor(np.stack([materials], axis=1), dtype=torch.float32),
        y=torch.tensor(norms, dtype=torch.float32).reshape(-1, 1),
    )
    return transform(data)


def main():
    # Collect all txt files (same logic as FsgMotorDriver)
    collect_txts = []
    for file in BASE_DIR.rglob("*Bnorm_matinfo.txt"):
        if ROTOR_POS in file.name:
            collect_txts.append(file)

    print(f"Found {len(collect_txts)} files matching rotor_pos='{ROTOR_POS}'")

    # Split (same logic as FsgMotorDriver)
    random.seed(RANDOM_SEED)
    collect_txts_test = random.sample(collect_txts, int(len(collect_txts) * TEST_RATIO))
    collect_txts_train = [f for f in collect_txts if f not in collect_txts_test]
    collect_txts_val = random.sample(collect_txts_train, int(len(collect_txts) * VAL_RATIO))
    collect_txts_train = [f for f in collect_txts_train if f not in collect_txts_val]

    splits = {
        "train": collect_txts_train,
        "val": collect_txts_val,
        "test": collect_txts_test,
    }

    for split_name, files in splits.items():
        print(f"\n--- Processing {split_name}: {len(files)} files ---")
        split_dir = CACHE_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        for i, f in enumerate(files):
            data = read_and_transform(f)
            out_path = split_dir / f"{f.stem}.pt"
            torch.save(data, out_path)

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(files) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(files)}] {elapsed:.1f}s elapsed, "
                      f"{rate:.1f} samples/s, ~{remaining:.0f}s remaining")
                sys.stdout.flush()

    print(f"\nDone! Cached data saved to {CACHE_DIR}")
    for split_name in splits:
        n = len(list((CACHE_DIR / split_name).glob("*.pt")))
        print(f"  {split_name}: {n} files")


if __name__ == "__main__":
    main()
