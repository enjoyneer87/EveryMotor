"""
Convert DOE H5 timeseries data to FSG Motor format (*Bnorm_matinfo.txt).

Each H5 timestep → one txt file containing (x, y, bnorm, material) per node.
Bnorm = sqrt(Bx² + By²), computed as element-to-node scatter average.

Output goes to: /workspace/multiscale-pde-operators/datasets/motor/BM2/
"""
import os
import sys
import json
import glob
import numpy as np
import h5py
from pathlib import Path


DOE_DIR  = Path("/workspace/doe_data")
OUT_DIR  = Path("/workspace/multiscale-pde-operators/datasets/motor/BM2")
MANIFEST = DOE_DIR / "doe_manifest.json"

# Only use OnLoadTorque (main result with rotating rotor + current)
H5_PATTERN = "Mag_OnLoadTorque_result_1.h5"


def parse_and_convert_h5(h5_path: Path, case_idx: int, max_steps: int = None):
    """Read H5, scatter element→node fields, output Bnorm_matinfo.txt files."""
    txt_files = []
    with h5py.File(h5_path, "r") as f:
        steps    = np.asarray(f["steps"][:], dtype=np.int32)
        node_id  = np.asarray(f["mesh/node_id"][:], dtype=np.int32)
        node_x0  = np.asarray(f["mesh/node_x_mm"][:], dtype=np.float64)
        node_y0  = np.asarray(f["mesh/node_y_mm"][:], dtype=np.float64)

        node_1   = np.asarray(f["mesh/node_1"][:], dtype=np.int32)
        node_2   = np.asarray(f["mesh/node_2"][:], dtype=np.int32)
        node_3   = np.asarray(f["mesh/node_3"][:], dtype=np.int32)
        reg_code = np.asarray(f["mesh/reg_code"][:], dtype=np.int32)

        bx_mat   = np.asarray(f["fields/bx"][:], dtype=np.float32)
        by_mat   = np.asarray(f["fields/by"][:], dtype=np.float32)

        # Moving node coords
        x_bs = np.asarray(f["mesh/node_x_mm_by_step"][:], dtype=np.float64) \
            if "mesh/node_x_mm_by_step" in f else None
        y_bs = np.asarray(f["mesh/node_y_mm_by_step"][:], dtype=np.float64) \
            if "mesh/node_y_mm_by_step" in f else None
        x_bsm = np.asarray(f["mesh/node_x_mm_by_step_moving"][:], dtype=np.float64) \
            if "mesh/node_x_mm_by_step_moving" in f else None
        y_bsm = np.asarray(f["mesh/node_y_mm_by_step_moving"][:], dtype=np.float64) \
            if "mesh/node_y_mm_by_step_moving" in f else None
        mov_idx = np.asarray(f["mesh/moving_node_indices"][:], dtype=np.int32) \
            if "mesh/moving_node_indices" in f else None

    # Build node LUT
    n_nodes = node_id.size
    sort_order = np.argsort(node_id)
    sorted_ids = node_id[sort_order]
    max_nid = int(node_id.max()) + 1
    lut = np.full(max_nid, -1, dtype=np.int64)
    for i, nid in enumerate(sorted_ids):
        lut[nid] = i

    # Element topology
    m = min(len(node_1), len(node_2), len(node_3), len(reg_code))
    i1 = lut[np.clip(node_1[:m], 0, max_nid - 1)]
    i2 = lut[np.clip(node_2[:m], 0, max_nid - 1)]
    i3 = lut[np.clip(node_3[:m], 0, max_nid - 1)]
    valid = (i1 >= 0) & (i2 >= 0) & (i3 >= 0)
    i1v, i2v, i3v = i1[valid], i2[valid], i3[valid]
    reg_v = reg_code[:m][valid].astype(np.float32)
    n = len(sorted_ids)

    # Scatter indices
    all_idx = np.concatenate([i1v, i2v, i3v])

    # Node region (majority vote)
    all_reg = np.tile(reg_v.astype(np.int32), 3)
    node_reg = np.zeros(n, np.float32)
    max_reg = int(all_reg.max()) + 1 if len(all_reg) > 0 else 1
    for nidx in np.unique(all_idx):
        mask = all_idx == nidx
        bc = np.bincount(all_reg[mask], minlength=max_reg)
        node_reg[nidx] = float(np.argmax(bc))

    # Moving node mask
    mov_mask = None
    if mov_idx is not None and mov_idx.size > 0:
        mov_mask = (mov_idx >= 0) & (mov_idx < n_nodes)

    n_steps = min(int(steps.shape[0]), max_steps) if max_steps else int(steps.shape[0])

    for si in range(n_steps):
        x = node_x0.copy()
        y = node_y0.copy()

        # Apply by_step coords
        if x_bs is not None and y_bs is not None:
            xs, ys = x_bs[si], y_bs[si]
            ok = np.isfinite(xs) & np.isfinite(ys)
            x[ok] = xs[ok]; y[ok] = ys[ok]

        # Apply moving-node
        if x_bsm is not None and y_bsm is not None and mov_mask is not None:
            xs_m, ys_m = x_bsm[si], y_bsm[si]
            vi = np.where(mov_mask)[0][:min(len(xs_m), mov_mask.sum())]
            ni = mov_idx[vi]
            xv, yv = xs_m[vi], ys_m[vi]
            ok_f = np.isfinite(xv) & np.isfinite(yv)
            x[ni[ok_f]] = xv[ok_f]
            y[ni[ok_f]] = yv[ok_f]

        # Sort by node_id
        pos_x = x[sort_order].astype(np.float32)
        pos_y = y[sort_order].astype(np.float32)

        # Element → Node scatter for Bnorm
        bx_v = bx_mat[si][:m][valid].astype(np.float32)
        by_v = by_mat[si][:m][valid].astype(np.float32)
        bnorm_elem = np.sqrt(bx_v**2 + by_v**2)

        all_bn = np.tile(bnorm_elem, 3)
        sum_bn = np.zeros(n, np.float32)
        np.add.at(sum_bn, all_idx, all_bn)
        cnt = np.zeros(n, np.float32)
        np.add.at(cnt, all_idx, 1.0)
        cnt = np.clip(cnt, 1.0, None)
        node_bnorm = sum_bn / cnt

        # Write Bnorm_matinfo.txt
        # Filename: case{case_idx:04d}_pos1_step{step:04d}_Bnorm_matinfo.txt
        step_key = int(steps[si])
        fname = f"case{case_idx:04d}_pos1_step{step_key:04d}_Bnorm_matinfo.txt"
        fpath = OUT_DIR / fname

        with open(fpath, "w") as fp:
            # 9 header lines (FSG format expects 9 lines to skip)
            fp.write(f"# DOE Case {case_idx}, Step {step_key}\n")
            fp.write(f"# Source: {h5_path.name}\n")
            fp.write(f"# Bnorm = sqrt(Bx^2 + By^2), element->node scatter avg\n")
            fp.write(f"# Coordinates in mm\n")
            fp.write(f"# Columns: x  y  bnorm  material\n")
            fp.write(f"# N_nodes: {n}\n")
            fp.write(f"# Step: {step_key}\n")
            fp.write(f"# Converted from Motor-CAD DOE H5\n")
            fp.write(f"# x[mm]  y[mm]  Bnorm[T]  RegionCode\n")

            for ni in range(n):
                fp.write(f"{pos_x[ni]:.6f} {pos_y[ni]:.6f} {node_bnorm[ni]:.8f} {node_reg[ni]:.0f}\n")

        txt_files.append(fpath)

    return txt_files, n


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load manifest
    with open(MANIFEST) as f:
        raw = json.load(f)
    manifest = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw

    print(f"Found {len(manifest)} DOE cases in manifest")
    print(f"Output dir: {OUT_DIR}")
    print(f"H5 pattern: {H5_PATTERN}")
    print()

    total_files = 0
    all_bnorms = []
    all_materials = []

    for entry in manifest:
        case_idx = entry["index"]
        case_dir = DOE_DIR / f"case_{case_idx:04d}" / "postproc"
        h5_path = case_dir / H5_PATTERN

        if not h5_path.exists():
            print(f"  SKIP case {case_idx}: {H5_PATTERN} not found")
            continue

        print(f"  Case {case_idx:04d}: {h5_path.name} ...", end="", flush=True)
        try:
            txt_files, n_nodes = parse_and_convert_h5(h5_path, case_idx)
            total_files += len(txt_files)
            print(f" -> {len(txt_files)} txt files ({n_nodes} nodes each)")

            # Collect stats from first step only (representative)
            if txt_files:
                sample = np.loadtxt(txt_files[0], skiprows=9)
                all_bnorms.append(sample[:, 2])
                all_materials.append(sample[:, 3])

        except Exception as e:
            print(f" ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\nTotal: {total_files} txt files generated")
    print(f"Output dir: {OUT_DIR}")

    # List files
    out_files = sorted(OUT_DIR.glob("*Bnorm_matinfo.txt"))
    print(f"Verification: {len(out_files)} Bnorm_matinfo.txt files found")
    if out_files:
        print(f"  First: {out_files[0].name}")
        print(f"  Last:  {out_files[-1].name}")

    # Compute dataset statistics for config
    if all_bnorms:
        all_bn = np.concatenate(all_bnorms)
        all_mt = np.concatenate(all_materials)
        print(f"\n=== Dataset Statistics (for config) ===")
        print(f"  Bnorm: mean={all_bn.mean():.6f}, std={all_bn.std():.6f}, "
              f"min={all_bn.min():.6f}, max={all_bn.max():.6f}")
        print(f"  Material: unique={np.unique(all_mt).tolist()}, "
              f"mean={all_mt.mean():.6f}, std={all_mt.std():.6f}")
        print()
        print(f"  >>> Update configs/dataset/motor.yaml:")
        print(f"      mean: {all_mt.mean():.6f}")
        print(f"      std: {all_mt.std():.6f}")
        print(f"  (Note: FSG normalizes material feature with these stats)")


if __name__ == "__main__":
    main()
