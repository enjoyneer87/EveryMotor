#!/usr/bin/env python3
"""TASK-P1B-1: Boundary chain and PBC edge visualization.

Usage (inside Docker):
    python /workspace/app/visualize_pbc_boundary.py \
        --npz /workspace/app/results/overfit_smoke.npz \
        --out /workspace/app/logs/pbc_boundary_vis.png

Produces:
  - Left: arc(blue)/radial(red)/mixed(gray) chain overlay on mesh nodes
  - Right: interior(+1.0, light gray) vs anti-periodic(-1.0, red) edge distribution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    print("[ERROR] matplotlib required. Install or run inside Docker.")
    sys.exit(1)

from phase1_static.motor_dataset import build_sector_pbc_edges_from_mesh
from phase1_static.pbc_boundary import extract_boundary_chains_from_mesh


CHAIN_COLORS = {
    "arc": "#2196F3",           # blue
    "radial": "#F44336",        # red
    "mixed_polyline": "#9E9E9E",# gray
    "unresolved": "#FF9800",    # orange
}


def build_demo_sector_mesh():
    """Synthetic 6-node annular sector for demo when no NPZ provided."""
    angles = np.deg2rad(np.linspace(0.0, 45.0, 6))
    inner = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    outer = 2.0 * inner
    pos = np.vstack([inner, outer]).astype(np.float64)

    # Triangulate sector strips
    n = len(angles)
    tris = []
    for i in range(n - 1):
        tris.append([i, i + 1, i + n])
        tris.append([i + 1, i + n + 1, i + n])
    triangles = np.array(tris, dtype=np.int32)
    region_code = np.ones(len(triangles), dtype=np.int32)
    region_code[len(triangles) // 2:] = 2
    return pos, triangles, region_code


def load_mesh_from_npz(npz_path: Path):
    """Load first sample from NPZ bundle."""
    arr = np.load(npz_path, allow_pickle=True)
    pos = np.asarray(arr["pos"][0], dtype=np.float64)
    ie = np.asarray(arr["interior_edge_index"][0], dtype=np.int64)  # [2, E]
    # Reconstruct triangles approximation from interior edges (undirected)
    src, dst = ie[0], ie[1]
    # Build adjacency for triangle extraction (not perfect but sufficient for vis)
    n_nodes = pos.shape[0]
    adj = [[] for _ in range(n_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        if v not in adj[u]:
            adj[u].append(v)

    tris = []
    seen = set()
    for u in range(n_nodes):
        for v in adj[u]:
            for w in adj[v]:
                if w in adj[u] and w != u and v != u:
                    key = tuple(sorted([u, v, w]))
                    if key not in seen:
                        seen.add(key)
                        tris.append(list(key))

    triangles = np.array(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32)
    region_code = np.ones(len(triangles), dtype=np.int32)
    return pos, triangles, region_code


def main():
    parser = argparse.ArgumentParser(description="Visualize PBC boundary chains")
    parser.add_argument("--npz", type=str, default=None, help="NPZ bundle path (optional)")
    parser.add_argument("--out", type=str, default="logs/pbc_boundary_vis.png")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    if args.npz and Path(args.npz).exists():
        pos, triangles, region_code = load_mesh_from_npz(Path(args.npz))
        title_suffix = f"(NPZ: {Path(args.npz).name})"
    else:
        pos, triangles, region_code = build_demo_sector_mesh()
        title_suffix = "(synthetic demo sector)"

    # --- Extract chains ---
    chain_set, err = extract_boundary_chains_from_mesh(pos, triangles, region_code)
    if err or chain_set is None:
        print(f"[WARN] chain extraction failed: {err}. Using empty result.")
        chain_set = None

    # --- Build PBC edges ---
    pbc_edge_index, pbc_edge_attr, pbc_diag = build_sector_pbc_edges_from_mesh(
        pos_xy=pos.astype(np.float32),
        triangles=triangles,
        rotation_deg=-45.0,
        anti_periodic=True,
    )

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"PBC Boundary Visualization {title_suffix}", fontsize=11)

    # Left: chain overlay
    ax = axes[0]
    ax.scatter(pos[:, 0], pos[:, 1], s=8, c="#BDBDBD", zorder=2, label="nodes")

    if chain_set is not None:
        origin = chain_set.rotation_origin_xy
        ax.scatter(origin[0], origin[1], marker="x", s=80, c="black", zorder=5, label="origin")

        counts = {}
        for chain in chain_set.chains:
            ct = chain.chain_type
            counts[ct] = counts.get(ct, 0) + 1
            color = CHAIN_COLORS.get(ct, "#000000")
            idx = np.asarray(chain.node_indices, dtype=np.int64)
            ax.plot(pos[idx, 0], pos[idx, 1], "-o",
                    color=color, markersize=3, linewidth=1.5,
                    label=ct if counts[ct] == 1 else "_")

    # PBC edges overlay
    if pbc_edge_index.shape[1] > 0:
        for e in range(pbc_edge_index.shape[1]):
            u, v = pbc_edge_index[0, e], pbc_edge_index[1, e]
            ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
                    color="purple", alpha=0.25, linewidth=0.6)

    ax.set_aspect("equal")
    ax.set_title(
        f"Chains: {len(chain_set.chains) if chain_set else 0}  |  "
        f"PBC edges: {pbc_edge_index.shape[1]}  |  "
        f"match_ratio: {pbc_diag['match_ratio']:.2f}"
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # Right: edge sign distribution
    ax2 = axes[1]
    if pbc_edge_attr.size > 0:
        signs = pbc_edge_attr.ravel()
        unique, counts = np.unique(signs, return_counts=True)
        colors = ["#F44336" if s < 0 else "#4CAF50" for s in unique]
        bars = ax2.bar([str(s) for s in unique], counts, color=colors)
        ax2.bar_label(bars, padding=2)
        ax2.set_xlabel("edge_attr sign")
        ax2.set_ylabel("count")
        ax2.set_title("PBC Edge Sign Distribution\n(red=-1.0 anti-periodic, green=+1.0 periodic)")
    else:
        ax2.text(0.5, 0.5, "No PBC edges\n" + pbc_diag.get("error_code", ""),
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("PBC Edge Sign Distribution")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    result = {
        "out": str(out_path),
        "chain_count": len(chain_set.chains) if chain_set else 0,
        "pbc_edge_count": int(pbc_edge_index.shape[1]),
        "match_ratio": pbc_diag["match_ratio"],
        "error_code": pbc_diag.get("error_code", ""),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
