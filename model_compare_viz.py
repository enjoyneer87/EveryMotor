from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


FieldPair = Tuple[np.ndarray, np.ndarray]


def load_node_comparison_npz(npz_path: str | Path) -> dict:
    """Load node-level model outputs from field_compare_nodes_allsteps.npz."""
    npz_path = Path(npz_path)
    d_node = np.load(str(npz_path), allow_pickle=True)

    data_dict: Dict[str, FieldPair] = {
        "GT": (d_node["gt_bx_node"], d_node["gt_by_node"]),
        "MGN": (d_node["mgn_bx_node"], d_node["mgn_by_node"]),
        "FNO": (d_node["fno_bx_node"], d_node["fno_by_node"]),
        "GINO": (d_node["gino_bx_node"], d_node["gino_by_node"]),
        "RNN": (d_node["rnn_bx_node"], d_node["rnn_by_node"]),
    }

    return {
        "raw": d_node,
        "node_x_all": d_node["node_x"],
        "node_y_all": d_node["node_y"],
        "n_steps": int(len(d_node["node_x"])),
        "case_idx": int(d_node.get("case_idx", 0)),
        "data_dict": data_dict,
    }


def load_mesh_triangles_from_h5(base_dir: str | Path, case_idx: int) -> Optional[np.ndarray]:
    """Load element connectivity from exported Mag_OnLoadTorque_result_1.h5."""
    base_dir = Path(base_dir)
    h5_path = base_dir / "doe_data" / f"case_{case_idx:04d}" / "postproc" / "Mag_OnLoadTorque_result_1.h5"
    if not h5_path.exists():
        return None

    with h5py.File(h5_path, "r") as f:
        n_id = np.asarray(f["mesh/node_id"][:], dtype=np.int32)
        n_1 = np.asarray(f["mesh/node_1"][:], dtype=np.int32)
        n_2 = np.asarray(f["mesh/node_2"][:], dtype=np.int32)
        n_3 = np.asarray(f["mesh/node_3"][:], dtype=np.int32)

    sorted_ids = np.sort(n_id)
    max_nid = int(sorted_ids.max()) + 1
    lut = np.full(max_nid, -1, dtype=np.int64)
    for i, nid in enumerate(sorted_ids):
        lut[int(nid)] = i

    m = min(len(n_1), len(n_2), len(n_3))
    i1 = lut[np.clip(n_1[:m], 0, max_nid - 1)]
    i2 = lut[np.clip(n_2[:m], 0, max_nid - 1)]
    i3 = lut[np.clip(n_3[:m], 0, max_nid - 1)]
    valid = (i1 >= 0) & (i2 >= 0) & (i3 >= 0)
    return np.column_stack([i1[valid], i2[valid], i3[valid]])


def _triangulation(x: np.ndarray, y: np.ndarray, mesh_triangles: Optional[np.ndarray]) -> mtri.Triangulation:
    if mesh_triangles is None:
        return mtri.Triangulation(x, y)

    max_idx = len(x)
    valid = (
        (mesh_triangles[:, 0] < max_idx)
        & (mesh_triangles[:, 1] < max_idx)
        & (mesh_triangles[:, 2] < max_idx)
    )
    return mtri.Triangulation(x, y, triangles=mesh_triangles[valid])


def _field(bx: np.ndarray, by: np.ndarray, quantity: str) -> np.ndarray:
    q = quantity.lower().strip()
    if q == "bx":
        return bx
    if q == "by":
        return by
    return np.sqrt(bx**2 + by**2)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    r2 = float(1.0 - np.sum(err**2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12))
    return mae, rmse, r2


def plot_model_comparison_grid(
    node_x_all: np.ndarray,
    node_y_all: np.ndarray,
    data_dict: Dict[str, FieldPair],
    *,
    step: int,
    quantity: str = "bmag",
    models: Iterable[str] = ("MGN", "FNO", "GINO", "RNN"),
    mesh_triangles: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
    dpi: int = 220,
) -> plt.Figure:
    """Plot per-model Pred/Error against GT for a given step and quantity."""
    model_list = [m for m in models if m in data_dict and m != "GT"]
    if not model_list:
        raise ValueError("No valid model names found in data_dict.")

    x = node_x_all[step]
    y = node_y_all[step]
    tri = _triangulation(x, y, mesh_triangles)

    gt_bx, gt_by = data_dict["GT"][0][step], data_dict["GT"][1][step]
    v_gt = _field(gt_bx, gt_by, quantity)

    pred_fields = {}
    errors = {}
    for m in model_list:
        bx_p, by_p = data_dict[m][0][step], data_dict[m][1][step]
        v_p = _field(bx_p, by_p, quantity)
        pred_fields[m] = v_p
        errors[m] = v_p - v_gt

    vmin = min([float(v_gt.min())] + [float(pred_fields[m].min()) for m in model_list])
    vmax = max([float(v_gt.max())] + [float(pred_fields[m].max()) for m in model_list])
    err_lim = max(max(abs(float(errors[m].min())), abs(float(errors[m].max()))) for m in model_list)
    err_lim = max(err_lim, 1e-12)

    fig, axes = plt.subplots(len(model_list), 3, figsize=(14, 4.0 * len(model_list)), dpi=dpi, layout="constrained")
    if len(model_list) == 1:
        axes = np.array([axes])

    qty_label = "|B| [T]" if quantity.lower() == "bmag" else f"{quantity.upper()} [T]"

    for i, m in enumerate(model_list):
        ax_gt, ax_pred, ax_err = axes[i, 0], axes[i, 1], axes[i, 2]

        c0 = ax_gt.tripcolor(tri, v_gt, shading="flat", cmap="jet", vmin=vmin, vmax=vmax)
        ax_gt.set_title(f"GT ({qty_label})")
        fig.colorbar(c0, ax=ax_gt, fraction=0.046, pad=0.04)

        c1 = ax_pred.tripcolor(tri, pred_fields[m], shading="flat", cmap="jet", vmin=vmin, vmax=vmax)
        mae, rmse, r2 = _metrics(v_gt, pred_fields[m])
        ax_pred.set_title(f"{m} Pred | MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        fig.colorbar(c1, ax=ax_pred, fraction=0.046, pad=0.04)

        c2 = ax_err.tripcolor(tri, errors[m], shading="flat", cmap="RdBu_r", vmin=-err_lim, vmax=err_lim)
        ax_err.set_title(f"{m} Error (Pred-GT)")
        fig.colorbar(c2, ax=ax_err, fraction=0.046, pad=0.04)

        for ax in (ax_gt, ax_pred, ax_err):
            ax.set_aspect("equal")
            ax.axis("off")

    fig.suptitle(f"Model Comparison @ step={step}, quantity={quantity}", fontsize=13)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig
