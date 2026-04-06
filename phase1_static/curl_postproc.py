"""Mesh-edge finite-difference curl(A) for post-processing.

Computes B = curl(A) using graph edge topology instead of autograd.grad.
This avoids TransformerEngine / @once_differentiable backward conflicts
and can be used as:
  1. Post-processing visualization (compare curl(A_pred) vs B_pred)
  2. Physics consistency metric (no autograd required)
  3. Future: trainable regularizer (pure tensor ops, 1st-order backward safe)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def mesh_edge_curl_b(
    pos: np.ndarray,
    a_values: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """Compute B = curl(A) via mesh-edge finite differences.

    For 2D magnetostatics:
        Bx = dA/dy,  By = -dA/dx

    Each edge (i→j) provides a directional derivative estimate.
    Per-node B is obtained by least-squares averaging over all incident edges.

    Parameters
    ----------
    pos : (N, 2) array
        Node coordinates [x, y].
    a_values : (N,) array
        Scalar magnetic vector potential A at each node.
    edge_index : (2, E) array
        Edge connectivity [source_nodes, target_nodes].

    Returns
    -------
    b_curl : (N, 2) array
        Estimated [Bx, By] at each node from curl(A).
    """
    pos = np.asarray(pos, dtype=np.float64)
    a_values = np.asarray(a_values, dtype=np.float64).ravel()
    edge_index = np.asarray(edge_index, dtype=np.int64)

    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be (N,2), got {pos.shape}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be (2,E), got {edge_index.shape}")

    n_nodes = pos.shape[0]
    src, dst = edge_index[0], edge_index[1]

    dx = pos[dst, 0] - pos[src, 0]  # (E,)
    dy = pos[dst, 1] - pos[src, 1]  # (E,)
    da = a_values[dst] - a_values[src]  # (E,)

    # Per-node least-squares gradient reconstruction.
    # For node i with neighbors j: ΔA_j = gx·Δx_j + gy·Δy_j
    # Normal equations: [[Σdx², Σdx·dy], [Σdx·dy, Σdy²]] · [gx, gy] = [Σdx·dA, Σdy·dA]
    sum_dx2 = np.zeros(n_nodes, dtype=np.float64)
    sum_dy2 = np.zeros(n_nodes, dtype=np.float64)
    sum_dxdy = np.zeros(n_nodes, dtype=np.float64)
    sum_dx_da = np.zeros(n_nodes, dtype=np.float64)
    sum_dy_da = np.zeros(n_nodes, dtype=np.float64)

    np.add.at(sum_dx2, src, dx * dx)
    np.add.at(sum_dy2, src, dy * dy)
    np.add.at(sum_dxdy, src, dx * dy)
    np.add.at(sum_dx_da, src, dx * da)
    np.add.at(sum_dy_da, src, dy * da)

    # Solve 2x2 system per node: det = dx2*dy2 - dxdy^2
    det = sum_dx2 * sum_dy2 - sum_dxdy * sum_dxdy
    valid = np.abs(det) > 1e-30
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]

    grad_x = (sum_dy2 * sum_dx_da - sum_dxdy * sum_dy_da) * inv_det
    grad_y = (sum_dx2 * sum_dy_da - sum_dxdy * sum_dx_da) * inv_det

    # Bx = dA/dy,  By = -dA/dx
    bx_curl = grad_y
    by_curl = -grad_x

    return np.stack([bx_curl, by_curl], axis=1)  # (N, 2)


def curl_consistency_metrics(
    pos: np.ndarray,
    a_pred: np.ndarray,
    b_pred: np.ndarray,
    edge_index: np.ndarray,
) -> dict:
    """Compute curl(A_pred) vs B_pred consistency metrics.

    Parameters
    ----------
    pos : (N, 2) node coordinates
    a_pred : (N,) predicted A values
    b_pred : (N, 2) predicted [Bx, By] values
    edge_index : (2, E) edge connectivity

    Returns
    -------
    dict with keys:
        curl_b : (N, 2) curl(A_pred) estimate
        rmse_bx : float — RMSE between curl_Bx and pred_Bx
        rmse_by : float — RMSE between curl_By and pred_By
        rmse_total : float — RMSE over both components
        max_abs_error : float — maximum |curl(A) - B_pred|
    """
    b_pred = np.asarray(b_pred, dtype=np.float64)
    if b_pred.ndim == 1:
        raise ValueError("b_pred must be (N,2)")

    curl_b = mesh_edge_curl_b(pos, a_pred, edge_index)
    diff = curl_b - b_pred

    rmse_bx = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
    rmse_by = float(np.sqrt(np.mean(diff[:, 1] ** 2)))
    rmse_total = float(np.sqrt(np.mean(diff ** 2)))
    max_abs_error = float(np.max(np.abs(diff)))

    return {
        "curl_b": curl_b,
        "rmse_bx": rmse_bx,
        "rmse_by": rmse_by,
        "rmse_total": rmse_total,
        "max_abs_error": max_abs_error,
    }
