"""FEM P1 element-wise curl: nodal A -> element B.

Why this replaces the autograd curl
-----------------------------------
The roadmap's `autograd.grad(A_pred, coords)` is not a spatial derivative for a
message-passing model: node i's prediction depends on its neighbours' coordinates
too, so the autograd derivative mixes in the message-passing dependency and is
not dA/dx|_i (methodology review F4).

For linear (P1) triangles the gradient of A is *exactly* constant per element and
is a fixed linear map of the three nodal values. Precomputing that map gives an
operator that is exact, cheap, and differentiable by construction — no autograd
through the mesh at all.

Why this also fixes the support mismatch
----------------------------------------
Motor-CAD stores B per element. A node-predicting model that outputs B must be
averaged back onto elements to be scored, and that element -> node -> element
round trip costs ~17% |B| nRMSE and ~60% torque nRMSE on this data — the airgap
band is one element thick, so its nodes pick up stator-iron flux density.

Predicting A at nodes and taking the curl lands B directly on elements, with no
round trip. Measured on the DOE export, the best P1-curl representation of the
FEM field reaches ~5-8% |B| nRMSE and **1.3-3.3% torque nRMSE**, against ~60% for
the round-trip pipeline.

Formulae
--------
For a triangle with nodes (x1,y1), (x2,y2), (x3,y3) and nodal values A1, A2, A3::

    b = (y2-y3, y3-y1, y1-y2)
    c = (x3-x2, x1-x3, x2-x1)
    2S = (x2*y3 - x3*y2) + (x3*y1 - x1*y3) + (x1*y2 - x2*y1)      (signed)

    dA/dx = sum_i A_i * b_i / 2S
    dA/dy = sum_i A_i * c_i / 2S

and for the 2D magnetostatic potential A = A_z * z_hat::

    Bx =  dA/dy      By = -dA/dx

The signed 2S is used deliberately: flipping node order flips both 2S and the
b/c coefficients, so the result is independent of triangle orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# Triangles with |2S| below this (m^2) are treated as degenerate and dropped.
DEGENERATE_AREA_TOL = 1e-14


class CurlOperatorError(ValueError):
    """Raised when a usable P1 curl operator cannot be built."""


@dataclass(frozen=True)
class P1CurlOperator:
    """Immutable, precomputed nodal-A -> element-B linear map.

    Built once per (mesh, rotor position); applying it is three gathers and a
    weighted sum, so it is cheap enough to sit inside a training loop and is
    differentiable with respect to the nodal values.

    Attributes
    ----------
    node_index:
        ``(K, 3)`` node indices of each retained element.
    coef_bx, coef_by:
        ``(K, 3)`` coefficients such that ``Bx = sum_j coef_bx[:, j] * A[node_index[:, j]]``.
    element_index:
        ``(K,)`` positions of the retained elements within the full element
        array, so element-wise truth can be sliced to match.
    n_nodes, n_elements:
        Sizes of the full mesh, retained or not.
    """

    node_index: np.ndarray
    coef_bx: np.ndarray
    coef_by: np.ndarray
    element_index: np.ndarray
    n_nodes: int
    n_elements: int

    @property
    def n_valid(self) -> int:
        return int(self.element_index.size)

    @property
    def coverage(self) -> float:
        """Fraction of mesh elements the operator produces B on."""
        return float(self.n_valid) / float(self.n_elements) if self.n_elements else 0.0

    def describe(self) -> dict:
        return {
            "method": "p1_element_curl",
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "n_valid_elements": self.n_valid,
            "coverage": self.coverage,
        }


def signed_double_area(
    node_x_m: np.ndarray,
    node_y_m: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Signed 2*area of every triangle, in m^2.

    The sign encodes node ordering; `mesh_validity_mask` uses a change of sign
    against a reference configuration to detect elements that have inverted.
    """
    i1, i2, i3 = (np.asarray(a, dtype=np.int64) for a in tri)
    x1, x2, x3 = node_x_m[i1], node_x_m[i2], node_x_m[i3]
    y1, y2, y3 = node_y_m[i1], node_y_m[i2], node_y_m[i3]
    return (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3) + (x1 * y2 - x2 * y1)


def mesh_validity_mask(
    node_x_m: np.ndarray,
    node_y_m: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_sign: Optional[np.ndarray] = None,
    exclude: Optional[np.ndarray] = None,
    area_tol: float = DEGENERATE_AREA_TOL,
) -> np.ndarray:
    """Boolean mask of elements whose geometry is usable at this configuration.

    An element is dropped when it is degenerate, when it has inverted relative to
    `reference_sign`, or when `exclude` marks it.

    Inversion is not hypothetical on this data: in the Motor-CAD moving-mesh
    export the rotor-side airgap layers are re-meshed by the solver at every
    step, but only the reference connectivity is written to the H5. Combining
    that connectivity with the per-step node coordinates tangles 180-246
    elements from step 3 onward. See `eval.mesh_regions.sliding_band_mask`.
    """
    area2 = signed_double_area(node_x_m, node_y_m, tri)
    ok = np.abs(area2) > float(area_tol)
    if reference_sign is not None:
        ok &= np.sign(area2) == np.sign(np.asarray(reference_sign))
    if exclude is not None:
        ok &= ~np.asarray(exclude, dtype=bool)
    return ok


def build_p1_curl_operator(
    node_x_m: np.ndarray,
    node_y_m: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray],
    valid: Optional[np.ndarray] = None,
) -> P1CurlOperator:
    """Precompute the nodal-A -> element-B map.

    Parameters
    ----------
    node_x_m, node_y_m:
        Node coordinates **in metres**. Passing millimetres silently scales B by
        1000, so convert before calling.
    tri:
        ``(i1, i2, i3)`` element connectivity as node positions.
    valid:
        Optional boolean element mask (see `mesh_validity_mask`). Elements not
        selected produce no output rather than producing a wrong one.
    """
    node_x_m = np.asarray(node_x_m, dtype=np.float64)
    node_y_m = np.asarray(node_y_m, dtype=np.float64)
    i1, i2, i3 = (np.asarray(a, dtype=np.int64) for a in tri)
    n_elements = int(i1.size)

    if valid is None:
        valid = mesh_validity_mask(node_x_m, node_y_m, tri)
    element_index = np.flatnonzero(np.asarray(valid, dtype=bool))
    if element_index.size == 0:
        raise CurlOperatorError("No valid elements remain; cannot build a curl operator")

    j1, j2, j3 = i1[element_index], i2[element_index], i3[element_index]
    x1, x2, x3 = node_x_m[j1], node_x_m[j2], node_x_m[j3]
    y1, y2, y3 = node_y_m[j1], node_y_m[j2], node_y_m[j3]

    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)
    two_area = ((x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3) + (x1 * y2 - x2 * y1))[:, None]

    return P1CurlOperator(
        node_index=np.stack([j1, j2, j3], axis=1),
        coef_bx=c / two_area,      # Bx =  dA/dy
        coef_by=-b / two_area,     # By = -dA/dx
        element_index=element_index,
        n_nodes=int(node_x_m.size),
        n_elements=n_elements,
    )


def curl_a_to_b(operator: P1CurlOperator, a_nodal: np.ndarray) -> np.ndarray:
    """Apply the operator: nodal A ``(n_nodes,)`` -> element B ``(n_valid, 2)``."""
    a = np.asarray(a_nodal, dtype=np.float64).reshape(-1)
    if a.size != operator.n_nodes:
        raise CurlOperatorError(
            f"A has {a.size} nodal values but the operator was built for {operator.n_nodes} nodes"
        )
    gathered = a[operator.node_index]                       # (K, 3)
    bx = np.einsum("kj,kj->k", operator.coef_bx, gathered)
    by = np.einsum("kj,kj->k", operator.coef_by, gathered)
    return np.stack([bx, by], axis=1)


def curl_a_to_b_torch(operator: P1CurlOperator, a_nodal, device=None):
    """Differentiable torch version of `curl_a_to_b`.

    Gradients flow to `a_nodal`; the coefficients are mesh constants. Import of
    torch is deferred so the numpy path stays dependency-free.
    """
    import torch

    if a_nodal.dim() == 2 and a_nodal.shape[1] == 1:
        a_nodal = a_nodal.squeeze(1)
    if a_nodal.dim() != 1:
        raise CurlOperatorError(f"A must be [n_nodes] or [n_nodes, 1], got {tuple(a_nodal.shape)}")

    device = device or a_nodal.device
    idx = torch.as_tensor(operator.node_index, dtype=torch.long, device=device)
    cbx = torch.as_tensor(operator.coef_bx, dtype=a_nodal.dtype, device=device)
    cby = torch.as_tensor(operator.coef_by, dtype=a_nodal.dtype, device=device)

    gathered = a_nodal[idx]                                  # (K, 3)
    return torch.stack([(cbx * gathered).sum(1), (cby * gathered).sum(1)], dim=1)


def fit_nodal_a(
    operator: P1CurlOperator,
    b_element: np.ndarray,
    damp: float = 0.0,
    iter_lim: int = 6000,
    tol: float = 1e-11,
) -> Tuple[np.ndarray, float]:
    """Least-squares nodal A whose curl best reproduces a given element B field.

    Used to establish the representation floor — the best any A-predicting model
    could do on a mesh — and to generate nodal A supervision targets from an
    export that only stores element-averaged A. (Curl of the *exported* A is a
    poor reference: it scores ~39% |B| nRMSE, because differentiating an already
    element-averaged potential amplifies the averaging error.)

    A is determined only up to an additive constant per connected component; the
    solver returns one representative, which is harmless since only its curl is
    ever used.

    Returns
    -------
    ``(a_nodal, relative_residual)`` where the residual is
    ``||curl(A) - B|| / ||B||``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    b_element = np.asarray(b_element, dtype=np.float64)
    if b_element.shape != (operator.n_valid, 2):
        raise CurlOperatorError(
            f"B must be ({operator.n_valid}, 2) to match the operator, got {b_element.shape}"
        )

    k = operator.n_valid
    rows = np.concatenate([np.repeat(np.arange(k), 3), np.repeat(np.arange(k, 2 * k), 3)])
    cols = np.tile(operator.node_index.ravel(), 2)
    vals = np.concatenate([operator.coef_bx.ravel(), operator.coef_by.ravel()])
    matrix = coo_matrix((vals, (rows, cols)), shape=(2 * k, operator.n_nodes)).tocsr()

    rhs = np.concatenate([b_element[:, 0], b_element[:, 1]])
    a_nodal = lsqr(matrix, rhs, damp=damp, atol=tol, btol=tol, iter_lim=iter_lim)[0]

    residual = matrix @ a_nodal - rhs
    denom = float(np.linalg.norm(rhs))
    return a_nodal, float(np.linalg.norm(residual) / denom) if denom > 0 else float("nan")
