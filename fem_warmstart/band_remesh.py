"""Rebuild the air-gap shear layer so every rotor position is solvable.

Motor-CAD re-meshes the sliding band at each rotor step but the H5 export stores
only the *reference* connectivity, so combining it with the per-step coordinates
folds the `a2` layer over itself from step 3 onward (180 of its 366 elements
invert; measured on case_0004, and matching `eval.mesh_regions.sliding_band_mask`).
Steps 0-2 are the only ones usable as exported -- too few rotor positions to say
anything about warm starting.

This module discards `a2` and re-triangulates the annular strip between the two
rings that bound it:

    outer ring   a1's inner arc, r = 68.682 mm, fixed to the stator, theta in [-45, 0]
    inner ring   a3's outer arc, r = 68.432 mm, rigid with the rotor

Both rings carry exactly one sector of nodes, so the rotor ring is first folded
into ``[-45, 0]``: a node at ``theta`` outside the sector is represented by a new
node at ``theta + 45k`` carrying ``a_new = (-1)^k a_old``. The sign is handed to
the caller as an ordinary anti-periodic constraint, so the assembler never needs
per-element signs.

The strip is then a standard two-ring triangulation: walk both sorted rings from
theta = -45 to theta = 0, always advancing the ring whose next node has the
smaller angle, and emit one triangle per advance. That produces a conforming,
positively-oriented, non-overlapping strip for any rotor angle -- which is what
the sliding band is supposed to be.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

SECTOR_DEG = 45.0


@dataclass
class BandRemesh:
    """Result of rebuilding the shear layer."""

    x_mm: np.ndarray                 # node coords, original nodes plus images
    y_mm: np.ndarray
    tri: np.ndarray                  # (E,3) connectivity with a2 replaced
    source_element: np.ndarray       # (E,) original element index, -1 for new band elements
    constraints: List[Tuple[int, int, float]]   # (image node, original node, sign)
    n_original_nodes: int
    report: dict


def _fold(theta_deg: np.ndarray, sector_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fold angles into (-sector, 0] and return ``(folded, sign)``."""
    k = np.ceil(-theta_deg / sector_deg - 1.0 + 1e-12)
    folded = theta_deg + k * sector_deg
    sign = np.where(np.abs(k) % 2 == 0, 1.0, -1.0)
    return folded, sign


def rebuild_band(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    tri: np.ndarray,
    reg_code: np.ndarray,
    name_of_code,
    sector_deg: float = SECTOR_DEG,
    arc_tol_mm: float = 0.02,
) -> BandRemesh:
    x_mm = np.asarray(x_mm, dtype=np.float64)
    y_mm = np.asarray(y_mm, dtype=np.float64)
    tri = np.asarray(tri, dtype=np.int64)
    reg_code = np.asarray(reg_code, dtype=np.int64)
    code_of = {str(v).strip().lower(): int(k) for k, v in name_of_code.items()}
    for layer in ("a1", "a2", "a3"):
        if layer not in code_of:
            raise ValueError(f"missing air-gap layer {layer!r}; cannot rebuild the band")

    radius = np.hypot(x_mm, y_mm)
    theta = np.degrees(np.arctan2(y_mm, x_mm))

    a1_nodes = np.unique(tri[reg_code == code_of["a1"]].ravel())
    a3_nodes = np.unique(tri[reg_code == code_of["a3"]].ravel())
    r_outer = float(radius[a1_nodes].min())     # a1's inner arc = stator side of the band
    r_inner = float(radius[a3_nodes].max())     # a3's outer arc = rotor side of the band
    outer = a1_nodes[np.abs(radius[a1_nodes] - r_outer) < arc_tol_mm]
    inner = a3_nodes[np.abs(radius[a3_nodes] - r_inner) < arc_tol_mm]

    # --- outer (stator) ring: already in [-45, 0] ---------------------------
    o_theta = theta[outer]
    order = np.argsort(o_theta)
    outer, o_theta = outer[order], o_theta[order]

    # --- inner (rotor) ring: fold into (-45, 0], creating image nodes -------
    i_theta_raw = theta[inner]
    i_theta, i_sign = _fold(i_theta_raw, sector_deg)
    order = np.argsort(i_theta)
    inner, i_theta, i_sign, i_theta_raw = (
        inner[order], i_theta[order], i_sign[order], i_theta_raw[order])

    new_x: List[float] = []
    new_y: List[float] = []
    constraints: List[Tuple[int, int, float]] = []
    n0 = int(x_mm.size)

    def image_node(node: int, target_theta: float, sign: float) -> int:
        """Node index carrying ``sign * a_node`` at ``target_theta`` on the ring."""
        if abs(sign - 1.0) < 1e-12 and abs(target_theta - theta[node]) < 1e-9:
            return int(node)
        idx = n0 + len(new_x)
        rad = radius[node]
        ang = np.radians(target_theta)
        new_x.append(float(rad * np.cos(ang)))
        new_y.append(float(rad * np.sin(ang)))
        constraints.append((idx, int(node), float(sign)))
        return idx

    ring_nodes: List[int] = []
    ring_theta: List[float] = []
    for node, t_folded, sgn, t_raw in zip(inner, i_theta, i_sign, i_theta_raw):
        ring_nodes.append(image_node(int(node), float(t_folded), float(sgn)))
        ring_theta.append(float(t_folded))

    # Extend the rotor ring past both ends of the sector so the strip closes:
    # the ring is periodic with period 45 deg and sign -1, so the node just past
    # theta = 0 is the image of the first node, and the one just before -45 is the
    # image of the last.
    first_node, first_t = int(inner[0]), float(i_theta[0])
    last_node, last_t = int(inner[-1]), float(i_theta[-1])
    pre = image_node(last_node, last_t - sector_deg, -float(i_sign[-1]))
    post = image_node(first_node, first_t + sector_deg, -float(i_sign[0]))
    ring_nodes = [pre] + ring_nodes + [post]
    ring_theta = [last_t - sector_deg] + ring_theta + [first_t + sector_deg]

    # --- two-ring strip triangulation over [-45, 0] -------------------------
    ring_theta_arr = np.asarray(ring_theta)
    j = int(np.searchsorted(ring_theta_arr, -sector_deg, side="right")) - 1
    j = max(j, 0)
    i = 0
    new_tri: List[Tuple[int, int, int]] = []
    while i < len(outer) - 1 or j < len(ring_nodes) - 1:
        take_outer = (
            j >= len(ring_nodes) - 1
            or (i < len(outer) - 1 and o_theta[i + 1] <= ring_theta_arr[j + 1])
        )
        if take_outer:
            new_tri.append((int(outer[i]), int(ring_nodes[j]), int(outer[i + 1])))
            i += 1
        else:
            new_tri.append((int(outer[i]), int(ring_nodes[j]), int(ring_nodes[j + 1])))
            j += 1
        if o_theta[min(i, len(outer) - 1)] >= 0.0 and ring_theta_arr[j] >= 0.0:
            break

    keep = reg_code != code_of["a2"]
    tri_out = np.concatenate([tri[keep], np.asarray(new_tri, dtype=np.int64)], axis=0)
    source = np.concatenate([
        np.flatnonzero(keep),
        np.full(len(new_tri), -1, dtype=np.int64),
    ])

    return BandRemesh(
        x_mm=np.concatenate([x_mm, np.asarray(new_x)]),
        y_mm=np.concatenate([y_mm, np.asarray(new_y)]),
        tri=tri_out,
        source_element=source,
        constraints=constraints,
        n_original_nodes=n0,
        report={
            "band_r_outer_mm": r_outer,
            "band_r_inner_mm": r_inner,
            "n_outer_ring": int(outer.size),
            "n_inner_ring": int(inner.size),
            "n_a2_removed": int(np.sum(~keep)),
            "n_band_elements": len(new_tri),
            "n_image_nodes": len(new_x),
        },
    )
