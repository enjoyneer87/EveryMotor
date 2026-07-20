"""Sector symmetry for the 1/8 anti-periodic Motor-CAD model.

The DOE mesh is a 45-degree sector of an 8-fold machine with anti-periodic
boundaries: rotating the rotor by one sector reproduces the field with the sign
flipped. Measured on the export, on the stationary airgap band:

    corr(B_r at -44 deg, B_r at 0 deg)  = -0.994     (one sector  -> sign flip)
    corr(B_r at -88 deg, B_r at 0 deg)  = +0.981     (two sectors -> sign back)
    corr(B_r at -66 deg, B_r at -22 deg) = -0.996

and across the two cut planes, ||A(0) + A(-45)|| / ||A|| = 0.017 while the
periodic hypothesis gives 2.007. The model is anti-periodic, not periodic.

None of this reached the DOE training path. Three consequences, all addressed
here:

1. **`rotate_step` carries no information.** It is the per-step increment and
   takes only the values {0, -2}, so it cannot tell the model which step it is
   on. `rotor_angle_features` supplies the cumulative angle instead.

2. **The export reports the rotor unwrapped.** The rotor rotates rigidly to
   -153 degrees over the sweep while the stator stays at [-45, 0], so at late
   steps the exported geometry is not the domain that was solved.
   `wrap_rotor_coordinates` folds it back, and the field target must be
   multiplied by the returned `sign`.

3. **The two cut planes are unconnected in the graph.** `cut_plane_pairs`
   matches them by radius so an anti-periodic edge (or an anti-periodicity
   penalty) can be built.

What this does *not* fix
------------------------
The rotor-side airgap layer `a2` is a shear layer that the solver redistributes
every step; its exported nodes take a continuum of displacements, and the
reference connectivity inverts 180-246 of its triangles regardless of wrapping.
Wrapping neither causes nor cures that — it is handled by excluding the sliding
band (`eval.mesh_regions.sliding_band_mask`). Verified: wrapping introduces zero
inversions outside the already-excluded band at every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_SECTOR_DEG = 45.0

# Radial tolerance when matching a node on one cut plane to its partner.
CUT_PLANE_RADIUS_TOL_MM = 1e-3
CUT_PLANE_ANGLE_TOL_DEG = 1e-4


class SectorSymmetryError(ValueError):
    """Raised when the sector symmetry cannot be established for a mesh."""


@dataclass(frozen=True)
class SectorGeometry:
    """Immutable description of the modelled sector."""

    sector_deg: float
    n_sectors: int
    anti_periodic: bool = True

    @property
    def sign_per_sector(self) -> float:
        return -1.0 if self.anti_periodic else 1.0


def sector_geometry(sector_deg: float = DEFAULT_SECTOR_DEG, tol: float = 0.02) -> SectorGeometry:
    """Sector description, checking that the span divides the full circle."""
    if sector_deg <= 0.0:
        raise SectorSymmetryError(f"Non-positive sector span {sector_deg}")
    ratio = 360.0 / sector_deg
    nearest = int(round(ratio))
    if nearest < 1 or abs(ratio - nearest) > tol * max(nearest, 1):
        raise SectorSymmetryError(
            f"Sector span {sector_deg} deg does not divide 360 deg (got {ratio:.4f} sectors)"
        )
    return SectorGeometry(sector_deg=float(sector_deg), n_sectors=nearest)


def wrap_rotor_angle(
    cumulative_deg: float,
    sector_deg: float = DEFAULT_SECTOR_DEG,
) -> Tuple[float, int, float]:
    """Fold a cumulative rotor angle into one sector.

    Returns ``(wrapped_deg, k, sign)`` with ``wrapped_deg`` in ``[-sector, 0)``,
    ``wrapped_deg == cumulative_deg + k * sector_deg`` and ``sign == (-1)**k``.

    The sign is the anti-periodic factor: a field solved at `cumulative_deg`
    equals `sign` times the field of the wrapped configuration, so wrapping the
    geometry without applying it to the target silently mislabels the data.
    """
    sector = float(sector_deg)
    wrapped = cumulative_deg - sector * np.floor(cumulative_deg / sector) - sector
    k = int(round((wrapped - cumulative_deg) / sector))
    return float(wrapped), k, float((-1.0) ** k)


def cumulative_rotor_angle(rotate_steps: Sequence[float]) -> np.ndarray:
    """Cumulative rotor angle per step from the per-step increments.

    `meta/rotate_step` stores the increment, not the absolute angle, so it must
    be accumulated. The first entry is the angle at step 0 (zero).
    """
    inc = np.asarray(rotate_steps, dtype=np.float64)
    return np.cumsum(np.nan_to_num(inc, nan=0.0)) - float(np.nan_to_num(inc[0], nan=0.0) if inc.size else 0.0)


def rotor_angle_features(
    cumulative_deg: float,
    sector_deg: float = DEFAULT_SECTOR_DEG,
) -> Tuple[float, float]:
    """``(sin, cos)`` of the rotor angle at the sector's angular frequency.

    Using ``2*pi/(2*sector)`` — i.e. a full electrical period of two sectors —
    makes the encoding respect the anti-periodicity: advancing by one sector
    moves the phase by pi, which is where the sign flip lives. A plain
    ``sin(theta)`` would make positions one sector apart look unrelated, and a
    ``2*pi/sector`` encoding would make them look identical, losing the sign.
    """
    phase = np.pi * float(cumulative_deg) / float(sector_deg)
    return float(np.sin(phase)), float(np.cos(phase))


def rigid_rotor_node_mask(
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray],
    reg_code: Sequence[int],
    name_of_code: Mapping[int, str],
    moving_reg_codes: Sequence[int],
    n_nodes: int,
) -> np.ndarray:
    """Nodes belonging to rigidly rotating rotor regions.

    Everything that moves except the `a2` shear layer, whose nodes are
    redistributed rather than rotated and must not be transformed.
    """
    from eval.mesh_regions import AIRGAP_NAME_PATTERN

    reg = np.asarray(reg_code, dtype=np.int64)
    moving = {int(c) for c in moving_reg_codes}

    shear = {
        c for c in moving
        if AIRGAP_NAME_PATTERN.match(str(name_of_code.get(int(c), "")))
        and str(name_of_code.get(int(c), "")).lower() == "a2"
    }
    rigid = moving - shear

    mask = np.zeros(int(n_nodes), dtype=bool)
    rigid_elems = np.isin(reg, np.asarray(sorted(rigid), dtype=np.int64)) if rigid else np.zeros(reg.size, bool)
    for idx in tri:
        mask[np.asarray(idx, dtype=np.int64)[rigid_elems]] = True

    # a2's own nodes are never moved, even where they are shared.
    if shear:
        shear_elems = np.isin(reg, np.asarray(sorted(shear), dtype=np.int64))
        for idx in tri:
            mask[np.asarray(idx, dtype=np.int64)[shear_elems]] = False
    return mask


def wrap_rotor_coordinates(
    node_x_mm: np.ndarray,
    node_y_mm: np.ndarray,
    rigid_mask: np.ndarray,
    cumulative_deg: float,
    sector_deg: float = DEFAULT_SECTOR_DEG,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fold the rigid rotor back into the modelled sector.

    Returns ``(x, y, sign)``. **The caller must multiply the field target by
    `sign`** — the wrapped geometry corresponds to `sign` times the exported
    field, and dropping that factor mislabels every wrapped sample.
    """
    _, k, sign = wrap_rotor_angle(cumulative_deg, sector_deg)
    x = np.array(node_x_mm, dtype=np.float64, copy=True)
    y = np.array(node_y_mm, dtype=np.float64, copy=True)
    if k == 0:
        return x, y, sign

    angle = np.radians(k * float(sector_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    sel = np.asarray(rigid_mask, dtype=bool)
    x[sel] = cos_a * node_x_mm[sel] - sin_a * node_y_mm[sel]
    y[sel] = sin_a * node_x_mm[sel] + cos_a * node_y_mm[sel]
    return x, y, sign


def cut_plane_pairs(
    node_x_mm: np.ndarray,
    node_y_mm: np.ndarray,
    sector_deg: float = DEFAULT_SECTOR_DEG,
    radius_tol_mm: float = CUT_PLANE_RADIUS_TOL_MM,
    angle_tol_deg: float = CUT_PLANE_ANGLE_TOL_DEG,
    min_radius_mm: float = 1e-6,
) -> np.ndarray:
    """Match nodes on the two cut planes by radius.

    Returns an ``(P, 2)`` array of node index pairs ``(on theta=0, on
    theta=-sector)``. These are the same physical location one sector apart, so
    their potentials satisfy ``A(theta=0) = -A(theta=-sector)``.

    Matching is by radius because the mesh is generated independently on the two
    faces and the node counts differ (69 vs 74 on the DOE meshes); only nodes
    with a partner within `radius_tol_mm` are paired.
    """
    x = np.asarray(node_x_mm, dtype=np.float64)
    y = np.asarray(node_y_mm, dtype=np.float64)
    theta = np.degrees(np.arctan2(y, x))
    radius = np.hypot(x, y)

    on_zero = np.flatnonzero((np.abs(theta) < angle_tol_deg) & (radius > min_radius_mm))
    on_sector = np.flatnonzero((np.abs(theta + sector_deg) < angle_tol_deg) & (radius > min_radius_mm))
    if on_zero.size == 0 or on_sector.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    r_sector = radius[on_sector]
    order = np.argsort(r_sector)
    sorted_r = r_sector[order]

    pairs = []
    for node in on_zero:
        j = int(np.searchsorted(sorted_r, radius[node]))
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < sorted_r.size and abs(sorted_r[cand] - radius[node]) <= radius_tol_mm:
                pairs.append((int(node), int(on_sector[order[cand]])))
                break
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def anti_periodic_edges(
    pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bidirectional edges across the sector cut, with their sign.

    Returns ``(edge_index, sign)`` where `edge_index` is ``(2, 2P)`` and `sign`
    is ``(2P,)`` of -1.0. Interior edges carry +1.0, so a model that multiplies
    each message by its edge sign reproduces the anti-periodic boundary
    condition (the convention `phase1_static/pbc_boundary.py` already uses).
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.size == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float64)

    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    edge_index = np.stack([src, dst], axis=0)
    return edge_index, np.full(edge_index.shape[1], -1.0, dtype=np.float64)


def anti_periodicity_residual(
    values: np.ndarray,
    pairs: np.ndarray,
) -> float:
    """Relative violation of ``v(theta=0) = -v(theta=-sector)``.

    Zero when the field is exactly anti-periodic. On the FEM export this is
    ~0.017, which is the level a trained model should be held to.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.size == 0:
        return float("nan")
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    a, b = v[pairs[:, 0]], v[pairs[:, 1]]
    denom = float(np.sqrt(np.mean(a**2)))
    return float(np.sqrt(np.mean((a + b) ** 2)) / denom) if denom > 0 else float("nan")
