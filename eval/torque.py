"""Maxwell-stress (Arkkio) torque from a predicted airgap flux density field.

Why torque
----------
Field nRMSE does not tell a motor designer whether the surrogate is usable. A 5%
field error concentrated in the airgap tangential component can be a 15% torque
error; a 10% error spread over the back-iron can be a 1% torque error. The
review (F6) makes torque the engineering acceptance metric, and gate G2 is
stated in torque terms.

Method
------
Arkkio's method integrates the Maxwell stress over an airgap *band* rather than
a single contour, which is far less sensitive to mesh faceting than a line
integral:

    T = (L * n_sym / (mu0 * (r_out - r_in))) * SUM_e r_e * B_r,e * B_theta,e * dA_e

The band is the stationary stator-side airgap mesh layer. In the Motor-CAD DOE
export the airgap is meshed as concentric layers ``a1..a4``; ``a2..a4`` are listed
in ``mesh/moving_reg_codes`` and rotate with the rotor, while ``a1`` stays fixed
to the stator and spans the full sector at every timestep. Integrating over a1
alone therefore gives a band with consistent angular coverage across the whole
rotation, which is what the formula assumes.

Absolute scale
--------------
``axial_length_m`` defaults to 1.0, i.e. torque per metre of stack. The Motor-CAD
stack length is not carried in the H5 export, and the metric that matters —
relative error between predicted and FEM torque computed by this same operator —
is invariant to it. Pass the real stack length when an absolute N*m number is
wanted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from eval.mesh_regions import AIRGAP_NAME_PATTERN, element_areas_m2, element_centroids_m

MU0 = 4.0e-7 * np.pi

# Sector span must divide the full circle to a near-integer number of sectors.
SECTOR_INTEGER_TOLERANCE = 0.02


class AirgapBandError(ValueError):
    """Raised when a usable airgap integration band cannot be built."""


@dataclass(frozen=True)
class AirgapBand:
    """Immutable, precomputed Arkkio integration operator for one mesh.

    Built once per case (the band is stationary, so it is valid for every
    timestep) and then applied to any number of flux-density fields.

    Attributes
    ----------
    element_index:
        Positions of the band elements within the full element arrays, so a
        caller can slice its own ``(n_elements,)`` field arrays.
    radius_m, area_m2, cos_theta, sin_theta:
        Per-element centroid radius, area and unit radial direction.
    r_inner_m, r_outer_m:
        Radial extent of the band, taken from its *node* radii (not centroid
        radii, which would understate the thickness and inflate the torque).
    symmetry_multiplier:
        Number of such sectors in the full machine.
    """

    element_index: np.ndarray
    radius_m: np.ndarray
    area_m2: np.ndarray
    cos_theta: np.ndarray
    sin_theta: np.ndarray
    r_inner_m: float
    r_outer_m: float
    sector_span_rad: float
    symmetry_multiplier: int
    region_codes: Tuple[int, ...]

    @property
    def n_elements(self) -> int:
        return int(self.element_index.size)

    @property
    def thickness_m(self) -> float:
        return float(self.r_outer_m - self.r_inner_m)

    def describe(self) -> Dict[str, object]:
        """Serializable description, recorded in the results JSON."""
        return {
            "method": "arkkio_band_integral",
            "n_elements": self.n_elements,
            "region_codes": list(self.region_codes),
            "r_inner_m": self.r_inner_m,
            "r_outer_m": self.r_outer_m,
            "thickness_m": self.thickness_m,
            "sector_span_deg": float(np.degrees(self.sector_span_rad)),
            "symmetry_multiplier": self.symmetry_multiplier,
            "area_m2": float(np.sum(self.area_m2)),
        }


def _sector_symmetry(span_rad: float) -> int:
    """Number of sectors in the full machine, from the modelled sector span."""
    if span_rad <= 0.0:
        raise AirgapBandError(f"Non-positive sector span {span_rad}")
    ratio = 2.0 * np.pi / span_rad
    nearest = int(round(ratio))
    if nearest < 1 or abs(ratio - nearest) > SECTOR_INTEGER_TOLERANCE * max(nearest, 1):
        raise AirgapBandError(
            f"Sector span {np.degrees(span_rad):.3f} deg does not divide 360 deg "
            f"(got {ratio:.4f} sectors). Pass symmetry_multiplier explicitly."
        )
    return nearest


def build_airgap_band(
    node_x_mm: np.ndarray,
    node_y_mm: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray],
    reg_code_of_element: Sequence[int],
    name_of_code: Mapping[int, str],
    moving_reg_codes: Optional[Sequence[int]] = None,
    symmetry_multiplier: Optional[int] = None,
) -> AirgapBand:
    """Build the Arkkio band from a Motor-CAD mesh.

    Parameters
    ----------
    node_x_mm, node_y_mm:
        Node coordinates in millimetres, indexed the way ``tri`` indexes them.
    tri:
        ``(i1, i2, i3)`` element connectivity as node positions.
    reg_code_of_element:
        ``mesh/reg_code``.
    name_of_code:
        ``regions/reg_code`` -> ``regions/name``.
    moving_reg_codes:
        ``mesh/moving_reg_codes``. Airgap layers listed here rotate with the
        rotor and are excluded, leaving the stationary stator-side layer.
    symmetry_multiplier:
        Override the sector count instead of deriving it from the span.
    """
    codes = np.asarray(reg_code_of_element, dtype=np.int64)
    airgap_codes = {
        int(c) for c in np.unique(codes) if AIRGAP_NAME_PATTERN.match(name_of_code.get(int(c), ""))
    }
    if not airgap_codes:
        raise AirgapBandError(
            "No airgap regions found (expected Motor-CAD layer names a1, a2, ...). "
            f"Available regions: {sorted(set(name_of_code.values()))[:12]}"
        )

    stationary = airgap_codes - {int(c) for c in (moving_reg_codes or [])}
    # Fall back to the whole airgap when the export carries no motion table;
    # the band is then only trustworthy at the reference rotor position.
    selected = stationary or airgap_codes

    mask = np.isin(codes, np.asarray(sorted(selected), dtype=np.int64))
    element_index = np.flatnonzero(mask)
    if element_index.size == 0:
        raise AirgapBandError(f"Airgap regions {sorted(selected)} contain no elements")

    i1, i2, i3 = (np.asarray(a, dtype=np.int64) for a in tri)
    sub_tri = (i1[mask], i2[mask], i3[mask])

    cx, cy = element_centroids_m(node_x_mm, node_y_mm, sub_tri)
    radius = np.hypot(cx, cy)
    if np.any(radius <= 0.0):
        raise AirgapBandError("Airgap band contains an element centroid at r = 0")

    band_nodes = np.unique(np.concatenate(sub_tri))
    node_r = np.hypot(node_x_mm[band_nodes], node_y_mm[band_nodes]) * 1e-3
    r_inner, r_outer = float(node_r.min()), float(node_r.max())
    if not r_outer > r_inner:
        raise AirgapBandError(f"Degenerate airgap band thickness: {r_inner} .. {r_outer} m")

    theta = np.arctan2(cy, cx)
    span = float(theta.max() - theta.min())
    n_sym = int(symmetry_multiplier) if symmetry_multiplier else _sector_symmetry(span)

    return AirgapBand(
        element_index=element_index,
        radius_m=radius,
        area_m2=element_areas_m2(node_x_mm, node_y_mm, sub_tri),
        cos_theta=cx / radius,
        sin_theta=cy / radius,
        r_inner_m=r_inner,
        r_outer_m=r_outer,
        sector_span_rad=span,
        symmetry_multiplier=n_sym,
        region_codes=tuple(sorted(selected)),
    )


def arkkio_torque(
    band: AirgapBand,
    bx_elem: np.ndarray,
    by_elem: np.ndarray,
    axial_length_m: float = 1.0,
) -> float:
    """Electromagnetic torque from an element-wise flux density field.

    Parameters
    ----------
    band:
        Operator from `build_airgap_band`.
    bx_elem, by_elem:
        Flux density in tesla, one value per mesh element, indexed over the
        *full* element array (the band slices out what it needs). Passing
        already-sliced band-length arrays is also accepted.
    axial_length_m:
        Stack length. Default 1.0 gives torque per metre of stack.

    Returns
    -------
    Torque in N*m (N*m per metre of stack at the default axial length), for the
    full machine — the sector result is already multiplied by
    ``band.symmetry_multiplier``.
    """
    bx = np.asarray(bx_elem, dtype=np.float64).ravel()
    by = np.asarray(by_elem, dtype=np.float64).ravel()
    if bx.shape != by.shape:
        raise ValueError(f"bx/by shape mismatch: {bx.shape} vs {by.shape}")

    if bx.size == band.n_elements:
        bx_band, by_band = bx, by
    else:
        bx_band, by_band = bx[band.element_index], by[band.element_index]

    b_r = bx_band * band.cos_theta + by_band * band.sin_theta
    b_theta = -bx_band * band.sin_theta + by_band * band.cos_theta

    integrand = band.radius_m * b_r * b_theta * band.area_m2
    finite = np.isfinite(integrand)
    if not np.any(finite):
        return float("nan")

    scale = axial_length_m * band.symmetry_multiplier / (MU0 * band.thickness_m)
    return float(scale * np.sum(integrand[finite]))


def element_to_nodal(
    values: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray],
    n_nodes: int,
) -> np.ndarray:
    """Scatter element values onto nodes by incident-element averaging.

    This is what the training loaders do to turn the Motor-CAD element fields
    into node targets. Pairing it with `nodal_to_element` reproduces the full
    element -> node -> element round trip a node-predicting model is subject to.
    """
    values = np.asarray(values, dtype=np.float64)
    i1, i2, i3 = (np.asarray(a, dtype=np.int64) for a in tri)
    idx = np.concatenate([i1, i2, i3])
    total = np.bincount(idx, weights=np.tile(values, 3), minlength=n_nodes)
    count = np.bincount(idx, minlength=n_nodes).astype(np.float64)
    return total / np.clip(count, 1.0, None)


def nodal_to_element(values: np.ndarray, tri: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Average node-wise values onto elements.

    Graph models predict per node while the Motor-CAD fields (and therefore the
    torque operator) are per element. This is the adjoint of the element->node
    scatter the training loaders perform.
    """
    values = np.asarray(values, dtype=np.float64)
    i1, i2, i3 = (np.asarray(a, dtype=np.int64) for a in tri)
    if values.ndim == 1:
        return (values[i1] + values[i2] + values[i3]) / 3.0
    return (values[i1] + values[i2] + values[i3]) / 3.0


@dataclass(frozen=True)
class TorqueComparison:
    """Immutable predicted-vs-FEM torque comparison for one sample."""

    torque_pred: float
    torque_true: float
    abs_error: float
    rel_error_pct: float
    axial_length_m: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "torque_pred": self.torque_pred,
            "torque_true": self.torque_true,
            "abs_error": self.abs_error,
            "rel_error_pct": self.rel_error_pct,
            "axial_length_m": self.axial_length_m,
        }


def compare_torque(
    band: AirgapBand,
    bx_pred: np.ndarray,
    by_pred: np.ndarray,
    bx_true: np.ndarray,
    by_true: np.ndarray,
    axial_length_m: float = 1.0,
) -> TorqueComparison:
    """Torque error of a prediction against the FEM field on the same band."""
    t_pred = arkkio_torque(band, bx_pred, by_pred, axial_length_m)
    t_true = arkkio_torque(band, bx_true, by_true, axial_length_m)
    abs_err = abs(t_pred - t_true)
    rel = 100.0 * abs_err / abs(t_true) if np.isfinite(t_true) and t_true != 0.0 else float("nan")
    return TorqueComparison(
        torque_pred=t_pred,
        torque_true=t_true,
        abs_error=abs_err,
        rel_error_pct=rel,
        axial_length_m=float(axial_length_m),
    )


# Below this ratio of |mean torque| to RMS torque, the mean is dominated by
# ripple and relative-to-the-mean errors stop being meaningful.
MEAN_TORQUE_SIGNIFICANCE = 0.05


def aggregate_torque(comparisons: Sequence[TorqueComparison]) -> Dict[str, float]:
    """Summarize torque error across samples.

    Headline metric is `nrmse_torque_pct` — torque RMSE normalized by the RMS of
    the FEM torque over the same samples. It is the torque analogue of the field
    nRMSE and is the number gate G2 should be read against.

    Instantaneous relative error (`rel_error_pct_*`) is reported too but is
    fragile by construction: over a rotor sweep the torque passes through zero
    whenever the operating point is near pure d-axis current, and DOE case 32
    (phase advance 89 deg) does exactly that. Dividing by an instantaneous
    torque near a zero crossing produces enormous percentages that say nothing
    about model quality, so those fields must not be used as an acceptance
    criterion on their own.

    Mean torque and ripple are reported separately because a surrogate can track
    the average torque while flattening the ripple, or vice versa, and the two
    failure modes have very different consequences for a DOE screen.
    """
    valid = [c for c in comparisons if np.isfinite(c.torque_true) and np.isfinite(c.torque_pred)]
    if not valid:
        nan = float("nan")
        return {
            "n": 0,
            "nrmse_torque_pct": nan,
            "rel_error_pct_mean": nan,
            "rel_error_pct_max": nan,
            "mean_torque_error_pct": nan,
        }

    t_pred = np.array([c.torque_pred for c in valid], dtype=np.float64)
    t_true = np.array([c.torque_true for c in valid], dtype=np.float64)
    rel = np.array([c.rel_error_pct for c in valid], dtype=np.float64)
    rel = rel[np.isfinite(rel)]

    rms_true = float(np.sqrt(np.mean(t_true**2)))
    rmse = float(np.sqrt(np.mean((t_pred - t_true) ** 2)))

    mean_true = float(np.mean(t_true))
    mean_pred = float(np.mean(t_pred))
    mean_abs_err = abs(mean_pred - mean_true)
    # Relative-to-the-mean only where the mean is not swamped by ripple.
    mean_significant = rms_true > 0.0 and abs(mean_true) >= MEAN_TORQUE_SIGNIFICANCE * rms_true

    ripple_true = float(np.max(t_true) - np.min(t_true))
    ripple_pred = float(np.max(t_pred) - np.min(t_pred))

    return {
        "n": len(valid),
        "nrmse_torque_pct": float(100.0 * rmse / rms_true) if rms_true > 0 else float("nan"),
        "rmse_torque": rmse,
        "rms_torque_true": rms_true,
        "rel_error_pct_mean": float(np.mean(rel)) if rel.size else float("nan"),
        "rel_error_pct_median": float(np.median(rel)) if rel.size else float("nan"),
        "rel_error_pct_max": float(np.max(rel)) if rel.size else float("nan"),
        "mean_torque_true": mean_true,
        "mean_torque_pred": mean_pred,
        "mean_torque_abs_error": mean_abs_err,
        "mean_torque_error_pct": (
            float(100.0 * mean_abs_err / abs(mean_true)) if mean_significant else float("nan")
        ),
        # Always well defined, even when the average torque is near zero.
        "mean_torque_error_norm_pct": (
            float(100.0 * mean_abs_err / rms_true) if rms_true > 0 else float("nan")
        ),
        "mean_torque_is_significant": bool(mean_significant),
        "ripple_true": ripple_true,
        "ripple_pred": ripple_pred,
        "ripple_error_pct": (
            float(100.0 * abs(ripple_pred - ripple_true) / ripple_true) if ripple_true > 0 else float("nan")
        ),
    }
