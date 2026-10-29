"""Synthetic-tested preparation prototype, NOT a production evaluator or qualification.

No I/O, model loading, inference, solver, implicit source discovery, or hash admission.
The caller pins the canonical ``eval`` source via PYTHONPATH and a separate contract.
Coordinates are explicitly mm, fields T, areas m2, axial length m; no unit guessing.
Full mesh truth must first be selected to the SAME declared admitted native IDs as
prediction. The strict field join requires that entire set, then selects a1 by ID.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from eval.mesh_regions import element_areas_m2, element_centroids_m
from eval.torque import AirgapBand, MU0, arkkio_torque


def _integers(value, name: str, ndim: int = 1) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != ndim or result.dtype.kind not in "iu":
        raise ValueError(f"{name}: expected {ndim}D integer array")
    if result.dtype.kind == "u" and result.size and result.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name}: native ID outside int64")
    return result.astype(np.int64, copy=True)


def _ids(value, name: str, *, allow_empty: bool = False) -> np.ndarray:
    result = _integers(value, name)
    if (not allow_empty and not result.size) or len(np.unique(result)) != len(result):
        raise ValueError(f"{name}: empty or duplicate IDs")
    return result


def _finite(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: nonfinite values")
    return result


def _positive(value, name: str) -> float:
    result = float(value)
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name}: expected positive finite scalar")
    return result


def _readonly(value) -> np.ndarray:
    array = np.ascontiguousarray(value)
    # An immutable bytes owner prevents callers from re-enabling writeability.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _positions(have: np.ndarray, need: np.ndarray, name: str) -> np.ndarray:
    order = np.argsort(have)
    sorted_ids = have[order]
    indices = np.searchsorted(sorted_ids, need)
    if np.any(indices >= len(have)) or not np.array_equal(sorted_ids[indices], need):
        raise ValueError(f"{name}: missing or unregistered native IDs")
    return order[indices]


@dataclass(frozen=True)
class StrictAirgap:
    band: AirgapBand
    admitted_element_ids: np.ndarray
    a1_element_ids: np.ndarray
    a1_node_ids: np.ndarray
    a1_node_xy_mm: np.ndarray
    a1_triangle_node_ids: np.ndarray
    a1_orientation: np.ndarray
    axial_length_m: float


def build_strict_airgap(
    node_ids, node_xy_mm, element_ids, triangle_node_ids, region_codes,
    region_names: Mapping[int, str], moving_region_codes, moving_node_ids, *,
    admitted_element_ids, axial_length_m: float, symmetry_multiplier: int,
    verified_stationary_node_ids=(),
) -> StrictAirgap:
    """Build an a1-only native operator; both motion tables are mandatory.

    Clockwise triangles are valid. Admitted triangles must have finite,
    positive *absolute* area. Excluded a2 geometry is not restored or scored.
    Sector count is an explicitly admitted input, never inferred from angles.
    """
    ids = _ids(node_ids, "node_ids")
    elements = _ids(element_ids, "element_ids")
    admitted = np.sort(_ids(admitted_element_ids, "admitted_element_ids"))
    admitted_rows = _positions(elements, admitted, "admitted_element_ids")
    xy = np.asarray(node_xy_mm, dtype=np.float64)
    native_tri = _integers(triangle_node_ids, "triangle_node_ids", 2)
    codes = _integers(region_codes, "region_codes")
    if xy.shape != (len(ids), 2) or native_tri.shape != (len(elements), 3) or codes.shape != elements.shape:
        raise ValueError("native geometry array shape mismatch")
    if any(int(c) not in region_names or not isinstance(region_names[int(c)], str) for c in codes):
        raise ValueError("unregistered region code or invalid region name")
    a2 = np.array([region_names[int(c)].lower() == "a2" for c in codes])
    if np.intersect1d(elements[a2], admitted).size:
        raise ValueError("a2 elements are excluded by this fixed reader contract")
    if moving_region_codes is None or moving_node_ids is None:
        raise ValueError("explicit motion metadata required")
    # Empty tables must be explicitly typed arrays (or lists, normalized below).
    move_codes = _ids(np.asarray(moving_region_codes, dtype=np.int64) if len(moving_region_codes) == 0 else moving_region_codes,
                      "moving_region_codes", allow_empty=True)
    move_nodes = _ids(np.asarray(moving_node_ids, dtype=np.int64) if len(moving_node_ids) == 0 else moving_node_ids,
                      "moving_node_ids", allow_empty=True)
    if any(int(c) not in region_names for c in move_codes):
        raise ValueError("unregistered moving region code")
    _positions(ids, move_nodes, "moving_node_ids")
    stable = _ids(np.asarray(verified_stationary_node_ids, dtype=np.int64)
                  if len(verified_stationary_node_ids) == 0 else verified_stationary_node_ids,
                  "verified_stationary_node_ids", allow_empty=True)
    _positions(move_nodes, stable, "stationary candidate evidence")
    length = _positive(axial_length_m, "axial_length_m")
    if isinstance(symmetry_multiplier, (bool, np.bool_)) or not isinstance(symmetry_multiplier, (int, np.integer)) or symmetry_multiplier < 1:
        raise ValueError("symmetry_multiplier must be an explicit positive integer")
    tri = _positions(ids, native_tri.ravel(), "triangle_node_ids").reshape(native_tri.shape)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            vertices = _finite(xy[tri[admitted_rows]], "admitted node_xy_mm")
            ab, ac = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
            admitted_signed = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
            admitted_areas = element_areas_m2(xy[:, 0], xy[:, 1], tuple(tri[admitted_rows].T))
            if not np.isfinite(admitted_signed).all() or not np.isfinite(admitted_areas).all() or np.any(admitted_areas <= 0):
                raise ValueError("nonfinite or degenerate admitted native triangle")
            signed, all_areas = np.full(len(elements), np.nan), np.full(len(elements), np.nan)
            signed[admitted_rows], all_areas[admitted_rows] = admitted_signed, admitted_areas
            mask = np.array([region_names[int(c)].lower() == "a1" for c in codes])
            if not mask.any():
                raise ValueError("native a1 absent; no airgap fallback")
            a1_codes = tuple(sorted(set(int(c) for c in codes[mask])))
            if np.intersect1d(codes[mask], move_codes).size:
                raise ValueError("a1 marked as a moving region")
            rows = np.flatnonzero(mask)
            rows = rows[np.argsort(elements[rows])]
            a1_ids = elements[rows]
            _positions(admitted, a1_ids, "a1 coverage")
            a1_nodes = np.unique(native_tri[rows])
            if np.setdiff1d(np.intersect1d(a1_nodes, move_nodes), stable).size:
                raise ValueError("a1 contains a moving native node")
            a1_node_positions = _positions(ids, a1_nodes, "a1 nodes")
            a1_xy = xy[a1_node_positions]
            radii = np.hypot(a1_xy[:, 0], a1_xy[:, 1]) * 1e-3
            inner, outer = float(radii.min()), float(radii.max())
            if not np.isfinite(radii).all() or inner <= 0 or outer <= inner:
                raise ValueError("invalid a1 radial extent")
            cx, cy = element_centroids_m(xy[:, 0], xy[:, 1], tuple(tri[rows].T))
            radius = np.hypot(cx, cy)
            if not np.isfinite(radius).all() or np.any(radius <= 0):
                raise ValueError("invalid a1 centroid radius")
            # Diagnostic span only. Explicit n_sym remains authoritative.
            theta = np.arctan2(cy, cx)
            band = AirgapBand(
                element_index=_readonly(np.arange(len(rows), dtype=np.int64)),
                radius_m=_readonly(radius), area_m2=_readonly(all_areas[rows]),
                cos_theta=_readonly(cx / radius), sin_theta=_readonly(cy / radius),
                r_inner_m=inner, r_outer_m=outer,
                sector_span_rad=float(theta.max() - theta.min()),
                symmetry_multiplier=int(symmetry_multiplier), region_codes=a1_codes,
            )
            # Preflight the actual scale too (tiny thickness / huge L may overflow).
            _positive(length * int(symmetry_multiplier) / (MU0 * band.thickness_m), "Arkkio scale")
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("geometry arithmetic overflow or invalid operation") from exc
    return StrictAirgap(band, _readonly(admitted), _readonly(a1_ids), _readonly(a1_nodes),
                        _readonly(a1_xy), _readonly(native_tri[rows]),
                        _readonly(np.sign(signed[rows])), length)


def align_airgap_field(operator: StrictAirgap, element_ids, field_B_T) -> np.ndarray:
    """Exact entire admitted-set join, followed by a1-ID ordering; never positional."""
    ids = _ids(element_ids, "field element_ids")
    field = _finite(field_B_T, "field_B_T")
    if field.shape != (len(ids), 2):
        raise ValueError("field shape must be (native elements, 2)")
    if not np.array_equal(np.sort(ids), operator.admitted_element_ids):
        raise ValueError("field native IDs differ from declared admitted set")
    return _readonly(field[_positions(ids, operator.a1_element_ids, "field a1 join")])


def strict_arkkio_torque(operator: StrictAirgap, element_ids, field_B_T) -> float:
    aligned = align_airgap_field(operator, element_ids, field_B_T)
    band = operator.band
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            br = aligned[:, 0] * band.cos_theta + aligned[:, 1] * band.sin_theta
            bt = -aligned[:, 0] * band.sin_theta + aligned[:, 1] * band.cos_theta
            integrand = band.radius_m * br * bt * band.area_m2
            _finite(integrand, "Arkkio integrand")
            # Already aligned band-sized B: no full-mesh positional slicing.
            result = arkkio_torque(band, aligned[:, 0], aligned[:, 1], operator.axial_length_m)
            if not np.isfinite(result):
                raise ValueError("nonfinite Arkkio torque")
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("field/torque arithmetic overflow") from exc
    return float(result)


def validate_all45_stationarity(operators: Mapping[int, StrictAirgap]) -> None:
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) for k in operators) or set(operators) != set(range(45)):
        raise ValueError("stationarity requires exactly native steps 0..44")
    ref = operators[0]
    for step in range(1, 45):
        other = operators[step]
        for name in ("admitted_element_ids", "a1_element_ids", "a1_node_ids", "a1_node_xy_mm", "a1_triangle_node_ids", "a1_orientation"):
            if not np.array_equal(getattr(ref, name), getattr(other, name)):
                raise ValueError(f"a1 stationarity failed at step {step}: {name}")
        for name in ("radius_m", "area_m2", "cos_theta", "sin_theta"):
            if not np.array_equal(getattr(ref.band, name), getattr(other.band, name)):
                raise ValueError(f"a1 band changed at step {step}: {name}")
        if (ref.band.r_inner_m, ref.band.r_outer_m, ref.band.region_codes, ref.band.symmetry_multiplier, ref.axial_length_m) != (
            other.band.r_inner_m, other.band.r_outer_m, other.band.region_codes, other.band.symmetry_multiplier, other.axial_length_m
        ):
            raise ValueError(f"a1 scalar contract changed at step {step}")


@dataclass(frozen=True)
class FieldStatistics:
    count: int
    weight_sum: float
    vector_sse: float
    magnitude_sse: float
    truth_energy: float


def field_sufficient_statistics(pred, truth, weights=None) -> FieldStatistics:
    pred, truth = _finite(pred, "pred T"), _finite(truth, "truth T")
    if pred.ndim != 2 or pred.shape[1:] != (2,) or not len(pred) or pred.shape != truth.shape:
        raise ValueError("B field shapes must both be nonempty (N,2)")
    weights = np.ones(len(pred), dtype=np.float64) if weights is None else _finite(weights, "weights m2")
    if weights.shape != (len(pred),) or np.any(weights <= 0):
        raise ValueError("weights must be positive with shape (N,)")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            truth2 = np.sum(truth * truth, axis=1)
            pred2 = np.sum(pred * pred, axis=1)
            values = np.array([weights.sum(), np.sum(weights * np.sum((pred - truth) ** 2, axis=1)),
                               np.sum(weights * (np.sqrt(pred2) - np.sqrt(truth2)) ** 2),
                               np.sum(weights * truth2)])
            _finite(values, "field sufficient statistics")
    except FloatingPointError as exc:
        raise ValueError("field statistics overflow") from exc
    return FieldStatistics(len(pred), *map(float, values))


def merge_statistics(statistics: Sequence[FieldStatistics]) -> FieldStatistics:
    if not statistics:
        raise ValueError("no field statistics")
    count = 0
    rows = []
    for s in statistics:
        if isinstance(s.count, bool) or not isinstance(s.count, (int, np.integer)) or s.count <= 0:
            raise ValueError("invalid field count")
        row = _finite([s.weight_sum, s.vector_sse, s.magnitude_sse, s.truth_energy], "field statistics")
        if row[0] <= 0 or np.any(row[1:] < 0):
            raise ValueError("invalid field statistic sign")
        count += int(s.count)
        rows.append(row)
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = _finite(np.sum(rows, axis=0), "pooled sufficient statistics")
    except FloatingPointError as exc:
        raise ValueError("pooled field statistics overflow") from exc
    return FieldStatistics(count, *map(float, result))


@dataclass(frozen=True)
class FieldSummary:
    count: int
    vector_nrmse_percent: float | None
    magnitude_nrmse_percent: float | None
    vector_rmse_T: float
    magnitude_rmse_T: float
    normalization_status: str


def finish_statistics(statistics: FieldStatistics) -> FieldSummary:
    s = merge_statistics([statistics])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            absolute = np.array([np.sqrt(s.vector_sse / s.weight_sum), np.sqrt(s.magnitude_sse / s.weight_sum)])
            _finite(absolute, "absolute field summary")
            if s.truth_energy == 0:
                return FieldSummary(s.count, None, None, *map(float, absolute), "UNDEFINED_ZERO_REFERENCE")
            normalized = np.array([100 * np.sqrt(s.vector_sse / s.truth_energy),
                                   100 * np.sqrt(s.magnitude_sse / s.truth_energy)])
            _finite(normalized, "normalized field summary")
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("field summary arithmetic overflow") from exc
    return FieldSummary(s.count, *map(float, normalized), *map(float, absolute), "DEFINED")


def validate_snapshot_coverage(pairs, expected_cases) -> None:
    cases = _ids(expected_cases, "expected_cases")
    pairs = _integers(pairs, "case-step pairs", 2)
    if pairs.shape[1:] != (2,):
        raise ValueError("case-step pairs must have shape (N,2)")
    received = [tuple(map(int, row)) for row in pairs]
    expected = {(int(case), step) for case in cases for step in range(45)}
    if len(received) != len(set(received)) or set(received) != expected:
        raise ValueError("missing, duplicate, or unregistered case-step snapshot")


@dataclass(frozen=True)
class TorqueSummary:
    count: int
    rmse_Nm: float
    truth_rms_Nm: float
    nrmse_percent: float | None
    pred_mean_Nm: float
    truth_mean_Nm: float
    mean_error_rms_normalized_percent: float | None
    pred_ripple_Nm: float
    truth_ripple_Nm: float
    normalization_status: str


def summarize_torque45(step_ids, pred, truth) -> TorqueSummary:
    steps = _ids(step_ids, "torque step_ids")
    if set(map(int, steps)) != set(range(45)):
        raise ValueError("torque summary requires exactly steps 0..44 for ONE case")
    pred, truth = _finite(pred, "torque pred Nm"), _finite(truth, "torque truth Nm")
    if pred.shape != (45,) or truth.shape != (45,):
        raise ValueError("torque shape must be (45,)")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            rms = np.sqrt(np.mean(truth ** 2))
            rmse = np.sqrt(np.mean((pred - truth) ** 2))
            pm, tm = np.mean(pred), np.mean(truth)
            pred_ripple, true_ripple = np.ptp(pred), np.ptp(truth)
            _finite([rmse, rms, pm, tm, pred_ripple, true_ripple], "absolute torque summary")
            if rms == 0:
                if np.any(truth != 0):
                    raise ValueError("nonzero torque reference energy underflow")
                return TorqueSummary(45, float(rmse), 0., None, float(pm), float(tm), None,
                                     float(pred_ripple), float(true_ripple), "UNDEFINED_ZERO_REFERENCE")
            nrmse, mean_norm = 100 * rmse / rms, 100 * abs(pm - tm) / rms
            _finite([nrmse, mean_norm], "normalized torque summary")
    except FloatingPointError as exc:
        raise ValueError("torque summary arithmetic overflow") from exc
    return TorqueSummary(45, float(rmse), float(rms), float(nrmse), float(pm), float(tm),
                         float(mean_norm), float(pred_ripple), float(true_ripple), "DEFINED")
