from __future__ import annotations

import math

import numpy as np
from scipy.spatial import KDTree

from phase1_static.pbc_boundary import extract_periodic_boundary_groups_from_mesh
from postproc_interop.model.MeshFrame import MeshFrame
from postproc_interop.model.PBCBoundaryCandidate import PBCBoundaryCandidate
from postproc_interop.model.PBCPairSet import PBCPairSet
from postproc_interop.model.PBCVisualizationCase import PBCVisualizationCase
from postproc_interop.model.SolverMetadata import SolverMetadata


def build_pbc_boundary_candidate(
    mesh_frame: MeshFrame,
    *,
    rotation_deg: float = -45.0,
) -> tuple[PBCBoundaryCandidate | None, str | None]:
    """Pure morphism: MeshFrame -> PBCBoundaryCandidateResult."""

    group_specs, _, chain_error = extract_periodic_boundary_groups_from_mesh(
        np.asarray(mesh_frame.pos_xy, dtype=np.float64),
        np.asarray(mesh_frame.triangles, dtype=np.int32),
        np.asarray(mesh_frame.reg_code, dtype=np.int32),
        moving_reg_codes=_meshframe_moving_reg_codes(mesh_frame),
        region_name_by_code=_meshframe_region_name_by_code(mesh_frame),
        rotation_deg=rotation_deg,
    )
    if not group_specs:
        return None, chain_error or "chain_extract_failed"

    master_groups = tuple(
        np.asarray(master_nodes, dtype=np.int64)
        for _, master_nodes, _ in group_specs
    )
    slave_groups = tuple(
        np.asarray(slave_nodes, dtype=np.int64)
        for _, _, slave_nodes in group_specs
    )
    group_labels = tuple(str(label) for label, _, _ in group_specs)
    master_line_idx = (
        np.concatenate(master_groups).astype(np.int64, copy=False)
        if master_groups else np.empty((0,), dtype=np.int64)
    )
    slave_line_idx = (
        np.concatenate(slave_groups).astype(np.int64, copy=False)
        if slave_groups else np.empty((0,), dtype=np.int64)
    )

    return (
        PBCBoundaryCandidate(
            rotation_origin_xy=np.zeros((2,), dtype=np.float64),
            master_line_idx=master_line_idx,
            slave_line_idx=slave_line_idx,
            rotation_deg=float(rotation_deg),
            master_line_groups=master_groups,
            slave_line_groups=slave_groups,
            group_labels=group_labels,
            extraction_method=(
                "phase1_static.extract_periodic_boundary_groups_from_mesh"
            ),
        ),
        None,
    )


def build_pbc_pair_set(
    mesh_frame: MeshFrame,
    boundary_candidate: PBCBoundaryCandidate,
    *,
    anti_periodic: bool = True,
    atol_mm: float = 5e-2,
    rtol: float = 1e-6,
) -> tuple[PBCPairSet | None, str | None]:
    """Pure morphism: (MeshFrame, PBCBoundaryCandidate) -> PBCPairSetResult."""

    pos_xy = np.asarray(mesh_frame.pos_xy, dtype=np.float64)
    chunks: list[np.ndarray] = []
    match_modes: list[str] = []
    residual_max = 0.0
    total_slave = 0
    total_matched = 0
    for master_idx, slave_idx in zip(
        boundary_candidate.master_line_groups,
        boundary_candidate.slave_line_groups,
    ):
        master_nodes = pos_xy[master_idx]
        slave_nodes = pos_xy[slave_idx]
        matched_master, matched_slave, match_mode = _match_periodic_nodes(
            master_nodes,
            slave_nodes,
            boundary_candidate.rotation_origin_xy,
            rotation_deg=boundary_candidate.rotation_deg,
            atol_mm=atol_mm,
            rtol=rtol,
        )
        if matched_master.size == 0 or matched_slave.size == 0:
            continue

        global_master = master_idx[matched_master]
        global_slave = slave_idx[matched_slave]
        chunks.append(
            np.stack([global_master, global_slave], axis=0).astype(
                np.int64,
                copy=False,
            )
        )
        total_slave += int(slave_idx.size)
        total_matched += int(matched_slave.size)
        match_modes.append(match_mode)

        rotated_slave = _rotate_points(
            pos_xy[global_slave],
            boundary_candidate.rotation_origin_xy,
            rotation_deg=boundary_candidate.rotation_deg,
        )
        residual = np.linalg.norm(pos_xy[global_master] - rotated_slave, axis=1)
        if residual.size > 0:
            residual_max = max(residual_max, float(np.max(residual)))

    if not chunks:
        return None, "pbc_match_empty"

    pbc_forward_index = np.concatenate(chunks, axis=1)

    return (
        PBCPairSet(
            pbc_forward_index=pbc_forward_index,
            match_ratio=float(total_matched / max(1, total_slave)),
            match_mode=(
                match_modes[0]
                if len(set(match_modes)) == 1 else "mixed"
            ),
            max_rotation_residual=float(residual_max),
            anti_periodic=bool(anti_periodic),
        ),
        None,
    )


def build_pbc_visualization_case(
    *,
    case_idx: int,
    mesh_frame: MeshFrame,
    boundary_candidate: PBCBoundaryCandidate,
    pair_set: PBCPairSet,
) -> PBCVisualizationCase:
    """Pure morphism: canonical core -> tutorial visualization payload."""

    metadata = mesh_frame.solver_metadata or SolverMetadata(
        source_file_name="",
        fidelity_type="Unknown",
        step_index=-1,
        step_semantics="implicit_single_step",
        coupling_policy="weak_coupled",
    )
    error_code = pair_set.error_code or boundary_candidate.error_code
    return PBCVisualizationCase(
        case_idx=int(case_idx),
        source_file_name=metadata.source_file_name,
        source_file_type=metadata.fidelity_type,
        step_index=int(metadata.step_index),
        step_semantics=metadata.step_semantics,
        coupling_policy=metadata.coupling_policy,
        pos_xy=mesh_frame.pos_xy,
        triangles=mesh_frame.triangles,
        reg_code=np.asarray(mesh_frame.reg_code, dtype=np.int32),
        master_line_idx=boundary_candidate.master_line_idx,
        slave_line_idx=boundary_candidate.slave_line_idx,
        master_line_groups=boundary_candidate.master_line_groups,
        slave_line_groups=boundary_candidate.slave_line_groups,
        group_labels=boundary_candidate.group_labels,
        pbc_forward_index=pair_set.pbc_forward_index,
        match_ratio=float(pair_set.match_ratio),
        error_code=error_code,
        match_mode=pair_set.match_mode,
        max_rotation_residual=float(pair_set.max_rotation_residual),
    )


def _meshframe_moving_reg_codes(mesh_frame: MeshFrame) -> np.ndarray | None:
    attrs = dict(mesh_frame.attrs)
    if "moving_reg_codes" in attrs:
        return np.asarray(attrs["moving_reg_codes"], dtype=np.int32)
    mesh_attrs = attrs.get("mesh_attrs")
    if isinstance(mesh_attrs, dict) and "moving_reg_codes" in mesh_attrs:
        return np.asarray(mesh_attrs["moving_reg_codes"], dtype=np.int32)
    return None


def _meshframe_region_name_by_code(mesh_frame: MeshFrame) -> dict[int, str]:
    if mesh_frame.regions is not None:
        return {
            int(code): (
                name.decode("utf-8", errors="replace")
                if isinstance(name, bytes) else str(name)
            )
            for code, name in zip(
                np.asarray(mesh_frame.regions.reg_code, dtype=np.int32).tolist(),
                np.asarray(mesh_frame.regions.name, dtype=object).tolist(),
            )
        }

    attrs = dict(mesh_frame.attrs)
    direct = attrs.get("region_name_by_code")
    if isinstance(direct, dict):
        return {int(code): str(name) for code, name in direct.items()}
    mesh_attrs = attrs.get("mesh_attrs")
    if isinstance(mesh_attrs, dict):
        nested = mesh_attrs.get("region_name_by_code")
        if isinstance(nested, dict):
            return {int(code): str(name) for code, name in nested.items()}
    return {}


def _match_periodic_nodes(
    master_nodes: np.ndarray,
    slave_nodes: np.ndarray,
    origin_xy: np.ndarray,
    *,
    rotation_deg: float,
    atol_mm: float,
    rtol: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    master = np.asarray(master_nodes, dtype=np.float64)
    slave = np.asarray(slave_nodes, dtype=np.float64)
    rotated_slave = _rotate_points(slave, origin_xy, rotation_deg=rotation_deg)

    tree = KDTree(master)
    distance, index = tree.query(rotated_slave, distance_upper_bound=atol_mm)

    finite = np.isfinite(distance)
    valid_idx = finite & (index < master.shape[0])
    rel_scale = np.zeros(slave.shape[0], dtype=np.float64)
    if np.any(valid_idx):
        rel_scale[valid_idx] = np.max(
            np.abs(master[index[valid_idx]] - rotated_slave[valid_idx]),
            axis=1,
        )
    within_tol = valid_idx & (distance <= atol_mm + rtol * rel_scale)
    matched_slave = np.nonzero(within_tol)[0].astype(np.int64, copy=False)
    matched_master = index[within_tol].astype(np.int64, copy=False)

    if matched_master.size > 0 and matched_slave.size > 0:
        return matched_master, matched_slave, "kdtree_rotation"

    fallback_master, fallback_slave = _match_nodes_by_radius(
        master,
        slave,
        origin_xy,
    )
    return fallback_master, fallback_slave, "radius_fallback"


def _match_nodes_by_radius(
    master_nodes: np.ndarray,
    slave_nodes: np.ndarray,
    origin_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if master_nodes.shape[0] == 0 or slave_nodes.shape[0] == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    origin = np.asarray(origin_xy, dtype=np.float64)
    master_order = np.argsort(np.linalg.norm(master_nodes - origin, axis=1))
    slave_order = np.argsort(np.linalg.norm(slave_nodes - origin, axis=1))
    n_pairs = min(master_order.size, slave_order.size)
    if n_pairs == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    master_pick = np.round(
        np.linspace(0, master_order.size - 1, n_pairs)
    ).astype(np.int64)
    slave_pick = np.round(
        np.linspace(0, slave_order.size - 1, n_pairs)
    ).astype(np.int64)
    return master_order[master_pick], slave_order[slave_pick]


def _rotate_points(
    xy: np.ndarray,
    origin_xy: np.ndarray,
    *,
    rotation_deg: float,
) -> np.ndarray:
    radians = math.radians(rotation_deg)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)
    rotation = np.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        dtype=np.float64,
    )
    points = np.asarray(xy, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64)
    return (points - origin) @ rotation.T + origin
