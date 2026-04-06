from __future__ import annotations

from pathlib import Path

import numpy as np

from postproc_interop.model.FieldObservation import FieldObservation
from postproc_interop.model.MeshFrame import MeshFrame
from postproc_interop.model.MeshMat import MeshMat
from postproc_interop.model.MeshSolution import MeshSolution
from postproc_interop.model.MeshTopology import MeshTopology
from postproc_interop.model.NodeTable import NodeTable
from postproc_interop.model.RegionTable import RegionTable
from postproc_interop.model.SolverMetadata import SolverMetadata


def classify_motorcad_h5_filename(path_like: str) -> str:
    """Classify MotorCAD H5 semantic source type from filename."""

    name = Path(str(path_like).replace("\\", "/")).name.lower()
    if "onloadtorque" in name:
        return "OnLoadTorque"
    if "losselement" in name or "onloadloss" in name:
        return "LossElement_OnLoadLoss"
    if "staticloadinductance" in name:
        return "StaticLoadInductance"
    if "staticload" in name:
        return "StaticLoad"
    if "staticoc" in name:
        return "StaticOC"
    return "Unknown"


def infer_step_semantics(fidelity_type: str) -> str:
    """Infer canonical step semantics from MotorCAD source type."""

    if fidelity_type == "OnLoadTorque":
        return "explicit_time_step"
    return "implicit_single_step"


def infer_coupling_policy(fidelity_type: str) -> str:
    """Infer canonical coupling policy from MotorCAD source type."""

    if fidelity_type == "OnLoadTorque":
        return "transient_coupled"
    return "weak_coupled"


def build_solver_metadata(
    *,
    source_path: Path,
    source_file_name: str | None = None,
    fidelity_type: str | None = None,
    step_index: int = -1,
    reader_name: str = "",
) -> SolverMetadata:
    """Build canonical solver metadata owned by postproc_interop."""

    resolved_name = source_file_name or Path(source_path).name
    resolved_type = (
        fidelity_type or classify_motorcad_h5_filename(resolved_name)
    )
    return SolverMetadata(
        source_file_name=resolved_name,
        fidelity_type=resolved_type,
        step_index=int(step_index),
        step_semantics=infer_step_semantics(resolved_type),
        coupling_policy=infer_coupling_policy(resolved_type),
        source_path=str(source_path),
        reader_name=reader_name,
    )


def meshsolution_to_meshframe(
    mesh_solution: MeshSolution,
    *,
    solver_metadata: SolverMetadata | None = None,
) -> MeshFrame:
    """Pure morphism: MeshSolution -> MeshFrame."""

    mesh = mesh_solution.mesh
    if not isinstance(mesh, MeshMat):
        raise TypeError("mesh_solution.mesh must be a MeshMat instance")

    topology = MeshTopology(
        tri_index=np.asarray(mesh.tri_index, dtype=np.int32),
        node_1=np.asarray(mesh.node_1, dtype=np.int32),
        node_2=np.asarray(mesh.node_2, dtype=np.int32),
        node_3=np.asarray(mesh.node_3, dtype=np.int32),
        reg_code=np.asarray(mesh.reg_code, dtype=np.int32),
    )
    nodes = NodeTable(
        node_id=np.asarray(mesh.node_id, dtype=np.int32),
        x_mm=np.asarray(mesh.x_mm, dtype=np.float64),
        y_mm=np.asarray(mesh.y_mm, dtype=np.float64),
    )

    regions = None
    if mesh.region_code is not None and mesh.region_name is not None:
        regions = RegionTable.from_mapping(
            reg_code=np.asarray(mesh.region_code, dtype=np.int32),
            name=np.asarray(mesh.region_name, dtype=object),
        )

    field_observations = {
        key: FieldObservation(
            name=str(getattr(solution, "label", "") or key),
            values=np.asarray(solution.field),
            association="element",
            unit=str(getattr(solution, "unit", "") or ""),
            source_key=key,
        )
        for key, solution in mesh_solution.solution_dict.items()
    }
    attrs = dict(mesh_solution.attrs)
    if mesh.attrs:
        attrs.setdefault("mesh_attrs", dict(mesh.attrs))

    return MeshFrame.from_components(
        topology=topology,
        nodes=nodes,
        regions=regions,
        field_observations=field_observations,
        solver_metadata=solver_metadata,
        attrs=attrs,
    )
