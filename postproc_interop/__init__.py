"""Interoperability layer for MotorCAD postprocess formats.

Canonical pipeline:
  source format (h5 / txt / vtu / ...)
    -> MeshReader.read()
    -> MeshSolution  (mesh: MeshMat  +  solution_dict: {key: SolutionMat})
    -> Exporter  (pyvista / babylon / vtu / ...)
"""

from .model import (
    FieldObservation,
    FrozenMapping,
    Mesh,
    MeshFrame,
    MeshMat,
    MeshSolution,
    MeshTopology,
    NodeTable,
    PBCBoundaryCandidate,
    PBCPairSet,
    Solution,
    SolutionMat,
    RegionTable,
    SolverMetadata,
)
from .adapters import (
    MeshReader,
    MotorCADMeshSolutionH5Reader,
    MotorCADMeshSolutionH5PyMCADReader,
    MotorCADMeshSolutionTxtReader,
    VTUMeshReader,
)
from .bridges import MotorCADPBCVisualizationBridge
from .conversions import (
    build_solver_metadata,
    classify_motorcad_h5_filename,
    infer_coupling_policy,
    infer_step_semantics,
    meshsolution_to_meshframe,
)
from .header_mapping import (
    MOTORCAD_TXT_HEADER_TO_H5_KEY_MAP,
    map_header_tokens_to_h5_keys,
)
from .pbc import (
    build_pbc_boundary_candidate,
    build_pbc_pair_set,
    build_pbc_visualization_case,
)
from .tabular import load_h5_as_tables, meshframe_to_dataframes

__all__ = [
    # model
    "FieldObservation",
    "FrozenMapping",
    "Mesh",
    "MeshFrame",
    "MeshMat",
    "Solution",
    "SolutionMat",
    "MeshSolution",
    "MeshTopology",
    "NodeTable",
    "PBCBoundaryCandidate",
    "PBCPairSet",
    "RegionTable",
    "SolverMetadata",
    # adapters
    "MeshReader",
    "MotorCADMeshSolutionH5Reader",
    "MotorCADMeshSolutionH5PyMCADReader",
    "MotorCADMeshSolutionTxtReader",
    "VTUMeshReader",
    "MotorCADPBCVisualizationBridge",
    # pure transforms
    "build_pbc_boundary_candidate",
    "build_pbc_pair_set",
    "build_pbc_visualization_case",
    "build_solver_metadata",
    "classify_motorcad_h5_filename",
    "infer_coupling_policy",
    "infer_step_semantics",
    "meshsolution_to_meshframe",
    # header mapping
    "MOTORCAD_TXT_HEADER_TO_H5_KEY_MAP",
    "map_header_tokens_to_h5_keys",
    # tabular (legacy)
    "meshframe_to_dataframes",
    "load_h5_as_tables",
]
