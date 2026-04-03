"""Interoperability layer for MotorCAD postprocess formats.

Canonical pipeline:
  source format (h5 / txt / vtu / ...)
    -> MeshReader.read()
    -> MeshSolution  (mesh: MeshMat  +  solution_dict: {key: SolutionMat})
    -> Exporter  (pyvista / babylon / vtu / ...)
"""

from .model import Mesh, MeshMat, Solution, SolutionMat, MeshSolution
from .adapters import (
    MeshReader,
    MotorCADMeshSolutionH5Reader,
    MotorCADMeshSolutionTxtReader,
    VTUMeshReader,
)
from .header_mapping import (
    MOTORCAD_TXT_HEADER_TO_H5_KEY_MAP,
    map_header_tokens_to_h5_keys,
)
from .tabular import load_h5_as_tables, meshframe_to_dataframes

__all__ = [
    # model
    "Mesh",
    "MeshMat",
    "Solution",
    "SolutionMat",
    "MeshSolution",
    # adapters
    "MeshReader",
    "MotorCADMeshSolutionH5Reader",
    "MotorCADMeshSolutionTxtReader",
    "VTUMeshReader",
    # header mapping
    "MOTORCAD_TXT_HEADER_TO_H5_KEY_MAP",
    "map_header_tokens_to_h5_keys",
    # tabular (legacy)
    "meshframe_to_dataframes",
    "load_h5_as_tables",
]
