from .MeshReader import MeshReader
from .MotorCADMeshSolutionH5Reader import MotorCADMeshSolutionH5Reader
from .MotorCADMeshSolutionH5PyMCADReader import (
    MotorCADMeshSolutionH5PyMCADReader,
)
from .MotorCADMeshSolutionTxtReader import MotorCADMeshSolutionTxtReader
from .VTUMeshReader import VTUMeshReader

# backward-compat aliases
MotorCADH5Adapter = MotorCADMeshSolutionH5Reader
VTUAdapter = VTUMeshReader

__all__ = [
    "MeshReader",
    "MotorCADMeshSolutionH5Reader",
    "MotorCADMeshSolutionH5PyMCADReader",
    "MotorCADMeshSolutionTxtReader",
    "VTUMeshReader",
    # aliases
    "MotorCADH5Adapter",
    "VTUAdapter",
]
