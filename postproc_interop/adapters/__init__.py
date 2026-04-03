from .MeshReader import MeshReader
from .MotorCADMeshSolutionH5Reader import MotorCADMeshSolutionH5Reader
from .MotorCADMeshSolutionTxtReader import MotorCADMeshSolutionTxtReader
from .VTUMeshReader import VTUMeshReader

# backward-compat aliases
MotorCADH5Adapter = MotorCADMeshSolutionH5Reader
VTUAdapter = VTUMeshReader

__all__ = [
    "MeshReader",
    "MotorCADMeshSolutionH5Reader",
    "MotorCADMeshSolutionTxtReader",
    "VTUMeshReader",
    # aliases
    "MotorCADH5Adapter",
    "VTUAdapter",
]
