from __future__ import annotations

from pathlib import Path

from postproc_interop.adapters.MeshReader import MeshReader
from postproc_interop.model.Mesh import Mesh


class VTUMeshReader(MeshReader):
    """Read VTU file into Mesh (scaffold — implement with meshio/vtk).

    Returns Mesh (geometry only), not MeshSolution, because VTU may or may
    not carry field data — the caller decides whether to wrap in MeshSolution.
    """

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() == ".vtu"

    def read(self, path: Path) -> Mesh:
        _ = path
        try:
            import meshio as _meshio  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "meshio is required for VTUMeshReader. "
                "Install with `pip install meshio`."
            ) from exc

        raise NotImplementedError(
            "VTUMeshReader.read is a scaffold. "
            "Map VTU cells/point_data into MeshMat."
        )
