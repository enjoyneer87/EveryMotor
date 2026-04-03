from __future__ import annotations

from pathlib import Path

from postproc_interop.adapters.base import PostprocAdapter
from postproc_interop.model import MeshFrame


class VTUAdapter(PostprocAdapter):
    """Placeholder VTU adapter.

    Implement with meshio/vtk based on your pipeline convention.
    """

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() == ".vtu"

    def read(self, path: Path) -> MeshFrame:
        _ = path
        try:
            import meshio as _meshio
        except ImportError as exc:
            raise ImportError(
                "meshio is required for VTUAdapter. "
                "Install with `pip install meshio`."
            ) from exc

        _ = _meshio

        raise NotImplementedError(
            "VTUAdapter.read is a scaffold. "
            "Map VTU cells/point_data into MeshFrame."
        )
