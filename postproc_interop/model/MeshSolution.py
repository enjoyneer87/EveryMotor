from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from postproc_interop.model.Mesh import Mesh
from postproc_interop.model.Solution import Solution


@dataclass
class MeshSolution:
    """Composition of Mesh geometry + Solution field dict (cf. pyleecan MeshSolution).

    Does NOT inherit from Mesh — uses composition so geometry and field data
    remain independently typed and swappable.

    Example
    -------
    meshsol.mesh                       # MeshMat instance
    meshsol.solution_dict["bx"].field  # np.ndarray of Bx values per element
    """

    mesh: Mesh
    solution_dict: dict[str, Solution] = field(default_factory=dict)
    label: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        from postproc_interop.model.MeshMat import MeshMat
        m = self.mesh
        return {
            "n_elements": int(m.tri_index.shape[0]) if isinstance(m, MeshMat) else None,
            "n_nodes": int(m.node_id.shape[0]) if isinstance(m, MeshMat) else None,
            "solution_keys": sorted(self.solution_dict.keys()),
        }
