from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from postproc_interop.model.Mesh import Mesh
from postproc_interop.model.MeshSolution import MeshSolution


class MeshReader(ABC):
    """Abstract base: reads a file and returns Mesh or MeshSolution."""

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read(self, path: Path) -> Mesh | MeshSolution:
        raise NotImplementedError
