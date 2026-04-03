from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from postproc_interop.model import MeshFrame


class PostprocAdapter(ABC):
    """Format-specific reader adapter."""

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read(self, path: Path) -> MeshFrame:
        raise NotImplementedError
