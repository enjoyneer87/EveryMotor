from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from postproc_interop.model.FieldObservation import FieldObservation
from postproc_interop.model.FrozenMapping import FrozenMapping
from postproc_interop.model.MeshTopology import MeshTopology
from postproc_interop.model.NodeTable import NodeTable
from postproc_interop.model.RegionTable import RegionTable
from postproc_interop.model.SolverMetadata import SolverMetadata


@dataclass(frozen=True)
class MeshFrame:
    """Canonical immutable mesh + field + semantics contract."""

    topology: MeshTopology
    nodes: NodeTable
    regions: RegionTable | None = None
    field_observations: FrozenMapping = field(default_factory=FrozenMapping)
    solver_metadata: SolverMetadata | None = None
    attrs: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if not isinstance(self.field_observations, FrozenMapping):
            object.__setattr__(
                self,
                "field_observations",
                FrozenMapping(self.field_observations),
            )
        if not isinstance(self.attrs, FrozenMapping):
            object.__setattr__(self, "attrs", FrozenMapping(self.attrs))

        n_nodes = int(self.nodes.node_id.shape[0])
        n_elements = int(self.topology.tri_index.shape[0])
        for observation in self.field_observations.values():
            if not isinstance(observation, FieldObservation):
                raise TypeError(
                    "field_observations values must be "
                    "FieldObservation instances"
                )
            expected = (
                n_nodes
                if observation.association == "node"
                else n_elements
            )
            if observation.sample_count != expected:
                raise ValueError(
                    f"FieldObservation {observation.name!r} size mismatch: "
                    f"expected {expected}, got {observation.sample_count}"
                )

    @property
    def pos_xy(self) -> np.ndarray:
        pos_xy = np.column_stack([self.nodes.x_mm, self.nodes.y_mm]).astype(
            np.float32,
            copy=False,
        )
        pos_xy.flags.writeable = False
        return pos_xy

    @property
    def triangles(self) -> np.ndarray:
        triangles = np.stack(
            [
                self.topology.node_1,
                self.topology.node_2,
                self.topology.node_3,
            ],
            axis=1,
        ).astype(np.int32, copy=False)
        triangles.flags.writeable = False
        return triangles

    @property
    def reg_code(self) -> np.ndarray:
        return self.topology.reg_code

    @property
    def fields(self) -> dict[str, np.ndarray]:
        return {
            key: observation.values
            for key, observation in self.field_observations.items()
        }

    def get_field(self, key: str) -> FieldObservation | None:
        observation = self.field_observations.get(key)
        if isinstance(observation, FieldObservation):
            return observation
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "n_elements": int(self.topology.tri_index.shape[0]),
            "n_nodes": int(self.nodes.node_id.shape[0]),
            "field_keys": sorted(self.field_observations.keys()),
            "has_regions": self.regions is not None,
            "fidelity_type": (
                self.solver_metadata.fidelity_type
                if self.solver_metadata is not None
                else ""
            ),
        }

    @classmethod
    def from_components(
        cls,
        *,
        topology: MeshTopology,
        nodes: NodeTable,
        regions: RegionTable | None = None,
        field_observations: Mapping[str, FieldObservation] | None = None,
        solver_metadata: SolverMetadata | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> "MeshFrame":
        return cls(
            topology=topology,
            nodes=nodes,
            regions=regions,
            field_observations=FrozenMapping(field_observations),
            solver_metadata=solver_metadata,
            attrs=FrozenMapping(attrs),
        )
