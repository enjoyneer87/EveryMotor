"""Phase 1 PBC preprocessing immutable contracts and value objects.

Category-theory-aligned OOP: all data objects are frozen dataclasses with
read-only numpy arrays. Transformations are pure functions returning
tuple[T | None, str | None] at API boundaries (no bare exceptions).

Reference: .github/instructions/design-principles.instructions.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Tuple, TypeAlias

import numpy as np


# ============================================================================
# Constants
# ============================================================================

PBC_BUNDLE_FORMAT_VERSION = "0.1.0"
PBC_BUNDLE_TYPE = "phase1_pbc_bundle"

# Chain type literals
ChainType = Literal["arc", "radial", "mixed_polyline", "unresolved"]


# ============================================================================
# Helper: FrozenMapping
# ============================================================================

class FrozenMapping:
    """Immutable dict wrapper for frozen dataclass attributes."""

    def __init__(self, data: Dict[str, Any]) -> None:
        """Wrap dict as read-only."""
        self._data = dict(data)  # shallow copy

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("FrozenMapping does not support item assignment")

    def __delitem__(self, key: str) -> None:
        raise TypeError("FrozenMapping does not support item deletion")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


# ============================================================================
# Data Contracts
# ============================================================================

@dataclass(frozen=True)
class CanonicalMeshObservation:
    """Immutable mesh geometry + field observation container.

    Represents a single motor mesh snapshot with spatial coordinates,
    element connectivity, region codes, and associated field data.
    All arrays are frozen (read-only) after construction.
    """

    nodes_xy: np.ndarray  # shape [N_nodes, 2], float64
    triangles: np.ndarray  # shape [N_tri, 3], int32 (node indices)
    region_code: np.ndarray  # shape [N_tri], int32 (region IDs)
    field_dict: FrozenMapping  # {field_name: array[N_nodes] or array[N_tri]}
    sector_id: str  # e.g., "1/8", "quarter", "full"

    def __post_init__(self) -> None:
        """Freeze numpy arrays after initialization."""
        # Set arrays as read-only (direct assignment works even in frozen dataclass)
        self.nodes_xy.flags.writeable = False
        self.triangles.flags.writeable = False
        self.region_code.flags.writeable = False


@dataclass(frozen=True)
class BoundaryChain:
    """Immutable boundary edge chain.

    A contiguous sequence of boundary edges forming a polyline or arc.
    Endpoint pair is a tuple of start and end node indices.
    """

    node_indices: np.ndarray  # shape [M,], int32, node sequence along chain
    chain_type: ChainType  # "arc", "radial", "mixed_polyline", "unresolved"
    endpoint_pair: Tuple[int, int]  # (start_node_idx, end_node_idx)

    def __post_init__(self) -> None:
        """Validate endpoint pair shape and freeze array."""
        if not isinstance(self.endpoint_pair, tuple) or len(self.endpoint_pair) != 2:
            raise ValueError(
                f"endpoint_pair must be a 2-tuple, got {self.endpoint_pair!r}"
            )

        self.node_indices.flags.writeable = False


@dataclass(frozen=True)
class BoundaryChainSet:
    """Immutable collection of boundary chains from a mesh.

    Represents all detected boundary features (external edges, region interfaces)
    clustered into contiguous chains with estimated rotation origin.
    """

    chains: Tuple[BoundaryChain, ...]  # immutable tuple of chains
    rotation_origin_xy: np.ndarray  # shape [2], float64 (x, y of rotation center)
    mesh_hash: str  # hash of mesh geometry for reproducibility

    def __post_init__(self) -> None:
        """Freeze rotation origin array."""
        self.rotation_origin_xy.flags.writeable = False


@dataclass(frozen=True)
class PriorContext:
    """Immutable global geometry prior from mesh analysis.

    Captures inferred symmetry properties and periodicity rules
    from boundary chain analysis.
    """

    n_geom_prior: int  # estimated number of-fold symmetry (e.g., 8 for 1/8 sector)
    anti_periodic_prior: bool  # whether anti-periodic BCs are expected
    evidence_notes: str  # human-readable explanation of inference


@dataclass(frozen=True)
class PeriodicChainPair:
    """Immutable pair of related boundary chains in periodic domain.

    Records a master-slave chain correspondence with spatial matching confidence.
    """

    master_chain_idx: int
    slave_chain_idx: int
    rotation_deg: float  # rotation angle from master to slave (degrees)
    match_ratio: float  # [0.0, 1.0] confidence of spatial match

    def __post_init__(self) -> None:
        """Validate match_ratio bounds."""
        if not (0.0 <= self.match_ratio <= 1.0):
            raise ValueError(
                f"match_ratio must be in [0.0, 1.0], got {self.match_ratio}"
            )


@dataclass(frozen=True)
class PeriodicPairSet:
    """Immutable set of periodic boundary condition definitions.

    Groups matched chain pairs with graph connectivity (pbc_edge_index, pbc_edge_attr)
    for downstream GNN input layer.
    """

    pairs: Tuple[PeriodicChainPair, ...]  # matched master-slave pairs
    pbc_edge_index: np.ndarray  # shape [2, 2P], int64 (source, target node indices)
    pbc_edge_attr: np.ndarray  # shape [2P, 1], float32 (+1.0 periodic, -1.0 anti-periodic)

    def __post_init__(self) -> None:
        """Validate edge dimensions and freeze arrays."""
        if self.pbc_edge_index.shape[0] != 2:
            raise ValueError(
                f"pbc_edge_index must have shape [2, E], got {self.pbc_edge_index.shape}"
            )
        if self.pbc_edge_attr.shape[1] != 1:
            raise ValueError(
                f"pbc_edge_attr must have shape [E, 1], got {self.pbc_edge_attr.shape}"
            )

        # Check bidirectionality: for each (u, v), (v, u) must exist
        edge_set = set(map(tuple, self.pbc_edge_index.T))
        for u, v in edge_set:
            if (v, u) not in edge_set:
                raise ValueError(
                    f"pbc_edge_index is not bidirectional: ({u}, {v}) has no reverse"
                )

        self.pbc_edge_index.flags.writeable = False
        self.pbc_edge_attr.flags.writeable = False


@dataclass(frozen=True)
class PBCBundle:
    """Immutable complete PBC preprocessing result.

    Bundles all preprocessing stages into a single versioned artifact
    for deterministic serialization and downstream consumption.
    """

    observation: CanonicalMeshObservation
    chain_set: BoundaryChainSet
    prior: PriorContext
    pair_set: PeriodicPairSet
    format_version: str
    bundle_type: str

    @classmethod
    def create(
        cls,
        observation: CanonicalMeshObservation,
        chain_set: BoundaryChainSet,
        prior: PriorContext,
        pair_set: PeriodicPairSet,
    ) -> PBCBundle:
        """Factory method with default version/type."""
        return cls(
            observation=observation,
            chain_set=chain_set,
            prior=prior,
            pair_set=pair_set,
            format_version=PBC_BUNDLE_FORMAT_VERSION,
            bundle_type=PBC_BUNDLE_TYPE,
        )


# ============================================================================
# Type Aliases (Error Wrapping for API Boundaries)
# ============================================================================

BoundaryChainSetResult: TypeAlias = Tuple[BoundaryChainSet | None, str | None]
"""Result of boundary chain extraction: (result, error_code) tuple.

On success: (chain_set, None)
On error: (None, error_code_string)
"""

PeriodicPairSetResult: TypeAlias = Tuple[PeriodicPairSet | None, str | None]
"""Result of periodic pair matching: (result, error_code) tuple.

On success: (pair_set, None)
On error: (None, error_code_string)
"""

PBCBundleResult: TypeAlias = Tuple[PBCBundle | None, str | None]
"""Result of full PBC bundle construction: (result, error_code) tuple.

On success: (bundle, None)
On error: (None, error_code_string)
"""
