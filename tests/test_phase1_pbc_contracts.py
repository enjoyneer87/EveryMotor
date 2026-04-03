"""Tests for phase1_static.pbc_contracts immutable dataclasses."""

import numpy as np
import pytest

from phase1_static.pbc_contracts import (
    BoundaryChain,
    BoundaryChainSet,
    CanonicalMeshObservation,
    FrozenMapping,
    PBCBundle,
    PeriodicChainPair,
    PeriodicPairSet,
    PriorContext,
)


class TestFrozenMapping:
    """FrozenMapping dict wrapper."""

    def test_frozen_mapping_read(self) -> None:
        data = {"a": 1, "b": 2}
        fm = FrozenMapping(data)
        assert fm["a"] == 1
        assert "b" in fm

    def test_frozen_mapping_setitem_raises(self) -> None:
        fm = FrozenMapping({"a": 1})
        with pytest.raises(TypeError, match="does not support item assignment"):
            fm["a"] = 2


class TestCanonicalMeshObservation:
    """Immutable mesh observation with frozen numpy arrays."""

    def test_canonical_mesh_observation_freezes_arrays_and_attrs(self) -> None:
        """Verify that node and triangle arrays become read-only post-init."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        region = np.array([1], dtype=np.int32)
        fields = FrozenMapping({"bx": np.array([0.1, 0.2, 0.3])})

        obs = CanonicalMeshObservation(
            nodes_xy=nodes,
            triangles=triangles,
            region_code=region,
            field_dict=fields,
            sector_id="1/8",
        )

        # Arrays must be read-only
        assert not obs.nodes_xy.flags.writeable
        assert not obs.triangles.flags.writeable
        assert not obs.region_code.flags.writeable

        # Cannot mutate array
        with pytest.raises(ValueError, match="read-only"):
            obs.nodes_xy[0, 0] = 99.0


class TestBoundaryChain:
    """Boundary chain with endpoint validation."""

    def test_boundary_chain_rejects_bad_endpoint_shape(self) -> None:
        """endpoint_pair must be a 2-tuple."""
        node_indices = np.array([0, 1, 2], dtype=np.int32)

        # Invalid: single int
        with pytest.raises(ValueError, match="endpoint_pair must be a 2-tuple"):
            BoundaryChain(
                node_indices=node_indices,
                chain_type="arc",
                endpoint_pair=0,  # type: ignore
            )

        # Invalid: 3-tuple
        with pytest.raises(ValueError, match="endpoint_pair must be a 2-tuple"):
            BoundaryChain(
                node_indices=node_indices,
                chain_type="arc",
                endpoint_pair=(0, 1, 2),  # type: ignore
            )

        # Valid: 2-tuple
        chain = BoundaryChain(
            node_indices=node_indices,
            chain_type="arc",
            endpoint_pair=(0, 2),
        )
        assert chain.endpoint_pair == (0, 2)


class TestPeriodicChainPair:
    """Periodic chain pair with match_ratio validation."""

    def test_periodic_chain_pair_rejects_out_of_range_match_ratio(self) -> None:
        """match_ratio must be in [0.0, 1.0]."""
        # Too low
        with pytest.raises(ValueError, match="match_ratio must be in"):
            PeriodicChainPair(
                master_chain_idx=0,
                slave_chain_idx=1,
                rotation_deg=-45.0,
                match_ratio=-0.1,  # Invalid
            )

        # Too high
        with pytest.raises(ValueError, match="match_ratio must be in"):
            PeriodicChainPair(
                master_chain_idx=0,
                slave_chain_idx=1,
                rotation_deg=-45.0,
                match_ratio=1.5,  # Invalid
            )

        # Valid: boundaries
        pair_low = PeriodicChainPair(
            master_chain_idx=0, slave_chain_idx=1, rotation_deg=-45.0, match_ratio=0.0
        )
        assert pair_low.match_ratio == 0.0

        pair_high = PeriodicChainPair(
            master_chain_idx=0, slave_chain_idx=1, rotation_deg=-45.0, match_ratio=1.0
        )
        assert pair_high.match_ratio == 1.0


class TestPeriodicPairSet:
    """PeriodicPairSet with bidirectional edge validation."""

    def test_periodic_pair_set_requires_bidirectional_edges(self) -> None:
        """Every edge (u, v) must have reverse edge (v, u)."""
        pair = PeriodicChainPair(
            master_chain_idx=0,
            slave_chain_idx=1,
            rotation_deg=-45.0,
            match_ratio=0.9,
        )

        # Non-bidirectional edge_index: (0,1) but no (1,0)
        bad_edge_index = np.array([[0], [1]], dtype=np.int64)
        bad_edge_attr = np.array([[1.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="not bidirectional"):
            PeriodicPairSet(
                pairs=(pair,),
                pbc_edge_index=bad_edge_index,
                pbc_edge_attr=bad_edge_attr,
            )

        # Bidirectional: (0,1) and (1,0)
        good_edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
        good_edge_attr = np.array([[1.0], [-1.0]], dtype=np.float32)

        pair_set = PeriodicPairSet(
            pairs=(pair,),
            pbc_edge_index=good_edge_index,
            pbc_edge_attr=good_edge_attr,
        )
        assert pair_set.pairs[0].match_ratio == 0.9


class TestPBCBundle:
    """Full PBC bundle with nested contracts."""

    def test_pbc_bundle_accepts_nested_contracts_and_freezes_context(self) -> None:
        """PBCBundle wraps all preprocessing stages."""
        # Build minimal components
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        region = np.array([1], dtype=np.int32)
        fields = FrozenMapping({})

        obs = CanonicalMeshObservation(
            nodes_xy=nodes,
            triangles=triangles,
            region_code=region,
            field_dict=fields,
            sector_id="1/8",
        )

        chain = BoundaryChain(
            node_indices=np.array([0, 1, 2], dtype=np.int32),
            chain_type="arc",
            endpoint_pair=(0, 2),
        )
        chain_set = BoundaryChainSet(
            chains=(chain,),
            rotation_origin_xy=np.array([0.5, 0.33], dtype=np.float64),
            mesh_hash="hash_abc123",
        )

        prior = PriorContext(
            n_geom_prior=8,
            anti_periodic_prior=False,
            evidence_notes="2 radial families inferred",
        )

        pair = PeriodicChainPair(
            master_chain_idx=0,
            slave_chain_idx=1,
            rotation_deg=-45.0,
            match_ratio=0.95,
        )
        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
        edge_attr = np.array([[1.0], [-1.0]], dtype=np.float32)
        pair_set = PeriodicPairSet(pairs=(pair,), pbc_edge_index=edge_index, pbc_edge_attr=edge_attr)

        # Bundle
        bundle = PBCBundle.create(
            observation=obs,
            chain_set=chain_set,
            prior=prior,
            pair_set=pair_set,
        )

        # Verify structure
        assert bundle.chain_set.chains[0].chain_type == "arc"
        assert bundle.prior.n_geom_prior == 8
        assert bundle.format_version == "0.1.0"
        assert bundle.bundle_type == "phase1_pbc_bundle"
        assert not bundle.observation.nodes_xy.flags.writeable
