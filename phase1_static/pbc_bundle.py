"""Phase 1 PBC bundle construction and validation.

Morphism: CanonicalMeshObservation → PBCBundle (immutable, versioned artifact)

Composes all preprocessing stages: contracts → boundary → prior → pairing → bundle.
"""

from __future__ import annotations

from phase1_static.pbc_boundary import extract_boundary_chains
from phase1_static.pbc_contracts import (
    CanonicalMeshObservation,
    PBCBundle,
    PBCBundleResult,
)
from phase1_static.pbc_pairing import build_periodic_pair_set
from phase1_static.pbc_prior import infer_global_prior


def build_pbc_bundle(observation: CanonicalMeshObservation) -> PBCBundleResult:
    """Build complete PBC preprocessing bundle from observation.

    Pure morphism: CanonicalMeshObservation → PBCBundleResult

    Pipeline:
      1. Extract boundary chains from mesh geometry
      2. Infer global prior from chain analysis
      3. Build periodic pair set from chains + prior + node coordinates
      4. Wrap all into versioned PBCBundle

    Args:
        observation: Canonical mesh observation (geometry + fields)

    Returns:
        (bundle, error_code) tuple
    """
    # Step 1: Extract boundary chains
    chain_set, err1 = extract_boundary_chains(observation)
    if err1 is not None:
        return None, err1

    assert chain_set is not None

    # Step 2: Infer global prior
    prior = infer_global_prior(chain_set)

    # Step 3: Build periodic pair set
    pair_set, err3 = build_periodic_pair_set(chain_set, prior, observation.nodes_xy)
    if err3 is not None:
        return None, err3

    assert pair_set is not None

    # Step 4: Create versioned bundle
    bundle = PBCBundle.create(
        observation=observation,
        chain_set=chain_set,
        prior=prior,
        pair_set=pair_set,
    )

    return bundle, None
