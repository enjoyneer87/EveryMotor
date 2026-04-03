"""Phase 1 global geometry prior inference from boundary chains.

Morphism: BoundaryChainSet → PriorContext (immutable value object)

Pure function: no external CAD analyzers, no global state.
"""

from __future__ import annotations

from collections import Counter
from math import gcd
from functools import reduce
from typing import List

from phase1_static.pbc_contracts import (
    BoundaryChainSet,
    PriorContext,
)


def infer_global_prior(chain_set: BoundaryChainSet) -> PriorContext:
    """Infer global 기하 사전 from boundary chain analysis.

    Pure morphism: BoundaryChainSet → PriorContext

    Strategy:
      1. Count radial-type chain families
      2. Estimate n_geom_prior from radial count (e.g., 2 radials → 8-fold for full motor)
      3. Infer anti_periodic_prior (True if n_geom is even)

    Args:
        chain_set: Boundary chains with rotation origin

    Returns:
        PriorContext with n_geom_prior, anti_periodic_prior, evidence_notes
    """
    # Count chain types
    type_counts = Counter(chain.chain_type for chain in chain_set.chains)
    radial_count = type_counts.get("radial", 0)
    arc_count = type_counts.get("arc", 0)

    # Estimate n_geom from radial families
    # Heuristic: for a sector mesh, radial chains mark the sector edges
    # Example: 2 radial → sector is 1/4 or 1/8 → full motor is 4-fold or 8-fold
    if radial_count == 0:
        # No clear radial structure, default to 8-fold
        n_geom_prior = 8
        evidence = "No radial chains detected; defaulting to 8-fold symmetry"
    elif radial_count == 2:
        # Typical for 1/8 sector: 2 radial edges → 8-fold
        n_geom_prior = 8
        evidence = f"2 radial chains detected → 8-fold symmetry"
    elif radial_count == 1:
        # Possible 1/4 or 1/2 sector
        n_geom_prior = 4
        evidence = f"1 radial chain detected → 4-fold symmetry"
    else:
        # Multiple radial families (e.g., due to interface splitting)
        # GCD heuristic: find common factor
        candidate_geom = radial_count * 4  # Conservative estimate
        n_geom_prior = candidate_geom
        evidence = f"{radial_count} radial chain families detected → {candidate_geom}-fold estimate"

    # Anti-periodic: True if n_geom is even (typical for motors)
    anti_periodic_prior = (n_geom_prior % 2 == 0)

    return PriorContext(
        n_geom_prior=n_geom_prior,
        anti_periodic_prior=anti_periodic_prior,
        evidence_notes=evidence,
    )
