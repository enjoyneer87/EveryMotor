"""Phase 1 boundary chain extraction from mesh geometry.

Morphism: Mesh geometry → BoundaryChainSet (immutable value object)

Key insight (from debugging):
  - Chain splitting MUST happen FIRST (via corner turn detection at 45°)
  - THEN estimate rotation origin from non-linear (arc-like) chains
  - Using boundary centroid directly causes all chains to classify as mixed_polyline

Reference: .github/instructions/design-principles.instructions.md
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.spatial import KDTree

from phase1_static.pbc_contracts import (
    BoundaryChain,
    BoundaryChainSet,
    BoundaryChainSetResult,
    CanonicalMeshObservation,
    ChainType,
)


LOG = logging.getLogger(__name__)

# Constants
CORNER_TURN_DEG = 45.0  # Corner detection threshold (degrees)
RADIAL_THETA_TOL_DEG = 5.0  # Radial classification tolerance
ARC_RADIUS_RTOL = 5e-2  # Arc classification tolerance (5%)
LINE_FIT_RESIDUAL_TOL = 1e-3  # Threshold for "non-linear" chain


# ============================================================================
# Layer 1: Edge Extraction
# ============================================================================


def triangles_to_undirected_edges(triangles: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Convert triangle elements to undirected edge counts.

    Args:
        triangles: shape [N_tri, 3], int32, triangle node indices

    Returns:
        edge_count: dict mapping (min(u,v), max(u,v)) → count
    """
    edge_count: Dict[Tuple[int, int], int] = {}
    for tri in triangles:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            edge = tuple(sorted([tri[i], tri[j]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    return edge_count


def find_external_boundary_edges(edge_count: Dict[Tuple[int, int], int]) -> Set[Tuple[int, int]]:
    """Find edges appearing exactly once (external boundary)."""
    return {edge for edge, count in edge_count.items() if count == 1}


def find_region_interface_edges(
    triangles: np.ndarray, region_code: np.ndarray
) -> Set[Tuple[int, int]]:
    """Find edges between different region codes (internal interfaces)."""
    interface: Set[Tuple[int, int]] = set()
    edge_regions: Dict[Tuple[int, int], Set[int]] = {}

    for tri_idx, tri in enumerate(triangles):
        region = region_code[tri_idx]
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            edge = tuple(sorted([tri[i], tri[j]]))
            if edge not in edge_regions:
                edge_regions[edge] = set()
            edge_regions[edge].add(region)

    # Interface edges separate at least 2 different regions
    for edge, regions in edge_regions.items():
        if len(regions) > 1:
            interface.add(edge)

    return interface


# ============================================================================
# Layer 2: Graph Construction
# ============================================================================


def build_boundary_graph(
    external_edges: Set[Tuple[int, int]], interface_edges: Set[Tuple[int, int]]
) -> Dict[int, List[int]]:
    """Build adjacency dict from boundary edges (external + interface).

    Returns:
        graph: {node_id: [neighbor_id, ...]}
    """
    all_edges = external_edges | interface_edges
    graph: Dict[int, List[int]] = {}

    for edge in all_edges:
        u, v = edge
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)

    return graph


def detect_non_manifold_nodes(graph: Dict[int, List[int]]) -> Set[int]:
    """Detect nodes with degree != 2 (junctions or dead-ends)."""
    return {node for node, neighbors in graph.items() if len(neighbors) != 2}


# ============================================================================
# Layer 3: Chain Splitting (FIRST step before origin estimation)
# ============================================================================


def _compute_turn_angle(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """Compute turn angle (degrees) at p2 in sequence p1→p2→p3.

    Angle is in [0, 180] representing the turn magnitude.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < 1e-10 or v2_norm < 1e-10:
        return 0.0

    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def split_boundary_graph_into_chains(
    graph: Dict[int, List[int]], nodes_xy: np.ndarray
) -> List[List[int]]:
    """Split boundary graph into chains at corner turns ≥ CORNER_TURN_DEG.

    Strategy: Trace paths through the boundary graph, starting from each unvisited
    edge. At each step, check the turn angle; if ≥ CORNER_TURN_DEG, split into
    a new chain.

    Returns:
        chains: list of node sequences [[n0, n1, ...], ...]
    """
    # Build undirected edge set
    visited_edges = set()
    edges_set = set()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edge = (min(node, neighbor), max(node, neighbor))
            edges_set.add(edge)

    chains: List[List[int]] = []

    while edges_set:
        # Pick any unvisited edge
        edge = edges_set.pop()
        start_node, next_node = edge[0], edge[1]

        # Build chain starting from start_node
        chain = [start_node, next_node]
        prev_node = start_node
        current_node = next_node

        while True:
            # Find next unvisited neighbors
            neighbors = graph.get(current_node, [])
            candidates = [
                n for n in neighbors
                if (min(current_node, n), max(current_node, n)) in edges_set
            ]

            if not candidates:
                # No more unvisited edges, chain complete
                break

            # If multiple candidates, pick one based on turn angle
            if len(candidates) == 1:
                next_candidate = candidates[0]
                turn_angle = _compute_turn_angle(
                    nodes_xy[prev_node], nodes_xy[current_node], nodes_xy[next_candidate]
                )
            else:
                # Multiple candidates: pick by smallest turn (prefer straight lines)
                next_candidate = None
                min_turn = None
                for cand in candidates:
                    turn = _compute_turn_angle(
                        nodes_xy[prev_node], nodes_xy[current_node], nodes_xy[cand]
                    )
                    if min_turn is None or turn < min_turn:
                        min_turn = turn
                        next_candidate = cand
                turn_angle = min_turn

            # Check for corner
            if turn_angle >= CORNER_TURN_DEG:
                # Split here: save current chain and start new one
                chains.append(chain)
                chain = [current_node, next_candidate]
            else:
                # Continue chain
                chain.append(next_candidate)

            # Mark edge as visited
            edge_key = (min(current_node, next_candidate), max(current_node, next_candidate))
            edges_set.discard(edge_key)

            prev_node = current_node
            current_node = next_candidate

        # Save final chain
        if len(chain) > 1:
            chains.append(chain)

    return chains


# ============================================================================
# Layer 4: Origin Estimation (SECOND step, using split chains)
# ============================================================================


def _line_fit_residual(xy: np.ndarray) -> float:
    """RMS residual of points from best-fit line.

    Used to detect "non-linear" chains for circle fitting.
    """
    if len(xy) < 2:
        return 0.0

    # Center points
    xy_centered = xy - xy.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(xy_centered, full_matrices=False)
    # Smallest singular vector is perpendicular to line
    normal = Vt[-1]

    # Distance from each point to line
    distances = np.abs(xy_centered @ normal)
    return float(np.sqrt(np.mean(distances**2)))


def _fit_circle_origin(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit circle to 2D points via least-squares.

    Returns:
        (center [2], radius)
    """
    if len(xy) < 3:
        center = xy.mean(axis=0)
        radius = float(np.linalg.norm(xy - center).mean())
        return center, radius

    # Set up linear system: (x - xc)^2 + (y - yc)^2 = r^2
    # Expand: x^2 + y^2 - 2*xc*x - 2*yc*y + (xc^2 + yc^2 - r^2) = 0
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)

    # Solve via normal equations
    AtA = A.T @ A
    Atb = A.T @ b
    try:
        xc, yc, c = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        # Singular matrix, fallback to mean
        center = xy.mean(axis=0)
        radius = float(np.linalg.norm(xy - center).mean())
        return center, radius

    center = np.array([xc, yc], dtype=np.float64)
    radius = float(np.sqrt(xc**2 + yc**2 - c))
    return center, radius


def estimate_rotation_origin(
    nodes_xy: np.ndarray, chain_paths: List[List[int]]
) -> np.ndarray:
    """Estimate rotation origin from arc-like chains via circle fitting.

    Strategy: identify non-linear chains (candidates for arcs), fit circles,
    average the centers.

    Args:
        nodes_xy: shape [N, 2]
        chain_paths: list of node index sequences

    Returns:
        origin: shape [2], estimated rotation center
    """
    candidate_origins = []

    for path in chain_paths:
        if len(path) < 3:
            continue

        xy_chain = nodes_xy[path]

        # Check if this chain is "non-linear" (good candidate for arc)
        residual = _line_fit_residual(xy_chain)

        if residual > LINE_FIT_RESIDUAL_TOL:
            # Likely an arc, fit circle
            center, _ = _fit_circle_origin(xy_chain)
            candidate_origins.append(center)

    if candidate_origins:
        return np.mean(candidate_origins, axis=0).astype(np.float64)
    else:
        # Fallback to boundary centroid (least ideal)
        return nodes_xy.mean(axis=0).astype(np.float64)


# ============================================================================
# Layer 5: Chain Classification
# ============================================================================


def compute_chain_descriptor(
    xy_chain: np.ndarray, origin: np.ndarray
) -> ChainType:
    """Classify chain type by polar character from rotation origin.

    Returns one of: "arc", "radial", "mixed_polyline", "unresolved"
    """
    if len(xy_chain) < 2:
        return "unresolved"

    # Compute polar coords relative to origin
    r = np.linalg.norm(xy_chain - origin, axis=1)
    theta = np.arctan2(xy_chain[:, 1] - origin[1], xy_chain[:, 0] - origin[0])
    theta = np.degrees(theta)

    # Check arc: constant radius, varying angle
    r_mean = r.mean()
    r_std = r.std()
    if r_mean > 1e-10:
        r_rtol = r_std / r_mean
    else:
        r_rtol = 0.0

    theta_span = np.abs(np.max(theta) - np.min(theta))
    if theta_span > 180:
        # Handle wraparound at ±180
        theta_wrapped = np.where(theta < 0, theta + 360, theta)
        theta_span = np.max(theta_wrapped) - np.min(theta_wrapped)

    # Heuristics
    is_arc = r_rtol < ARC_RADIUS_RTOL and theta_span > 5.0
    is_radial = theta_span < RADIAL_THETA_TOL_DEG and r_std > 1e-3

    if is_arc:
        return "arc"
    elif is_radial:
        return "radial"
    elif theta_span > 2.0 or r_std > 1e-3:
        return "mixed_polyline"
    else:
        return "unresolved"


# ============================================================================
# Layer 6: Public API
# ============================================================================


def extract_boundary_chains_from_mesh(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    region_code: np.ndarray,
) -> BoundaryChainSetResult:
    """Extract boundary chains from raw mesh geometry.

    Pure morphism: numpy arrays → BoundaryChainSet

    Args:
        nodes_xy: shape [N, 2], float64
        triangles: shape [T, 3], int32
        region_code: shape [T], int32

    Returns:
        (chain_set, error_code) tuple
    """
    # Validate inputs
    if len(nodes_xy) == 0 or len(triangles) == 0:
        return None, "E-MESH-BOUNDARY-001"

    # Extract boundary edges
    edge_count = triangles_to_undirected_edges(triangles)
    external = find_external_boundary_edges(edge_count)
    interface = find_region_interface_edges(triangles, region_code)

    boundary_edges = external | interface
    if not boundary_edges:
        return None, "E-MESH-BOUNDARY-001"

    # Build graph and split at corners FIRST
    graph = build_boundary_graph(external, interface)
    chain_paths = split_boundary_graph_into_chains(graph, nodes_xy)

    if not chain_paths:
        return None, "E-MESH-BOUNDARY-001"

    # Estimate rotation origin from arc-like chains SECOND
    origin = estimate_rotation_origin(nodes_xy, chain_paths)

    # Classify each chain
    chains = []
    for path in chain_paths:
        xy_chain = nodes_xy[path]
        chain_type = compute_chain_descriptor(xy_chain, origin)

        endpoint = (path[0], path[-1])
        chain = BoundaryChain(
            node_indices=np.array(path, dtype=np.int32),
            chain_type=chain_type,
            endpoint_pair=endpoint,
        )
        chains.append(chain)

    # Compute mesh hash for reproducibility
    mesh_hash = hashlib.sha256(
        (triangles.tobytes() + region_code.tobytes())
    ).hexdigest()[:16]

    chain_set = BoundaryChainSet(
        chains=tuple(chains),
        rotation_origin_xy=origin,
        mesh_hash=mesh_hash,
    )

    return chain_set, None


def extract_boundary_chains(
    observation: CanonicalMeshObservation,
) -> BoundaryChainSetResult:
    """Extract boundary chains from CanonicalMeshObservation (immutable wrapper).

    Pure morphism: CanonicalMeshObservation → BoundaryChainSet
    """
    return extract_boundary_chains_from_mesh(
        nodes_xy=observation.nodes_xy,
        triangles=observation.triangles,
        region_code=observation.region_code,
    )
