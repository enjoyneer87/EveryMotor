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
PERIODIC_EDGE_THETA_TOL_DEG = 2.0
PERIODIC_CLUSTER_THETA_TOL_DEG = 3.0
PERIODIC_GAP_TOL_DEG = 7.0
PERIODIC_MIN_CLUSTER_EDGE_COUNT = 2


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


def _wrap_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    return (np.asarray(angle_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _mean_angle_deg(angle_deg: np.ndarray) -> float:
    wrapped = _wrap_angle_deg(angle_deg)
    radians = np.deg2rad(wrapped)
    return float(
        np.degrees(
            np.arctan2(
                np.sin(radians).mean(),
                np.cos(radians).mean(),
            )
        )
    )


def _angular_span_deg(angle_deg: np.ndarray) -> float:
    wrapped = _wrap_angle_deg(angle_deg)
    span = float(np.max(wrapped) - np.min(wrapped))
    if span <= 180.0:
        return span
    lifted = np.where(wrapped < 0.0, wrapped + 360.0, wrapped)
    return float(np.max(lifted) - np.min(lifted))


def build_external_edge_region_lookup(
    triangles: np.ndarray,
    region_code: np.ndarray,
) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Map each external boundary edge to the adjacent element reg_code."""
    tri = np.asarray(triangles, dtype=np.int32)
    reg = np.asarray(region_code, dtype=np.int32)
    if tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] == 0:
        return set(), {}

    if reg.shape[0] != tri.shape[0]:
        reg = np.zeros((tri.shape[0],), dtype=np.int32)

    edge_count = triangles_to_undirected_edges(tri)
    external = find_external_boundary_edges(edge_count)
    edge_to_reg: Dict[Tuple[int, int], int] = {}
    for tri_idx, tri_nodes in enumerate(tri.tolist()):
        tri_reg = int(reg[tri_idx])
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(tri_nodes[i]), int(tri_nodes[j]))))
            if edge in external and edge not in edge_to_reg:
                edge_to_reg[edge] = tri_reg

    return external, edge_to_reg


def _infer_moving_reg_code_set(
    region_code: np.ndarray,
    *,
    moving_reg_codes: np.ndarray | None = None,
    region_name_by_code: Dict[int, str] | None = None,
) -> Set[int]:
    if moving_reg_codes is not None:
        moving = {
            int(code)
            for code in np.asarray(moving_reg_codes, dtype=np.int32).tolist()
            if int(code) > 0
        }
        if moving:
            return moving

    if region_name_by_code:
        keywords = (
            "rotor",
            "shaft",
            "magnet",
            "pocket",
            "a2",
            "a3",
            "a4",
        )
        inferred = {
            int(code)
            for code, name in region_name_by_code.items()
            if any(token in str(name).strip().lower() for token in keywords)
        }
        if inferred:
            return inferred

    _ = np.asarray(region_code, dtype=np.int32)
    return set()


def _build_radial_edge_fragments(
    nodes_xy: np.ndarray,
    external_edges: Set[Tuple[int, int]],
    edge_to_reg: Dict[Tuple[int, int], int],
    *,
    origin_xy: np.ndarray,
    theta_tol_deg: float,
) -> List[Dict[str, object]]:
    fragments: List[Dict[str, object]] = []
    points = np.asarray(nodes_xy, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64)

    for edge in external_edges:
        reg_code = int(edge_to_reg.get(edge, 0))
        edge_points = points[list(edge)]
        rel = edge_points - origin
        theta_deg = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
        theta_span = _angular_span_deg(theta_deg)
        radius = np.linalg.norm(rel, axis=1)
        radius_span = float(np.max(radius) - np.min(radius))
        if theta_span >= theta_tol_deg or radius_span <= 1e-3:
            continue

        fragments.append(
            {
                "edge": edge,
                "reg_code": reg_code,
                "theta_deg": _mean_angle_deg(theta_deg),
                "radius_span": radius_span,
            }
        )

    return fragments


def _cluster_periodic_edge_fragments(
    fragments: List[Dict[str, object]],
    nodes_xy: np.ndarray,
    *,
    origin_xy: np.ndarray,
    theta_tol_deg: float,
) -> List[Dict[str, object]]:
    if not fragments:
        return []

    sorted_fragments = sorted(
        fragments,
        key=lambda item: float(item["theta_deg"]),
    )
    grouped: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = []
    for fragment in sorted_fragments:
        theta_deg = float(fragment["theta_deg"])
        if not current:
            current = [fragment]
            continue
        center_deg = float(
            np.mean([float(item["theta_deg"]) for item in current])
        )
        if abs(theta_deg - center_deg) <= theta_tol_deg:
            current.append(fragment)
            continue
        grouped.append(current)
        current = [fragment]

    if current:
        grouped.append(current)

    clusters: List[Dict[str, object]] = []
    points = np.asarray(nodes_xy, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64)
    for group in grouped:
        node_ids = np.unique(
            np.asarray(
                [
                    node_id
                    for item in group
                    for node_id in tuple(item["edge"])
                ],
                dtype=np.int64,
            )
        )
        if node_ids.size == 0:
            continue
        radius = np.linalg.norm(points[node_ids] - origin, axis=1)
        order = np.argsort(radius)
        node_ids = node_ids[order]
        radius = radius[order]
        clusters.append(
            {
                "theta_deg": float(
                    np.mean([float(item["theta_deg"]) for item in group])
                ),
                "edge_count": int(len(group)),
                "node_ids": node_ids,
                "radial_span": (
                    float(np.max(radius) - np.min(radius))
                    if radius.size > 0 else 0.0
                ),
            }
        )

    return clusters


def _select_periodic_cluster_pair(
    clusters: List[Dict[str, object]],
    *,
    expected_gap_deg: float,
    gap_tol_deg: float,
    min_cluster_edge_count: int,
) -> Tuple[Dict[str, object], Dict[str, object]] | None:
    eligible = [
        cluster
        for cluster in clusters
        if int(cluster["edge_count"]) >= int(min_cluster_edge_count)
    ]
    if len(eligible) < 2:
        return None

    best_score = None
    best_pair: Tuple[Dict[str, object], Dict[str, object]] | None = None
    for low_idx in range(len(eligible)):
        for high_idx in range(low_idx + 1, len(eligible)):
            low_cluster = eligible[low_idx]
            high_cluster = eligible[high_idx]
            low_theta = float(low_cluster["theta_deg"])
            high_theta = float(high_cluster["theta_deg"])
            gap_deg = high_theta - low_theta
            gap_error = abs(gap_deg - expected_gap_deg)
            if gap_error > gap_tol_deg:
                continue

            score = (
                gap_error,
                -(
                    float(low_cluster["radial_span"])
                    + float(high_cluster["radial_span"])
                ),
                -(
                    int(low_cluster["edge_count"])
                    + int(high_cluster["edge_count"])
                ),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (low_cluster, high_cluster)

    return best_pair


def extract_periodic_boundary_groups_from_mesh(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    region_code: np.ndarray,
    *,
    moving_reg_codes: np.ndarray | None = None,
    region_name_by_code: Dict[int, str] | None = None,
    origin_xy: np.ndarray | None = None,
    rotation_deg: float = -45.0,
) -> Tuple[Tuple[Tuple[str, np.ndarray, np.ndarray], ...], Dict[str, object], str | None]:
    """Extract reg-aware periodic master/slave node groups from external edges."""
    diagnostics: Dict[str, object] = {
        "group_count": 0,
        "group_labels": tuple(),
        "external_radial_edge_count": 0,
        "origin_xy": (0.0, 0.0),
    }

    points = np.asarray(nodes_xy, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int32)
    reg = np.asarray(region_code, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        return tuple(), diagnostics, "E-PBC-PERIODIC-GEOMETRY"
    if points.shape[0] == 0 or tri.shape[0] == 0:
        return tuple(), diagnostics, "E-PBC-PERIODIC-EMPTY"
    if reg.shape[0] != tri.shape[0]:
        reg = np.zeros((tri.shape[0],), dtype=np.int32)

    origin = (
        np.asarray(origin_xy, dtype=np.float64)
        if origin_xy is not None else np.zeros((2,), dtype=np.float64)
    )
    diagnostics["origin_xy"] = tuple(float(value) for value in origin.tolist())

    external_edges, edge_to_reg = build_external_edge_region_lookup(tri, reg)
    if not external_edges:
        return tuple(), diagnostics, "E-PBC-PERIODIC-EDGE-NOT-FOUND"

    fragments = _build_radial_edge_fragments(
        points,
        external_edges,
        edge_to_reg,
        origin_xy=origin,
        theta_tol_deg=PERIODIC_EDGE_THETA_TOL_DEG,
    )
    diagnostics["external_radial_edge_count"] = int(len(fragments))
    if not fragments:
        return tuple(), diagnostics, "E-PBC-PERIODIC-RADIAL-EDGE-NOT-FOUND"

    moving_set = _infer_moving_reg_code_set(
        reg,
        moving_reg_codes=moving_reg_codes,
        region_name_by_code=region_name_by_code,
    )

    groups: List[Tuple[str, np.ndarray, np.ndarray]] = []
    family_specs = (
        (("moving", True), ("stationary", False))
        if moving_set else (("all", None),)
    )
    expected_gap_deg = abs(float(rotation_deg))
    for family_label, moving_flag in family_specs:
        family_fragments: List[Dict[str, object]] = []
        for fragment in fragments:
            reg_code_value = int(fragment["reg_code"])
            if moving_flag is None:
                family_fragments.append(fragment)
                continue
            if (reg_code_value in moving_set) == bool(moving_flag):
                family_fragments.append(fragment)

        clusters = _cluster_periodic_edge_fragments(
            family_fragments,
            points,
            origin_xy=origin,
            theta_tol_deg=PERIODIC_CLUSTER_THETA_TOL_DEG,
        )
        pair = _select_periodic_cluster_pair(
            clusters,
            expected_gap_deg=expected_gap_deg,
            gap_tol_deg=PERIODIC_GAP_TOL_DEG,
            min_cluster_edge_count=PERIODIC_MIN_CLUSTER_EDGE_COUNT,
        )
        if pair is None:
            continue

        master_cluster, slave_cluster = pair
        master_nodes = np.asarray(master_cluster["node_ids"], dtype=np.int64)
        slave_nodes = np.asarray(slave_cluster["node_ids"], dtype=np.int64)
        if master_nodes.size < 2 or slave_nodes.size < 2:
            continue
        groups.append((family_label, master_nodes, slave_nodes))

    if not groups:
        return tuple(), diagnostics, "E-PBC-PERIODIC-GROUP-NOT-FOUND"

    diagnostics["group_count"] = int(len(groups))
    diagnostics["group_labels"] = tuple(label for label, _, _ in groups)
    return tuple(groups), diagnostics, None


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
