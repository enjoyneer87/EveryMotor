# Phase 1 PBC Preprocessing Design

This document freezes the design direction for Phase 1 periodic-boundary preprocessing.

It covers three items in sequence:
- immutable data contract drafts for `BoundaryChainSet`, `PeriodicPairSet`, and `PBCBundle`
- pseudocode for periodic-pair inference with `global prior` and `local matcher` separated
- proposed `phase1_static` module layout and file skeletons

## Why This Plan Exists
- Current DOE sample building emits empty `pbc_edge_index` and `pbc_edge_attr`, so raw DOE training does not yet consume PBC structure.
- Existing KDTree matching in `phase1_static/data_preprocessing.py` is useful only after candidate master/slave boundary sets are already known.
- We need a mesh-native preprocessing path that respects the repository's category-theory-aligned OOP rules and ADR functor boundaries.

## Architecture Fit Check

### Compatible with ADR functor architecture
- Parsing remains local to the EveryMotor side.
- Boundary extraction and periodic pairing operate on canonical mesh values, not on `eMach` CAD DTOs.
- `eMach` and `pyleecan` logic may inform `global prior` rules, but must not become a hard runtime dependency of mesh preprocessing.
- Bundle serialization is treated as an adapter from internal pure objects to a storage format.

### Compatible with category-theory-aligned OOP
- Objects are immutable value carriers.
- Transforms are pure morphisms with explicit inputs and outputs.
- Composition is preserved as:
  - `raw H5 -> CanonicalMeshObservation`
  - `CanonicalMeshObservation -> BoundaryChainSet`
  - `BoundaryChainSet + PriorContext -> PeriodicPairSet`
  - `CanonicalMeshObservation + BoundaryChainSet + PeriodicPairSet -> PBCBundle`

### Compatible with phase contract-first harnessing
- `step_semantics`, `fidelity_type`, and `coupling_policy` stay attached at sample-build time.
- `PBCBundle` extends the canonical sample contract rather than bypassing it.
- Validation gates can be attached to each stage boundary.

## Hard Constraints
- Do not import `postproc_interop` into `eMach`.
- Do not make mesh preprocessing depend on CAD `EntityInfo` objects.
- Do not mutate upstream sample dicts in place.
- Preserve solver-domain metadata across preprocessing and inference outputs.
- Treat topology, boundary, and pairing as first-class contracts, not incidental arrays.

## Non-Goals
- Full CAD reconstruction from mesh.
- GUI-first boundary labeling workflow.
- Dynamic multi-step rotor-gap edge refresh for later phases.
- Long-run training or inference implementation in this document.

## Acceptance Checks
- A topology group produces deterministic boundary chains from the same mesh input.
- Periodic pair inference can explain why a pair was accepted or rejected.
- Bundle format can feed `phase1_static/motor_dataset.py` without hidden metadata loss.
- Failure states are representable as explicit result values or error codes.

---

## 1. Immutable Data Contract Drafts

### Design rule
- These contracts are `Object`s in the category-theory sense.
- They must be immutable after construction.
- They must be independent of file format concerns.

### 1.1 CanonicalMeshObservation

This is the upstream canonical input for preprocessing. It is not a new external format. It is the internal value object built from DOE H5 records.

```python
@dataclass(frozen=True)
class CanonicalMeshObservation:
    topology_group_id: str
    topology_hash: str
    sequence_id: int
    step_index: int
    source_file_name: str
    source_file_type: str
    fidelity_type: str
    step_semantics: str
    coupling_policy: str
    time_s: float
    rotate_step: float
    node_id: np.ndarray           # shape (N,)
    pos_xy: np.ndarray            # shape (N, 2)
    triangles: np.ndarray         # shape (E, 3), canonical 0-based node indices
    reg_code: np.ndarray          # shape (E,)
    moving_node_indices: np.ndarray | None
    target_bx_by_a_j: np.ndarray | None   # shape (N, 4) when available
    attrs: Mapping[str, Any]
```

#### Notes
- `CanonicalMeshObservation` is sample-scoped.
- `topology_hash` is invariant for all observations sharing the same mesh topology.
- `pos_xy` may vary by step, while `triangles` and `topology_hash` remain fixed within one topology group.

### 1.2 BoundaryChain

```python
@dataclass(frozen=True)
class BoundaryChain:
    chain_id: str
    family_id: str
    node_index: np.ndarray        # shape (K,), ordered path nodes
    edge_index: np.ndarray        # shape (K-1, 2), ordered chain edges
    chain_type: str               # radial | arc | mixed_polyline | unresolved
    theta_mean_deg: float
    theta_span_deg: float
    r_min: float
    r_max: float
    length: float
    endpoint_index: np.ndarray    # shape (2,)
    endpoint_tangent_xy: np.ndarray   # shape (2, 2)
    adjacent_region_hist: Mapping[int, int]
    quality_score: float
    attrs: Mapping[str, Any]
```

#### Notes
- `family_id` groups chains that appear to belong to the same semantic side family.
- `chain_type` is geometric classification, not solver semantics.
- `quality_score` expresses extractor confidence, not pair validity.

### 1.3 BoundaryChainSet

```python
@dataclass(frozen=True)
class BoundaryChainSet:
    topology_group_id: str
    topology_hash: str
    origin_xy: np.ndarray                 # shape (2,)
    outer_boundary_node_index: np.ndarray
    interface_boundary_node_index: np.ndarray
    non_manifold_node_index: np.ndarray
    chains: tuple[BoundaryChain, ...]
    family_summary: Mapping[str, Mapping[str, Any]]
    extraction_method: str               # mesh_boundary_graph_v1
    extraction_error_code: str | None
    extraction_warnings: tuple[str, ...]
    attrs: Mapping[str, Any]
```

#### Notes
- `BoundaryChainSet` is topology-scoped, not step-scoped.
- `extraction_error_code` allows explicit failure reporting without collapsing the pipeline.

### 1.4 PriorContext

```python
@dataclass(frozen=True)
class PriorContext:
    topology_group_id: str
    topology_hash: str
    n_slots_prior: int | None
    n_poles_prior: int | None
    n_geom_prior: int | None
    anti_periodic_prior: bool | None
    prior_source: str            # mesh_rule | cad_metadata | pyleecan_rule | manual_override | unknown
    confidence: float
    attrs: Mapping[str, Any]
```

#### Notes
- This object isolates prior knowledge from local matching.
- It is intentionally small so it can be produced by several independent upstream rules.

### 1.5 PeriodicChainPair

```python
@dataclass(frozen=True)
class PeriodicChainPair:
    pair_id: str
    master_chain_id: str
    slave_chain_id: str
    expected_rotation_deg: float
    applied_rotation_deg: float
    sign: float                  # +1.0 periodic, -1.0 anti-periodic
    sign_source: str
    chain_match_score: float
    node_match_ratio: float
    max_match_distance: float
    mean_match_distance: float
    unmatched_slave_count: int
    diagnostics: Mapping[str, Any]
```

### 1.6 PeriodicPairSet

```python
@dataclass(frozen=True)
class PeriodicPairSet:
    topology_group_id: str
    topology_hash: str
    period_angle_deg: float | None
    anti_periodic: bool | None
    global_prior: PriorContext
    chain_pairs: tuple[PeriodicChainPair, ...]
    master_node_index: np.ndarray     # shape (P,)
    slave_node_index: np.ndarray      # shape (P,)
    pbc_edge_index: np.ndarray        # shape (2, 2P) bidirectional
    pbc_edge_attr: np.ndarray         # shape (2P, 1)
    pairing_method: str               # rotate_and_kdtree_v1
    pairing_error_code: str | None
    pairing_warnings: tuple[str, ...]
    attrs: Mapping[str, Any]
```

#### Notes
- `chain_pairs` keeps explainability.
- `master_node_index` and `slave_node_index` provide compact direct access.
- `pbc_edge_index` and `pbc_edge_attr` are downstream-compatible materializations.

### 1.7 PBCBundle

```python
@dataclass(frozen=True)
class PBCBundle:
    format_version: str
    bundle_type: str                  # phase1_pbc_bundle
    topology_group_id: str
    topology_hash: str
    solver_context: Mapping[str, Any]
    mesh_context: Mapping[str, Any]
    boundary_chain_set: BoundaryChainSet
    periodic_pair_set: PeriodicPairSet
    observation_index: Mapping[str, np.ndarray]
    observation_payload_ref: Mapping[str, Any]
    validation_summary: Mapping[str, Any]
    attrs: Mapping[str, Any]
```

#### Notes
- `PBCBundle` is topology-group-scoped.
- Observations are attached by index or payload reference, not duplicated blindly.
- Storage adapters may flatten this object, but the logical contract stays the same.

### 1.8 Serialization target fields

The storage adapter must preserve at least the following fields even if the on-disk format is flattened.

- `format_version`
- `bundle_type`
- `topology_group_id`
- `topology_hash`
- `source_file_type`
- `fidelity_type`
- `step_semantics`
- `coupling_policy`
- `origin_xy`
- `period_angle_deg`
- `anti_periodic`
- `master_node_index`
- `slave_node_index`
- `pbc_edge_index`
- `pbc_edge_attr`
- `chain_ids`
- `family_ids`
- `pairing_error_code`
- `validation_summary`

### 1.9 Result wrapping rule

Expected preprocessing failures must be representable without destroying composition.

```python
BoundaryChainSetResult = tuple[BoundaryChainSet | None, str | None]
PeriodicPairSetResult = tuple[PeriodicPairSet | None, str | None]
PBCBundleResult = tuple[PBCBundle | None, str | None]
```

Example error code categories:
- `E-MESH-BOUNDARY-001`: external boundary extraction failed
- `E-MESH-BOUNDARY-002`: non-manifold boundary ratio too high
- `E-PBC-PRIOR-001`: no valid global prior could be inferred
- `E-PBC-PAIR-001`: chain family candidate pair not found
- `E-PBC-PAIR-002`: KDTree node matching below threshold
- `E-PBC-BUNDLE-001`: serialization field validation failed

---

## 2. Periodic Pair Decision Algorithm Pseudocode

### Design rule
- `global prior` decides expected periodicity and sign candidates.
- `local matcher` confirms actual boundary correspondence.
- Neither stage should silently absorb the responsibility of the other.

## 2.1 Stage map

```text
CanonicalMeshObservation(grouped by topology)
    -> extract_boundary_chains
    -> infer_global_prior
    -> propose_chain_family_candidates
    -> confirm_chain_pairs_via_rotation
    -> confirm_node_pairs_via_kdtree
    -> materialize_pbc_edges
    -> build_pbc_bundle
```

## 2.2 Topology grouping

```python
def group_observations_by_topology(observations):
    groups = {}
    for obs in observations:
        groups.setdefault(obs.topology_hash, []).append(obs)
    return groups
```

#### Rule
- All expensive boundary extraction is done once per topology group.
- Step-varying positions may be sampled for diagnostics, but topology-scoped contracts stay invariant.

## 2.3 Boundary extraction pseudocode

```python
def extract_boundary_chains(obs: CanonicalMeshObservation) -> BoundaryChainSetResult:
    undirected_edges = triangles_to_undirected_edges(obs.triangles)
    edge_use_count = count_edge_usage(undirected_edges)

    external_edges = [edge for edge in undirected_edges if edge_use_count[edge] == 1]
    interface_edges = find_region_interface_edges(obs.triangles, obs.reg_code)

    if not external_edges:
        return None, "E-MESH-BOUNDARY-001"

    boundary_graph = build_boundary_graph(external_edges)
    origin_xy = estimate_rotation_origin(obs.pos_xy, external_edges)
    chain_paths = split_graph_into_ordered_chains(boundary_graph)

    chains = []
    for path in chain_paths:
        descriptor = compute_chain_descriptor(path, obs.pos_xy, origin_xy, obs.triangles, obs.reg_code)
        chains.append(descriptor)

    family_summary = cluster_chain_families(chains)
    non_manifold_nodes = detect_non_manifold_nodes(boundary_graph)

    return BoundaryChainSet(...), None
```

#### Local rules
- Boundary edges are defined topologically, not geometrically.
- Chain splitting must treat branch points as boundaries between chains.
- `family_summary` should record which families look like radial sector-cut candidates.

## 2.4 Global prior inference pseudocode

```python
def infer_global_prior(chain_set: BoundaryChainSet, obs_group: Sequence[CanonicalMeshObservation]) -> PriorContext:
    cad_prior = try_read_machine_level_metadata(obs_group)
    if cad_prior is valid:
        return cad_prior

    mesh_slot_count = estimate_slot_count_from_stator_boundary(chain_set, obs_group)
    mesh_pole_count = estimate_pole_count_from_rotor_boundary(chain_set, obs_group)
    aux_periodicity = estimate_auxiliary_periodicity(chain_set)

    n_geom = reduce_gcd(mesh_slot_count, mesh_pole_count, aux_periodicity)
    anti_periodic = infer_anti_periodic_sign(mesh_slot_count, mesh_pole_count, obs_group)

    return PriorContext(...)
```

#### Rule
- Reuse the `gcd`-style periodicity logic conceptually.
- Do not import `eMach` CAD entity analyzers into this stage.
- Accept `unknown` prior when insufficient information exists.

## 2.5 Candidate family proposal pseudocode

```python
def propose_chain_family_candidates(chain_set: BoundaryChainSet, prior: PriorContext):
    radial_families = [f for f in chain_set.family_summary if f.is_radial_candidate]

    if prior.n_geom_prior is None:
        expected_angles = infer_angles_from_family_distribution(radial_families)
    else:
        expected_angles = [360.0 / prior.n_geom_prior]

    proposals = []
    for left_family in radial_families:
        for right_family in radial_families:
            if left_family.id == right_family.id:
                continue
            for angle in expected_angles:
                if family_descriptors_are_compatible(left_family, right_family, angle):
                    proposals.append((left_family.id, right_family.id, angle))

    return rank_family_proposals(proposals)
```

#### Rule
- This stage uses descriptors only.
- It must not perform final node-level acceptance.

## 2.6 Chain-level confirmation pseudocode

```python
def confirm_chain_pair(master_chain, slave_chain, angle_deg, pos_xy, origin_xy, sign):
    rotated_slave = rotate_chain(slave_chain.node_index, pos_xy, origin_xy, angle_deg)
    score = compare_chain_profiles(master_chain, slave_chain, rotated_slave)
    if score < CHAIN_SCORE_THRESHOLD:
        return None

    return PeriodicChainPair(
        ...,
        expected_rotation_deg=angle_deg,
        applied_rotation_deg=angle_deg,
        sign=sign,
        ...,
    )
```

#### Comparison features
- normalized radial profile overlap
- normalized cumulative arc-length profile
- endpoint tangent compatibility
- adjacent region histogram compatibility
- optional step-sampled stability check for moving-node cases

## 2.7 Node-level KDTree confirmation pseudocode

```python
def confirm_node_pairs(chain_pair, master_chain, slave_chain, pos_xy, origin_xy, local_scale):
    master_nodes = pos_xy[master_chain.node_index]
    slave_nodes = pos_xy[slave_chain.node_index]

    atol = LOCAL_EDGE_SCALE_MULT * local_scale
    result = match_periodic_boundary(
        master_nodes=master_nodes,
        slave_nodes=slave_nodes,
        angle_deg=chain_pair.applied_rotation_deg,
        atol=atol,
        rtol=RTOL,
    )

    if result.matched_slave.size / slave_nodes.shape[0] < NODE_MATCH_RATIO_THRESHOLD:
        return None, "E-PBC-PAIR-002"

    return result, None
```

#### Rule
- `match_periodic_boundary` remains a local matcher, not a boundary detector.
- Tolerances must scale with mesh resolution, not rely on a global hard-coded constant only.

## 2.8 Materialization pseudocode

```python
def materialize_pbc_edges(master_global_idx, slave_global_idx, sign):
    edge_index = build_bidirectional_edges(master_global_idx, slave_global_idx)
    edge_attr = np.full((edge_index.shape[1], 1), sign, dtype=np.float32)
    return edge_index, edge_attr
```

## 2.9 Main orchestration pseudocode

```python
def build_periodic_pair_set(obs_group):
    reference_obs = pick_reference_observation(obs_group)
    chain_set, err = extract_boundary_chains(reference_obs)
    if err:
        return None, err

    prior = infer_global_prior(chain_set, obs_group)
    proposals = propose_chain_family_candidates(chain_set, prior)
    if not proposals:
        return None, "E-PBC-PAIR-001"

    accepted_chain_pairs = []
    accepted_master = []
    accepted_slave = []

    for proposal in proposals:
        pair = confirm_chain_pair(...)
        if pair is None:
            continue

        kd_result, kd_err = confirm_node_pairs(...)
        if kd_err:
            continue

        accepted_chain_pairs.append(update_pair_with_kd_stats(pair, kd_result))
        accepted_master.append(...)
        accepted_slave.append(...)

    if not accepted_chain_pairs:
        return None, "E-PBC-PAIR-002"

    edge_index, edge_attr = materialize_pbc_edges(...)
    return PeriodicPairSet(...), None
```

## 2.10 Decision thresholds

Recommended initial thresholds:
- chain score threshold: `>= 0.90`
- node match ratio threshold: `>= 0.99`
- max match distance: `<= 0.25 * median_boundary_edge_length`
- duplicate node collision count: `0`
- unresolved non-manifold ratio: `< 0.01`

These values should be stored as explicit config, not hidden constants.

## 2.11 Harness plan for the algorithm

- contract harness
  - verify chain output determinism on one fixed topology group
  - verify bidirectional edge materialization shape `[2, E]` and attr shape `[E, 1]`
- overfit harness
  - verify a tiny precomputed PBC bundle can be consumed by `phase1_static.train`
- smoke harness
  - run one deterministic DOE topology group through extraction and pairing
- regression harness
  - compare accepted pair counts and match-distance stats to a frozen baseline JSON

---

## 3. phase1_static Module Layout and File Skeleton Proposal

### Design rule
- Keep parsing, preprocessing, and storage concerns separated.
- Place mesh-native PBC preprocessing in `phase1_static`, because its output contract is consumed there.
- Avoid leaking CAD-specific abstractions into this package.

## 3.1 Proposed files

```text
phase1_static/
  contracts.py                     # existing, remains canonical training contract
  data_preprocessing.py            # existing, keep KDTree matcher + edge materialization helpers
  pbc_contracts.py                 # new immutable dataclasses for BoundaryChainSet / PeriodicPairSet / PBCBundle
  pbc_boundary.py                  # new pure mesh boundary extraction morphisms
  pbc_prior.py                     # new global prior inference morphisms
  pbc_pairing.py                   # new local pair confirmation morphisms
  pbc_bundle.py                    # new bundle builder + validation morphisms
  pbc_io.py                        # new storage adapter for .pt/.npz bundle serialization
  pbc_cli.py                       # optional thin CLI for reproducible preprocessing runs
```

## 3.2 Responsibility split

### `pbc_contracts.py`
- defines immutable value objects only
- no file IO
- no KDTree dependency

### `pbc_boundary.py`
- converts canonical mesh arrays into `BoundaryChainSet`
- owns boundary graph reconstruction and chain descriptors
- pure transforms only

### `pbc_prior.py`
- derives `PriorContext`
- may read explicit machine metadata if already present in canonical attrs
- must not import `eMach` CAD analyzers

### `pbc_pairing.py`
- selects chain-family proposals
- invokes local rotation and KDTree matching
- builds `PeriodicPairSet`

### `pbc_bundle.py`
- combines canonical observations and pairing outputs
- validates that solver-domain metadata is preserved
- returns `PBCBundle`

### `pbc_io.py`
- adapter layer only
- serializes and deserializes `PBCBundle`
- keeps format-version handling isolated from domain logic

### `pbc_cli.py`
- thin orchestration only
- suitable command shape:
  - `python -m phase1_static.pbc_cli build --data-dir doe_data --case-idx 0 --out out/case0_pbc_bundle.pt`

## 3.3 File skeleton drafts

### `phase1_static/pbc_contracts.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np


@dataclass(frozen=True)
class BoundaryChain:
    ...


@dataclass(frozen=True)
class BoundaryChainSet:
    ...


@dataclass(frozen=True)
class PriorContext:
    ...


@dataclass(frozen=True)
class PeriodicChainPair:
    ...


@dataclass(frozen=True)
class PeriodicPairSet:
    ...


@dataclass(frozen=True)
class PBCBundle:
    ...
```

### `phase1_static/pbc_boundary.py`

```python
from __future__ import annotations

from phase1_static.pbc_contracts import BoundaryChainSet


def extract_boundary_chains_from_mesh(...):
    ...


def triangles_to_boundary_edges(...):
    ...


def split_boundary_graph_into_chains(...):
    ...


def compute_chain_descriptor(...):
    ...
```

### `phase1_static/pbc_prior.py`

```python
from __future__ import annotations

from phase1_static.pbc_contracts import BoundaryChainSet, PriorContext


def infer_global_prior(...):
    ...


def estimate_slot_count_from_mesh(...):
    ...


def estimate_pole_count_from_mesh(...):
    ...


def infer_sign_from_prior(...):
    ...
```

### `phase1_static/pbc_pairing.py`

```python
from __future__ import annotations

from phase1_static.data_preprocessing import match_periodic_boundary
from phase1_static.pbc_contracts import BoundaryChainSet, PeriodicPairSet, PriorContext


def build_periodic_pair_set(...):
    ...


def propose_chain_family_candidates(...):
    ...


def confirm_chain_pair(...):
    ...


def materialize_pbc_edges(...):
    ...
```

### `phase1_static/pbc_bundle.py`

```python
from __future__ import annotations

from phase1_static.pbc_contracts import PBCBundle


def build_pbc_bundle(...):
    ...


def validate_pbc_bundle(...):
    ...


def summarize_pbc_bundle(...):
    ...
```

### `phase1_static/pbc_io.py`

```python
from __future__ import annotations

from pathlib import Path
from phase1_static.pbc_contracts import PBCBundle


def save_pbc_bundle_pt(bundle: PBCBundle, path: Path):
    ...


def load_pbc_bundle_pt(path: Path):
    ...


def save_pbc_bundle_npz(bundle: PBCBundle, path: Path):
    ...
```

## 3.4 Integration points with existing code

### Existing code to preserve
- `phase1_static/data_preprocessing.py`
  - keep `match_periodic_boundary`
  - keep `build_pbc_edges`
  - keep `combine_edges`
- `phase1_static/contracts.py`
  - keep the training boundary contract canonical
- `phase1_static/motor_dataset.py`
  - extend input acceptance from raw DOE sample dict to bundle-backed sample dict

### Planned dataset integration

Recommended loader path:

```text
raw DOE manifest
    -> build CanonicalMeshObservation list
    -> optionally preprocess to PBCBundle per topology group
    -> expand bundle-backed sample dicts
    -> StaticMotorDataset
```

#### Compatibility rule
- `motor_dataset.py` should continue to accept pre-materialized `pbc_edge_index` and `pbc_edge_attr`.
- Bundle-backed integration should only enrich how those fields are produced, not change how the dataset consumes them.

## 3.5 Storage recommendation

### Canonical storage
- prefer `.pt` for canonical bundle storage because ragged chain and diagnostics structures are easier to preserve

### Compatibility export
- allow `.npz` only as a flattened adapter export for interoperability or quick debugging

### Versioning
- `format_version` is mandatory
- breaking changes must bump `format_version`
- loader must fail explicitly if version is unsupported

## 3.6 Rejected alternatives

### Rejected: direct `eMach` runtime dependency in `phase1_static`
- violates ADR parsing boundary intent
- couples mesh preprocessing to CAD entity abstraction
- increases risk of hidden drift between DOE mesh and CAD semantics

### Rejected: storing only `pbc_edge_index` and `pbc_edge_attr`
- too little provenance for debugging
- hides why a pair was accepted
- makes regression testing brittle

### Rejected: GUI-first labeling workflow
- not scalable for DOE-wide preprocessing
- can remain an optional override path later

---

## Suggested Execution Order

1. create `pbc_contracts.py` and freeze value object names and required fields
2. implement `pbc_boundary.py` with deterministic chain extraction on one topology group
3. implement `pbc_prior.py` using mesh-derived rules first and optional metadata injection second
4. implement `pbc_pairing.py` on top of the existing KDTree matcher
5. implement `pbc_bundle.py` and `pbc_io.py`
6. integrate bundle-backed samples into `motor_dataset.py` without changing downstream graph consumption
7. add contract, smoke, and regression harnesses before broader rollout

## Immediate Next Slice

The next concrete implementation slice should be:
- add `phase1_static/pbc_contracts.py`
- add `phase1_static/pbc_boundary.py`
- add a narrow test fixture using one fixed topology group from DOE data

That slice is small enough to validate object contracts and boundary extraction determinism before we touch pairing or storage.