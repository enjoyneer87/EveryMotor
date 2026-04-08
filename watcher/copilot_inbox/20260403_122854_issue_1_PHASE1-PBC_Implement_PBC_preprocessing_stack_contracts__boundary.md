# Copilot Issue Task

Repository: `enjoyneer87/EveryMotor`
Issue: `#1` (open)
Issue URL: https://github.com/enjoyneer87/EveryMotor/issues/1
Updated At: 2026-04-03T12:28:31Z
Title: [PHASE1-PBC] Implement PBC preprocessing stack: contracts → boundary → prior → pairing → bundle

## Original Issue Body
## Branch
`codex/phase-1-static-1-8-model`

## Context
Phase 1 static 1/8 motor PBC preprocessing pipeline을 `phase1_static/data_preprocessing.py` 위에 구축합니다.
(기존 파일에 `match_periodic_boundary`, `rotate_points`, `PBCMatchResult` 이미 존재)

**Design rules** (`.github/instructions/design-principles.instructions.md` 준수):
- Data objects = `frozen=True` immutable dataclass, numpy 배열은 `__post_init__`에서 read-only 설정
- Transformations = pure function (side effect 없음, global state 쓰기 없음)
- Error return style = `tuple[T | None, str | None]` — API 경계에서 bare exception 금지

---

## Tasks (순서대로 구현)

### 1. `phase1_static/pbc_contracts.py`
불변 값 객체 레이어.

필수 dataclass:
- `CanonicalMeshObservation(nodes_xy, triangles, region_code, field_dict, sector_id)`
- `BoundaryChain(node_indices, chain_type, endpoint_pair)` — `chain_type` ∈ `{"arc","radial","mixed_polyline","unresolved"}`
- `BoundaryChainSet(chains, rotation_origin_xy, mesh_hash)`
- `PriorContext(n_geom_prior, anti_periodic_prior, evidence_notes)`
- `PeriodicChainPair(master_chain_idx, slave_chain_idx, rotation_deg, match_ratio)` — `0.0 ≤ match_ratio ≤ 1.0` 검증
- `PeriodicPairSet(pairs, pbc_edge_index, pbc_edge_attr)` — `pbc_edge_index` shape `[2, 2P]`, `pbc_edge_attr` shape `[2P, 1]`
- `PBCBundle(observation, chain_set, prior, pair_set, format_version, bundle_type)`
- `FrozenMapping` (dict 래퍼)

TypeAliases:
```python
BoundaryChainSetResult = tuple[BoundaryChainSet | None, str | None]
PeriodicPairSetResult  = tuple[PeriodicPairSet | None, str | None]
PBCBundleResult        = tuple[PBCBundle | None, str | None]
```
Constants:
```python
PBC_BUNDLE_FORMAT_VERSION = "0.1.0"
PBC_BUNDLE_TYPE = "phase1_pbc_bundle"
```
### 2. phase1_static/pbc_boundary.py
메시 네이티브 경계 체인 추출.

Public API:
```python
def extract_boundary_chains(observation: CanonicalMeshObservation) -> BoundaryChainSetResult: ...
def extract_boundary_chains_from_mesh(nodes_xy, triangles, region_code) -> BoundaryChainSetResult: ...
```

구현 단계 (순서 중요):
1. `triangles_to_undirected_edges` — 삼각형의 모든 엣지 열거
2. `find_external_boundary_edges` — 정확히 1회 등장하는 엣지
3. `find_region_interface_edges` — 서로 다른 region_code 사이 엣지
4. `build_boundary_graph` — 인접 dict 구성
5. `split_boundary_graph_into_chains` — **코너 턴 ≥ 45°에서 먼저 분할** (origin 추정 전)
6. `estimate_rotation_origin` — arc-like 체인에 circle fit (경계 centroid 사용 금지)
7. `compute_chain_descriptor` — 추정된 origin 기준 극좌표로 분류
   - `arc`: 반지름 일정 (rtol 5%), 각도 변함
   - `radial`: 각도 스팬 ≤ 5°, 반지름 변함
8. `cluster_chain_families` — 타입별 체인 그룹화

Error code: `"E-MESH-BOUNDARY-001"` (빈 경계 시)

> **⚠️ Critical design note**: 체인 분할을 **반드시 먼저** 수행 후 origin 추정.
> 경계 centroid를 rotation origin으로 직접 사용하면 모든 체인이 `mixed_polyline`으로 분류되는 버그 발생.
> `_line_fit_residual > 1e-3`인 비선형 체인들에서만 circle fit → 평균하여 origin 추정.
> 이 순서는 실제 디버깅을 통해 검증된 설계임.

---

### 3. `phase1_static/pbc_prior.py`
글로벌 기하 사전 추론.

```python
def infer_global_prior(chain_set: BoundaryChainSet) -> PriorContext: ...
```

규칙:
- `radial` 타입 체인 패밀리 수 세어 `n_geom_prior` 추정 (예: radial 패밀리 2개 → 8-fold 대칭)
- `anti_periodic_prior = True` if n_geom 짝수이고 field antisymmetry 예상
- eMach 또는 CAD analyzer import 금지

---

### 4. `phase1_static/pbc_pairing.py`
체인 패밀리 제안 + KDTree 매칭.

```python
def build_periodic_pair_set(
    chain_set: BoundaryChainSet,
    prior: PriorContext,
    nodes_xy: np.ndarray,
) -> PeriodicPairSetResult: ...
```

- `expected_rotation_deg = 360.0 / prior.n_geom_prior`
- data_preprocessing.py의 기존 `match_periodic_boundary` 재사용
- `pbc_edge_index` shape `[2, 2P]`, `pbc_edge_attr` shape `[2P, 1]` (주기: `+1.0`, 반주기: `-1.0`)
- Error code: `"E-PBC-PAIR-001"` (매칭 쌍 없을 때)

---

### 5. `phase1_static/pbc_bundle.py`
번들 빌더.

```python
def build_pbc_bundle(observation: CanonicalMeshObservation) -> PBCBundleResult: ...
```

2→3→4 단계를 순서대로 조합하여 `PBCBundle` 반환.

---

### 6. `tests/test_phase1_pbc_contracts.py`
5개 테스트:
- frozen 배열 변경 시 ValueError 발생
- `BoundaryChain` 잘못된 endpoint shape 거부
- `PeriodicChainPair` match_ratio 범위 [0,1] 외 거부
- `PeriodicPairSet` 양방향 엣지 검증
- `PBCBundle` 중첩 contracts 수용

---

### 7. `tests/test_phase1_pbc_boundary.py`
synthetic 6-node annular sector fixture 사용:
- 노드: inner ring r=1 (0°/22.5°/45°), outer ring r=2 (동일 각도)
- 4개 삼각형, `region_code = [1,1,2,2]`
- 예상 체인: `((0,1,2), (2,5), (5,4,3), (3,0))` → 타입: `[arc, radial, arc, radial]`
- 결정론성 테스트, region interface edge 테스트, 빈 경계 에러 테스트 (3개)

---

### 8. `tests/test_phase1_pbc_prior.py` + `tests/test_phase1_pbc_pairing.py`
prior inference 및 pairing 단계 단위 테스트.

---

## Validation
```bash
PYTHONPATH=. pytest tests/test_phase1_pbc_contracts.py tests/test_phase1_pbc_boundary.py tests/test_phase1_pbc_prior.py tests/test_phase1_pbc_pairing.py -v
```
모든 테스트 통과 후 완료 처리.

## Commit format
```
[PHASE1-PBC] implement pbc_contracts, pbc_boundary, pbc_prior, pbc_pairing, pbc_bundle
```
```
```



## Paste This Into VS Code Copilot Chat
```text
You are working in the local repository enjoyneer87/EveryMotor.
Please implement GitHub issue #1: "[PHASE1-PBC] Implement PBC preprocessing stack: contracts → boundary → prior → pairing → bundle".

Constraints:
1. Read AGENTS.md and follow repository instructions.
2. Make minimal, safe code changes needed for this issue.
3. Run relevant tests/checks.
4. Summarize changed files and validation results.

Issue body:
## Branch
`codex/phase-1-static-1-8-model`

## Context
Phase 1 static 1/8 motor PBC preprocessing pipeline을 `phase1_static/data_preprocessing.py` 위에 구축합니다.
(기존 파일에 `match_periodic_boundary`, `rotate_points`, `PBCMatchResult` 이미 존재)

**Design rules** (`.github/instructions/design-principles.instructions.md` 준수):
- Data objects = `frozen=True` immutable dataclass, numpy 배열은 `__post_init__`에서 read-only 설정
- Transformations = pure function (side effect 없음, global state 쓰기 없음)
- Error return style = `tuple[T | None, str | None]` — API 경계에서 bare exception 금지

---

## Tasks (순서대로 구현)

### 1. `phase1_static/pbc_contracts.py`
불변 값 객체 레이어.

필수 dataclass:
- `CanonicalMeshObservation(nodes_xy, triangles, region_code, field_dict, sector_id)`
- `BoundaryChain(node_indices, chain_type, endpoint_pair)` — `chain_type` ∈ `{"arc","radial","mixed_polyline","unresolved"}`
- `BoundaryChainSet(chains, rotation_origin_xy, mesh_hash)`
- `PriorContext(n_geom_prior, anti_periodic_prior, evidence_notes)`
- `PeriodicChainPair(master_chain_idx, slave_chain_idx, rotation_deg, match_ratio)` — `0.0 ≤ match_ratio ≤ 1.0` 검증
- `PeriodicPairSet(pairs, pbc_edge_index, pbc_edge_attr)` — `pbc_edge_index` shape `[2, 2P]`, `pbc_edge_attr` shape `[2P, 1]`
- `PBCBundle(observation, chain_set, prior, pair_set, format_version, bundle_type)`
- `FrozenMapping` (dict 래퍼)

TypeAliases:
```python
BoundaryChainSetResult = tuple[BoundaryChainSet | None, str | None]
PeriodicPairSetResult  = tuple[PeriodicPairSet | None, str | None]
PBCBundleResult        = tuple[PBCBundle | None, str | None]
```
Constants:
```python
PBC_BUNDLE_FORMAT_VERSION = "0.1.0"
PBC_BUNDLE_TYPE = "phase1_pbc_bundle"
```
### 2. phase1_static/pbc_boundary.py
메시 네이티브 경계 체인 추출.

Public API:
```python
def extract_boundary_chains(observation: CanonicalMeshObservation) -> BoundaryChainSetResult: ...
def extract_boundary_chains_from_mesh(nodes_xy, triangles, region_code) -> BoundaryChainSetResult: ...
```

구현 단계 (순서 중요):
1. `triangles_to_undirected_edges` — 삼각형의 모든 엣지 열거
2. `find_external_boundary_edges` — 정확히 1회 등장하는 엣지
3. `find_region_interface_edges` — 서로 다른 region_code 사이 엣지
4. `build_boundary_graph` — 인접 dict 구성
5. `split_boundary_graph_into_chains` — **코너 턴 ≥ 45°에서 먼저 분할** (origin 추정 전)
6. `estimate_rotation_origin` — arc-like 체인에 circle fit (경계 centroid 사용 금지)
7. `compute_chain_descriptor` — 추정된 origin 기준 극좌표로 분류
   - `arc`: 반지름 일정 (rtol 5%), 각도 변함
   - `radial`: 각도 스팬 ≤ 5°, 반지름 변함
8. `cluster_chain_families` — 타입별 체인 그룹화

Error code: `"E-MESH-BOUNDARY-001"` (빈 경계 시)

> **⚠️ Critical design note**: 체인 분할을 **반드시 먼저** 수행 후 origin 추정.
> 경계 centroid를 rotation origin으로 직접 사용하면 모든 체인이 `mixed_polyline`으로 분류되는 버그 발생.
> `_line_fit_residual > 1e-3`인 비선형 체인들에서만 circle fit → 평균하여 origin 추정.
> 이 순서는 실제 디버깅을 통해 검증된 설계임.

---

### 3. `phase1_static/pbc_prior.py`
글로벌 기하 사전 추론.

```python
def infer_global_prior(chain_set: BoundaryChainSet) -> PriorContext: ...
```

규칙:
- `radial` 타입 체인 패밀리 수 세어 `n_geom_prior` 추정 (예: radial 패밀리 2개 → 8-fold 대칭)
- `anti_periodic_prior = True` if n_geom 짝수이고 field antisymmetry 예상
- eMach 또는 CAD analyzer import 금지

---

### 4. `phase1_static/pbc_pairing.py`
체인 패밀리 제안 + KDTree 매칭.

```python
def build_periodic_pair_set(
    chain_set: BoundaryChainSet,
    prior: PriorContext,
    nodes_xy: np.ndarray,
) -> PeriodicPairSetResult: ...
```

- `expected_rotation_deg = 360.0 / prior.n_geom_prior`
- data_preprocessing.py의 기존 `match_periodic_boundary` 재사용
- `pbc_edge_index` shape `[2, 2P]`, `pbc_edge_attr` shape `[2P, 1]` (주기: `+1.0`, 반주기: `-1.0`)
- Error code: `"E-PBC-PAIR-001"` (매칭 쌍 없을 때)

---

### 5. `phase1_static/pbc_bundle.py`
번들 빌더.

```python
def build_pbc_bundle(observation: CanonicalMeshObservation) -> PBCBundleResult: ...
```

2→3→4 단계를 순서대로 조합하여 `PBCBundle` 반환.

---

### 6. `tests/test_phase1_pbc_contracts.py`
5개 테스트:
- frozen 배열 변경 시 ValueError 발생
- `BoundaryChain` 잘못된 endpoint shape 거부
- `PeriodicChainPair` match_ratio 범위 [0,1] 외 거부
- `PeriodicPairSet` 양방향 엣지 검증
- `PBCBundle` 중첩 contracts 수용

---

### 7. `tests/test_phase1_pbc_boundary.py`
synthetic 6-node annular sector fixture 사용:
- 노드: inner ring r=1 (0°/22.5°/45°), outer ring r=2 (동일 각도)
- 4개 삼각형, `region_code = [1,1,2,2]`
- 예상 체인: `((0,1,2), (2,5), (5,4,3), (3,0))` → 타입: `[arc, radial, arc, radial]`
- 결정론성 테스트, region interface edge 테스트, 빈 경계 에러 테스트 (3개)

---

### 8. `tests/test_phase1_pbc_prior.py` + `tests/test_phase1_pbc_pairing.py`
prior inference 및 pairing 단계 단위 테스트.

---

## Validation
```bash
PYTHONPATH=. pytest tests/test_phase1_pbc_contracts.py tests/test_phase1_pbc_boundary.py tests/test_phase1_pbc_prior.py tests/test_phase1_pbc_pairing.py -v
```
모든 테스트 통과 후 완료 처리.

## Commit format
```
[PHASE1-PBC] implement pbc_contracts, pbc_boundary, pbc_prior, pbc_pairing, pbc_bundle
```
```
```


```
