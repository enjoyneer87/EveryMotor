# Motor GNN Roadmap (Phase 0-3)

This document is the shared execution contract for Codex, Claude, GitHub Copilot, and Antigravity.

## Why This Plan Exists
- A prior Gemini prompt outlined a practical roadmap for static-to-dynamic motor GNN development.
- This plan validates that roadmap, applies repository constraints, and defines an implementation-ready sequence.
- Use this file as the single source of truth for multi-agent handoff.

## Prompt Review (Gemini Proposal)
The proposal is largely valid and aligned with this repo.

Accepted as-is:
- KDTree-based master/slave matching for 1/8 periodic boundary mapping
- Anti-periodic message sign handling on boundary edges
- Physics-informed constraint from curl(A) to Bx/By
- Overfit-single sanity test and boundary continuity visualization
- Lambda annealing for physics terms

Required adjustments for this repo:
- Training target contract is fixed to channel order: `[Bx, By, A, J]`
- Primary loss strategy is hybrid: supervised A + supervised B + curl consistency
- Inference/training runtime should be validated in Docker PhysicsNeMo environment
- Keep `eMach/` independent (no `postproc_interop` dependency injection)

## Phase 0 (Required): Contract Freeze Before Coding
### Objective
Freeze shared data and transformation contracts before feature growth.

### Why this is mandatory
- Category-theory-aligned development requires stable objects and pure morphisms first.
- If dynamic logic is added before contracts stabilize, refactor cost grows quickly in Phase 2-3.

### Contract freeze tasks
- Object contract freeze:
  - Define canonical sample schema for training data with explicit channel order `[Bx, By, A, J]`.
  - Treat phase data objects as immutable values after construction.
- Morphism contract freeze:
  - Ensure parsing, graph building, normalization, and loss input mapping are pure transforms.
  - No hidden global writes or in-place mutation of shared upstream inputs.
- Composition contract freeze:
  - Lock the pipeline shape as composable stages: `raw -> canonical sample -> graph -> batch -> loss`.
  - Declare input/output type expectations at each stage.
- Test gate freeze:
  - Add or update tests by layer: object shape, transform correctness, pipeline composition.

### Exit criteria (must pass before Phase 1 coding continues)
- Canonical schema and channel order are written and referenced by training code.
- All critical transforms are side-effect-free at API boundaries.
- Contract-level tests pass for at least one real sample path.

## Phase 1 (Now): Static 1/8 + PBC + Hybrid Loss
### Objective
Build a stable static training baseline that can overfit one sample and satisfy boundary symmetry.

> **"Static" means memoryless, not angle-free.** The model *does* learn the field as a
> function of rotor angle — `rotor_angle_sin/cos` are node inputs and the 45 rotor
> steps span a full electrical period. What makes Phase 1 *static* is that it fits a
> **pointwise map** `(θ, operating point, geometry) → field`: each `(case, step)` is an
> independent, shuffled i.i.d. sample, there is **no `field(t-1)` input** and **no
> rollout**, and the **connectivity is read from the H5 file per step, not generated
> from θ**. Rotor angle enters as a *condition coordinate*, not a time axis — which is
> also why `time_s` (corr +1.0000 with θ; all cases run at one speed) was a redundant
> shortcut and was removed. Angle-dependence is common to Phase 1 and Phase 2 and is
> therefore *not* the static/dynamic axis. The dynamic step (Phase 2) is two specific
> additions: connectivity generated from θ(t) via the sliding band, and `field(t-1)`
> fed as state. Physically the data is a **multi-static rotor-position sweep**
> (`Mag_OnLoadTorque`: an independent magnetostatic solve per position — no eddy
> currents, no time integration, no history), so a memoryless fit is the correct model
> of it.

### Copilot scaffold prompts
Use these comments directly in code files to drive GitHub Copilot generation:

```python
# TODO: Use scipy.spatial.KDTree to match master/slave boundary nodes for 1/8 sector PBC.
# Rotate slave nodes by -45 deg, find nearest master nodes within tolerance, and create bidirectional edge_index.
# Set interior edge_attr=+1.0 and anti-periodic PBC edge_attr=-1.0.
```

```python
# TODO: Build a StaticMotorDataset that returns PyG Data(x, pos, edge_index, edge_attr, y).
# x must include coordinates and physical/context features, with coords autograd-ready for curl loss.
# y channel order must be [Bx, By, A, J].
```

```python
# TODO: Implement hybrid physics loss:
# total = wA*MSE(A_pred,A_gt) + wB*MSE([Bx_pred,By_pred],[Bx_gt,By_gt]) + wCurl*MSE(curl(A_pred),[Bx_gt,By_gt]).
# Use autograd.grad on A_pred wrt coordinates to compute Bx=dA/dy and By=-dA/dx.
```

### Implementation tasks
- Data preprocessing:
  - Identify master/slave boundary node pairs (1/8 sector rotation basis)
  - Build bidirectional PBC edges
  - Assign edge sign flags: interior `+1`, anti-periodic PBC `-1`
- Dataset:
  - Build PyG `Data` objects with node features, edge indices, edge attributes, and targets
  - Ensure coordinate tensors are autograd-compatible for curl loss
- Model:
  - Use anti-periodic message handling (`message * edge_attr`)
  - Preserve compatibility with PhysicsNeMo MeshGraphNet forward signatures
- Loss:
  - `L = wA * L_A + wB * L_B + wCurl * L_curl`
  - `L_A`: MSE(A_pred, A_gt)
  - `L_B`: MSE([Bx_pred, By_pred], [Bx_gt, By_gt])
  - `L_curl`: MSE(curl(A_pred), [Bx_gt, By_gt])
  - Anneal `wCurl` by epoch
- Training checks:
  - Overfit-single loss should drop near `1e-4` target range
  - Boundary continuity and sign behavior must be visually verified

### Verification checklist (mandatory before Phase 2)
- ✅ No isolated nodes after PBC edge insertion
- ✅ `edge_index` shape is valid `[2, E]`
- ✅ PBC sign handling is correctly applied on boundary edges
- ✅ Overfit-single converges without divergence (best_loss=0.000344 ≤ target=0.01)
- ✅ Boundary continuity plots are archived (`logs/pbc_boundary_vis.png` via `visualize_pbc_boundary.py`)
- ✅ PBC test suite: 23 tests passing (pbc_bundle + motor_dataset_pbc)
- ✅ `results/regression_baseline.json` recorded

### NPZ Inference Visualization Policy
- NPZ 추론 결과(`pos_x/pos_y/gt_*/pred_*`)에는 삼각형(connectivity) 정보를 **저장하지 않는다**.
  - 삼각형은 형상 전용이므로 다른 형상 추론 시 무의미
  - NPZ는 경량 추론 기록용으로 유지
- **NPZ 시각화**: `scatter(pos_x, pos_y, c=field)` 사용 (Delaunay 아티팩트 방지)
- **정밀 메시 렌더링**: H5 원본 → `tripcolor(triang, field, shading="gouraud")` 참조
- 셀 33 (H5 reference)은 `tripcolor`, 셀 47 (NPZ GT/Pred)는 `scatter`가 올바른 조합

## Phase 2: Dynamic Sliding-Band Time-Series Pipeline
### Objective
Support time-varying topology/conditions without memory instability.

### Architecture: 3-Module Separation (Geometry / Graph / Field)

Phase 2 이후 아키텍처는 **세 모듈로 명확히 분리**한다.
SB connectivity는 로터 회전각 θ의 결정론적 함수이므로 학습 대상이 아니다.

```
┌──────────────────────────────────────────────────────────────┐
│  [Geometry Encoder]  (학습)                                  │
│    Input:  pos(θ), node_type, region_code                    │
│    Output: node_embed  (형상 latent)                         │
│    - 동일 형상이면 인코딩 1회 → 모든 step에서 재사용        │
│    - 다른 DOE 케이스(다른 슬롯/극수)에도 일반화 가능        │
│                                                              │
│  [Graph Builder]  (비학습, 알고리즘적)                       │
│    Input:  θ(t), stator mesh, rotor mesh                     │
│    Output: edge_index(t) = static_edges + SB_edges(θ)        │
│    - per-step SB connectivity를 결정론적으로 생성            │
│    - stator mesh 고정, rotor mesh 좌표만 회전                │
│    - SB 영역: θ에 따라 stator-rotor 연결 재계산             │
│                                                              │
│  [Field GNN]  (학습)                                         │
│    Input:  node_embed + edge_index(t) + field(t-1)           │
│    Output: field(t) = [Bx, By, A, Je]                        │
│    - connectivity 생성 부담 없이 필드 예측에 집중            │
│    - Phase 3에서 autoregressive rollout 확장                 │
└──────────────────────────────────────────────────────────────┘
```

**분리 근거:**
- SB connectivity = f(θ) 이며 기하학적 결정론이므로 학습 불필요
- NPZ 추론 결과에 삼각형 정보를 저장해도 새 형상에 범용적이지 않음
- 시각화 시 NPZ는 scatter로, 정밀 렌더링은 H5 원본 connectivity 참조
- Phase 4+ (미래)에서 완전히 새로운 형상(다른 모터 토폴로지)에 대해
  connectivity까지 예측하려면 별도 Connectivity Predictor Network 필요

**Phase별 connectivity 전략:**

| Phase | Connectivity 출처 | 접근 |
|-------|-------------------|------|
| Phase 1 (Static) | H5에서 per-step 고정 merge | 현재 구현 (doe_data_utils.py) |
| Phase 2 (Dynamic) | θ(t) → Graph Builder 알고리즘 생성 | 모듈 분리 |
| Phase 3 (Rollout) | 이전 step θ → 다음 θ 예측 → Graph Builder | autoregressive 확장 |
| Phase 4+ (미래) | 새 형상 일반화 | Connectivity Predictor (학습) |

### Contract freeze (mandatory — see phase2_dynamic_edge_pipeline.md)
- ✅ TemporalMotorSample frozen dataclass with shape invariants
- ✅ NormStats + pure normalize/denormalize roundtrip
- ✅ build_sliding_band_edges pure function (KDTree, bidirectional, dedup)
- ✅ collate_temporal_batch numpy stacker with sequence_len guard
- ✅ Contract tests: 11 passed (test_phase2_contracts.py)
- ✅ Memory profile: collation loop no-accumulation + no-aliasing (test_phase2_memory_profile.py)

### Implementation tasks
- Implement Graph Builder module: θ(t) → edge_index(t) (deterministic, non-learned)
- Implement Geometry Encoder: pos + node_type → node_embed (learned, per-shape cached)
- Refactor Field GNN to consume (node_embed, edge_index(t), field(t-1))
- Build dynamic edge refresh for gap/sliding regions per timestep
- Reuse static region topology to reduce overhead
- Build time-aware dataloader returning `(x_t, edge_t, y_t)`
- Apply normalization/standardization policy for mixed-scale physical features

### Verification checklist
- ✅ No memory leak under 100-batch collation loop (host numpy path)
- [ ] No memory leak under long dataloader iteration (GPU path — Docker)
- [ ] Stable GPU/CPU usage during multi-step epoch execution
- [ ] Interpolated rotor angles do not produce boundary discontinuities

## Phase 3: Autoregressive Rollout Stability
### Objective
Enable robust long-horizon rollout with controlled error accumulation.

### Implementation tasks
- Transition from absolute prediction to residual prediction (`delta A_t` or equivalent)
- Introduce teacher forcing schedule
- Add noise injection for rollout robustness
- Add multi-step training loss (2-5+ rollout steps)

### Verification checklist
- 10/50/100-step rollout error trends are quantified
- No exponential blow-up in rollout MSE
- Torque/energy trend consistency against FEM baseline is tracked
- Inference speed-up ratio versus FEM is measured and logged

## Agent Handoff Protocol
All agents should follow these rules when continuing work:
- Read `AGENTS.md` first
- Read `.github/instructions/design-principles.instructions.md` before editing `.py`
- Read `.github/instructions/phase-dev-context-harness.instructions.md` before running phase tasks
- Read this roadmap before editing phase code
- Keep commit scope focused to one subtask
- Keep commit message format: `[TASKKEY] short action summary`
- Do not rewrite history and do not revert unrelated local changes

## Context + Harness Engineering (Operational)
Use these techniques to reduce drift across Codex, Claude, Copilot, and Antigravity.

### Context engineering techniques
- Context pack per task:
  - Goal, invariants, interface contracts, non-goals, and acceptance checks in one short block.
- Retrieval-first handoff:
  - Always link the exact plan/instruction files instead of paraphrasing from memory.
- Constraint pinning:
  - Keep immutable constraints in explicit bullets (channel order, purity rules, environment).
- Decision log:
  - Record key design decisions and rejected alternatives in commit messages or task notes.

### Harness engineering techniques
- Deterministic run harness:
  - Fixed random seed, explicit data slice, explicit epoch count for reproducible smoke tests.
- Contract test harness:
  - Fast checks validating shape/type/order at stage boundaries before long runs.
- Overfit harness:
  - Single-batch or single-graph overfit command as the first gate for every training change.
- Regression harness:
  - Keep a minimal baseline metrics snapshot and fail when divergence exceeds thresholds.
- Runtime harness:
  - Validate in Docker PhysicsNeMo first for environment consistency.

## Suggested Next Task Slice
Complete Phase 1 to "verification complete":
- Complete Phase 0 contract freeze gates for phase pipeline
- Stabilize dataset+model+loss integration
- Pass overfit-single target
- Produce boundary continuity artifact
- Record run config and metrics in commit notes
