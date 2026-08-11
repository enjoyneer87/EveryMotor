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

## Phase 1 (마감 실험 중): Static 1/8 + PBC + Hybrid Loss
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

## Phase 1 실제 결과 — Phase 2가 상속하는 불변식 (2026-08-12 갱신)

> 이 문서의 Phase 2/3 절은 2026-07-22에 마지막으로 쓰였고, 그 뒤 §24~§31의
> 방법론 아크(여자배선 결함, 게이트 재정의, 전류축, 물리 prior, 공극 바닥)가
> 전부 지나갔다. Phase 1은 계획보다 훨씬 긴 서사가 됐고, 아래는 Phase 2를
> 시작하기 전에 **반드시 승계해야 하는 것들**이다. 근거는
> `.github/plans/methodology_review_20260720.md`의 해당 절.

| 승계 항목 | 내용 | 근거 |
|---|---|---|
| **게이트** | G1 폐기(표현 바닥 아래). 현행은 **G1' = 공극 \|B\| < 5% AND 토크 < 3%** | §25 |
| **노드 계약** | V2(9) → **V3(12)**: prior_a/bx/by가 뒤 3열. 열 순서 강제, `wrap_rotor=False` 필수 | §27 |
| **여자 배선** | `CurrentDefinition=0` per-case 주입 + **A-turns 게이트 상설**. 생성 배치마다 전수 | §24/§26 |
| **회전 스텝** | 45점은 order 18 아래에서 스펙트럼 청정. 24f는 21f로 앨리어스 → **리플 지표에 각주** | §28 |
| **집계** | 게이트·챔피언 비교는 **풀드**, 운전점당 주장은 케이스별(근제로 토크 케이스 명시 제외) | §29 |
| **공극 지표의 실체** | 게이트가 채점하는 공극은 **a1 한 층뿐**(a2/a3/a4는 슬라이딩밴드 마스크에서 제외). a1의 거칢은 고정자 슬로팅이고, 48슬롯의 48/96/144는 허용차수(4의 홀수배) 집합에 **없다** | §31d |
| **토크 연산자** | Arkkio(환형평균 MST)와 Coulomb 국소 가상일이 동일 필드에서 **0.475% 일치(산포 0.024pp)**. eMach `tools/torque_operators` | §31c |

### Phase 2에 직접 영향을 주는 Phase 1 교훈 셋

1. **전역 유한차분 가상일은 동기 스윕에서 무효**다. `dW'/dθ`는 전류 고정 미분인데
   온로드 동기 스윕은 전류파가 로터와 함께 진상해 기계·전기 기여가 상쇄된다
   (실측 O(10) N·m/m, 참값 O(2500)). **Phase 2에서 전류를 고정한 스텝이
   생긴다면 그때는 유효해진다** — 트랜지언트 설계 시 의도적으로 확보할 가치가 있다.
2. **생성 프로세스 단일성**이 §13~§30 비교가능성의 근거다. rev-1/rev-2 때 480케이스를
   통째로 재생성한 이유가 이것이다. **트랜지언트 데이터는 새 생성 프로세스이므로
   Phase 1의 480과 섞어 스케일링 주장을 만들 수 없다** — 별도 캠페인 ID/split로 간다.
3. **비적합 이동/정지 계면**이 필드 매핑 오차의 자리다(§31c). Phase 2는 이 계면을
   매 스텝 다루므로, Phase 1에서 "제외해서 회피"했던 문제를 정면으로 만난다.

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

### 2026-08-12 추가 과제 (Phase 1 결과 반영)

- **Geometry Encoder 캐시와 prior를 분리할 것.** 형상 latent는 케이스당 1회
  재사용이 맞지만, **prior 3채널은 (케이스, 스텝, 전류)의 함수**다. 로터각과
  여자가 바뀌면 값이 바뀌고, 캐시 키도 h5 실제 경로 기준이다(§27, prior.py).
  형상 캐시에 prior를 딸려 넣으면 전 스텝에 같은 prior를 먹이는 조용한 버그가 된다.
- **트랜지언트 데이터 생성 계약을 Phase 1과 동급으로 세울 것.** A-turns 게이트
  전수, 경로 게이트, `CurrentDefinition=0`, 생성기 rev 기록. Phase 1에서 이 게이트가
  없었다면 19개 형상이 전류 0으로 풀린 것을 못 잡았다.
- **2D/3D 트랜지언트에서 Continuum Air on/off 대조를 설계에 포함할 것.**
  Maxwell 2D 내보내기는 `results/logs/mcad_to_maxwell_export.py`로 준비돼 있고
  (case 4에서 17,838줄, Rotating_Band 포함), 베타 옵션 1회 활성화만 남았다.
  이것이 §31d가 남긴 미결(a1의 잔여 6.95%가 물리인가 메싱인가)을 가르는 유일한 실험이다.
- **전류 고정 스텝을 의도적으로 확보할 것.** 위 교훈 1 참조. 확보되면 전역 가상일이
  독립 토크 연산자로 되살아나고, Coulomb·Arkkio와 3-way 교차검증이 가능해진다.

### Verification checklist
- ✅ No memory leak under 100-batch collation loop (host numpy path)
- [ ] No memory leak under long dataloader iteration (GPU path — Docker)
- [ ] Stable GPU/CPU usage during multi-step epoch execution
- [ ] Interpolated rotor angles do not produce boundary discontinuities
- [ ] **층별 매끄러움 추적**: 매 스텝 a1 vs a3/a4의 조화 잔차를 기록
      (`results/logs/airgap_layer_breakdown.py`). Phase 1 기준선은 a1 9.02% /
      a3·a4 2.2%. 슬라이딩 밴드를 매 스텝 재구성하는 Phase 2에서 이 대비가
      무너지면 계면 처리가 필드를 오염시키고 있다는 신호다.
- [ ] **연산자 교차검증**: 스텝별 토크를 Arkkio와 Coulomb VW 양쪽으로 계산해
      Phase 1의 0.475% 일치가 유지되는지 확인. 벌어지면 필드가 비물리적으로
      드리프트한 것이다.

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

### 2026-08-12 추가 — 롤아웃 진단을 물리량으로 (Phase 1 자산 활용)

- **코에너지 드리프트를 롤아웃 안정성 지표로 쓸 것.** `tools/torque_operators.coenergy`가
  조립기와 동일한 구성식으로 W'를 계산한다(공기 ν\|B\|²/2, 자석 ν(\|B\|²/2 − Br·B),
  강판은 BH 보간의 닫힌형 적분). MSE는 어디서 틀렸는지 말해주지 않지만 코에너지가
  단조 발산하면 롤아웃이 비물리로 가고 있다는 것을 바로 말해준다.
- **토크는 반드시 검증된 연산자로.** 롤아웃 필드에 Arkkio와 Coulomb VW를 둘 다 걸고,
  두 값이 벌어지는 스텝을 발산의 조기 신호로 쓴다. Phase 1 기준 일치도 0.475%.
- **근제로 토크 형상을 지표에서 살릴 것.** 케이스별 % 는 g32 계열에서 분모가
  붕괴한다(§29). 롤아웃 보고는 **절대오차 N·m/m를 % 와 병기**하는 것을 표준으로 한다.
- **리플 기반 롤아웃 지표에는 §28 각주를 유지.** 45점은 order 18 아래에서만 청정하고
  24f는 21f로 접힌다. 장기 롤아웃의 리플 스펙트럼을 주장하려면 검증 서브셋을
  120점으로 재솔브해 두는 편이 안전하다.
- **게이트는 G1'으로 읽되 공극은 a1 한 층임을 명시**할 것(§31d). 롤아웃이 공극을
  얼마나 유지하는지 볼 때 a3/a4까지 뭉뚱그리면 5배 매끄러운 층이 섞여 실제보다
  좋아 보인다.

## 도구·데이터 축 결정 기록 (2026-08-12)

### gmsh — 승인, 도입한다

**목적:** Motor-CAD 라이선스 없이 형상 → 메시를 만들 수 있게 되면 두 가지가 열린다.
1. **선형 prior로 넓은 설계공간을 싸게 채워 사전학습 후 480 실데이터로 미세조정.**
   prior는 스텝당 0.24s이므로 샘플 수가 사실상 제약이 아니다. 지금 막는 것은 오직
   "신규 형상마다 메시가 없다"는 것뿐이다.
2. Phase 2에서 공극 메시를 직접 통제할 수 있다. Motor-CAD의
   `AirgapMesh_NumLayers`는 이 모델에서 **무효**임이 실측됐고(요청과 무관하게
   `AirgapMesh_NumLayers_Used=4`), 반경방향 층수를 건드릴 경로가 현재 없다(§31a).

**출처:** pyleecan에 실제 gmsh 경로가 있다 —
`Functions/GMSH/draw_GMSH.py`, `comp_gmsh_mesh_dict.py`, `gen_3D_mesh.py`,
`Methods/Simulation/MagElmer/gen_elmer_mesh.py`. 로컬 `D:/KDH/gitPyleecan`에 있다.
**SyR-e는 gmsh를 쓰지 않는다**(FEMM 자체 mesher: `draw_motor_in_FEMM.m`,
`dimMesh.m`) — 여기서 가져올 것은 없다.

**주의:** gmsh로 만든 메시는 Motor-CAD 메시와 **다른 생성 프로세스**다. Phase 1의
480과 섞어 스케일링 곡선을 만들 수 없다. 사전학습 → 미세조정 구조로만 쓰고,
평가는 끝까지 Motor-CAD 산출 홀드아웃으로 한다.

### 복소 퍼미언스(Žarko) 입력 feature — 보류

`force-harmonics` 레포(TU Delft, Klop 2022)에 `Zarko.m` / `B_PM_slotless.m` /
`B_arm_slotless.m`이 있으나 도입하지 않는다.

**근거:** ① 그 모델은 **표면부착 PM + 분수슬롯 집중권** 전제이고 포화를 무시한다
(README 명시). 우리는 매입형(IPM) 2층 자석 + 48슬롯 분포권이고 치 \|B\| p95가
2.07 T다. ② 특히 DOE 축이 `PhaseAdvance 0–90°`인데 이는 자석토크↔릴럭턴스토크
배분 축이고, SPM 모델은 돌극성이 없어 **연구 대상 축 자체를 표현하지 못한다.**
③ 현행 선형 FEM prior가 이미 정확한 형상·슬로팅·배리어를 담고 포화만 놓치므로
상위호환이다. λ(θ)를 입력에 얹는 변형은 위험은 낮으나 기대값도 낮다.

**재개 조건:** 위 ①②가 해소되는 IPM 전용 해석 모델(비선형 퍼미언스 네트워크 등)을
확보했을 때. 그때도 "덩어리 자속 → 노드 필드" 재구성이 포맷 문제를 되살린다는 점을
먼저 해결해야 한다.

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

## Suggested Next Task Slice (2026-08-12 갱신)

Phase 1은 "verification complete"를 지났고 현재는 **마감 실험 큐**에 있다.
§30이 순서와 판정 규칙을 실행 전에 박아뒀다.

1. **R9 판정** — 480케이스 학습(진행 중). 레거시 6-test 풀드 공극으로 읽는다:
   < 6.368% → 데이터축 유효, 신규 챔피언 / 6.368~6.868% → 런간 밴드, 기울기 둔화 /
   > 6.868% → 회귀, 학습 예산부터 의심.
2. **E2** — 50 → 75에폭 연장(~20h). 공극 ≥0.25pp 개선이면 50ep 부족으로 판정하고
   이후 비교를 75ep로 재기준. <0.10pp면 예산 적정.
3. **E1** — `--airgap-weight 5` 단독(~48h). 성공 = 동일 데이터 50ep 기준선 대비
   공극 −0.25pp 초과. 해악 가드: 전체 |B|·토크 +0.25pp 이내.
4. **(비 GPU) 487.9 A 보간 프로브 채점** — 6케이스, 평가 전용, 게이트 통과 완료.
   게이트가 아니라 계측: 풀드 토크가 325.3 A(2.404%)와 650.5 A(3.270%) 사이면
   보간 성립.

### 사전등록 대기 (실행 전 판정 규칙 필요)

- **밴드 스펙트럴 손실에 슬롯 조화 48/96/144 추가.** §31d가 근거를 만들었다 —
  게이트가 읽는 a1의 거칢이 슬로팅인데 `BAND_SPECTRAL_ORDERS`(4의 홀수배)가
  그 차수를 배제하고 있다. a1 단독 측정에서 **파라미터 6개가 잔차 개선폭의 36%를
  회수**했다. 손실 축이므로 E1 뒤에 단일축으로 붙인다.
- **Continuum Air on/off** — §31d의 미결(a1 잔여 6.95%)을 가르는 유일한 실험.

### Phase 2 착수 전 필수

- 위 큐 종료 후 **논문 초안 작성**(캠페인 서사가 R9/E2/E1로 닫힌다).
- **트랜지언트 데이터 생성 계약 수립** — Phase 1과 동급의 게이트를 새 캠페인에 이식.
- gmsh 도입(승인됨) 후 사전학습 경로 설계.
