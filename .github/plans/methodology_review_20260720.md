# Methodology Review & Revised Plan (2026-07-20)

기존 방법론(이전 세션에서 수립)을 실험 결과 기반으로 비판적으로 재검토하고, 재개 시점의 우선순위 계획을 재정의한다.
이 문서는 `.github/plans/phase1_phase2_phase3_motorgnn_roadmap.md`를 대체하지 않고 **수정(amend)** 한다.

검토 근거 아티팩트:
- `model_comparison_summary.json`, `comparison_results.json`
- `mgn_fem_comparison/test_summary.json` (180 test samples, physical units)
- `results/regression_baseline.json` (overfit-single baseline)
- `train_doe_meshgraphnet.py` (split/feature 구성 코드 직접 확인)
- `eval_all_models_output.txt` (400 records, 크래시 로그 포함)
- 로드맵 / PBC 전처리 설계 문서

---

## 1. 실측 결과 요약 (있는 그대로)

| 모델 | best val MSE (norm.) | 물리 단위 평가 | 파라미터 | 비고 |
|---|---|---|---|---|
| MeshGraphNet | 1.40e-3 | nRMSE ~9.7%, R² 0.77 (Bnorm, 180 samples) | 2.3M / 721k | 두 개의 서로 다른 학습 경로 존재 |
| FNO | 6.01e-3 | nRMSE 0.18%, R² 0.9925 (?) | 16.8M | **의심스러운 수치 — §2.2 참조** |
| GINO | 8.04e-2 | — | 2.5M | 사실상 학습 실패, 원인 미규명 |
| Seq2SeqRNN | 6.59e-3 | — | 2.8M | — |

- 데이터: DOE 40 케이스 × 10 timesteps = 400 records, 노드 ~6–8k (2D 1/8 sector)
- Overfit-single gate: 통과 (7.9e-6), PBC 테스트 23개 통과 — 파이프라인 배관 자체는 건전
- `eval_all_models.py`는 `torch.load(weights_only)` 오류로 **완주 실패**, H5 파싱 경고(StaticLoad/StaticOC "missing steps") 다수 — 조용히 데이터 스킵 중

---

## 2. 비판적 발견 (Critical Findings)

### F1. 데이터 누수 2종 — 가장 심각

**(a) Split 누수.** `train_doe_meshgraphnet.py:355` — "Train / Val split (shuffle across all cases)".
400 records를 랜덤 셔플하므로 **같은 형상(case)의 다른 timestep이 train과 val에 동시에 존재**한다.
같은 메시, 같은 형상 파라미터, 로터 각도만 다른 샘플은 사실상 준-중복이다.
→ 현재 보고된 모든 val MSE는 "새 형상에 대한 일반화"가 아니라 "본 형상의 각도 보간" 능력이다.
서로게이트의 존재 이유(새 DOE 후보 스크리닝)를 검증한 실험이 **아직 없다**.

**(b) 문서-코드 불일치로 인한 누수 오인 위험.** `train_doe_meshgraphnet(_aj).py`의 docstring은
node features에 `A, J` 포함(11개)을 명시하지만, **실제 코드는 9개 특징만 구성하며 A/J를 입력에
넣지 않는다** (검증 완료: `column_stack` 실제 컬럼 확인). 즉 현재 MGN 학습 자체는 깨끗하다.
그러나 (i) 스테일 docstring과 `# 11` 주석은 후속 에이전트/작업자가 A/J 입력을 "의도된 설계"로
오인해 재도입할 위험이 있고, (ii) FNO 그리드 파이프라인(`eval_fno_aj.py`의 "A, J 채널 확인")은
입력 채널 구성이 별도 경로라 동일 검증이 필요하다. FNO의 nRMSE 0.18%는 (a)의 split 누수 +
프로토콜 차이(F2)로 설명될 가능성이 높지만, 입력 채널 감사 전까지 신뢰 불가.
참고로 phase1_static 경로(`regression_baseline.json`)는 input 3 / output 5로 깨끗하다 —
**두 학습 파이프라인이 공존하며 서로 다른 문제를 풀고 있다.**

### F2. 평가 프로토콜 불일치

MGN 9.7% vs FNO 0.18%는 같은 벤치마크의 숫자가 아니다(다른 split, 다른 정규화, 다른 타깃,
다른 그리드). 현재 상태로는 모델 선택 결론을 내릴 수 없다. 또한 eval 스크립트 크래시와 H5
파싱 경고가 방치되어 "전 모델 공정 비교"는 실제로 한 번도 완료된 적이 없다.

### F3. 타깃 설계 오류: J는 출력이 아니다

채널 계약 `[Bx, By, A, J]`(로드맵) / `[Bx, By, A, J, Je]`(baseline) — 계약 자체가 이미 드리프트했고,
물리적으로도 문제가 있다:
- **J (소스 전류밀도)는 권선 여자로 주어지는 입력**이다. 예측 대상이 아니라 조건 특징이다.
- **B는 A로부터 유도 가능** (B = ∇×A). Bx, By를 독립 채널로 예측하면 div B = 0 보장이 없고,
  A–B 간 불일치가 허용된다.

권고: 출력 = **[A] (+ 도체 영역 Je)**, B는 이산 curl로 후처리 유도. 이러면 L_B와 L_curl이
하나로 합쳐지고 A–B 일관성이 구조적으로 보장된다.

### F4. autograd curl은 그래프 모델에서 수학적으로 부정확

로드맵의 `autograd.grad(A_pred, coords)` 방식: MGN에서 노드 i의 A는 이웃 노드 좌표에도
의존하므로, ∂A_i/∂pos_i는 공간 미분 ∂A/∂x|_i 가 아니다 (message passing을 통한 의존성 누락).
권고: **FEM P1 요소별 이산 gradient** (삼각형 요소마다 상수 ∇A, 면적 가중 노드 평균)로 대체.
메시 connectivity는 이미 있고, 순수 함수로 구현 가능하며 미분 가능(사전 계산된 sparse 행렬 곱).

### F5. 모델 포트폴리오 문제

- GINO: val MSE 0.08로 실패했는데 원인 분석 없이 방치. 디버그하거나 명시적으로 drop 결정 필요.
- FNO: 비정형 메시를 그리드로 보간해야 하며(경계 아티팩트), 16.8M 파라미터는 2D 필드 대비 과대.
  공정 벤치마크에서 이기지 못하면 drop.
- RNN: 노드 순서 의존 구조라면 permutation invariance가 없어 메시 기반 문제에 원리적으로 불리.

### F6. 정확도 수준과 엔지니어링 지표 부재

nRMSE ~10%, R² 0.75는 DOE 스크리닝 서로게이트로 불충분하다 (통상 필드 <5%, 파생량 <2-3% 요구).
더 중요한 문제: **모터 설계자가 쓰는 지표(토크, 쇄교자속, 코깅, 철손)로 평가한 적이 없다.**
필드 nRMSE가 5%여도 토크 오차가 15%일 수 있고 그 역도 성립한다. Maxwell stress 기반 토크를
평가 지표에 넣어야 서로게이트의 실용 가치를 판단할 수 있다.

### F7. PBC 전처리 설계 과잉 (over-engineering)

`phase1_pbc_preprocessing_design.md`의 chain 추출 + global prior 추론 + 설명가능 pairing 체계는
잘 설계됐지만, **현재 데이터는 Motor-CAD 산출물이라 slot/pole 수와 주기성이 메타데이터로 이미
알려져 있다.** 메시에서 주기성을 역추론할 필요가 없다. 알려진 n_geom + KDTree 매칭(이미 구현,
23 테스트 통과)이면 Phase 1–3에 충분하다. 범용 mesh-native 추론은 Phase 4(신규 토폴로지
일반화)로 연기. 문서는 보존하되 지금 구현하지 않는다.

### F8. Phase 2/3 조기 진입 위험 + 운영 위생

정적 일반화가 미검증(F1a)인 상태에서 동적 시계열/rollout으로 가면, 정적 단계의 결함이 시간축
오차 누적과 얽혀 원인 분리가 불가능해진다. 부수적으로: 루트 디렉토리에 스크립트/로그/체크포인트
~200개 산적(tmp_*, check_*, patch_* 등), 채널 계약 4ch/5ch 혼재, eval 스크립트 PyTorch 2.6
비호환 — 재현성을 갉아먹는 상태.

---

## 3. 유지할 것 (Keep)

- **MeshGraphNet 백본** — 비정형 메시에 자연스럽고, 공정 비교에서도 현재 최선
- **PBC anti-periodic edge 처리** (`message * edge_attr`, sign ±1) — 물리적으로 올바름
- **Overfit-single / contract test / regression baseline 하니스 문화** — 그대로 확대 적용
- **Phase 2의 3-모듈 분리 설계** (Geometry Encoder / 결정론적 Graph Builder / Field GNN) —
  SB connectivity = f(θ)를 비학습으로 두는 결정은 옳다. 시기만 늦춘다.
- **불변 데이터 계약 + 순수 함수 원칙** — 다만 계약 문서를 코드보다 앞세우는 관성은 경계
  (PBC 설계 문서 800줄 vs 미구현. 문서:코드 비율 관리 필요)

---

## 4. 수정 로드맵 (R0 → R4, 우선순위 순)

### R0. 평가 신뢰 회복 — 다른 모든 것보다 먼저 (~1주)

지금 있는 어떤 숫자도 의사결정에 쓸 수 없다는 것이 본 리뷰의 결론이므로, 첫 작업은 학습이 아니라 측정이다.

1. `eval/benchmark.py` 단일 하니스 신설:
   - **Case-level split 고정**: 40 케이스를 train 30 / val 4 / test 6으로 **형상 단위** 분리,
     split manifest를 JSON으로 저장·버전 관리 (모든 모델이 동일 split 사용)
   - 물리 단위 지표: 채널별 nRMSE/R², region별(공극/치/요크) 분해 지표
   - **토크 지표 추가**: Maxwell stress 선적분 기반, FEM 대비 상대오차
   - 입력 특징 화이트리스트 강제: `pos, region, geometry params, operating point (Ipk, phase adv, θ)` —
     **A, J, B 등 해 유래 특징 입력 금지**를 assert로 차단
2. PyTorch 2.6 `weights_only` 대응, H5 파싱 경고 원인 규명 (StaticLoad/StaticOC 파일 포맷 차이)
3. 4개 모델 재평가 → `results/benchmark_v2_baseline.json` 재작성 (이것이 새 기준선)

**게이트 G0**: 동일 하니스에서 모든 모델의 case-holdout 성적표 확보. 이전 숫자 전량 폐기 선언.

### R1. 계약 v2 재동결 + 누수 제거 재학습 (~1-2주)

1. 채널 계약 v2: 입력 = `[pos, region_code, geom params, Ipk, phase_adv, θ(rotate_step)]`,
   출력 = `[A]` (+ 도체 영역 `Je`), **B는 이산 curl로 유도** (F3)
2. `phase1_static/discrete_curl.py`: P1 요소별 ∇A sparse 연산자 (사전 계산, 순수 함수, torch 미분 가능)
3. 손실 v2: `L = wA·MSE(A) + wB·MSE(curl(A_pred) − B_gt)` — autograd curl 제거 (F4),
   λ annealing은 유지
4. MGN 재학습 (phase1_static 경로로 일원화, DOE 학습 스크립트의 A/J 입력 경로 폐기)

**게이트 G1**: case-holdout에서 B nRMSE, 토크 오차 측정 → 기준선 대비 개선 확인.

### R2. 일반화·데이터 효율 (~2-4주, R1과 부분 병행)

1. 현 40 케이스로 **학습곡선** 실험 (10/20/30 케이스 학습 → holdout 성능): 데이터 추가의
   한계효용을 정량화 → DOE 추가 실행 규모 결정 (Motor-CAD 배치는 이미 자동화되어 있음)
2. 형상 파라미터 공간 커버리지 점검 (Ratio_Bore, Ratio_SlotDepth 등 DOE 경계 근처 성능 분해)
3. 필요시 데이터 증강: 회전 대칭 활용 (1/8 sector의 회전 복제는 PBC와 정합적)

**게이트 G2 (핵심 의사결정 지점)**: case-holdout에서 **B nRMSE < 5% AND 토크 오차 < 3%**.
- 통과 → R4 (Phase 2 동적 파이프라인 진행)
- 미달 + 학습곡선이 우상향 → DOE 확대 후 재시도
- 미달 + 학습곡선 포화 → 모델 계열 재검토 (R3), 또는 목표 재정의
  (예: 절대 서로게이트 대신 FEM 보정/warm-start 모델, coarse-FEM + ML correction)

### R3. 모델 포트폴리오 정리 (조건부)

- GINO: 1일 debug 타임박스 → 원인 못 찾으면 공식 drop (기록 남김)
- FNO: R0 공정 벤치마크에서 MGN을 유의하게 이기지 못하면 drop (16.8M 파라미터 유지 비용)
- 후보 추가 검토(G2 미달 시에만): graph transformer 계열 (Transolver/GNOT류),
  geometry latent conditioning 강화 — PhysicsNeMo 내 구현 존재 여부 우선 확인
- RNN: 정적 문제에서 제외, Phase 3 시계열 비교군으로만 보존

### R4. Phase 2/3 진행 (G2 통과 후에만)

- 기존 로드맵의 3-모듈 설계 그대로 진행 (수정 불요)
- Phase 3 rollout은 residual 예측 + teacher forcing + noise injection 기존 계획 유지
- PBC mesh-native 추론(설계 문서)은 Phase 4 신규 토폴로지 단계에서 재개봉

### 운영 위생 (백그라운드, 아무 때나)

- 루트 스크립트 정리: `tools/`, `scripts/archive/`로 이동, 로그/체크포인트 gitignore 정비
- 로드맵 문서의 채널 계약 서술을 v2로 갱신 (4ch/5ch 혼재 해소)
- 커밋 컨벤션 유지 (`[TASKKEY] ...`), 본 문서를 `AGENTS.md`의 Active Work Plans에 링크

---

## 5. 요약: 무엇이 바뀌는가

| 항목 | 기존 | 수정 |
|---|---|---|
| Split | 400 records 랜덤 셔플 | **case(형상) 단위 holdout 고정** |
| 입력 특징 | docstring-코드 불일치 (오인 위험) | 해 유래 특징 금지 (assert 강제) + 문서 정정 |
| 출력 채널 | [Bx,By,A,J(,Je)] | **[A(,Je)]**, B는 이산 curl 유도 |
| Curl 손실 | autograd.grad(A, pos) | FEM P1 요소별 이산 gradient |
| 평가 | 모델별 제각각 | 단일 하니스, 물리 단위, **토크 지표** |
| PBC 전처리 | mesh-native 추론 설계 구현 예정 | 메타데이터 기반으로 축소, 설계는 Phase 4 보류 |
| Phase 2 진입 | 체크리스트 완료 시 | **G2 게이트 (B<5%, 토크<3%, case-holdout) 통과 시** |
| GINO/FNO | 계속 유지 | 타임박스 debug / 공정 비교 후 drop 판단 |

---

## 6. R0/R1 실행 결과 추가 (2026-07-20, 하니스 구축 후)

`eval/` 하니스를 구축하며 실측한 내용. §2의 발견을 일부 **정정·강화**한다.

### A. F1b 정정 — 문서-코드 불일치가 아니라 실제 타깃 누수였다

리뷰 시점에 "현재 코드는 9특징이라 깨끗"으로 판정했으나, **저장된 체크포인트를 열어보니
`doe_meshgraphnet_ckpt.pt`와 `doe_meshgraphnet_ckpt_4d.pt`는 node encoder 입력이 11이다.**
즉 §1 표의 헤드라인 MGN(val MSE 1.40e-3, nRMSE 9.7%)은 A와 J를 입력으로 받은 모델이다.
B = ∇×A 이므로 이는 split 누수가 아니라 정답 주입이다.
`physicsnemo_train_from_pyMCAD.py`는 A, J를 **입력이자 출력**으로 쓴다(4채널 중 2채널이 항등).

→ 하니스가 11-feature 체크포인트를 구조적으로 거부한다(`CheckpointFeatureMismatch`).
→ 입력이 깨끗한 체크포인트는 `physicsnemo_meshgraphnet_ckpt.pt`(9 in / 4 out) 하나뿐.

### B. 데이터가 알려진 것보다 4.8배 많다

`Mag_StaticLoad/StaticOC`는 `pyMCAD.magnetic.static_mesh` 포맷(스칼라 `step`, 1D 필드)이라
학습 파서가 raise → 호출부가 삼켜 조용히 드롭되고 있었다(§1의 "missing steps" 경고 정체).
또한 OnLoadTorque는 케이스당 **45 스텝**이며, "400 records"는 `--max-steps-per-case 10`으로
잘린 값이다. 실 가용: OnLoadTorque 1800 + static 116 = **1916 샘플**. R2 학습곡선 전제가 바뀐다.

### C. F4 보강 — 노드 지지점 자체가 토크 지표를 파괴한다 (신규, 가장 중요)

Motor-CAD는 B를 **요소별**로 저장한다. 노드 예측 모델은 element→node로 학습하고
node→element로 채점되는데, **이 왕복만으로** |B| nRMSE 17%, 토크 nRMSE 60%(토크 57% 과대)가
발생한다. 공극 밴드가 요소 한 겹이라 그 노드가 고정자 철의 |B|를 끌어오기 때문이다.
모델 품질과 무관한 파이프라인 손실이며, **G2의 토크 3% 게이트는 이 파이프라인에서 도달 불가**다.

측정된 대안(테스트 케이스, 슬라이딩 밴드 제외):

| 파이프라인 | \|B\| nRMSE | 토크 nRMSE |
|---|---|---|
| 노드 왕복 (현행) | 16.7% | 60–77% |
| **nodal A → P1 요소 curl** | **5.3%** | **1.3–3.3%** |

→ R1의 "출력 = A, B는 이산 curl 유도"는 물리적 일관성뿐 아니라 **수치적으로 필수**다.
→ 단, export의 element A를 그대로 미분하면 39%로 오히려 나쁘다(이미 요소평균된 값을 미분하면
   평균화 오차가 증폭). nodal A는 **curl을 통해 element B로 supervise**해야 한다 — R1 손실 v2 그대로.

### D. 신규 발견 — 회전 스텝의 슬라이딩 밴드는 기하학적으로 무효

export는 기준 connectivity + 스텝별 노드 좌표만 담는데, Motor-CAD는 공극 슬라이딩 밴드를
매 스텝 **재분할**한다. 둘을 결합하면 스텝 3부터 요소 180~246개의 부호가 뒤집히며,
**전부 a2(회전자측 공극층)**이다. 고정자측 a1은 한 번도 뒤집히지 않는다(→ 토크 밴드로 적합).

→ a2/a3/a4는 회전 스텝의 모든 기하 연산(요소 gradient, edge 벡터, 면적)에서 제외해야 한다.
   MGN의 edge feature(dx, dy, dist)도 45스텝 중 44스텝에서 이 영역이 오염되어 있다.
→ `eval.mesh_regions.sliding_band_mask`로 식별.

### E. 토크 지표 설계 — 순간 상대오차는 게이트로 쓸 수 없다

test case 32는 PhaseAdvance 89.07°(DOE 최대)로 평균 토크가 ~0이며 스윕 중 부호가 바뀐다.
순간 상대오차는 분모가 0을 지나며 발산한다. 헤드라인 지표는
**`nrmse_torque_pct` = RMSE(T) / RMS(T_fem)** 로 고정한다. G2는 이 숫자로 판정.

### F. 고정된 분할

`eval/splits/doe40_case_split.json` (seed 42, DOE digest f62202a0):
train 30 / val 4 / **test [4, 7, 18, 32, 37, 39]**. 전 모델이 이 파일을 공유한다.
