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

### G. anti-periodic 엣지가 "제약"이 아니라 "힌트"로 들어갔다 (미해결)

`phase1_static/custom_mgn.py`의 `AntiPeriodicMessagePassing`은 `msg * signed_attr`로
**메시지에 부호를 곱한다**. 이것이 원래 설계이고 물리적으로 올바른 반주기 전파다.

그러나 현재 DOE 학습기는 PhysicsNeMo stock `MeshGraphNet`을 쓰고, 이 구현의 소스에는
`* edge`, `mul(`, `sign` 이 전혀 없다 -- 표준 Pfaff 구조대로 [src, dst, edge]를 concat해
MLP에 넣을 뿐이다. 따라서 `edge_attr`에 넣은 `edge_sign` 채널은 **모델이 학습해야 할
특징**이지 강제되는 제약이 아니다.

즉 §6.D의 "anti-periodic 엣지 추가"는 연결성(두 절단면을 잇는 엣지)은 확보했지만
부호 전파는 확보하지 못했다. 사전/사후 비교는 "부호를 힌트로 준 것이 도움이 되는가"를
답할 뿐이며, 제대로 된 반주기 전파의 효과는 아직 측정되지 않았다.

해소하려면 stock MeshGraphNet 대신 custom_mgn 경로로 바꿔야 하는데, 그러면 아키텍처가
달라져 기존 비교 기준선과 단절된다. 별도 실험으로 분리하는 것이 옳다.

---

## 7. R1 최종 성적표와 결론 (2026-07-21)

case-holdout 270샘플(test 케이스 4/7/18/32/37/39), 물리 단위, 전 모델 동일 요소
부분집합(coverage 0.865). 30 epoch, batch 4, stride 3, 동일 하이퍼파라미터.

| 모델 | Bx | By | \|B\| | 토크 nRMSE | 평균토크 오차 | 리플비 |
|---|---|---|---|---|---|---|
| curl 표현 바닥값 | — | — | 5.31% | **1.58%** | 1.28% | 1.00x |
| node 왕복 바닥값 | — | — | 16.98% | 59.54% | 51.57% | 2.20x |
| G0 node (사전, **노드 손실**) | 33.38% | 29.01% | 24.49% | 66.55% | 59.21% | 2.50x |
| curl A (사전, 요소 손실) | 35.52% | 32.30% | 25.26% | **29.59%** | 6.38% | 2.94x |
| curl A (사후, 섹터수정) | 47.16% | 46.36% | 30.57% | 45.05% | 11.86% | 5.67x |
| **node B (사후, 요소 손실)** | 33.71% | 30.70% | **22.67%** | **29.73%** | **5.19%** | 4.11x |

FEM 평균토크 −2281 N·m/m, 리플 862.

### 결론 1 — 병목은 "A vs B"가 아니라 **손실의 지지점**이었다

§6.C에서 "노드 예측은 왕복 바닥값 59.5%에 갇힌다"고 했으나 **틀렸다**.
node B가 29.73%로 바닥값을 크게 밑돌았다.

`node_resampling_floor`는 *진실값을 기계적으로 왕복시킨* 값이다. 학습된 모델은 거기
묶이지 않는다 — 요소 지지점에서 손실을 받으면 평균화를 되돌리는 노드값을 학습한다.

증거는 G0(66.55%)와 node B(29.73%)의 차이다. 둘 다 노드에서 B를 예측하지만:
- G0: 노드로 흩뿌린 타깃에 **노드에서** 손실 → 이중 평활화 → 바닥값 위에 갇힘
- node B: **요소에서** 손실 → 평균화 보상을 학습 → 바닥값 돌파

curl A(사전) 29.59% ≈ node B 29.73%로 사실상 동일한 것이 이를 확증한다.
**교훈: 손실은 FEM이 값을 저장한 지지점(요소)에서 계산한다.** A 예측은 이를 달성하는
한 방법이지 유일한 방법이 아니다. P1 curl 연산자는 여전히 유용하다(표현 바닥값 1.58%가
가장 낮고, div B = 0이 구조적으로 보장된다) — 다만 필수는 아니다.

### 결론 2 — 섹터 수정은 현 구현으로는 손해다 (음성 결과)

통제된 비교는 curl A 사전 vs 사후(같은 학습기·손실·하이퍼파라미터, 섹터 처리만 차이):

  |B| 25.26% -> 30.57%,  토크 29.59% -> 45.05%,  리플비 2.94x -> 5.67x

물리적으로 옳은 처리인데 전 지표가 악화됐다. 원인 두 가지:

1. **wrap이 부호 모호성을 만든다.** cum=-10도와 -55도가 기하학적으로 완전히 동일해지고
   타깃 부호만 반대가 된다. 모델은 전역 스칼라 각도 특징 하나로 구분해야 하는데,
   모든 노드에 브로드캐스트된 상수는 지역 기하 대비 약하게 작용한다. wrap 전에는
   로터 위치 자체가 달라 구분이 쉬웠다.
2. **anti-periodic 엣지가 힌트로만 들어갔다** (§6.G). stock MeshGraphNet은 edge_attr를
   MLP 입력으로 concat할 뿐 부호를 곱하지 않는다. 부호를 전파할 구조적 경로가 없는
   상태에서 부호 판별을 요구한 셈이다.

즉 2번 없이 1번만 적용하면 손해다. 섹터 수정은 `custom_mgn.py`의 곱셈식 부호 전파
(`msg * signed_attr`)와 **함께** 재시도해야 하며, 그때까지는 기본값을
`--no-wrap-rotor --no-anti-periodic-edges`로 둔다.

### 결론 3 — 이제 리플이 지배적 오차다 (신규)

전 모델이 평균토크는 5-12%로 맞추지만 리플은 3-6배 과대다. **표현 바닥값의 리플비는
1.00x**이므로 리플 오차는 전적으로 모델 오차이지 표현 오차가 아니다.

토크 nRMSE는 사실상 리플 오차가 지배한다(node B: 평균 5.19% vs 전체 29.73%).
DOE 스크리닝에서 평균토크 5%는 경계선상 유용하지만, 리플 4배는 코깅/리플 평가에
쓸 수 없다.

영역별로 보면 공극이 가장 나쁘다(바닥값 2.68% vs 모델 29.48%). 토크는 공극 밴드에서
B_r·B_theta의 곱으로 계산되므로 공극 정확도가 직접적인 레버다.

### 다음 단계

1. **요소 지지점 손실을 기본으로 채택.** `train_doe_curl_mgn.py --target B`가 현재 최선
   (22.67% / 29.73%)이고 curl보다 2배 빠르다(32초 vs 62초/epoch).
2. **수렴 확인.** 30 epoch에서 val이 아직 하강 중이다. 장기 학습으로 "학습 부족"과
   "데이터 부족"을 먼저 분리해야 R2 학습곡선이 의미를 갖는다.
3. **공극 가중.** 토크가 공극에서 결정되므로 공극 요소에 손실 가중을 주는 것이
   전체 |B|를 균등하게 낮추는 것보다 토크에 직접적이다.
4. **섹터 수정 재시도는 custom_mgn 곱셈 부호와 묶어서.** 단독으로는 손해다.

G2(토크 3%)까지는 29.6% -> 3%로 10배가 남았다. 표현 바닥값이 1.58%이므로 여지는
있으나, 현재 격차는 모델·데이터·수렴의 문제이지 표현의 문제가 아니다.

---

## 8. 토크 지표 검증: Motor-CAD 대조 (2026-07-21)

"우리 FEM 토크 파형이 Motor-CAD 것과 다르다"는 지적을 조사한 결과.

### 결론: 연산자는 맞고, 표기 단위가 달랐다

우리 "FEM" 곡선은 Motor-CAD의 네이티브 토크가 아니라 **하니스가 export된 요소
필드를 직접 Arkkio 적분한 값**이다. 라벨은 정확했으나 단위가 **N·m per m of
stack**(적층장 1 m 기준)이었다. 이 설계의 실제 적층장은 150 mm이므로
Motor-CAD 표기와 **6.67배** 차이가 난다.

  우리 표기  -2462.7 N·m/m
  x 0.150 m    -369.4 N·m   <- Motor-CAD와 같은 단위

### Motor-CAD .mot에서 교차 확인된 것 (case_0004)

  MagneticSymmetryFactor = 8      우리가 섹터 span에서 도출한 8과 일치
  Pole_Number = 8, Slot_Number=48  1/8 섹터 = 45도 확인
  Stator_Lam_Length = 150 mm       적층장
  PeakCurrent = 224.047777         backup case_0004와 동일
  PhaseAdvance = 6.228413          backup case_0004와 동일
  TorquePointsPerCycle = 45, TorqueNumberCycles = 1

스텝-사이클 정합도 확인: 45 스텝 x 2.0도 = 90.0도 기계각이고, 8극에서
1 전기사이클 = 90.0도 기계각이다. 우리 45스텝이 정확히 1 전기사이클이다.

### 연산자 내부 일관성 (독립 검증)

step 0(메시 전체 유효)에서 4개 공극층을 각각 독립 적분:

  a1  744 elem  T = -2482.4 N·m/m
  a2  366 elem  T = -2437.8
  a3  720 elem  T = -2481.6
  a4  720 elem  T = -2481.3
  a1..a4 통합 (공극 1 mm 전체)  T = -2479.8

a1/a3/a4가 0.04% 이내로 일치하고, 통합값과 a1 단독이 0.1% 차이다.
"a1 밴드가 한 겹이라 부정확할 것"이라는 우려는 기각된다. a2만 1.8% 벗어나는데
이는 전단층이라 요소 수가 절반이기 때문이다.

### 확인 불가능한 것

**Motor-CAD의 E-Magnetic 온로드 토크 파형은 export에 없다.** .mot에는 Lab
결과만 있고(Envelope_Torque_Array = 토크-속도 포락선 151점,
LoadPoint_Torque_Array = 650 A / 43~79도의 5개 운전점 -- 우리 동작점
224 A / 6.2도가 아니다), 과도해석 토크는 바이너리 .mes에만 존재한다.
DOE export 파이프라인이 메시+필드만 저장했다.

따라서 절대 토크의 최종 대조는 Motor-CAD에서 해당 동작점의 토크를 다시
뽑아야 가능하다. 필요하면 pyMCAD로 `LoadPoint_Torque` 계열을 export에
추가하는 것이 근본 해결이다.

### G2 게이트 영향: 없음

`nrmse_torque_pct = RMSE(T) / RMS(T_fem)`는 분자·분모가 모두 적층장에
비례하므로 **스케일 불변**이다. 평균토크 오차(mean_torque_error_norm_pct)도
같다. 즉 지금까지 보고한 토크 지표(9.20% 등)는 **재보정이 불필요**하다.
바뀌는 것은 표시용 절대값뿐이다.

### 조치

- `tools/make_field_gif.py`에 `--axial-length-m` 추가, 기본 0.150 m.
  GIF의 토크 축이 이제 N·m으로 표기된다.
- 부호: 우리 값은 음수(로터가 -theta로 회전). Motor-CAD는 양수 모터링
  토크로 표기한다. 규약 차이이며 오류가 아니다.

### 8b. Motor-CAD 절대 대조 완료 (2026-07-21)

pyMCAD(PyMotorEnv_310, pymotorcad 0.8.4, Motor-CAD v261)로 case_0004 사본을
열어 E-Magnetic 해석을 재실행하고 네이티브 토크 파형을 뽑았다. 원본 solve
디렉토리는 건드리지 않았다(케이스 전체를 스크래치패드로 복사해 작업).

Motor-CAD가 이 모델에서 제공하는 온로드 토크 그래프는 `TorqueVW`
(가상일법) 하나뿐이다. `TorqueMST`는 존재하지 않으므로 우리 Arkkio
(Maxwell 응력)와는 **서로 다른 두 방법의 대조**가 된다.

  지표          Motor-CAD (VW)   우리 (Arkkio x8 x0.15m)   차이
  평균 토크        370.13 N.m          367.93 N.m         -0.60%
  리플              57.14 N.m           57.00 N.m         -0.2%
  점별 RMSE                          1.99 N.m (0.54%)
  상관계수                              0.99971
  계통 편의                          +1.95 N.m (MC가 높음)

**잔여 -0.60%는 §8에서 예측한 메시 faceting 편의(-0.49%)와 일치한다.**
곡선 공극 경계를 직선 삼각형으로 근사하면서 밴드 면적을 0.49% 적게
잡는 것이 원인이며, 계통적이고 부호가 일정하다(오차 그래프가 전 구간
+1.9~2.2 N.m로 평탄).

부수 확인:
- API가 반환한 값이 .mot 텍스트 파싱과 일치: Stator_Lam_Length=150,
  MagneticSymmetryFactor=8, Pole_Number=8, TorquePointsPerCycle=45
- 우리 H5 export(41스텝)는 1 전기사이클의 89%만 담고 있다(320/360 elec deg).
  Motor-CAD 그래프는 46점 x 8 elec deg = 360도 전체. backup DOE는 45스텝
  x 2 mech deg = 90 mech = 360 elec으로 전체 사이클이다.
- 위상 정렬에 232 elec deg 오프셋이 필요했는데, 이는 H5 step 0과
  Motor-CAD 그래프 x=0의 각도 기준이 다르기 때문이며 물리적 의미는 없다.

### 결론: 토크 지표 절대 검증 완료

우리 Arkkio 연산자는 Motor-CAD 가상일법 대비 **평균 0.60%, 파형 상관
0.9997**로 일치한다. 1/8 대칭(x8)과 적층장(x0.15m) 환산이 모두 올바르며,
G2 게이트가 읽는 `nrmse_torque_pct`는 재보정 없이 그대로 유효하다.

남은 계통 편의 0.5%를 없애려면 밴드 면적을 정확한 환형 면적으로
정규화하면 되지만, 예측/FEM 양쪽에 동일하게 작용하므로 상대 지표에는
영향이 없다.

### 8c. 3자 비교와 부호 규약 확정 (2026-07-21)

Motor-CAD / FEM-Arkkio / 서로게이트-Arkkio 세 파형을 한 축에 올렸다
(`results/viz/torque_three_way_case0004.png`, backup case_0004, 45스텝 = 1
전기사이클 전체).

  곡선                        평균        리플     MC 대비 평균   RMSE
  Motor-CAD TorqueVW        370.72      51.53         --          --
  Arkkio on FEM B           369.41      54.32       -0.35%       2.02
  Arkkio on surrogate B     349.59     142.14       -5.70%      34.89

Motor-CAD와 FEM-Arkkio는 그림에서 구별되지 않는다. 서로게이트는 평균은
5.7% 낮고 리플은 2.8배 크며, **오차가 사이클 앞부분(0~100 전기각)에
집중**된다(최대 -110 N.m). 이는 step 0의 rotate_step=0, time_s=NaN->0으로
첫 몇 샘플의 특징이 비정형인 것과 관련될 수 있으며, 별도 확인 가치가 있다.

#### 부호 규약 (확정)

`arkkio_torque`는 **+theta(반시계) 기준** 토크를 반환한다. 이 export에서
로터는 시계방향으로 돈다(rotate_step 음수, 회전자 평균각 -43 -> -120도
실측). 모터링이므로 전자기 토크가 운동 방향과 같고, 따라서 우리 적분값이
**음수인 것이 물리적으로 옳다.** Motor-CAD는 모터링 토크를 양의 크기로
보고하므로 두 값은 -1배 관계다.

**위상 이동이 부호 반전을 가릴 수 없음을 수치로 확인했다:**

  |우리|로 맞춤        shift 232도, RMSE   1.99 N.m
  우리(음수)로 맞춤    shift  60도, RMSE 738.38 N.m   <- 370배 나쁨

파형의 평균(368)이 리플 진폭(+-28.5)을 압도해 0을 절대 지나지 않으므로,
순환 이동은 평균을 보존하고 +370을 -370으로 만들 수 없다. 부호와 위상은
독립이다. 232도 오프셋은 H5 step 0과 Motor-CAD 그래프 x=0의 각도 기준
차이일 뿐이다(단, 리플 주기가 ~36 전기각이라 180도 어긋난 지점의 RMSE가
5.34로 3배 차이에 그친다 -- 위상 결정에는 이 정도 여유가 있음을 감안할 것).

규약은 `eval/torque.py` 모듈 docstring에 명시하고 두 개의 테스트로
고정했다(`test_sign_follows_the_plus_theta_convention`,
`test_a_dc_dominated_waveform_cannot_be_sign_flipped_by_a_phase_shift`).

---

## 9. 사이클 앞부분 오차 진단 (2026-07-21)

3자 비교에서 서로게이트 오차가 0~100 전기각에 집중되는 것이 관찰돼
세 가설을 실험으로 갈랐다.

### 관찰 1 — 보편적이지 않고 학습 케이스에도 있다

케이스별 (첫 5스텝 |오차| / 나머지) 비율:

  test  4  4.44 | train 3  3.03 | test 39  3.00 | test  7  2.03
  train 2  1.20 | train 0  1.10 | test 32  1.05 | train 6  0.98
  train 1  0.80 | train 8  0.52 | test 18  0.46 | test 37  0.39

전체 평균 프로파일은 첫 묶음 29.8 / 나머지 ~20 (비율 1.46)로 완만한
상승일 뿐, case_0004의 4.44는 예외적이다. **학습 케이스 3이 3.03을
보이므로 일반화 실패가 아니다.** 운전점과의 상관도 약하다
(ratio vs Ipk +0.147, vs PhaseAdvance -0.402, n=12).

### 관찰 2 — 원래 의심한 특징 아티팩트는 성립하지 않는다

이 체크포인트는 V2 특징이라 **`rotate_step`이 아예 입력이 아니다**.
`time_s`는 step0에서 NaN->0.0인데, 실제로 t=0이므로 값 자체는 정합적이고
이후 등간격이다. 즉 "step 0 특징이 비정형"이라는 원래 가설은 기각된다.

### 관찰 3 (결정적) — `time_s`가 지름길 특징이다

step 0에서 `time_s`만 바꿔 예측을 재계산했더니:

  FEM            -372.36 N.m
  pred (t=0)     -264.49        오차 +107.87
  t -> 2.778e-05 -229.96        (+34.5 이동)
  t -> 2.778e-04  +47.64        (+312.1 이동)   <- 부호까지 뒤집힘
  t -> 1.000e-03  -13.97        (+250.5 이동)

**`time_s` 하나만 바꿔도 토크가 -264에서 +48로 뒤집힌다.** 그리고
time_s는 전 케이스에서 동일하며 회전자각과 상관계수 **+1.0000**으로
완전 공선이다(모든 케이스가 같은 속도라 time_s = 각도 x 상수).

즉 `time_s`는 회전자각 인코딩과 **독립 정보가 0인 중복 특징**인데,
모델이 이것을 주 시간 단서로 학습했다. step 0은 time_s 분포의 최소
경계라 지름길이 가장 취약한 지점이고, 이것이 사이클 앞부분 오차의
직접 원인이다.

### 관찰 4 — 각도 인코딩 별칭(aliasing)

`rotor_angle_features`의 주기는 2섹터(90도)이므로 cum=0과 cum=-88의
각도 특징이 거의 같다(+0.000,+1.000 vs +0.139,+0.990). 반주기성에 의해
두 지점의 실제 자기장도 거의 같다. 그런데 모델의 오차는 케이스마다
크게 다르고 부호도 일정하지 않다(case 4: +107.9 vs +35.4, case 37:
-11.7 vs -85.5). **각도 특징이 같은데 답이 다르다는 것은 모델이 각도가
아닌 다른 것(time_s, 되감기지 않은 위치)으로 구별하고 있다는 뜻이다.**

### 관찰 5 — 학습 데이터도 못 맞춘다

학습 케이스 첫 5스텝 평균 오차 21.22 N.m = 정격의 5.7%. 일반화 이전에
**적합 자체가 안 되고 있다**(underfit). 용량 부족 또는 표현 문제다.

### 결론

가설 (a) 특징 아티팩트는 **원래 형태로는 기각**되지만, 변형된 형태로
**확인**된다: 문제는 step-0 값이 비정형인 것이 아니라 **`time_s`가
회전자각과 완전 공선인 중복 특징이고 모델이 이를 지름길로 사용**한다는
것이다. 가설 (c) MGN 한계도 부분적으로 성립한다(학습 오차 5.7%).
가설 (b) 기하 아티팩트는 근거 없다.

### 권고 조치

1. **`time_s`를 노드 특징에서 제거.** 회전자각 인코딩과 독립 정보가 0이고,
   모델이 지름길로 쓰고 있으며, 속도가 다른 DOE로 확장하면 곧바로
   틀린 특징이 된다. `eval.feature_guard`의 허용 목록에서도 빼고
   "중복/지름길 특징" 범주를 신설해 차단하는 것이 옳다.
2. 제거 후 재학습해 사이클 앞부분 오차가 사라지는지 확인.
3. 학습 오차 5.7%는 별개 문제이므로, time_s 제거 후에도 남으면
   용량(hidden_dim/processor_size) 또는 epoch을 올려 확인.

---

## 10. time_s 제거 재학습 완주와 §9 판정 (2026-07-21)

> **실행 머신: `HPC_134` / `192.168.0.134` — L40S 48GB (TCC), Windows Server 2022, 네이티브 실행**
> (이전 §6–9는 다른 머신에서 WSL2 + physicsnemo:26.03 컨테이너로 수행됐다.
> 공유 폴더는 `192.168.0.165`.)
>
> 이후 이 문서에 절을 추가할 때는 **어느 머신에서 실행했는지 위와 같은
> 형식으로 반드시 기입한다.** 스택이 머신마다 다르고(컨테이너 vs 네이티브),
> 결과 비교 시 그 차이를 아는 것이 전제이기 때문이다.

§9 권고 조치 2를 실행했다. 이전 머신에서 GPU 경합으로 중단됐던
no-time_s 런을 이 머신에서 60에폭 완주하고 채점·진단했다.

### 실행 환경 — Docker가 아니라 네이티브

새 머신(Windows Server 2022, L40S)에서는 **MIGRATION.md의 Docker 경로를
쓸 수 없다.** 호스트에 Docker가 없고, L40S가 **TCC 드라이버 모드**인데
WSL2 GPU 패스스루는 WDDM을 요구한다(WSL 안에 `nvidia-smi`조차 없다).
공유 서버의 드라이버 모델 변경은 침습적이라 네이티브 `.venv`로 갔다.

    python 3.12.9 / torch 2.11.0+cu128 / numpy 2.4.6
    physicsnemo 2.1.1 / torch_geometric 2.8.0 / torch_scatter 2.1.2+pt211cu128

**등가성은 게이트로 증명했다.** 기준 스코어카드
`results/benchmark_v2_nodeB_long.json`은 WSL2 리눅스 컨테이너(python
3.12.3, numpy 1.26.4)에서 만들어졌는데, 같은 체크포인트를 이 네이티브
환경에서 채점하면 **|B| 13.324% / 토크 9.207%**로 기준 13.32/9.20과
소수 셋째 자리까지 일치한다. 즉 평가 경로는 스택에 둔감하다.

비자명한 함정: physicsnemo의 MeshGraphNet은 `torch_scatter`를 요구하는데,
없으면 체크포인트를 **로드는 성공하고 0 샘플만 채점**한다(예외도 비정상
종료도 없이 `no samples`, exit 0). `check_runtime_contract.py`의 검사
목록에도 `torch_scatter`/`torch_geometric`이 없어 사전 점검을 통과한다.

### 학습

`results/logs/run_no_timefeat.sh`와 동일한 플래그를 네이티브에서 실행
(`results/logs/run_no_timefeat_native.ps1`, `.sh`가 `/workspace/app`을
하드코딩하기 때문). 60에폭 3,496초, 배치 4, seed 42.

노드 특징 **9개**(`MGN_NODE_FEATURES_V2`, time_s 없음), 파라미터
2,332,802개 — 기존 10특징 모델(2,332,930)보다 정확히 128개(1특징 ×
hidden 128) 적어 A/B가 격리됐음이 확인된다.

체크포인트에 저장된 학습 곡선을 같은 에폭끼리 비교하면:

    epoch    with time_s (train/val)   no time_s (train/val)
       10        24.93 / 39.40            23.75 / 36.84
       30        16.93 / 31.93            16.59 / 29.25
       50        14.43 / 27.03            14.17 / 25.25
      best        26.91 (ep51)             25.02 (ep55)

train 곡선이 매 에폭 1%p 이내로 겹친다. 컨테이너 학습과 네이티브 학습이
서로 다른 스택인데도 궤적이 사실상 같다는 뜻이다.

### 채점 결과

    curl:mgn_nodeB_long     |B| 13.324%   torque 9.207%   (with time_s)
    curl:mgn_nodeB_notime   |B| 13.016%   torque 8.240%   (no time_s)

지름길 특징을 제거했는데 두 지표가 모두 개선됐다. 토크는 상대 −10.5%.

### §9 판정 — 사이클 앞부분 오차 집중은 사라졌다

판정 전에 지표를 검증했다. §9가 발표한 12개 케이스 비율(4.44, 3.03,
3.00, 2.03, 1.20, 1.10, 1.05, 0.98, 0.80, 0.52, 0.46, 0.39), 전체 1.46,
학습 앞5스텝 21.22 N·m = 5.7%를 커밋된 `results/early_cycle_diagnosis.json`
에서 **전부 재현**했다. 같은 기준을 쓰고 있음이 확인된다.

대조군으로 with-time_s 체크포인트를 이 머신에서 재진단한 결과도
이전 머신과 세 자리까지 일치한다(case 4: 4.43 vs 4.44, 전체 1.461 vs
1.460, 학습 앞5스텝 21.29 vs 21.22 N·m). 평가 스택은 통제됐다.

    지표                     with time_s    no time_s
    전체 비율 (앞5/나머지)      1.461         1.054
    case_0004 비율              4.43          0.99
    비율 > 2.0 케이스           4 / 12        1 / 12
    학습 앞5스텝 오차          21.29 N·m     10.74 N·m
                              (5.72% 정격)   (2.89% 정격)

**권고 조치 2는 확인됐다.** 전체 비율 1.054는 앞부분과 나머지의 오차가
같아졌다는 뜻이고, 조사의 발단이던 case_0004의 4.43이 0.99로 평탄해졌다.

가장 직접적인 증거는 case_0004 step 0이다(§9가 +107.87을 보고한 지점):

    FEM          -372.4 N·m
    with time_s  -264.3        오차 +108.1   <- §9의 +107.87 재현
    no  time_s   -392.2        오차  -19.8   <- 5.5배 감소

`results/viz/case0004_step0_{withtime,notime}.png`로 시각화했다
(`tools/make_field_gif.py --case 4 --max-steps 1`로 재생성 가능).

**관찰 5는 부분적으로 뒤집힌다.** §9는 학습 오차 5.7%를 "별개의 용량
문제"로 분리하고 time_s 제거 후에도 남으리라 예상했으나, 실제로는
5.72% → 2.89%로 절반이 됐다. 지름길 특징이 과소적합의 상당 부분을
만들고 있었다. 다만 필드 B nRMSE는 14.4% → 13.9%로 거의 그대로여서,
time_s는 특히 **사이클 초반 토크**를 망가뜨리고 있었다는 해석이 맞다.

### 남는 것

1. **case_0032는 회귀했다.** 비율 1.05 → 2.02, 앞부분 오차 40.6 →
   86.7 N·m로 유일하게 남은 >2.0 케이스이자 최악 케이스가 됐다.
   전체 토크는 개선됐지만 이 케이스는 따로 봐야 한다.
2. **학습 스택은 통제되지 않았다.** 위 대조군이 증명한 것은 *평가*
   환경의 동등성이다. with-time_s는 여전히 컨테이너 학습이고
   no-time_s는 네이티브 학습이며, 조건당 런이 1개뿐이라 엄밀히는
   "time_s 제거 효과"와 "런 간 변동"이 분리되지 않는다. §9가 인과
   증거(step 0에서 time_s만 바꿔 토크가 −264→+48로 반전)를 갖고 있고
   효과 크기가 커서 결론이 뒤집힐 가능성은 낮지만, with-time_s를
   네이티브로 재학습하면 이 구멍이 닫힌다.

### 이 문서의 정정 사항 (실측)

- `mgn_nodeB_notime.pt`(이전 머신 산출물)는 MIGRATION.md가 3곳에서
  "epoch 20에서 중단"이라 하지만 파일 내용은 **epoch 34**다.
- MIGRATION.md는 체크포인트가 "best weights만, optimizer state 없음"
  이라 하지만 **`optimizer_state_dict`가 262개 엔트리로 존재**한다.
  없는 것은 resume *코드 경로*이지 상태 자체가 아니다. 문서를 믿고
  "resume 불가"로 판단하면 복구 가능한 상태를 버리게 된다.
- `mgn_nodeB_long.pt`의 최적 에폭은 51/60이다(미기재였다).

### 이식성 — 신선한 클론 감사에서 확인된 차단 요인

다른 머신에서 `git clone --recurse-submodules`를 실제로 실행해 확인:

    exit 128
    fatal: Fetched in submodule path 'eMach', but it did not contain 0fd6023d...

상위가 고정한 eMach 커밋 `0fd6023d4de2…`가 **원격 어느 ref에서도 도달
불가**하다(`upload-pack: not our ref`). 실존 커밋은 `0fd6023c75ff…`
(origin/develop)로 **8번째 16진수 한 글자만 다르다.** 도입 커밋 e8f26bc가
7자리 접두사만 기록해 두 커밋이 구분되지 않았다. MIGRATION.md에는 eMach
언급이 아예 없어 경고도 없다. 학습·채점은 eMach를 import하지 않으므로
무시하고 진행은 가능하지만, 스크립트화된 셋업은 여기서 정지한다.

또한 공유 폴더의 `mgn_nodeB_notime.pt`(epoch 34, sha256 `4ef2c475…`)와
새로 만든 `mgn_nodeB_notime.pt`(epoch 55, sha256 `6937a2e2…`)는 **같은
이름에 다른 가중치**이고, 스코어카드는 체크포인트 해시를 기록하지 않아
받는 쪽이 구분할 수 없다. 공유 전에 이름·매니페스트 규약이 필요하다.
