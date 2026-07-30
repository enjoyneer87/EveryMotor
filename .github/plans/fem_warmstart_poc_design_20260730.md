# FEM warm-start PoC 설계 (2026-07-30, 보고·승인 대기)

> Monumo식: 서로게이트가 예측한 nodal a_z를 비선형 2D 자기정 solve의 Newton
> 초기값 u₀로 넣어 반복수·시간을 줄이되, Newton이 오차를 보정하므로 FEM 정확도
> 보장은 유지. **§19에서 음성이던 target-A curl 체크포인트에 제 역할을 준다**
> — 그 nodal A 출력이 정확히 u₀. "실패 모델의 두 번째 삶."

## 1. 물리 문제
2D 자기정(자속 벡터포텐셜 a_z):

    −∇·(ν(B) ∇a_z) = j_z,   B = |∇a_z|

- **강판(스테이터·로터)**: ν(B) 비선형. `.bh` autofile에서 (H,B) 26점 →
  ν=H/B, 단조 보간 + 포화 tail 외삽.
- **공기·권선**: ν = 1/μ₀ (선형).
- **자석(V-IPM)**: recoil ν ≈ 1/(μ₀μ_r) + **잔류자속 Br 소스항**(∇×(ν Br) → RHS).
  자석 데이터는 Motor-CAD 자석 물성(eMach `converMCADMagnetTable.m` 참고) 또는
  .mot에서. **PoC 최대 모델링 난점** — 초기엔 자석을 Br 등가 소스로 포함.
- **경계조건**: 외곽 스테이터 호 = Dirichlet a_z=0; 두 반경 컷면 = **반주기**
  (1/8 섹터 45°, a(θ=0) = −a(θ=−45)). 우리 `cut_plane_pairs`/`anti_periodic_edges` 재사용.

## 2. 보유 vs 필요

**보유(우리 파이프라인):**
- 메시: 삼각형 12100, 노드 6282(`mesh/node_1/2/3`, `node_x/y_mm`), region code(요소별).
- 요소 면적 `eval/mesh_regions.element_areas_m2`, P1 gradient(curl_coef / 표준식
  ∇φ_i = 수직변/2A), 반주기 쌍 `cut_plane_pairs`.
- **소스 j_z**: `fields/j`(요소별, 45스텝). **참값 A**: `fields/a`(요소별→노드 변환
  `_element_to_nodal`). 둘 다 h5에 존재(FIELD_KEYS=(bx,by,a,j) 확인).
- **warm-start 후보**: §19 curl-A ckpt(nodal A 예측 = u₀).

**필요(신규 구현):**
- **ν(B) 파서**: `.bh` autofile 파싱(region 블록, `idx H B`, latin-1/errors=replace,
  비숫자 라인 skip — `pyMCAD/magnetic_parse.py:15-48` 인코딩 로직 참고). ν=H/B,
  B=0은 원점 기울기, 단조 큐빅(PCHIP) 보간, tail은 μ₀ 점근 선형 외삽. (기존 코드
  없음 — MATLAB은 raw H,B만. 소량 신규.) 강판은 케이스 불변(동일 NO18-1160) →
  .bh 1개 재사용.
- **FEM 조립 + Newton 루프**: 강성 K(ν(a)), 하중 f(j_z, 자석), 잔차 R=K·a−f,
  선형화(Newton J=K+dK 또는 Picard로 ν lag). Dirichlet + 반주기 구속.

## 3. 솔버 옵션 (4-way 비교)

| 옵션 | 비용 | 장점 | 단점 | 역할 |
|---|---|---|---|---|
| **(A) scipy 자체 Newton** | **낮음** | 전 machinery 보유, scipy.sparse+spsolve/cg, 정확성 최속 | CPU, 케이스별 직렬 | **Phase 1 PoC** |
| **(B) NVIDIA Warp** | 중 | **이미 컨테이너에 있음**(1.11.1). bsr_from_triplets·cg·gmres·**Tape(미분가능)**. GPU 배치 DOE + **Newton-in-the-loop 파인튜닝** 재개 | 커널 작성 학습곡선 | **Phase 2**(이득 확인 시) |
| (C) GetDP/ONELAB | 중 | EM 전용 레퍼런스, 네이티브 반주기 Constraint | Motor-CAD tri→.msh 변환, u₀ 주입 배선 | 외부 교차검증 |
| (D) Elmer | 중상 | 범용 다물리 | 셋업 무거움, EM엔 GetDP가 더 타깃 | 보류(GetDP로 충분) |

**권고 (내 판단):**
1. **Phase 1 = (A) scipy 자체 Newton** — 전 machinery가 손안에 있어 정확성 우선
   PoC를 가장 빨리 만든다. "curl-A u₀가 반복을 줄이나?"를 즉답.
2. **Phase 2 = (B) Warp 포트**(PoC가 이득 보이면). Warp가 scipy 수준의 희소·CG를
   GPU+**미분가능**으로 제공(프리미티브 컨테이너 확인 완료). → DOE 배치 솔브 +
   Monumo "Newton-in-the-loop" 파인튜닝(솔버 통과 그래디언트로 서로게이트 학습).
3. **(C) GetDP = 외부 검증**(선택). 우리 수렴해가 독립 EM 솔버와 일치하는지 1-2
   케이스 교차확인. mesh 변환 비용 있어 필수 아님.
4. **(D) Elmer 보류.**

## 4. 성공 지표 (정직한 판정)
동일 케이스·동일 수렴 허용오차(‖R‖/‖f‖ < 1e-6)에서 **비선형 반복수 + wall time**을
u₀별로 비교:
- **zero-init**(a₀=0) — 하한 baseline.
- **직전 회전각 warm-start**(전 스텝 수렴해) — 표준 실무 baseline, **curl-A가 이겨야
  가치 있음**(Monumo의 핵심 비교).
- **AI-init**(curl-A ckpt 예측 u₀).
목표: Monumo 15~30% 반복 절감. **curl-A가 직전각도 warm-start를 유의미하게 이기지
못하면 음성으로 보고**(정직).

## 5. 검증
- 솔버 자체 정합: 내 수렴 a_z ↔ Motor-CAD `fields/a`(요소→노드) 일치 확인(할인:
  discretization/BC 차이). 이걸로 솔버 정확성 먼저 확립한 뒤 warm-start 실험.
- 게이트/결정성과 무관(별개 모듈), 기존 학습 파이프라인 무영향.

## 6. 단계·비용
- **Phase 1(지금 착수, CPU)**: .bh 파서 + ν(B) → 조립/Newton → 1 케이스 검증 →
  u₀ 3종 반복수 벤치. curl-A u₀ 생성은 모델 추론(GPU 필요) → **R5 재학습이 GPU를
  비운 뒤**. 그 전엔 zero/직전각도/참값-A(good-u₀ proxy)로 솔버·벤치 골격 완성.
- **Phase 2(조건부)**: Warp 포트 + 배치 + Newton-in-the-loop.
- 컴퓨트 경량(Phase 1 CPU), 학습과 머신 공유 가능.

## 7. 정지점
설계 보고 후 진행(사용자가 "코드도 지금 시작" 승인) — Phase 1 구현을 병렬 착수,
AI-init 벤치는 GPU 여유 시. Phase 2/GetDP는 Phase 1 판정 후 별도 결정.

## 부록 — 앵커
- .bh: `D:\KDH\Sim_4SolverX\DOE_Ext\case_0000\TestCAD1\FEResultsData\Steel_Material_BH_Magnetic_Properties_Autofile.bh`(26점, Stator Code:1/Rotor Code:17, NO18-1160)
- 인코딩 참고 `eMach/tools/motorCAD/pyMCAD/magnetic_parse.py:15-48`; 자석 물성 `eMach/tools/varyToolCompatibility/converMCADMagnetTable.m`
- 우리 machinery: `eval/mesh_regions.element_areas_m2`, `phase1_static/curl_postproc.py`(gradient), `train_doe_curl_mgn._element_to_nodal`, `cut_plane_pairs`
- warm-start 모델: §19 curl-A ckpt `results/mgn_nodeB_r4_curl_spectral.pt`
- Warp: 컨테이너 1.11.1, `warp.sparse`(bsr)·`warp.optim.linear`(cg/gmres)·`wp.Tape`
