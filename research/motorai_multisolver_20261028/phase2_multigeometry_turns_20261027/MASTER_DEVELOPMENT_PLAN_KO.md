# EveryMotor MotorAI 전체 개발·검증 계획

계획 v1.0 · 기준 2026-10-28T02:05:00+09:00 · FROZEN_PLAN_V1_NOT_EXECUTION_RECEIPT

> **복원 기록:** Astra 턴 `01a17e3d…`은 2026-10-27T17:04Z 자체검증 PASS 뒤 Codex 사용량 한도로 멈춰 파일을 쓰지 못했다. Claude가 같은 JS 호출을 수정 없이 재생해 복원했다(23:28Z). 근거 해시는 재생 시점 값이다. E08/E24는 8903d3d 커밋 blob 해시이고, E21은 비공개 경로 대신 논리키로 표기했다. 08:00 KST 창은 종료되었다. 이후 사용자 결정으로 Claude가 Astra 판정과 Sol 실행을 겸하며, 진행은 `SOL_EXECUTION_LOG.json`에 기록한다.

이 문서는 Astra의 과학적 계획과 Sol의 실행 추적 기준이다. 현재 작성 범위는 이 문서와 동반 JSON 두 파일이다. WBS에 적힌 후속 산출물은 별도 표시가 없는 한 아직 생성되지 않았다. 장기 WBS는 08시 이후 실행 권한을 부여하지 않는다.

## 1. 지금 확정할 방향

기존 frozen 진단은 **완료**되었다. 32 innerdev × 45시각, 총 1440graph를 처리했고 저장된 guard receipt는 exit0·owned process 종료·checkpoint 불변을 기록한다. 직접 읽은 CSV 행수와 SHA도 확인했다. 준비·전송·추론을 다시 실행하지 않는다.

다음 과학 후보는 **동일 구조·동일 train328 split·동일 업데이트 수에서 phase9와 phase45를 비교**하는 것이다. Sol은 공통 평가와 실행계약을 먼저 작성한다. 신규 학습은 Astra가 구체적 계약을 검토해 release하기 전까지 HOLD다. B² loss, width/depth, global context를 첫 비교에 함께 바꾸지 않는다.

E1 weights/teacher는 세 admission UNKNOWN 때문에 HOLD다. 새 FEA, 4/8도체 field 확보, exact bridge 및 추가 모달 활성화는 각각 별도 관문 뒤에 둔다. 예약 형상 R1_0199/R1_0076과 alias는 pretrain/train/tune/eval 전 과정에서 제외한다.

## 2. 소유권·호스트·마감

- 저장소: enjoyneer87/EveryMotor
- branch: `codex/motorai-multisolver-20261028`
- 로컬 연구 worktree: `E:/PJT/EveryMotor/motorai-research-20261028`
- 연구 namespace: `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028`
- 계획·상태 기준 폴더: `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027`
- iso 연구 worktree: `C:/work/EveryMotor_motorai_research_20261028`; 기존 `C:/work/integrate/EveryMotor` 보존
- 이관 commit `cdaca004c66f9f2152f7ceee827b6481f270d32b`, source `65dda206e28770b242fa0dd888fd394f7a155ca0`, 467파일 SHA 일치는 사용자 전달 이관 기록이다. 본 계획 작성에서 Git 상태를 재검증하거나 변경하지 않았다.

Astra=전체 계획·과학적 주요 판정. Sol=구현·실행·검증·기존 승인 범위 Git. 이 계획은 기존 완료 실행의 재실행, 신규 학습/FEA 또는 마감 연장을 승인하지 않는다.

- local: relay·작은 문서/수집 metadata만, 학습·scoring 없음
- iso: 별도 연구 worktree의 실행 연결·commit
- host133: 대규모 GPU 학습·기존 frozen 추론
- host234: bounded read/preparation·작은 비교

저장소 em.sh/기존 bridge 사용. inline SSH 우회 없음.

Sol만 기존 승인된 연구 namespace를 iso에서 commit하고 relay에서 비강제 push. enjoyneer87, Enjoyneer87 <45963466+enjoyneer87@users.noreply.github.com>. JEET 신규 commit/push, main merge, history rewrite 없음. 기존 공개 JEET19 commit 보존; raw/checkpoint/private control 제외.

**마감 2026-10-28 08:00 KST, guard 07:59 KST(2026-10-27 22:59Z).** 07:30부터 정리한다. 현재 신규 scientific run release=0. 별도 승인된 run도 전체 wall cap+120초 정리 여유가 guard까지 남아야 시작. 07:30부터 문서·종료·부분결과 정리 우선. 기존 PAUSED 및 사용자 native app 보존.

STOP 또는 guard/wall/resource 위반 시 소유 identity 일치 작업만 중단. 종료 확인이 없으면 baton을 넘기지 않는다. 10GiB Torch allocated는 그래프 간 soft check이며 GPU hard quota가 아니다.

새 transient의 128step 제한은 **history당 총 step**이다. 두 개의 각128step history를 합256이라는 이유로 위반 처리하지 않는다. 이 계획의 새 학습 및 새 FEA 실행 승인 수는 각각0이며, 장기 예산·마감·자동화를 자동으로 늘리지 않는다.

## 3. 증거 수준별 현재 상태

**F01** EveryMotor DOE 480조건/선언360그룹은 45step moving transient, ProximityLossModel3·개별도체 mesh. magnetostatic 데이터로 분류하지 않는다. 440 HDF5 원본↔사본 hash 일치; initial40 native source 결손, CurrentAxis80 geometry 불일치 격리. [E15, E17, E20]

**F02** 현재 eligible360조건/320그룹 = train328조건/288그룹 + innerdev32조건/32그룹. dev는460Arms, 낮은115/230Arms는 train에만 존재. 20그룹이3OP이고 나머지는 단일OP. pilot16/4 역할 보존; current dev32는 old E1 TRAIN 노출. [E14, E15]

**F03** R1은 예약형상 제외272geometry ×4/6/8도체=816 geometry-only. field qualified 4/8 자료는 미확보. DOE6도체와 동일 geometry 집합이라고 간주하지 않는다. HalfSC32/4 별도 split 보존. [E07, E15]

**F04** SCALE360: from-scratch width32/depth3, 9/45시각,30epoch/88565update,3233.78초 완료. E1weights/AC/MSteacher 사용0, exact bridge0. devBvector18.076472622%, fixeddev4 16.603528484%; 이전pilot58.186% 대비 표본/업데이트/LR가 함께 바뀌어 단일 개선 원인 주장 불가. [E09, E10, E11]

**F05** E1 실제 target=B: nodalBx/By→element average. nodalA/curl baseline이라는 이전 설명은 철회. h256/p15≈9.29M, linear-solve prior+band spectral30+airgap5. legacy6 전체|B|11.096697%/airgap4.956319%/torque3.214573%는 현재 areaweighted vector와 다른 지표·집합. [E12, E20, E21]

**F06** E1 checkpoint/prior 존재는 확인되었으나 prior-native geometry/OP/mesh binding, 모든 donor reserved exclusion, normalizer compatibility가 UNKNOWN이므로 weights/features/teacher/distillation HOLD. clean code reuse는 별도 가능하나 prior 생성 linear solve를 no-FEA 작업으로 숨기지 않는다. [E12]

**F07** N128P1/N64P2는 각각 total128step numerical diagnostic. N512P3대비+19.403%/+3.752%는 formal mesh/time/period qualification이 아니다. 두 history 합256은 per-history128 위반이 아니다. frozen newOP8 중6완료/2SC미실행,5current/nonlinearPASS·1wholecurrentFAIL. SC12000 whole0.001101196735%>0.001%; 마지막period PASS가 전체실패를 대체하지 않는다. [E16, E18, E19]

**F08** frozen newOP6 AF3.2644678235%, scalar1.7819708147%, MS3seed mean0.9658%는 정해진 평가집합의 기록이다. 알려진2MSgroup, formal truth qualification UNKNOWN. 재튜닝/추가해석 없이 보존. AF는 f(omega)*kappa(I,beta)^p(omega)×canonical Hybrid spatial mean-square+skin excess; source B² 진단은 분모가 아니다. [E17, E18, E19]

**F09** 최신 동결 진단1440graph 완료, saved guard exit0·owner_live=false·checkpoint 동일. 직접 읽은 CSV 행수3456/1440 및 SHA 확인. 추가 GPU 실행 없이 기존결과를 계획에 반영. 새45phase 훈련/architecture 실험은 시작하지 않음. [E25, E26, E27, E28]

## 4. 완료된 frozen 진단과 D01-v1 판정

| 평가 범위 | B-vector nRMSE % | B magnitude nRMSE % |
|---|---:|---:|
| seen9 전체 | 18.076473 | 11.579548 |
| unused36 전체 | 20.915342 | 13.944028 |
| all45 전체 | 20.378967 | 13.504077 |
| all45 copper | 35.047435 | 27.464474 |

- 도체별 spatial meanB² all45 상대MAE: 14.480253%; signed bias: -9.458019%.
- truth9 대 truth45 B² sampling 상대MAE: 0.261877%; bias: 0.061240%.
- seen9 anchor 차이: 1.903343616049824e-8 percentage-point. 기존 0.0001pp 산술 허용오차 PASS이며 물리 PASS가 아니다.
- checkpoint SHA-256: `f47a32b665239ecb00479e8b66fe3707551ee15f38f2a6a3b5e07ab2d378ad6e`
- model SHA-256: `ae1966b5d660b8647ece32da3caf9af2e3dfd2ae11752738dfaba7d75e6663ed`

unused36의 Bvector20.9153%는 seen9의18.0765%보다 크다. all45 copperB² bias -9.4580%에 비해 truth9-vs45 sampling bias는+0.06124%로 작다. phase coverage 검토를 우선하되, 본 관찰은 인과증명/시간적분 AC/독립geometry final test가 아니다.

**D01-v1:** 다음 후보는 phase coverage만 바꾸는 same-architecture equal-update 비교 계약. B² loss·width·global context 동시변경 금지. 계약 작성은 즉시 가능, 새 학습은 별도 Astra release 전 HOLD.

수치 진단은 E25~E28에서 직접 읽었다. 저장 종료 receipt를 현재 호스트의 live process 조회로 표현하지 않는다. AC formal qualification은 UNKNOWN이다.

## 5. 단계별 진행표

완료 단계 W01/W02를 다시 시작하지 않는다. W08의 마감·중단은 의존 단계 완료를 기다리지 않는다. G04/G06/G07은 해당 가중치/paired/AC 기능을 사용할 때 적용하며, 미사용 기능의 gate를 이유로 독립적인 B-only 계약 작성을 막지 않는다.

| 단계 | 목적 | 선행 | 상태 | MI |
|---|---|---|---|---|
| W00 | 소유권·출처·현재 실행 상태 고정 | 없음 | SNAPSHOT_REVIEWED_LIVE_REFRESH_REQUIRED | MI-012, MI-014, MI-015 |
| W01 | 기존 frozen all45 진단 bundle 전송·입수 검증 | W00 | COMPLETED_RECEIPT_VERIFIED_NO_REPEAT | MI-004, MI-009, MI-010, MI-014 |
| W02 | 동결 SCALE360 32×45 단일 추론 | W01 | COMPLETED_ARITHMETIC_PASS_PHYSICS_UNQUALIFIED | MI-004, MI-009, MI-010, MI-002 |
| W03 | 진단 해석·전체 자계 개선 축 결정 | W02 | READY_REPORT_AND_NEXT_RUN_CONTRACT | MI-004, MI-009, MI-010, MI-011 |
| W04 | 동결 새OP6/8 반환과 품질층 보존 | W00 | EVIDENCE_COMPLETE_REPORT_ONLY | MI-012, MI-013 |
| W05 | E1 transfer admission의 세 미확인 gate 해소 감사 | W00 | HOLD_WEIGHTS_READ_ONLY_AUDIT_ALLOWED | MI-003, MI-005, MI-012 |
| W06 | 기존 4/8도체 Field·초기40·CurrentAxis80 계보 탐색 | W00 | BOUNDED_METADATA_AUDIT_ONLY | MI-001, MI-007, MI-008 |
| W07 | 공통 평가·노출 감사·비용 ledger 계약 | W00 | CONTRACT_TO_FREEZE | MI-003, MI-004, MI-009, MI-012 |
| W08 | 07:59 종료·소유권 확인·EveryMotor 인계 | W00 | REQUIRED_AT_DEADLINE | MI-012, MI-014 |
| L10 | E1 코드·recipe clean-room 재사용과 common eval 통합 | W05, W07 | PLANNED_NOT_LAUNCHED | MI-003, MI-004, MI-005 |
| L11 | E1 frozen backbone 전이 micro-experiment | W05, W07 | HOLD_ADMISSION_AND_ASTRA_RELEASE | MI-004, MI-005, MI-012 |
| L12 | field 표현·용량·위상 개선의 단일축 실험 | W03, W07 | PHASE_COVERAGE_CONTRACT_FIRST_NO_RUN_RELEASED | MI-004, MI-010, MI-011 |
| L13 | J/Je·도체·회로·손실 의미 계약 | W06, W07 | UNKNOWN_MODALITIES_MASKED | MI-001, MI-006, MI-009 |
| L14 | 다중 template·4/6/8도체 native configuration 계약 | W06, W07 | GEOMETRY_ONLY_FIELDS_MISSING | MI-007, MI-008 |
| L15 | MS/Full 및 solver 내부 fidelity 최소 보강 | L13, L14 | NO_NEW_FEA_AUTHORIZED_BY_MASTER | MI-007, MI-008, MI-009, MI-012 |
| L16 | 최소 exact cross-solver bridge | L13, L14, L15 | EXACT_BRIDGE_COUNT_ZERO_HOLD_ALIGNMENT | MI-006, MI-007, MI-012 |
| L17 | 두 solver·field/scalar 멀티모달 학습 | W07, L13 | DESIGN_ONLY_MASK_UNQUALIFIED_MODALITIES | MI-005, MI-006, MI-009 |
| L18 | AF 대비 공정한 AC 손실 benchmark | W07, L13, L17 | FUTURE_BENCHMARK_NOT_FROZEN_TEST_RETUNING | MI-009, MI-012, MI-013 |
| L19 | lineup surrogate 기반 설계 선택·최적화 | L14, L18 | DOWNSTREAM_NOT_STARTED | MI-004, MI-006, MI-007, MI-009, MI-012 |

## 6. 관문과 판정 책임

**G01 소유권·출처** — 적용: ALL

research namespace·branch·source hash·실행 소유 PID/start-time 또는 container ID 일치. saved receipt는 live 조회와 구분.

미충족 처리: MISMATCH_HOLD

**G02 계보·분할·예약 형상** — 적용: ALL_DATA_USE

R1_0199/R1_0076 및 보수적 alias를 모든 donor/pretrain/train/tune/eval에서 제외. native↔manifest 형상/OP 일치; initial40와 CurrentAxis80 미복구자료 격리; 그룹 단위 분할 고정.

미충족 처리: UNKNOWN_HOLD_AFFECTED_DATA

**G03 물리량·지원점·정밀도** — 적용: EACH_MODALITY

B는 native element Bx/By T. A/J/Je는 support·unit·physical meaning·phase·frame·material·mesh/time/initialization fidelity 확인 후 개별 활성화. elementA를 nodalA로 취급하지 않음.

미충족 처리: MASK_UNQUALIFIED_MODALITY

**G04 E1 가중치·teacher 입수** — 적용: ONLY_E1_WEIGHTS_FEATURES_DISTILLATION

prior-native geometry/OP/mesh binding, donor reserved exclusion, normalizer compatibility 3개 PASS. target=B인 실제 recipe와 layer/channel mapping 검증. 원 checkpoint 존재만으로 통과 불가.

미충족 처리: HOLD_WEIGHTS

**G05 노출과 전이 효익 주장** — 적용: ANY_PRETRAINED_ARM

기존 gradient/tuning/eval 전체 노출 ledger. 현 dev32는 oldTRAIN이므로 E1 독립 형상 시험 주장 불가. 깨끗한 미노출 평가군 별도 동결.

미충족 처리: LIMIT_CLAIM_OR_HOLD

**G06 정확한 cross-solver pair** — 적용: ONLY_PAIRED_ALIGNMENT_OR_PAIRED_TRUTH

exact geometry/material/circuit/OP/phase/frame/fidelity·projection 검증. nominal/near는 exact 아님. exact count0이면 paired loss weight0; unpaired domain training은 다른 gate 충족 시 별도 가능.

미충족 처리: DISABLE_PAIRED_ALIGNMENT

**G07 AC 손실·분모** — 적용: ANY_AC_LABEL_BASELINE_OR_LOSS

f(omega)*kappa(I,beta)^p(omega)와 canonical Hybrid mean-square 분모, skin excess, solver별 보정/version/적분window/Full provenance를 묶음으로 고정. MS prediction/derivedB²를 Full label로 세지 않음.

미충족 처리: AC_QUALIFICATION_UNKNOWN

**G08 새 native 해석** — 적용: ONLY_NEW_FEA

형상·회로·drawn conductor count·mesh·material 검증, 별도 case 수/총비용 계약. 새 transient 각 history 총128step 상한; campaign 합128이 아님. formal 수렴 미확인 truth는 diagnostic.

미충족 처리: NO_NEW_FEA_RELEASED

**G09 실행·중단** — 적용: EVERY_RUN

host/owner/입출력 hash/wall/메모리/동시성/STOP/절대마감/재시도0 사전선언. 소유 작업만 중단, 종료 미확인이면 다음 baton 금지.

미충족 처리: STOP_AND_PRESERVE

**G10 공정 평가·주장** — 적용: EVERY_COMPARISON

같은 ID/split/support/평가 mask/지표/예산을 고정; 실패·결측도 보고. 산술 재현, innerdev 경향, 물리/최종독립 검증을 분리. accuracy threshold는 실행 전 사전등록.

미충족 처리: NO_RETROACTIVE_PASS

**D00 hash·ownership·state 충돌** — 영향받은 단계만 HOLD, 최신 receipt와 충돌 목록 제출 (판정: Astra)

**D01 전체 자계 다음 실험** — v1 방향은 phase9→45 equal-update 비교 계약 우선. 새 학습은 예산·마스크·seed 계약 승인 전 HOLD (판정: Astra)

**D02 예약 alias·노출·분할 불명** — 해당 donor/case 격리. UNKNOWN을 통과로 바꾸지 않음 (판정: Astra)

**D03 E1 admission 해소 및 전이 release** — 3개 HOLD gate 모두 PASS 근거와 비교계약 제출; teacher/distillation도 동일 (판정: Astra)

**D04 새 field/4·8도체/native run 필요** — 물리 정의·기존자료 부족 근거·최소 획득안 제출. 이 계획으로 새 FEA 발사 금지 (판정: Astra; 사용자 승인 범위 밖 예산은 사용자)

**D05 새 모달/paired alignment 활성화** — semantics/native reintegration 또는 exact bridge evidence 후 해당 mask만 해제 (판정: Astra)

**D06 지표·허용오차·공정예산 변경** — 원 계약 결과 보존, 새 계약에서 delta/no-harm/비용 기준을 사전등록 (판정: Astra)

**D07 범위·예산·마감 확장** — 08시 이후 지속/새 자동화/새 총예산은 사용자 명시 지시 필요. Astra는 이를 자동 승인하지 않음 (판정: 사용자)

**D08 실행 실패·중단·종료 불명** — 자동 재시도0, verified owned process만 중단, 부분결과/identity 보존; 다음 작업 HOLD (판정: Sol 즉시 보호 조치, Astra에 근거 제출)

## 7. 공통 평가·노출·비용 계약

상태: W07에서 실행 전 고정할 계약; 기존 진단의 지표를 소급 변경하지 않음

주 지표: 전체 B-vector area-weighted nRMSE; 전체 |B| area-weighted nRMSE. 보조 지표: airgap·copper 별 vector/magnitude/절대RMSE; 도체별 spatial meanB² 오차/bias; 형상별 최악오차/실패율; qualified AC loss/라인업 의사결정.

- vector: `100*sqrt(sum(area*||Bpred-Btrue||^2)/sum(area*||Btrue||^2))`
- magnitude: `100*sqrt(sum(area*(|Bpred|-|Btrue|)^2)/sum(area*|Btrue|^2))`
- spatial_mean_B2: `Q=sum(area*|B|^2)/sum(area)`
- square_of_mean_B: `R=||sum(area*B)/sum(area)||^2`
- consistency: `Q-R>=0 within declared floating tolerance; Q != R`
- zero_denominator: `분모0/near0 규칙을 사전동결하고 절대RMSE 병기. 사후 epsilon 선택 금지.`

case/geometry·영역·시각의 weight와 pooled 대 macro 값을 명시. 현재 diagnostic은 area + uniform snapshot pooling, 시간가중 적분 아님. legacy E1 unweighted flattened magnitude는 이력으로 보존하고 공통 metric 별도 계산.

모델 공통 고정 평가 ID/물리영역·support mask. 실패/NaN으로 어려운 표본을 모델별 제거하지 않음. true-dependent 입력 금지.

동일 geometry family·aliases는 같은 split. OP/phase는 그룹으로 관리. dev32는 개발용이고 dev의 unused36은 phase 일반화 진단만 제공한다. E1 oldTRAIN과 겹치므로 E1 최종독립 시험 불가. 예약2형상/alias는 최종시험에서도 제외.

과학적 정확도 PASS, 최소개선 delta, 영역별 비악화 margin은 **UNKNOWN(null)**이다. 운영 요구와 공통평가를 확정한 뒤 실행 전에 사전등록한다. 진단결과를 보고 새 PASS 숫자를 소급 생성하지 않는다. 그룹 단위 불확실성/실패와 전체 vector·magnitude 및 copper/airgap/B²를 함께 판단. 정해지지 않은 no-harm margin을 사후 임의로 채워 승자를 선언하지 않는다.

raw label 생성 총비용·기존 pretraining sunk cost·이번 incremental 비용을 분리. unique geometry/OP/phase/Full해석 수와 group-union 노출을 중복 없이 기록. 같은 label budget과 same wall/update 비교는 구분하여 보고.

## 8. 전이학습·멀티모달·새 데이터 획득 계약

**E1 전이:** legacy450train/12val/18test 전부 별도 ledger. legacy6 source missing40, 다른12test CurrentAxis80 mismatch. 480 전체 접근이력을 무시하지 않음.

현재 허용: recipe/channel/normalizer/prior metadata 감사 및 clean recipe 계획. 가중치 사용 없음.

admission 이후 후보: 동일 train-only74/98 후보와9시각 micro계약, originalE1/frozenfeatures+head/randomfrozen+head 비교; 동일 head init/order/LR/update와 feature normalization. 현 dev를 독립시험으로 포장하지 않음.

linear-solve prior의 출처·물리량·비용·생성조건도 모달/계산이다. native Full target의 변환을 prior 입력으로 사용하지 않는다.

E1 checkpoint SHA-256 `f58636620d5601f202d0203bff06d3f29c601e2885811c271edded28123bcba7` (epoch47,111835629bytes). 존재 확인은 전이 승인과 다르다.

**멀티모달:** shared target-free geometry/OP representation + solver-specific encoder/adapter/normalizer/decoder + 별도 mesh/time/initialization fidelity conditioning. Motor-CAD, Elmer, GetDP identity를 혼동하지 않음.

사용 가능한 모달: qualified native element Bx/By [T]. |B|와 B²는 파생량이며 독립 label 수로 세지 않는다. A support/gauge; J/Je total/source/induced current; Joule/AC loss and canonical denominator; cross-solver paired alignment는 개별 검증 전 mask한다.

sum_domain sum_modality lambda[d,m]*mask[d,m]*normalized_loss[d,m] + lambda_pair*mask_exact*L_pair. domain/modality count normalization과 weight는 train-only 사전동결. 정확pair0이면 lambda_pair=0; qualifiedAC없으면 lambda_AC=0.

Je440 존재/단위 A/mm²만으로 total/induced 의미확정 불가. A/mm²→A/m²는1e6이지만 물리정의와 지원점이 먼저다. native integrated reference와 단위/영역/axial length/time-window를 대조.

정답FullB·그 정답에서 파생된B²/손실을 입력으로 넣지 않음. geometry/OP만의 운영모드와 실제 관측모달이 있는 모드는 별도 선언.

비교 arm: clean B-only → admitted E1 warm-start → qualified unpaired multi-domain B → qualified current/loss modalities → exact-bridge alignment when available.

AF/scalar/MS/Full/MM 비교는 동일 frozen baseline versions와 새로 잠근 평가셋. donor predictions는 label이 아니며 Ref 보정계수를 타 solver 분모에 혼용하지 않음.

**획득 순서:** 4/6/8도체 template coverage→geometry/native audit→최소 field 획득→mesh/time/period qualification→exact bridge→증분확대

4/6/8 도체 equalAT 후보전류는375/250/187.5Arms다. 회로·parallel path 확인 전 제안값. equalAT가 equalcopperarea/currentdensity/thermal 비교를 보장하지 않음. 새 FEA case 예산은 UNKNOWN이며 승인0이다. 한 history128step 안에 수렴 불가하면 unqualified diagnostic으로 남기고 자동 연장하지 않음. 필요 추가 예산/기간은 D07.

남은 결손: 4/8 native drawn conductor count와mesh; reserved-free native source40 복구 가능성; CurrentAxis80 geometry/manifest repair; J/Je semantics; exact crosssolver commongeometry availability; formal convergence and new independent final set.

## 9. WBS 실행·완료 계약 상세

모든 단계의 planning/scientific acceptance는 Astra, implementation/execution/verification은 Sol이다. 아래 산출물 중 PROPOSED_NOT_CREATED는 계획 이름만 제시한 것이다. limits의 null은 무제한이 아니라 실행계약 미확정이다.

### W00. 소유권·출처·현재 실행 상태 고정

상태: SNAPSHOT_REVIEWED_LIVE_REFRESH_REQUIRED; 범위: CURRENT_WINDOW; 우선순위: P0.

선행: 없음 · 입력: E01, E02, E03, E04, E05, E08, E11, E12 · 근거: E01, E02, E03, E04, E05, E08, E11, E12 · gate: G01, G09 · Astra 판정: D00, D08.

**할 일**

- Sol은 현재 owned task/PID/start-time 또는 container ID와 기존 배포 hash를 먼저 확인한다. 저장 STATE를 live process proof로 쓰지 않는다.
- EveryMotor 독립 연구 worktree를 사용하고 canonical source branch, JEET 19공개 커밋, main을 변경하지 않는다.
- 최신 MI-010은 동결 평가 완료, MI-013은 6건 회수, MI-015는 계획 수립 중이다. 기존 STATE/CONTINUATION과 시각이 다르면 E25/E26와 최신 receipt를 우선하며 과거 이력은 보존한다.

**완료 기준**

- 소유 작업의 RUNNING/COMPLETED/FAILED/NOT_STARTED와 근거 시각이 구분됨
- 현재 inference가 이미 시작/완료했으면 W02를 새로 발사하지 않음

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/OWNERSHIP_AND_ACTIVE_RUN_SNAPSHOT.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: bounded_read_only_audit; host: 234 for remote sources; local relay for saved text/metadata only; wall_seconds_max: 600; processes_max: 1; GPU_jobs: 0; new_FEA_runs: 0; training_updates: 0; automatic_run_repeats: 0; memory_estimate_GiB: 2; output_estimate_GiB: 1.

### W01. 기존 frozen all45 진단 bundle 전송·입수 검증

상태: COMPLETED_RECEIPT_VERIFIED_NO_REPEAT; 범위: CURRENT_WINDOW; 우선순위: P0.

선행: W00 · 입력: E13, E14, E26 · 근거: E13, E14, E26 · gate: G01, G02, G09 · Astra 판정: D00, D01, D08.

**할 일**

- E26에서 기존 전송·배포와 bundle/model/plan/selection hash가 확인되었다. 전송·준비 재실행은 없다.
- 원본 receipt를 결과 해시 목록에 포함하고 raw bundle은 Git에 넣지 않는다.

**완료 기준**

- 133에서 실제 bundle와 checkpoint의 hash·크기·manifest 검증 receipt 확보
- 모델 source hash가 ae196...와 일치하고 frozen 모델 변경 없음

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_FIELD_DIAGNOSTIC_STATUS.json` — EXISTING_READ_VERIFIED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: existing_plan_transfer_and_hash_verification; host: relay to 133; wall_seconds_max: 1800; processes_max: 1; GPU_jobs: 0; new_FEA_runs: 0; training_updates: 0; automatic_run_repeats: 0; memory_estimate_GiB: 2; output_estimate_GiB: 1; prep_reruns_authorized: 0; disk_free_required_GiB: 6; run_count_remaining: 0.

### W02. 동결 SCALE360 32×45 단일 추론

상태: COMPLETED_ARITHMETIC_PASS_PHYSICS_UNQUALIFIED; 범위: CURRENT_WINDOW; 우선순위: P0.

선행: W01 · 입력: E13, E22, E10, E11, E14, E25, E26, E27, E28 · 근거: E13, E22, E10, E11, E14, E25, E26, E27, E28 · gate: G01, G02, G09, G10 · Astra 판정: D01, D08.

**할 일**

- 기존 사전선언 진단 1440graph 완료: exit0, 저장 receipt의 owned_process_live=false, checkpoint hash 동일. 새로운 추론 실행은 없다.
- seen9 anchor 차이 1.90334e-8 percentage-point는 기존 0.0001pp 산술 허용오차 PASS. 물리 정확도나 AC 승인으로 확대하지 않는다.
- E25/E26의 저장 종료근거와 현재 live 상태를 구분한다. 신규 작업은 이 완료 작업을 재사용해 발사하지 않는다.

**완료 기준**

- 1440graph 모두 처리, training_updates=0, new_FEA=0
- seen9 산술 anchor PASS 또는 실패 이유가 보존됨
- 결과·guard·checkpoint hash와 종료 확인 확보

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_FIELD_DIAGNOSTIC_RESULT.json` — EXISTING_READ_VERIFIED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_FIELD_DIAGNOSTIC_STATUS.json` — EXISTING_READ_VERIFIED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_DIAGNOSTIC_conductor_B2.csv` — EXISTING_READ_VERIFIED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_DIAGNOSTIC_snapshot_errors.csv` — EXISTING_READ_VERIFIED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: EXISTING_PREDECLARED_INFERENCE_ONLY; host: 133; run_count_max: 1; GPU_jobs: 1; wall_seconds_max: 1800; training_updates: 0; new_FEA_runs: 0; torch_allocated_soft_limit_GiB: 10; memory_estimate_GiB: 4; disk_estimate_GiB: 6; source_plan: E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_FIELD_DIAGNOSTIC_PLAN.json; run_count_remaining: 0.

### W03. 진단 해석·전체 자계 개선 축 결정

상태: READY_REPORT_AND_NEXT_RUN_CONTRACT; 범위: CURRENT_WINDOW; 우선순위: P0.

선행: W02 · 입력: E09, E10, E13, E22, E25, E27, E28 · 근거: E09, E10, E13, E22, E25, E27, E28 · gate: G03, G10 · Astra 판정: D01, D06.

**할 일**

- Sol은 고정 dev4와 dev32, 9/36/45 시각, 영역·형상별 절대/상대오차와 구리 B² bias를 같은 표로 제출한다.
- uniform snapshot pooling을 시간적분·AC loss·canonical Hybrid 분모로 바꾸지 않는다. 구리 B² 파생량은 독립 label로 세지 않는다.
- Astra는 위상 누락, 구리영역 편향, 전체 표현/최적화 문제의 근거를 구분한다. 수치 오차가 크다는 이유만으로 실패한 solver/새 FEA를 발사하지 않는다.
- 공통 전체 |B|/vector 개선과 구리/공극 비악화의 과학적 threshold는 별도 사전등록한다. 현 진단에는 새 accuracy PASS threshold를 소급 적용하지 않는다.
- D01-v1 판정: 다음 과학 후보는 동일 구조·동일 train328 split에서 9→45 위상 노출 범위만 바꾸는 통제 비교이다. update 수/optimizer/LR/seed/평가 마스크를 고정한 실행계약을 먼저 제시한다. 결과만으로 위상 범위가 원인이라고 단정하지 않는다.
- 구리 B² all45 bias -9.4580%는 truth9-vs45 sampling bias +0.06124%보다 크므로 단순 9시각 적분 오차로 설명되었다고 하지 않는다. B² loss 추가와 global context/width 변경은 첫 비교에서 제외한다.

**완료 기준**

- D01이 우선순위와 허용 claim/남은 UNKNOWN을 기록
- 후속 실험이 필요한 경우 별도 실행 계약 전까지 HOLD

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/FROZEN_FIELD_DIAGNOSTIC_REVIEW_KO.md` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/DECISION_D01_FIELD_DIRECTION.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: bounded_read_only_audit; host: 234 for remote sources; local relay for saved text/metadata only; wall_seconds_max: 1200; processes_max: 1; GPU_jobs: 0; new_FEA_runs: 0; training_updates: 0; automatic_run_repeats: 0; memory_estimate_GiB: 2; output_estimate_GiB: 1.

### W04. 동결 새OP6/8 반환과 품질층 보존

상태: EVIDENCE_COMPLETE_REPORT_ONLY; 범위: CURRENT_WINDOW; 우선순위: P0.

선행: W00 · 입력: E17, E18, E19 · 근거: E17, E18, E19 · gate: G07, G10 · Astra 판정: D06, D07.

**할 일**

- Full63·8방법·예측·seed를 바꾸지 않고 6회수/SC2미실행, whole-current PASS5/FAIL1을 보존한다.
- 전체6 포함 wMAE와 품질 통과 보조 subset을 분리한다. SC 12000rpm P2 PASS가 whole-history FAIL을 취소하지 않는다.
- 독립 손실채널·formal mesh/time/period UNKNOWN, known2MSgroup 범위를 보고한다. 누락2건을 실패 label 또는 zero loss로 만들지 않는다.

**완료 기준**

- 6/8 및 품질/수렴 범위·hash가 일치
- 새 tuning/추가 SC 실행 없음

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/FROZEN_NEWOP_FINAL_SCOPE_KO.md` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: bounded_read_only_audit; host: 234 for remote sources; local relay for saved text/metadata only; wall_seconds_max: 900; processes_max: 1; GPU_jobs: 0; new_FEA_runs: 0; training_updates: 0; automatic_run_repeats: 0; memory_estimate_GiB: 2; output_estimate_GiB: 1.

### W05. E1 transfer admission의 세 미확인 gate 해소 감사

상태: HOLD_WEIGHTS_READ_ONLY_AUDIT_ALLOWED; 범위: CURRENT_WINDOW; 우선순위: P0.

선행: W00 · 입력: E07, E12, E15, E20, E21 · 근거: E07, E12, E15, E20, E21 · gate: G02, G04, G05 · Astra 판정: D02, D03.

**할 일**

- checkpoint와 r3/prior source 존재 PASS는 유지하되 prior-native geometry/OP/mesh binding, 전체 donor reserved 제외, feature-normalizer compatibility를 각각 감사한다.
- 원본 E1은 target=B nodal Bx/By→element 평균, h256/p15, prior+band spectral30+airgap5이다. 파일명 curl만으로 A-head라 부르지 않는다.
- 원래 normalizer/feature 순서/output scale와 node/edge support를 model metadata에 연결한다. 부족한 prior를 새 선형 solve로 생성하지 않는다.
- 초기40 source UNKNOWN을 포함한 ancestor 데이터 노출과 R1_0199/R1_0076 alias를 확인한다. 누락/애매함은 UNKNOWN을 유지한다.

**완료 기준**

- 세 gate의 증거 경로/hash와 PASS/FAIL/UNKNOWN이 각각 기록
- Astra D03 없이는 weights/teacher/distillation 실행 없음

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/TRANSFER_ADMISSION_DELTA.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/E1_MODULE_COMPATIBILITY.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: bounded_read_only_audit; host: 234 for remote sources; local relay for saved text/metadata only; wall_seconds_max: 1800; processes_max: 1; GPU_jobs: 0; new_FEA_runs: 0; training_updates: 0; automatic_run_repeats: 0; memory_estimate_GiB: 2; output_estimate_GiB: 1.

### W06. 기존 4/8도체 Field·초기40·CurrentAxis80 계보 탐색

상태: BOUNDED_METADATA_AUDIT_ONLY; 범위: CURRENT_WINDOW; 우선순위: P1.

선행: W00 · 입력: E15, E20, E05 · 근거: E15, E20, E05 · gate: G02, G03 · Astra 판정: D02, D04.

**할 일**

- 알려진 source/private/NAS manifest부터 scoped metadata 검색한다. 전 드라이브 무제한 조사·source 덮어쓰기·archive 이동은 하지 않는다.
- 4/8 actual drawn conductor count, 회로·치수·메시·solver·OP·raw hash를 확보한 기존 field만 후보로 등록한다. 슬롯당 도체수를 검증 없는 직렬 턴수로 치환하지 않는다.
- initial40는 원래 메시의 TXT/MES/HDF5 회수를 우선한다. 다른 메시 재해석본은 별도 사례로 두며 기존 field를 대체하지 않는다.
- CurrentAxis80은 저장 .mot가 실제 field 생성 input인지 확인한 후 별도 dataset revision을 제안한다. 현 SCALE360 split/결과를 소급 수정하지 않는다.

**완료 기준**

- 찾은 자료와 찾지 못한 범위를 구분한 후보 inventory
- 신규 독립 형상 수/조건 수/도체수와 중첩·alias를 분리

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/EXISTING_FIELD_GAPS_AND_RECOVERY.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: bounded_read_only_audit; host: 234 for remote sources; local relay for saved text/metadata only; wall_seconds_max: 1800; processes_max: 1; GPU_jobs: 0; new_FEA_runs: 0; training_updates: 0; automatic_run_repeats: 0; memory_estimate_GiB: 2; output_estimate_GiB: 1.

### W07. 공통 평가·노출 감사·비용 ledger 계약

상태: CONTRACT_TO_FREEZE; 범위: CURRENT_WINDOW; 우선순위: P0.

선행: W00 · 입력: E02, E07, E14, E15, E17, E18, E20 · 근거: E02, E07, E14, E15, E17, E18, E20 · gate: G02, G03, G07, G10 · Astra 판정: D02, D06.

**할 일**

- 모델별 source support 차이를 native element의 공통 유효영역/좌표계/단위/위상/가중치에 맞춰 비교하도록 eval/ adapter 계획을 고정한다. 기존 imported 점수는 scoped diagnostic으로 유지한다.
- 450 train/12 validation/18 test의 역사적 E1 access와 현재328 train의 중첩을 원천키로 deduplicate한다. 실제 compute cost와 label-access count는 구분한다.
- 현재 dev32의 E1 pretraining 노출과 old18의 initial40/CurrentAxis80 계보 문제를 명시한다. 현재 old val/test payload는 새 승인 없이 열지 않는다.
- 전체 |B|/vector, 공극, 구리, B², AC·설계선택 지표를 묶되 물리 gate와 arithmetic gate, operational gate를 분리한다. 후보 선택/효과크기/비악화 margin과 final split은 결과 열람 전에 Astra가 확정한다.

**완료 기준**

- 모든 비교 arm에 공통 evaluator·support·budget·exposure ledger 정의
- 미확정 science margin은 null/UNKNOWN이며 PASS를 생성하지 않음

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/COMMON_EVALUATION_CONTRACT.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/EXPOSURE_AND_COST_LEDGER.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/COMMON_EVALUATION_PLAN_KO.md` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: bounded_read_only_audit; host: 234 for remote sources; local relay for saved text/metadata only; wall_seconds_max: 1800; processes_max: 1; GPU_jobs: 0; new_FEA_runs: 0; training_updates: 0; automatic_run_repeats: 0; memory_estimate_GiB: 2; output_estimate_GiB: 1.

### W08. 07:59 종료·소유권 확인·EveryMotor 인계

상태: REQUIRED_AT_DEADLINE; 범위: CURRENT_WINDOW; 우선순위: P0.

선행: W00 · 입력: E03, E05, E11, E13 · 근거: E03, E05, E11, E13 · gate: G01, G09, G10 · Astra 판정: D08.

**할 일**

- 마감은 다른 단계 완료와 무관하게 우선한다. 22:59Z/07:59KST에 verified owned compute를 멈추고 23:00Z/08:00KST 이후 새 계산을 금지한다.
- 부분 결과·실패 receipt·checkpoint·원본 해시를 보존한다. owner가 다르거나 종료확인이 실패하면 임의 kill/baton 해제 대신 Astra에 보고한다.
- Sol은 이미 승인된 EveryMotor 연구 branch에서만 iso commit·relay nonforce push를 수행한다. 계정/author·branch/diff·raw 제외 확인; main merge/JEET 신규commit/history rewrite 금지.
- 기존 PAUSED 예약과 다른 캠페인은 유지한다. 새 자동화·마감연장·budget 증액을 이 계획으로 생성하지 않는다.

**완료 기준**

- 각 단계 완료/미완료/HOLD와 미해결 gate가 명시됨
- owned process 종료 또는 확인불가가 정직하게 기록
- 소유 문서/code 변경만 게시되며 native raw/private mappings 미게시

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/WINDOW_CLOSEOUT_20261028.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/WINDOW_CLOSEOUT_20261028_KO.md` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: closeout_documentation_and_previously_authorized_Git; host: 234 for remote sources; local relay for saved text/metadata only; wall_seconds_max: 900; processes_max: 1; GPU_jobs: 0; new_FEA_runs: 0; training_updates: 0; automatic_run_repeats: 0; memory_estimate_GiB: 2; output_estimate_GiB: 1; trigger: 07:30KST closeout preparation; owned compute stop no later than07:59KST, independent of other dependencies; new_launches_after_guard: 0; dependencies_do_not_delay_stop: true.

### L10. E1 코드·recipe clean-room 재사용과 common eval 통합

상태: PLANNED_NOT_LAUNCHED; 범위: LONG_TERM_OR_SEPARATE_APPROVED_WINDOW; 우선순위: P0.

선행: W05, W07 · 입력: E07, E12, E20, E21 · 근거: E07, E12, E20, E21 · gate: G02, G03, G04, G10 · Astra 판정: D03, D06.

**할 일**

- admitted manifest만 로드하고 새 train 통계·from-scratch weights로 시작한다. old split을 default로 호출하지 않는다.
- target=B 경로와 E1 loss를 재사용하는 arm, target=A/P1 curl 대안은 별도 실험으로 분리한다. 후자를 E1 재현이라 부르지 않는다.
- legacy prior 캐시가 물리입력·native source와 연결되지 않으면 E1 동일 recipe 학습을 HOLD한다. prior를 뺀 변형은 명명·비용을 별도로 기록한다.
- native graph mapping, input contract, train-only normalizer, batch native ID offset, finite loss fail-fast를 검증하고 eval/에 adapter를 통합한다.

**완료 기준**

- 기존 모델 재현 경로와 새 common-split baseline이 혼동되지 않음
- 고정 데이터/metric·실제 update/time/param count로 비교

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/E1_RECIPE_ADAPTATION_PLAN.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/E1_CLEAN_BASELINE_COMPARISON.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none..

### L11. E1 frozen backbone 전이 micro-experiment

상태: HOLD_ADMISSION_AND_ASTRA_RELEASE; 범위: CONDITIONAL_CURRENT_WINDOW_ELSE_LONG_TERM; 우선순위: P0.

선행: W05, W07 · 입력: E07, E12, E14 · 근거: E07, E12, E14 · gate: G02, G04, G05, G09, G10 · Astra 판정: D03, D08.

**할 일**

- 전체 donor reserved 제외·prior-native binding·normalizer 호환 PASS 및 D03의 서면 과학 판정 후 별도 bounded run을 선언한다.
- E1 원본 예측, frozen E1 feature+작은 residual head, random frozen feature+동일 head를 같은 초기 head/순서/LR/update로 비교한다. exact state_dict/feature 계약을 유지한다.
- 원 nodal graph를 유지한다. element width32 모델에 shape만 맞춰 직접 주입하지 않는다. teacher와 distillation은 같은 admission을 상속한다.
- train-only2 사례의 적합/증분비용 진단이며 dev32/old18의 독립 일반화 주장은 금지한다. 여유시간 부족하면 시작하지 않는다.

**완료 기준**

- 학습된 표현 재사용의 train-only 적합/비용 증거 확보 또는 명시적 HOLD
- 무노출 final test가 없음을 보존

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/TRANSFER_MICRO_RUN_PLAN.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/TRANSFER_MICRO_RESULT.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none.; host: 133; candidate_wall_seconds_max: 3600; candidate_preparation_seconds_max: 1800; candidate_GPU_jobs: 1; candidate_torch_allocated_soft_limit_GiB: 10; candidate_train_cases: 74,98; candidate_phase_steps: 0,5,10,15,20,25,30,35,40; new_FEA_runs: 0.

### L12. field 표현·용량·위상 개선의 단일축 실험

상태: PHASE_COVERAGE_CONTRACT_FIRST_NO_RUN_RELEASED; 범위: CONDITIONAL_CURRENT_WINDOW_ELSE_BACKLOG; 우선순위: P0.

선행: W03, W07 · 입력: E10, E22, E23 · 근거: E10, E22, E23 · gate: G02, G03, G09, G10 · Astra 판정: D01, D06.

**할 일**

- 첫 후보 A0/A1은 동일 SCALE360 구조에서 train phase9 대 phase45의 equal-update 비교로 제한한다. E1 전이가 HOLD여도 이 clean from-scratch 계약 작성은 독립 진행 가능하다. 기존30epoch 결과는 이력 기준이며 새 seed/일정과 섞어 인과 비교하지 않는다.
- 진단에 따라 phase coverage, material-balanced loss, global context, width/depth를 하나씩 사전등록한다. 동시에 변경하면 recipe 비교로만 해석한다.
- global pooling은 graph별 검증 면적·support만 사용하며 target field나 다른 batch graph가 섞이지 않게 한다. shared계층 초기 weights와 데이터 순서를 가능한 한 맞춘다.
- train-only 작은 적합 진단 성공/실패를 일반화 또는 표현력의 단독 증명으로 쓰지 않는다.
- 공극 개선만으로 성공하지 않고 전체 |B|/vector와 구리/B²의 비악화 gate를 함께 판정한다.

**완료 기준**

- 고정 비교 arm과 손실/성능/자원 효과가 분리됨
- source-derived truth를 inference input으로 사용하지 않음
- 첫 실행 전에 arm별 데이터/업데이트·시간·메모리·seed·평가시각·중단·재시도0을 동결하고 D01/D06 release를 받음

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/FIELD_ABLATION_PLAN.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/FIELD_ABLATION_RESULTS.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none..

### L13. J/Je·도체·회로·손실 의미 계약

상태: UNKNOWN_MODALITIES_MASKED; 범위: LONG_TERM_OR_SEPARATE_APPROVED_WINDOW; 우선순위: P0.

선행: W06, W07 · 입력: E15, E17, E19, E20 · 근거: E15, E17, E19, E20 · gate: G02, G03, G07 · Astra 판정: D04, D05.

**할 일**

- 대표 native TXT/exporter/HDF5의 J/Je total/source/induced 정의, units, element/node support, conductor/circuit ID와 material temperature를 연결한다.
- A/mm²→A/m² 단위변환, peak/RMS, phase/parallel convention, 2D area×active length×symmetry, time window를 명시한다.
- 정의가 확인된 전체 전류밀도로만 Joule loss를 재적분한다. Je-only를 총J로 대체하지 않는다. native independent loss channel 및 circuit current와 비교한다.
- normalization·missing modality mask·검증 tolerance는 train/admission 근거로 사전등록한다. same-waveform arithmetic agreement를 independent physics proof로 부르지 않는다.

**완료 기준**

- quantity별 PASS/FAIL/UNKNOWN dictionary와 적분 재현 evidence
- 통과한 modality에만 supervision 허용

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/FIELD_CURRENT_LOSS_SEMANTICS.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/NATIVE_REINTEGRATION_CHECKS.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none..

### L14. 다중 template·4/6/8도체 native configuration 계약

상태: GEOMETRY_ONLY_FIELDS_MISSING; 범위: LONG_TERM_OR_SEPARATE_APPROVED_WINDOW; 우선순위: P0.

선행: W06, W07 · 입력: E08, E15, E16 · 근거: E08, E15, E16 · gate: G02, G03, G08 · Astra 판정: D04, D07.

**할 일**

- R1 272×3=816은 geometry-only이며 native mesh/회로/제조/solver PASS로 세지 않는다. EveryMotor320그룹과 같은 집합이라고 가정하지 않는다.
- 대표 비reserved base에서4/6/8 actual drawn count, 절연/간격/폭/높이/코너/슬롯영역과 회로 series/parallel을 검증한다. hidden fit 금지.
- equal slot ampere-turn pilot375/250/187.5 Arms는 제안값이며 회로 audit 후 사용한다. copper area 변화와 equal-current 비교를 분리한다.
- 새 template의 치수·재료·냉각·운전범위와 scaling similarity 가정을 명시한다. whole template holdout은 데이터 확보 후 별도 잠금한다.

**완료 기준**

- 해석 전 물리 configuration와 source lineage가 재현 가능
- 4/8 Field 확보 여부가 원시 증거로 구분됨

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/MULTITEMPLATE_CONFIGURATION_ADMISSION.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/CANDIDATE_CASE_MATRIX.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none..

### L15. MS/Full 및 solver 내부 fidelity 최소 보강

상태: NO_NEW_FEA_AUTHORIZED_BY_MASTER; 범위: LONG_TERM_OR_SEPARATE_APPROVED_WINDOW; 우선순위: P0.

선행: L13, L14 · 입력: E16, E17, E20 · 근거: E16, E17, E20 · gate: G03, G07, G08, G09 · Astra 판정: D04, D05, D07.

**할 일**

- 기존 raw 재사용→누락 보강 순서로 선정한다. 하나의 nonreserved6도체 base/OP를 먼저 사용하고,4/8 및 전류/속도 축은 단일변수로 확장한다.
- MS는 자계 teacher이지 independent AC truth가 아니다. Full-field donor와 MS donor를 ledger에서 구분한다.
- 새 transient history는 각 실행 총128스텝 상한이다. N128P1/N64P2 기존 진단을 재실행하지 않는다. 초기화/평균구간이 해결되지 않으면 low-fidelity/diagnostic으로만 사용한다.
- preflight mesh/current/region/solver identity 및 수렴·중단 criterion 통과 후에만 새 case ID/run count/wall/RAM/disk/STOP 계획을 Astra에 제출한다. master는 새 FEA 횟수를 부여하지 않는다.

**완료 기준**

- 각 새 관측에 fidelity와 formal qualification이 분리됨
- 독립label 예산과 실패·partial 원가를 모두 기록

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/MINIMAL_MS_FULL_ACQUISITION_PLAN.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/FIDELITY_RECEIPTS.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none.; per_new_transient_history_total_steps_max: 128; total_case_budget: UNKNOWN / 실행 불가; no_automatic_extension: true.

### L16. 최소 exact cross-solver bridge

상태: EXACT_BRIDGE_COUNT_ZERO_HOLD_ALIGNMENT; 범위: LONG_TERM_OR_SEPARATE_APPROVED_WINDOW; 우선순위: P0.

선행: L13, L14, L15 · 입력: E07, E23 · 근거: E07, E23 · gate: G02, G03, G06, G08 · Astra 판정: D05, D06.

**할 일**

- 첫 후보는 한 nonreserved native6도체 형상·한OP로 정하고 MC/Elmer의 geometry/material/temp/winding/circuit/waveform/window/quantity를 일치시킨다.
- GetDP는 별도 solver identity이며 Elmer로 이름만 바꾸지 않는다. 같은 mesh가 필수는 아니지만 각 native mapping·적분 support·fidelity가 재현돼야 한다.
- mesh 차이가 있으면 field pairing에는 검증된 physical projection/conservative integration이 필요하다. nominal/nearest coordinate만으로 도체/영역을 대응하지 않는다.
- 1점은 mapping 계약 확인용이다. 학습 alignment는 train-group exactpair만; 독립 교차형상/OP 검증점은 별도 잠금한다.

**완료 기준**

- 양쪽 hash·quantity·mapping·split의 exact admission PASS
- bridge count는 독립 physical pair로 집계되며 일반화 주장은 추가 시험까지 보류

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/EXACT_BRIDGE_REGISTRY.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/BRIDGE_ADMISSION_REPORT_KO.md` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none..

### L17. 두 solver·field/scalar 멀티모달 학습

상태: DESIGN_ONLY_MASK_UNQUALIFIED_MODALITIES; 범위: LONG_TERM_OR_SEPARATE_APPROVED_WINDOW; 우선순위: P0.

선행: W07, L13 · 입력: E07, E17, E23 · 근거: E07, E17, E23 · gate: G02, G03, G04, G06, G07, G09, G10 · Astra 판정: D03, D05, D06.

**할 일**

- solver별 native input adapter/encoder/normalizer/field decoder와 shared geometry/OP representation을 두고 solver와 numerical fidelity를 분리한다.
- 지금 확정된 Bx/By만 우선 사용한다. A/J/Je/AC mask는 quantity별 PASS 후 켠다. 결측과 관측된 비정상값을 구분한다.
- field/latent summary는 predicted quantities만 scalar head에 공급한다. actual target Full B 입력은 금지한다.
- paired loss는 L16의 exact train bridge가 있을 때만 켠다. bridge0에서는 unpaired domain-wise supervised learning만 허용한다. shared latent similarity를 solver discrepancy 정답으로 주장하지 않는다.
- field→loss 물리 consistency는 J/Je 정의·적분 또는 canonical baseline 계약이 검증된 경로에만 적용한다.

**완료 기준**

- 모달리티별 observation/admission mask와 inference contract 검증
- single-domain vs shared-domain 비교에서 source cost와 정확도 기여가 분리됨

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/MULTIMODAL_RUN_CONTRACT.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/MULTIMODAL_ABLATION_RESULTS.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none..

### L18. AF 대비 공정한 AC 손실 benchmark

상태: FUTURE_BENCHMARK_NOT_FROZEN_TEST_RETUNING; 범위: LONG_TERM_OR_SEPARATE_APPROVED_WINDOW; 우선순위: P0.

선행: W07, L13, L17 · 입력: E17, E18, E19 · 근거: E17, E18, E19 · gate: G02, G03, G07, G10 · Astra 판정: D06, D07.

**할 일**

- 새 독립 geometry/OP final test와 같은 nested target Full-label ID를 모든 arm에 적용한다. 기존Full63/newOP8/HalfSC32·4는 별도 동결 reference로 유지한다.
- 무보정 canonicalHybrid, 최신 exponent-separable AF, scalar residual, MS-pretrained field residual, Full-donor E1-transfer, multi-solver multimodal을 분리한다.
- target incremental label budget parity와 total historical Full access parity를 별도 비교한다. unmatched totals는 data-augmented comparison으로 표시한다.
- canonical spatial mean-square Hybrid evaluator·skin-excess·window·scope가 일치할 때만 AF를 적용한다. 새 형상은 동일 예산의 새 AF 재앵커와 fixed-zero-shot을 구분한다.

**완료 기준**

- 전체 independent test에서 AC오차·안정성·효율의 preregistered gate 판정
- known-template 결과를 새template/4·8도체 generalization으로 확대하지 않음

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/AC_BENCHMARK_PLAN.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/AC_BENCHMARK_RESULTS.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none..

### L19. lineup surrogate 기반 설계 선택·최적화

상태: DOWNSTREAM_NOT_STARTED; 범위: LONG_TERM_OR_SEPARATE_APPROVED_WINDOW; 우선순위: P1.

선행: L14, L18 · 입력: E17 · 근거: E17 · gate: G02, G03, G07, G08, G10 · Astra 판정: D06, D07.

**할 일**

- 여러 기준template·scaling·4/6/8구성의 torque/speed/duty/envelope/thermal/voltage/manufacturing constraints를 사용자 요구로 잠근다. 없는 성능값을 채우지 않는다.
- 지원범위 밖 geometry/OP는 OOD flag와 추가 native 검증 후보로 반환한다. uncertainty는 검증 없이 confidence로 포장하지 않는다.
- 동일 설계 후보·동일 선택규칙에서 AF와 새모델의 ranking, false-feasible rate, constraint error, top-k regret, pipeline wall/Full budget을 비교한다.
- 선택된 후보의 native 독립 검증을 통해 surrogate 최적점이 실제 개선인지 확인한다. 원시 자계 개선만으로 system energy–mass 최적화 완료를 주장하지 않는다.

**완료 기준**

- 다중template/turns 독립검증과 실제 설계선택 효용이 연결됨
- 불확실성과 영역외 입력에 대한 보수적 사용 정책 확정

**산출물**

- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/LINEUP_OPTIMIZATION_CONTRACT.json` — PROPOSED_NOT_CREATED
- `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/master_plan_artifacts/LINEUP_DECISION_VALIDATION.json` — PROPOSED_NOT_CREATED

**실행 제한**

automatic_retries: 0; guard_deadline_UTC: 2026-10-27T22:59:00Z; hard_deadline_KST: 2026-10-28T08:00:00+09:00; stop: STOP, wall cap, deadline, invalid owner identity or failed admission; preserve partial artifacts; stop only verified owned PID/start-time descendants or immutable container ID; new_automation: false; mode: NOT_EXECUTABLE_WITHOUT_RUN_DECLARATION; authorized_runs_now: 0; new_FEA_runs_authorized_now: 0; GPU_jobs_authorized_now: 0; wall_seconds_max: UNKNOWN / 실행 불가; memory_limit_GiB: UNKNOWN / 실행 불가; run_count_max: UNKNOWN / 실행 불가; execution_after_deadline_authorized: false; requirement: Astra scientific decision plus concrete bounded run contract within existing user authority. Deadline/budget/scope expansion requires explicit user instruction; this plan grants none..

## 10. 즉시 인계·종료 기준

- W00: 기존 완료 receipt/현재 소유 작업만 짧게 재확인. W01/W02는 완료로 소비하고 재전송·재추론하지 않음.
- W03+W04: 9/36/45·copperB² 표와 frozen6/8 결과/한계를 함께 정리.
- W07: 공통 metric/mask/exposure·cost ledger를 고정하고 L12 phase-only 비교 실행계약 초안 제출. 새 GPU 학습은 release 전 HOLD.
- W05/W06: 기존 자료 metadata 감사만 bounded 수행. 새 데이터 탐색/코드는 계획 승인 후 Sol 범위에서; 이 계획 작성자가 해결하지 않음.
- W08: 07:30 정리 시작, 07:59 owned guard, 08:00 종료. 기존 승인된 EveryMotor 연구 namespace에 작은 결과·계획·해시만 게시.

Astra release 요청에는 다음 묶음이 필요하다.

- stepid와현재gate별PASS/HOLD/UNKNOWN
- 입력/코드/checkpoint/selection hash와 geometry/OP/phase·노출 ledger
- arm별 host/run/update/label/wall/memory/disk/seed·STOP·retry0·deadline
- 공통eval mask/metric/threshold와실패처리
- 예상 산출물과현재 사용자 승인 범위; 초과분은 명시

진단 완료 receipt와 frozenOP6/8 기록, D01 방향, W07 실행계약 초안, E1/모달/새FEA HOLD 이유, 실제 종료·잔여작업을 남기면 window 종료 가능. 신규 실험을 위해 끝없이 연장하지 않는다.

전체 연구 완료 조건: 독립미노출 geometry/template·OP 검증과 qualified loss/native evidence를 확보하고 예산동일 비교 및 라인업 결정 검증까지 마쳐야 전체 연구 완료. 현재 도달 아님.

최신 E25~E28 완료 receipt > 오래된 준비/전송 대기 설명. 각 evidence hash는 읽은 시점 스냅샷이며 동시 Sol 갱신 가능. source 경로 내 private 원자료는 공개게시 대상 아님. 수치/관문 충돌은 D00으로 해결.

## 11. 개선대장 추적

| MI | 담당 WBS |
|---|---|
| MI-001 | W06, L13 |
| MI-002 | W02 |
| MI-003 | W05, W07, L10 |
| MI-004 | W01, W02, W03, W07, L10, L11, L12, L19 |
| MI-005 | W05, L10, L11, L17 |
| MI-006 | L13, L16, L17, L19 |
| MI-007 | W06, L14, L15, L16, L19 |
| MI-008 | W06, L14, L15 |
| MI-009 | W01, W02, W03, W07, L13, L15, L17, L18, L19 |
| MI-010 | W01, W02, W03, L12 |
| MI-011 | W03, L12 |
| MI-012 | W00, W04, W05, W07, W08, L11, L15, L16, L18, L19 |
| MI-013 | W04, L18 |
| MI-014 | W00, W01, W08 |
| MI-015 | W00 |

## 12. 근거 파일 스냅샷

JSON에는 각 파일의 경로·SHA-256·확인된 크기가 포함된다. E27/E28 행수는3456/1440이다. source 파일 내용이나 raw를 이 계획에 복제하지 않았다. 동일 경로가 후속 Sol 작업으로 갱신되면 원 계획의 해시는 이 시점 증거로 유지한다.

- **E01** `E:/PJT/EveryMotor/AGENTS.md`
  SHA-256 `79fb0fe1e158d124402918197a62656b22aaf52782626a60d5cce64e39451b6f` · relay ownership

- **E02** `E:/PJT/EveryMotor/motorai-research-20261028/AGENTS.md`
  SHA-256 `7887f40d4179837fb1a0e43a75318e46fdd4c5d2937e9d0fa2edcbf9816d1129` · repository evaluation conventions

- **E03** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/AGENTS.md`
  SHA-256 `d86683553589de3abaf1f229f1beaddbce1041f1249d03f7d1b9580ac246bdc4` · research ownership and current overrides

- **E04** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/README_KO.md`
  SHA-256 `26bdc8dc01ebce7296b83e3b7e9fb721350f67dd554bcad6312bf778d27abfae` · migration scope

- **E05** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/MIGRATION_MANIFEST.json`
  SHA-256 `1ac83562c45f0806e68c1652e6e5043e37432f2f9c76ed631a79a5b9a8a38c22` · migration provenance

- **E06** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/IMPROVEMENT_REGISTER.json`
  SHA-256 `2d2b805dee1a498362bd7ddfbfbc0e7ad80076498ac4824f2a020d7db03c91a3` · 最新 MI-001~015 register, frozen diagnostic complete

- **E07** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/ASTRA_TRANSFER_MULTIMODAL_KO.md`
  SHA-256 `6de9af3bb8d3e1cb0d51a02949cccf7b2d0e5971cbb171d685efdb6a0740b179` · transfer scientific decisions

- **E08** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/STATE.json`
  SHA-256 `3ecdf770288acbead311858abe56632e8ac3cfc3bb6ab63eafe2bb3adefce794` · saved state not live process status

- **E09** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/SCALE360_REPORT_KO.md`
  SHA-256 `d3ce6728070f67a19d626e43a823443e8a7b442d88a079352e8705b0f13c2a1d` · completed field training

- **E10** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/SCALE360_TRAIN_RESULT.json`
  SHA-256 `620e544cfbc84c4f3a8f0f79bcce859b9da74feb7dd5f01a2664d19706d05025` · field metrics and training evidence

- **E11** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/SCALE360_TRAIN_STATUS.json`
  SHA-256 `59b74c75da8b732c1d42f74c93e21cdec6be7aef5b741f89f72a2cdc0cb62d55` · checkpoint and stopped owner receipt

- **E12** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/TRANSFER_ADMISSION.json`
  SHA-256 `53ed76465e275385781525360a3e69509669526aee15c6b8dd78e5e41564f4ea` · weights hold gates

- **E13** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_FIELD_DIAGNOSTIC_PLAN.json`
  SHA-256 `8eafa8ccafbdb575b93eccaf6c52098764d4591725a7bcbe139b0eec1038cfb8` · existing bounded inference contract

- **E14** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/SCALE360_SELECTION_FROZEN.json`
  SHA-256 `21e78f734250ace514f2d2e646e9a382ca271cc6cf703a54068b7c3e0a9acb7d` · frozen group split

- **E15** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/NATIVE_DOE_CATALOG.json`
  SHA-256 `6411f353dc016726fb38d4dd58d02a704d315dc9c03412d5cdc12ad43c628cd6` · native source and alias audit

- **E16** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/STEP128_COMPARISON.json`
  SHA-256 `e058cd9aff31e2f3b24ff0b48c47e8b8acc661d88eb80146056db98b5379f16e` · 128 history numerical fidelity

- **E17** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/BASELINE_CONTRACT_KO.md`
  SHA-256 `348f0ad8761f12587d00206968a3b3705e7d8a993a4d39e3332ecc17cd2caaac` · AF denominator and loss budget

- **E18** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/new_op_validation_20261027/EVALUATION.json`
  SHA-256 `05620866f77340279377083e82d4c039bb78feb7e6a62c26add99e1cf0cb6a7d` · frozen 6 of 8 metrics

- **E19** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/new_op_validation_20261027/quality_summary.json`
  SHA-256 `b3ac1c36ae09d737bfd615fe285984cc09b8312cfa00513a3ab988c181f6df6c` · quality layers and blocked SC cases

- **E20** `E:/PJT/EveryMotor/motorai-research-20261028/.github/plans/methodology_review_20260720.md`
  SHA-256 `c3c0b049a18cfe463e9e4beeaa58ae207bbeb42bf50a8e63ea359df6ae2aa461` · sections 37-39 supersede stale static and target claims

- **E21** `private:prior_motorai_evidence/results/logs/run_e1_airgap_docker.sh`
  SHA-256 `16b00f06e2486dd22d4870d55a9e7facf193b5f7f79375cb9ad874d0ff830132` · private historical target B recipe

- **E22** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/evaluate_frozen_field_diagnostic.py`
  SHA-256 `94ccc9c752eee02ce123a6900c697d44647c4fee9b8fe7ee16b2ec2f23987d08` · frozen diagnostic metric definitions

- **E23** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/dual_solver_mgn.py`
  SHA-256 `ae1966b5d660b8647ece32da3caf9af2e3dfd2ae11752738dfaba7d75e6663ed` · existing solver branches and field only path

- **E24** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/CONTINUATION_KO.md`
  SHA-256 `bc06f7595dafd2cad1bb17ed518a5cbdbc43c160bc11393ab8350d0c7474e34c` · historical continuation; superseded operational statements identified

- **E25** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_FIELD_DIAGNOSTIC_RESULT.json`
  SHA-256 `6c739483b5aec20669bad65d9a646e42df98715d77d912c5a9b17a2571b7e567` · completed frozen all45 diagnostic

- **E26** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_FIELD_DIAGNOSTIC_STATUS.json`
  SHA-256 `3179ff9f66de5290bec622a894a2463c78e8027c9ee7cc176f37c0d94ab48294` · completed guard and unchanged checkpoint receipt

- **E27** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_DIAGNOSTIC_conductor_B2.csv`
  SHA-256 `922d2b512faa5a5e81cdf7dd5c7a8c7d9785843260a553925030a491858bf650` · conductor B2 diagnostics, 3456 rows

- **E28** `E:/PJT/EveryMotor/motorai-research-20261028/research/motorai_multisolver_20261028/phase2_multigeometry_turns_20261027/FROZEN_DIAGNOSTIC_snapshot_errors.csv`
  SHA-256 `d8162a97531e842270cc6768c16e7ac0e1b36935220a5acfb47740ad2ca4a19f` · snapshot diagnostics, 1440 rows

