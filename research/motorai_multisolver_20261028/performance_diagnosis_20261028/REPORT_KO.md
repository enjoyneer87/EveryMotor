# E1 공극·토크 성과와 최근 Field 모델의 차이 진단

2026-10-28 KST. 분석·회수는 관리 PC DESKTOP-6QIVKIC, 기존 graph 검사는 133 CPU에서 수행했다. 과학 판정은 사용자 지정 **Astra xHigh**, 실행·문서화는 Sol이 담당한다. [Astra 판정문](ASTRA_REVIEW_KO.md), [수치·출처 manifest](DIAGNOSTIC_EVIDENCE.json), [원 graph 검사](GRAPH_TOPOLOGY_AUDIT.json)를 함께 보존한다.

## 판단

최근 모델의 토크·공극 정확도가 E1보다 나빠졌다는 결론은 아직 **UNKNOWN**이다. 최근 모델은 해당 두 지표를 측정하지 않았다. 이전의 5% 이하는 **공극 B크기 4.9563%와 토크 3.2146%**였고, 전체 B크기는 11.0967%였다. 최근에 주로 보고한 18~20%는 전체 **B벡터** 오차다.

하지만 최근 모델은 E1을 그대로 확장한 모델도 아니다. E1의 큰 모델, 물리 prior, 전기각 입력, 공극 중심 손실을 대부분 사용하지 않은 별도 경량 모델이었다. 학습 graph 표본에서는 회전자–고정자의 메시지 전달 경로가 실제로 끊겨 있었다. 이 차이들을 고치거나 분리 측정하기 전에 데이터 수만 늘리는 방식은 우선순위가 낮다.

## 비교되는 수치와 비교할 수 없는 범위

| 지표 | E1 legacy6 | A0 phase9 / 2 seed | A1 phase45 / 2 seed |
|---|---:|---:|---:|
| 전체 B크기 nRMSE (%) | 11.0967 | 13.504 / 13.290 | 12.531 / 12.517 |
| 전체 B벡터 nRMSE (%) | 위 수치와 다른 지표 | 20.379 / 20.160 | 19.582 / 19.331 |
| 공극 전용 B크기 (%) | 4.9563 | 미평가 | 미평가 |
| 토크 nRMSE (%) | 3.2146 | 미평가 | 미평가 |

위 표는 지표의 뜻을 확인하는 표이며 모델 순위표가 아니다. E1은 legacy6 test에서 요소 수 기준으로 집계했고 최근은 E1 oldTRAIN에 포함됐던 innerdev32에서 요소 면적으로 가중했다. 출력 위치·mask·분할·가중치·checkpoint 선택도 다르다. E1의 공극<5%는 확인됐지만 고정 legacy6 토크<3% gate는 미달이었다. 이번 감사에서는 기존 원점수 요약만 읽었고 old val/test payload나 E1 weights를 개봉하지 않았다.

## 확인한 구현 차이

| 축 | E1 | 최근 A0/A1 |
|---|---|---|
| 모델 | h256 / processor15, 기록상 약 9.29M | width32 / depth3, 실제 자계 경로 30,594 parameter |
| 전체 인스턴스 | 하나의 자계 예측 모델 | 두 solver/AC/alignment 포함 총 70,182 parameter; B 손실에서 사용되지 않는 경로 39,588 |
| 출력 위치 | nodal Bx/By → 요소 평균 | native FE triangle에 Bx/By 직접 출력 |
| 물리 prior | 기존 선형 solve prior A/Bx/By | 사용 안 함 |
| graph | band를 loss에서 제외하고 message graph에는 유지 | a2를 먼저 제거하고 공유 native node ID edge 생성 |
| 전역 정보 | 기존 15단 메시지 전달과 prior | 3단 국소 전달. pooling 값은 field head로 돌아가지 않음 |
| 각도 | sin/cos(4×기계각) | A0/A1은 sin/cos(기계각); A2가 전기각 복원 중 |
| 정규화 | train 통계 기반 feature/edge 정규화, B scale | 물리 단위와 정해진 scale을 사용; E1 normalizer를 계승하지 않음 |
| 손실 | 공극 가중5 + band spectral30, B scale 정규화 | 전체 면적가중 B벡터 MSE, 공극 전용 가중 없음 |
| 학습 조건 | 450조건 × 45시각, 50epoch | 328조건, A0 9시각 / A1 45시각, 각 88,565 update |
| checkpoint | val 기준 best epoch47 | 사전 고정된 마지막 update; best seed/epoch 선택 없음 |

파라미터 수 차이는 실제 구성 차이이며, 그 자체로 오차 기여율을 뜻하지 않는다. E1도 시간 rollout 모델로 평가한 것이 아니므로 단순히 “새 모델에 시간 상태가 없어서 퇴보했다”는 설명을 쓰지 않는다. E1 실제 recipe는 `target=B`이며 파일명 `curl`을 근거로 nodal A→curl 모델이라고 부르지 않는다. E1 recipe는 rotor wrapping/추가 anti-periodic edge를 끄므로 그 두 기능이 E1에 유리했다는 주장도 하지 않는다.

## 원 graph 직접 검사

학습 manifest에서 label을 보지 않고 case ID 순으로 6조건을 균등 선택한 뒤, 각 조건의 이미 학습된 step0/20/40을 읽었다. 원 NPZ 18개 모두 크기·SHA-256이 manifest와 같았다. 읽은 배열은 node feature와 edge index이며 field target은 읽지 않았다.

- **18/18 graph:** 회전자/자석과 고정자가 서로 다른 component. 각각 두 개의 연결 성분을 확인했다.
- case40 예: 고정자·구리 등 12,078개 triangle과 회전자·자석 등 6,062개 triangle이 분리됐다.
- 공통 design/OP 조건은 양쪽에 들어간다. 따라서 “전자기 상호영향이 전혀 없다”는 뜻은 아니다.
- 6조건 표본의 구조 검사이며 전수 통계나 정확도 ablation은 아니다. 연결 복원의 오차 개선량은 아직 미측정이다.

133 CPU에서 4.58초, exit0으로 끝났다. 이번 진단의 새 FEA / 학습 update / GPU 추론 / old val·test payload 접근은 모두 0이다. 첫 inline 전송은 Windows command-length 제한으로 원격 실행 전에 실패했고, 파일 전송 방식으로 바꾼 뒤 감사 1회를 수행했다. 해석·학습 실패의 자동 재시도는 없었다.

## L12가 이미 알려준 것

같은 88,565 update에서 phase9→45는 두 seed 평균 전체 B벡터를 0.813pp 낮췄지만 seen9를 1.544pp 악화했다. 기존의 개선폭 1.0pp 및 seen9 비악화 0.5pp 조건을 만족하지 못했으므로 **NO_IMPROVEMENT_CLAIM**을 유지한다. A1의 `epochs_completed=30`은 A0 기준 환산치이며 실제 45시각 데이터 순회는 6회다.

끝부분 41~44시각 오차는 개선됐다. 그러나 노출 빈도·훈련 분포 변화가 정확도에 기여한 양은 분리되지 않았으므로 “희석이 원인으로 확정됐다”고 표현하지 않는다. 고위상각군 잔여 오차와 구리 B² 음의 bias도 남는다. B²는 파생 물리량이며 AC 손실 검증 결과가 아니다.

## 인계와 다음 실행 순서

Claude Code **MotorAI 학습 계속 진행**, session `c7aa7e1b-3e76-4741-a258-071716aa4e0a`의 마지막 human 지시는 2026-10-28 15:14 KST의 A2 3-seed 옵션 승인이다. 15:22 KST에 Claude 사용량 한도로 대화 실행이 멈췄고 원격 guard는 계속 실행 중이었다. 현재 사용자 지시로 Sol이 이어받으며 판단은 Astra xHigh 이상으로 되돌렸다. 예전 08시 작업창과 예약은 재활성화하지 않는다.

1. **승인된 A2 그대로 완료:** A2 seed17/29/43와 추가 대조군 A0 seed43를 회수한다. A0 seed17/29는 기존 완료 결과를 재사용한다. 입력 전기각만 바꾼 축이며 기존 data/split/update/마지막 checkpoint/실패 보존 규칙을 유지한다. 중복 실행·설정 변경·추가 seed 선택을 하지 않는다.
2. **공통 평가:** 기존 dev32와 동일 native element ID/mask에서 B크기·B벡터의 면적가중/비가중, stationary a1 공극, 구리, 동일 Arkkio 토크를 함께 측정하는 계약을 작성한다. `air`를 공극 지표로 대체하지 않는다. 추가 추론은 기존 소유 작업 종료 후 별도 실행 계약에 따른다.
3. **다음 단일축 후보:** 고정 각도 인코딩·width/depth·손실·update 아래에서 전역 요약을 field head에 되돌리는 유무를 비교하고 가능하면 parameter-matched 대조군을 둔다. 잘못된 a2를 손실에 넣거나 좌표 최근접 edge로 임의 연결하지 않는다.
4. **그 뒤:** 측정된 공극 오차를 근거로 공극 가중, 용량, admission을 통과한 prior/전이를 각각 분리한다. E1 donor reserved 노출·prior native binding·normalizer 호환성은 UNKNOWN이어서 weights/teacher/distillation HOLD를 유지한다.

모든 solver·턴·OP·파생 label의 geometry/alias 그룹, R1_0199/R1_0076 제외, HalfSC32/4, exact bridge0과 최신 AF 분모 계약을 유지한다. 이번 결과는 B 모델의 구조·평가 진단이며 AF보다 우월한 AC 손실 모델이라는 주장이 아니다.

## 재현

기존 133 환경에서 `python audit_graph_connectivity.py --bundle <admitted_bundle> --code-dir <pinned_model_code_dir>`를 실행하면 stdout에 감사 JSON이 나온다. 현재 실행 receipt가 있으므로 단순 상태 확인을 위해 다시 실행하지 않는다. 로컬 문서 집계는 `python build_evidence.py`이며 새 계산을 하지 않는다. raw graph/checkpoint는 원격에 보존하고 Git에는 코드·작은 JSON·보고서만 게시한다.
