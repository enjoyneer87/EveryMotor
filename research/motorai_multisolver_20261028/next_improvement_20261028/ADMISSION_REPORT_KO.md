# 연결 복원·E1 재사용 선행 감사

2026-10-28 KST. 사용자 요청에 따라 다음 개선 실험은 **Astra Max 판정 → Sol 구현·검증·실행**으로 진행한다. A2는 기존 승인된 설정 그대로 끝낸다. 이 문서는 새 학습의 성능 결과가 아니며 다음 실험의 최종 변경 축은 별도 Astra 판정문과 실행 계약에서 고정한다.

## 직접 확인한 native 연결

234 CPU에서 9.38초, exit0. 기존 admitted TRAIN 조건40·111·184·340·411·479의 HDF56개가 frozen source SHA-256과 모두 일치했다. field 배열은 읽지 않았고 새 solve·학습·추론은0회다.

| 확인 항목 | 결과 | 의미와 한계 |
|---|---|---|
| full export 삼각형 공유변 graph | 6/6 모두1 component | 원본 reference topology에 연결 경로가 존재 |
| a2를 제거한 동일 graph | 6/6 모두2 components | 최근 reader의 제거 단계가 경로를 끊는 현상 재현 |
| full export의 사용 native node graph | 6/6 모두1 component | E1처럼 node 기반 reference graph를 구성할 정보가 있음 |
| stationary a1 | 6/6 모두45시각 좌표 동일 | 공극·Arkkio 공통 평가용 ID 연결 가능성을 지지 |
| invalid non-a2 element | 6/6 모두0 | 기존 geometry reader 검사와 일치 |

case40~411 표본의 a1은744개 element, case479는1368개다. 동일 개수나 좌표 최근접으로 연결하지 않고 각 source의 native ID를 사용해야 한다. **Reference graph 연결성은 재메시된 a2의 실제 각 시각 FE mesh가 유효하다는 증명이 아니다.** a2의 잘못된 field target·면적·curl을 loss나 metric에 다시 넣지 않는다. 모델의 context graph와 물리 감독/적분 mask를 구분해야 한다.

## E1 재사용을 구분할 조건

| 대상 | 현재 판단 | 필요한 조건 |
|---|---|---|
| 구조·code를 from scratch 재사용 | 조건부 후보 | 출력 위치·입력 순서·train-only normalizer·실제 processor·자원 smoke·공통 평가 일치 |
| 기존 E1 weights/teacher/distillation | HOLD | 전체 donor reserved 제외·prior-native binding·normalizer 호환성 UNKNOWN 유지 |
| 기존 linear prior 캐시 | HOLD | filename/shape/current-scale 외 native ID·geometry/OP/material/source binding을 입증 |
| 공극 가중5·spectral30 | 별도 단일축 후보 | stationary a1·공통 B/토크 metric·손실 분모/정규화/고조파 basis를 먼저 고정. 원래 recipe와 같은 성과로 간주하지 않음 |
| native 연결 보존 | 별도 단일축 후보 | reference topology와 실제 solver 재메시를 구별하고 invalid a2 감독을 제외. 임의 최근접 edge 금지 |
| 전역 context→field head | 별도 구조 후보 | 정보 경로 보완이며 native FE topology를 복원한 것이라고 부르지 않음 |

기존 캐시는 raw array header만 읽었다. node ID나 source hash가 없으면 모양이 맞더라도 canonical prior로 승격하지 않는다. `nodal_prior_features()`는 cache miss·불일치 때 solver를 호출하고 캐시를 쓰므로 이번 감사에서 호출하지 않았다.

`fem_warmstart/prior.py`의 유효 투자율20은 코드 주석상 과거 case4/7 자계 sweep로 정했다. `winding_template.json`은 case0의 J에서 교정했다고 기록한다. 따라서 기존 prior는 실행 시 per-sample 정답을 직접 읽지 않아도 **역사적 상수 선택·교정의 label 노출**이 있다. 이 비용과 노출을 기록하고, 독립 비교에서 완전히 label-free인 순수 물리 상수로 설명하지 않는다. 이번에 old val/test payload를 다시 읽은 것은 아니다.

## 보존·실행 경계

원 HDF5와 prior NPZ를 변경하지 않았고, 원본 크기·수정시각 불변을 확인했다. graph/checkpoint/raw를 Git에 복사하지 않는다. [계획](ADMISSION_AUDIT_PLAN.json), [실행 코드](audit_native_topology_prior.py), [원 결과·hash](NATIVE_TOPOLOGY_PRIOR_AUDIT.json)를 보존한다. A2의 마지막 checkpoint와3seed 판정, 같은 Full donor 비용·geometry split·reserved/alias 제외·HalfSC32/4·exact bridge0은 유지한다.
