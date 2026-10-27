# 현재 결과가 바꾸는 연구 순서

현재 자료에서는 **출처가 확인된 scalar 손실 대리모델과 sparse Full-FEA 보정부터 구축하고, Field 모델은 추가 효용을 비교하는 경로로 유지**하는 편이 타당하다. 아래 개발 결과가 이 순서를 뒷받침하지만, 최신 JEET AF보다 정확한 Full-FEA 대리모델을 검증한 상태는 아니다.

## 동일한 MS 자료에서 확인한 것

기존 R1 유효274형상 중 reserved2를 제외한272형상의 numeric MS 자료와 original-B 근접손 재현을 확인했다. 학습8⊂32⊂128⊂256과 개발검증16을 고정했고, 검증16은 기존2형상과 추가14형상을 구분했다. reserved0199/0076은 모델 학습·성능평가에 쓰지 않았다. 과거 schema/finite QC와0199 그림 이력은 있으므로 전역 never-opened 시험이라고 부르지는 않는다.

아래 값은 모두 **같은 GetDP MS 근접손 교사에 대한 개발검증16형상의 wMAE**다. Full-FEA AC 손실 오차가 아니다.

|학습 형상 수|B 예측 Point graph → 근접손,20epochs|B 예측 Node MLP → 근접손,20epochs|형상 → 근접손 log-ridge|형상 → 근접손 log-RBF|
|---:|---:|---:|---:|---:|
|8|29.33%|25.92%|10.59%|13.71%|
|32|7.30%|6.96%|4.27%|2.51%|
|128|7.77%|5.78%|3.90%|0.73%|
|256|6.47%|4.58%|3.75%|0.84%|

Field 모델은 도체의 좌표·면적·범주와 기하·위상으로 평균 B를 예측한다. Scalar 기준선은 여섯 기하 변수로 하나의 근접손 값을 직접 예측한다. 출력과 목적함수·입력 정보가 다르므로 Field 재현 능력까지 같은 문제로 비교한 표는 아니다. Scalar 목표만 필요할 때 Field 단계를 추가할 실익을 먼저 입증해야 한다는 개발상의 판단이다. RBF의 길이 척도는 train 거리의 median으로 고정했고 검증 오차로 고르지 않았다.

신경망8모델의 총63,600updates는 CPU1thread에서 약394초에 끝났다. 학습 형상 수가 늘면20epochs의 update 수도600/2,400/9,600/19,200으로 증가한다. 같은600updates를 비교한 결과는 별도로 남겼으며 성능이 형상 수에 따라 단조롭게 좋아지지는 않았다. 따라서128 또는256을 보편적 필요 표본 수로 선언하지 않는다. N8의 두 예산점은 동일 모델이다.

앞선8형상 실험에서는 B+log-proximity 목적함수가 학습 오차는 줄였지만, 같은 초기 모델·추가300updates의 B-only 대조군보다 재사용한 검증2형상의 손실 오차가 커졌다. 이 결과로 해당 목적함수를 채택하지 않았으며, 물리 기반 목적함수 일반의 실패라고 확대하지 않는다.

세부 값과 재현 근거는 [Field 학습곡선](ms_expanded_evaluation/RESULTS_REPORT_KO.md), [직접 scalar 기준선](ms_scalar_reference/REPORT_KO.md), [독립 scalar 검토](ms_scalar_review/REPORT_KO.md), [목적함수 대조](physics_loss_control/REVIEW_THREE_ARM_KO.md)에 있다.

## 실제 AC 손실 정확도로 넘어가는 다음 비교

1. 최신 JEET의 sparse Full-FEA 보정 AF를 필수 기준선으로 고정한다. 현재 재현한 AF는 mean-square-field Hybrid와 결합된 모델이다. 평균 B만 저장된 R1의 다른 근접손 평가식에 그 계수를 바로 곱하지 않는다.
2. 실제 형상·회로·재료·온도·파형·OP·출력 범위가 대응하는 MS/Full-FEA 관측을 연결한다. 같은 ID의 다른 solver 관측은 독립 표본으로 세지 않고 같은 geometry group에 묶는다.
3. 같은 Full-FEA 예산과 평가 집합에서 직접 scalar 모델, 기존 AF, Full-FEA 잔차 모델, MS 사전학습 후 Full-FEA 보정 모델을 비교한다. Zero-shot과 소수 재앵커, 알려진 형상의 source 변환은 별도 시험이다.
4. Field 모델은 도체 내부 B²·고조파·Joule 적분 또는 여러 물리 출력의 전이에서 얻는 이득을 확인한다. 현재36도체 평균 B의 kNN 모델은 원래 FE 연결을 사용하는 MeshGraphNet이나 도체 내부 J 모델이 아니다.
5. 손실 외 토크·dq 자속·전압·열·제조 제약이 검증된 뒤 최적점 검증으로 간다. 현재 inference contract는 필요한 조건이 없으면 data_required를 반환한다.

실험 자료를 추가할 때도 specimen, 측정법, 불확실성, 파형·온도·평균 구간과 물리 범위를 별도 관측으로 유지한다. 총 측정 손실을 활성부 AC excess와 혼합하거나 pseudo-label을 새 독립 Full-FEA/실측으로 세지 않는다.

## MS-FEA와 Full-FEA의 규모 해석

|단계|형상 범위의 계획값|MS phase sweep 누적 목표|Full-FEA 독립 조건 누적 목표|
|---|---:|---:|---:|
|P1, 여러 기준 형상의 작은 통합 검증|12형상|144|96|
|P2, 라인업·제조 정책 확장|48형상|1,152–1,440|384–600|
|P3, 더 넓은 적용 범위 검토|약80–150형상|2,400–4,500|800–1,800|

이 표는 단계별 예산 계획이며 정확도 보장이나 통계적으로 입증된 최소치가 아니다. 이번272형상은 하나의 template·고정 OP의 MS 자료다. 전류·진각·온도, 독립 기준 형상, 공극·절연 유지 정책 및 정수 턴수의 부족을 메우지는 않는다. 기존 자료를 재사용할 때는 조건·출처·품질이 맞는 실제 case만 차감한다. MS1건은 필요한 회전 위상 sweep을 뜻하고, Full1건도 수렴에 필요한 시간 스텝과 주기를 포함한다. 실패·수렴 반복은 별도 비용이다.

별도의38 Full 진단 예시는 이 P1과 다른 설계이며 자동 합산하지 않는다. [단계별 후보와 예산](case_selection/REPORT_KO.md)의 case/OP/정책을 먼저 대조한다.

이전 framework_plan의 계획값과 출처 해시는 [예산 출처 기록](PLANNED_BUDGET_SOURCE.json)에 보존했다. 이번 학습 결과로 기존 예산을 사후 최소치로 바꾸지 않았다.

## 새 계산의 필요성 판단과 외부 인계

필요성이 확인된 R1_0167의 cycle1/2 Full-FEA 두 해석은 완료했다. 같은 형상·같은 물리 OP의 수치 설정 비교에서 API AC 값이2.244728% 달랐다. 이를 새 독립 형상2개나 수렴 검증으로 세지 않는다. 저장 MES·관련 metadata와 공식 문서를 확인했지만 API의 실제 평균 창은 확정하지 못했다. [시간 metadata 감사](fea_average_metadata/REPORT_KO.md)와 [공식 문서의 적용 범위](fea_average_metadata/OFFICIAL_DOC_BOUNDARY_KO.md)에 unknown을 보존했다.

현재 작은 모델은 CPU로 충분히 실행됐고 GPU 학습은0회다. 필요한 FEA/GPU를 허용한 사용자 지시는 유지되지만, 다음 고비용 계산은 정확한 bridge와 목표 label 정의를 확보한 뒤 결손을 채우는 case로 제한한다. 현재 증거만으로 추가 GPU 학습이나 FEA 물량 확대를 시작할 이유는 확인되지 않았다.

[GitHub 인계 이슈](https://github.com/enjoyneer87/JEET-repro/issues/1)와 [원시 자료 반환 계약](REMOTE_HANDOFF_KO.md)에 최신 AF 원본, 대표 Field 샤드, source별 scalar, 평균 구간·전류·수렴 근거, exact/nominal 형상 대응을 요청했다. 다른 로컬의 반환 자료가 도착하면 원본 해시·품질·group split을 검토한 뒤 다음 비교를 진행한다.
