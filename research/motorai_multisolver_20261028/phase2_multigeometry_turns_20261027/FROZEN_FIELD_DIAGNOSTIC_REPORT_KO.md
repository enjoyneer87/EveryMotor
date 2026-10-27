# 동결 모델의 전체 자계·구리 B²·미사용 위상 진단

32개 내부검증 형상에서45개 저장위상1440graph를 동결 SCALE36030epoch checkpoint로 평가했다. 새 FEA0·학습update0, 실행85초/exit0, 모델·checkpoint hash는 유지됐다.

| 시각 범위 | 전체 B벡터 nRMSE | 전체 B크기 nRMSE | 구리 B벡터 nRMSE | 구리 B크기 nRMSE |
|---|---:|---:|---:|---:|
| 기존9위상 | 18.0765% | 11.5795% | 34.6131% | 26.8823% |
| 미사용36위상 | 20.9153% | 13.9440% | 35.1553% | 27.6082% |
| 전체45위상 | 20.3790% | 13.5041% | 35.0474% | 27.4645% |

기존9위상의 B벡터18.076472603%는 저장최종성적과 1.9e-08pp 차이로 산술 재현됐다. 이 검사는 파일/실행 경로의 재현이며 물리 수렴 검증이 아니다.

## 도체별 공간 평균 B²

전체45위상의 도체별 공간mean B² 예측은 면적가중 상대MAE 14.4803%, signed bias -9.4580%다. B벡터의 방향·크기 오차와 함께 본다.

정답의9위상/45위상 B² 샘플 차이는 상대MAE 0.2619%, bias 0.0612%다. 이 숫자를 AC손실 보존·고조파 정확도라고 해석하지 않는다.

각 native conductor region에서 ∑area·|B|²/∑area와 |∑area·B/∑area|²를 별도 열로 기록했다. CSV는이값의균일위상평균을제공한다. 시간적분·end winding·J/Je·Joule/AC손실·JEETcanonicalHybrid분모는계산하지않았다.

## 해석과 다음 판단

같은 내부검증 형상의 미사용 위상에서 오차가 증가했다. 원인이 위상 범위·표현·학습량 중 무엇인지는 Astra가 판정한다. 현재32그룹은 최종 독립시험이 아니며, 과거E1의다른split/가중치/노드후처리점수와 직접 우열비교하지 않는다. 전체 |B| 개선과 구리 B² 편향을 후속 공통 평가의 근거로 제공한다.

원소면적 가중치와 a2제외는 기존reader계약을유지했다. conductor_B2.csv는 native case/reg ID, snapshot_errors.csv는 case/step ID로 연결한다. 큰Graph와원시HDF5는private mapping에보존하고작은결과/CSV/manifest만EveryMotor에게시한다.

관련파일: FROZEN_FIELD_DIAGNOSTIC_PLAN.json, RESULT.json, STATUS.json, ARTIFACTS.json 및 FROZEN_DIAGNOSTIC_*.csv. 과학적 후속우선순위는 MASTER_DEVELOPMENT_PLAN과개선대장에반영한다.
