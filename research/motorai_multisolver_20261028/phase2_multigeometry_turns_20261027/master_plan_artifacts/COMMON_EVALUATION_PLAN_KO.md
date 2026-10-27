# W07 · 공통 평가·노출·비용 계약 요약

계획 W07 · 작성 Claude(Sol 실행, Astra 대행 판정) · 2026-10-28 KST. 이 계약은 L12 결과가 하나도 없는 상태에서 동결했다. 원문은 `COMMON_EVALUATION_CONTRACT.json`과 `EXPOSURE_AND_COST_LEDGER.json`이다.

## 평가 집합과 지표

- 평가 집합: SCALE360 dev 32그룹·32조건(전부 460Arms)의 45위상 전체, 1440 snapshot. E1 oldTRAIN에 노출된 개발집합이며 최종 독립시험이 아니다.
- 평가기: 동결 진단과 같은 `evaluate_frozen_field_diagnostic.py`(`94ccc9c7…`)를 쓴다. 평가기를 바꾸면 계약 버전을 올리고 모든 arm을 다시 평가한다.
- 주 지표: all45 전체 면적가중 B벡터 nRMSE와 |B| nRMSE.
- 보조 지표:
  - seen9/unused36
  - 위상 영역(seen·사이·41–44)
  - 구리·공기 영역
  - 구리 meanB² 상대MAE와 bias
  - 사례별 최악값·중앙값
  - PhaseAdvance≥70° 7형상과 나머지 25형상
- 모든 % 지표 옆에 RMSE[T]를 함께 보고한다. 실패나 NaN은 해당 arm의 실패로 세며, 어려운 사례를 arm별로 빼지 않는다.

## L12 판정 규칙(결과 열람 전 등록)

- **개선 인정:** all45 전체 B벡터 nRMSE가 A0보다 1.0pp 이상 낮아야 한다. arm별 seed가 2개 이상이면 관측된 seed 차이의 2배도 넘어야 한다.
- **비악화 조건:**
  - seen9 전체 B벡터: 0.5pp 이내
  - all45 구리 B벡터: 1.0pp 이내
  - PA≥70° 그룹 평균: 1.0pp 이내
- **근거:** pilot에서 seed17과 seed29의 차이는 nRMSE 약 59% 수준에서 2.1pp(상대 3.6%)였다. 20% 수준으로 환산하면 약 0.7pp이다.
- seed가 1개뿐이면 통과해도 "단일 seed 개발 결과"로만 보고한다.

## 비교 조건

다음 조건은 모든 arm에서 같다. 단, 선언한 단일 변경축은 예외이다.

- 모델 코드 `ae1966b5…`
- train328/dev32 selection `21e78f73…`
- Adam, batch 1 graph, width 32, depth 3, clip 10
- update 88565회
- update 순번별 동일 학습률
- 마지막 checkpoint만 사용(dev로 고르지 않음)

## 노출·비용

- phase45는 새 Full 해석을 추가하지 않는다. 기존 transient 해석이 45스텝을 이미 담고 있기 때문이다.
- 다만 읽는 label snapshot은 5배(2952→14760)가 된다. 보고서에는 해석 수(불변)와 snapshot 접근 수(×5)를 모두 적는다.
- E1 이력 비용과 DOE 생성 비용은 sunk cost로 기록한다. L12 증분 비용은 준비·전송·GPU·평가 wall time뿐이다.
