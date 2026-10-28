# A2 완료와 첫 C/G 개선 실험

2026-10-28. A2 계산은 정상 완료됐지만 전기각 입력의 정확도 개선은 확인되지 않았다. 최종 3시드 전체 B벡터 오차는 A0 20.5993%, A2 22.8470%, paired delta는 -2.2652/-4.1725/-0.3054 pp다. highPA 악화 +2.5991 pp도 기존 +1pp 한도를 넘었다. **NO_IMPROVEMENT_CLAIM**이다. 학습·평가 누락/실패0, 회수42개 파일 SHA 검증 완료. [Astra Max 최종 판정](ASTRA_A2_FINAL_REVIEW_KO.md).

최근 모델의 공극·토크는 아직 공통 평가하지 않았다. E1 공극4.9563%·토크3.2146%와 최근 전체B벡터 점수를 비교해 퇴보 원인을 확정하지 않는다. [차이 진단](../performance_diagnosis_20261028/REPORT_KO.md).

후속 C/G는 연결 성분 내부(C)와 같은 graph 전체(G)의 면적 가중 hidden 요약을 자계 head로 전달한다. 두 arm 모두 bias 없는32×32 projection을 zero-init하고 총71,206 parameter로 맞췄다. 기계각·width32/depth3·동일328train/32dev·phase9·seed17/29/43·88565update·Adam/LR·final checkpoint를 유지한다. 이 실험은 **정보 전달 경로 개선**이며 native FE mesh 복원이 아니다. 잘못된 a2 label/좌표최근접 edge는 추가하지 않는다.

기존 CPU 검사는 초기값·W0 원본 출력·batch 누출·G에만 상대 성분 자계 미분 경로가 있음을 확인했다. 360조건 native-edge component sidecar도 준비했다. 평가기 직렬화 복원 검사를 추가하고, GPU overfit·profile 전체15분 gate 후 처리량이 허용할 때만 최대3작업씩6run을 시작한다. 훈련 각2h·전체6h·UTC guard·STOP·PID/start-time·retry0을 고정했다. [계약](CONTEXT_CG_PLAN.json), [실제 현황](STATUS.json).

이번 범위 새FEA0, prior solve0, E1weights/teacher0, oldval/test0이다. E1 구조/code는 후속 from-scratch 단일축 후보로 두고 prior·기존weights는 donor/source/nativeID/normalizer gate가 UNKNOWN이어서 HOLD다. 공극 가중5와 spectral30도 이번 비교에 섞지 않는다.

dev32는 E1 oldTRAIN에 노출된 내부개발 집합이다. B²는 파생 감독량으로 AC손실이나 JEET canonical Hybrid 분모가 아니다. reserved R1_0199/R1_0076 및 alias·HalfSC32/4·exact bridge0을 유지한다.

재현: 원본 graph/checkpoint는133에 보존하고 private source-key 경로 매핑을 사용한다. 공유 코드의 --bundle/--components/--checkpoint 입력을 hash가 일치하는 자료에 연결한다. train_context.py와 evaluate_context.py는 수정된 공통계약과 동일 scope를 검사하며, guard_context.py가 정해진6run만 실행한다. raw/대형NPZ/weights는 Git에 포함하지 않는다.

## 최신 실행 증거

UTC 2026-10-28T10:09:01.050536+00:00: RUNNING. 평가기 복원 PASS, overfit PASS_ENGINEERING_ONLY, 본학습 시작 6/6. 이 상태는 정확도 개선 판정이 아니다. [상태와 소유 신원](STATUS.json), [회수 SHA receipt](artifacts/COLLECTION_RECEIPT.json).
