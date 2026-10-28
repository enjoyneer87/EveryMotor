# C/G 단일축 비교 완료 · 개선 주장 없음

2026-10-28. 최종 3시드 전체 B벡터 nRMSE는 **C 20.425025% / G 20.472808%**다. 사전등록 Δ=C−G는 **-0.047783 pp**로 최소1pp와2×spread 기준을 충족하지 못했다. 산술 판정은 **NO_IMPROVEMENT_CLAIM**이다. [Astra Max 최종 판정](ASTRA_CONTEXT_CG_FINAL_REVIEW_KO.md). 평균 no-harm 통과와 정확도 개선 통과는 별개다.

6개 모두88,565 update와각1,440 snapshot 평가를 정상 완료했다. 누락/실패0, 소유 live PID0, 원격GPU idle을 확인했다. 작은결과64파일, frozen source15파일, native component sidecar360개, 최종checkpoint6개의 SHA/크기를 검증했다. [결과 JSON](CONTEXT_CG_RESULT.json), [출처 감사](TERMINAL_SOURCE_AUDIT.json), [항목별 품질](quality_summary.json).

## 사전등록 비교

| seed | C 전체B벡터 % | G 전체B벡터 % | C−G pp |
|---|---:|---:|---:|
| 17 | 19.896033 | 20.406160 | -0.510127 |
| 29 | 20.599103 | 20.132632 | +0.466471 |
| 43 | 20.779938 | 20.879632 | -0.099694 |

C range=0.883906pp, G range=0.747000pp, 2×spread=1.767812pp다. 모든3시드의 고정 마지막 checkpoint를 사용했고 seed/epoch 선택은 하지 않았다. range는 관측된 max−min 기술통계이며 신뢰구간이 아니다. arm 평균은 각시드 지표의 단순평균이다.

| 평균 no-harm 항목 | G−C pp | 허용악화 pp | 판정 |
|---|---:|---:|---|
| seen9_vector | +0.402660 | 0.5 | PASS |
| all45_copper_vector | -0.563286 | 1.0 | PASS |
| PA_ge_70_mean_case_vector | +0.303254 | 1.0 | PASS |

전체 B크기 평균은 C 13.431054% / G 13.448874%, 구리 B² signed bias는 C -6.677134% / G -9.586432%다. G의 구리 B벡터가 낮아져도 B² bias가 개선됐다는 뜻은 아니다. B²는 균일 snapshot의 공간평균 파생량이며 AC손실 또는 JEET canonical Hybrid 분모가 아니다.

## 고정한 실험 축과 범위

C는 같은 native-edge 연결 성분 내부, G는 같은 물리graph 전체의 면적가중 hidden mean을 `field(h+W·context)`에 전달한다. 양쪽 모두 bias없는32×32 W를zero-init해 총71,206 parameter로 맞췄다. 기계각·width32/depth3·328train/32dev·phase9·seed17/29/43·88,565update·Adam/LR·데이터순서·final checkpoint를 고정했다. [동결 계약](CONTEXT_CG_PLAN.json), [Astra Max 설계](ASTRA_MAX_REVIEW_KO.md).

이 비교는 **정보 전달 범위 실험**이며 native FE mesh 연결 복원이 아니다. invalid a2 label이나 좌표최근접 edge를 추가하지 않았다. 이 결과로 원 graph 단절이 무관하다고 결론 내리거나 성능 오차 기여율을 확정하지 않는다. E1공극4.9563%·토크3.2146%와 최근 전체B벡터 성적도 직접 비교할 수 없다. 공통dev32/nativea1/동일Arkkio토크 평가는 아직미수행이다. [최초 성능 차이 진단](../performance_diagnosis_20261028/REPORT_KO.md).

앞선 A2 전기각 비교도 A0평균20.5993%→A2 22.8470%로 개선기준을 충족하지 못했으므로 이번C/G는 기계각을 유지했다. [A2 최종 판정](ASTRA_A2_FINAL_REVIEW_KO.md).

## 실행과 재사용 경계

전체단계는 2.929시간에 끝나 6h 마감을 지켰고 각train2h/eval1800s/engineering900s/retry0을 유지했다. 정식train총531,390update, engineering8,914update, 마지막동결평가8,640snapshot이다. worker시간 합계는 병렬중첩되므로 stage wall과 구분한다. [비용장부](EXECUTION_COST_LEDGER.json).

새FEA0, prior solve0이다. 동일 historical Full train328조건+dev32조건/320geometrygroup을 재사용했다. 원 graph/sidecar/checkpoint는133에 보존하며 작은코드/receipt/CSV/hash만Git에 게시한다. [회수 manifest](artifacts/COLLECTION_RECEIPT.json), [재현 안내](README_KO.md).

dev32는 E1 oldTRAIN노출 내부개발 집합이며 독립최종시험이 아니다. R1_0199/R1_0076/alias·oldval/test 제외, HalfSC32/4, exactcrosssolver0과paired alignment 비활성을 유지한다. E1 weights/prior/teacher는 source/nativeID/donor exposure/normalizer admission까지HOLD다. 공극가중5·spectral30·용량·prior를 이번C/G와 섞지 않았다. 새로운AF/scalar/MS/Field AC손실 비교는 동일Full예산과canonical분모 계약에서 별도로 수행해야 한다.

이 단계 완료 후 추가학습·FEA·seed·재시도는 시작하지 않는다. 최종 Astra 판정과 반환게시를 마치면 이번 소유 실행예약만 종료한다.

## 보조 관측과 후속 검토 순서

구리 B² 상대MAE 평균은 C 14.221327% / G 16.440306%로 G에서커졌다. 사후gate로추가하지않고부정적관측으로보존한다. [보조산술](SECONDARY_DIAGNOSTICS.json). 평균no-harm통과는모든seed의무악화를뜻하지않는다. seed43 seen9 +0.810pp/구리 +1.242pp악화도원결과에남아있다.

Astra Max의후속순서는공통dev32 nativea1·동일Arkkio평가 → E1구조/code from-scratch와용량/연결의분리검토 → 근거를확인한공극가중·spectral손실이다. prior와weights/teacher는각자의admission까지HOLD다. [검토우선순위](NEXT_REVIEW_QUEUE.json)는새실행계약이아니며이단계에서추가계산을시작하지않는다.
