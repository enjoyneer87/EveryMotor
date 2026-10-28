# Astra Max · 다음 개선 실험 판정

이 문서는 읽기 전용 설계 판정이다. 반환 시점의 A2 진행 상태와 source를 보존하며, 이후 A2 완료 상태는 별도 receipt로 갱신한다. 파일 링크만 상대 경로로 바꿨다.

**A2는 아직 최종 판정할 수 없습니다. 현재 상태는 `RUNNING_PENDING_VERDICT`입니다.** 새 학습·평가·FEA는 실행하지 않았습니다.

확인 가능한 최근 소유자 기록인 **16:24 KST**에는 A0_s43가 58,700, A2_s17/s29/s43가 각각 약 59,000/88,565 업데이트로 실행 중이었습니다. 현재 로컬에도 이 네 실행의 최종 평가·완료 receipt가 없습니다. 기존 A0_s17/s29를 포함하면 **최종 결과가 확보된 것은 6건 중 2건**입니다. 로컬 집계의 `INCOMPLETE_NO_CLAIM`은 아직 미회수된 실행을 포함한 상태이며, 종료 실패로 판정하지 않습니다.

전달된 후속 설계 검토까지 반영하면, **첫 실험은 ‘전체 그래프의 요약 정보를 자계 출력에 전달하는 효과’만 비교하는 것으로 권고합니다.** native 연결 복원, 모델 확대, prior, 공극 손실은 각각 분리합니다.

**현재 근거에 대한 판정은 다음과 같습니다.**

| 항목 | 판정 | 확인된 범위 |
|---|---|---|
| 현재 그래프의 rotor–stator 경로 단절 | **PASS: 존재 확인** | TRAIN 표본 18/18에서 경로 없음 |
| a2 제거와 연결 성분 분리의 관계 | **PASS: 표본에서 확인** | 추가 TRAIN 원본 6/6에서 전체 reference graph는 1개, a2 제외 후 2개 성분 |
| 전역 요약의 자계 출력 전달 | **현재 경로 없음** | `pooled`는 계산하지만 자계는 `self.field(h)`에서 바로 출력 |
| 단절 때문에 발생한 정확도 손실의 크기 | **UNKNOWN** | ablation 전에는 기여율을 산정할 수 없음 |
| 기존 E1 prior·가중치 재사용 | **HOLD** | native 결합·과거 라벨 노출·normalizer 확인 필요 |

공유 design/OP 조건은 현재도 양쪽 성분에 전달됩니다. 따라서 “전역 정보가 전혀 없다”는 결론은 부정확합니다. 확인된 문제는 **상대 영역의 메시지 처리 결과가 자계 출력으로 전달되지 않는다는 것**입니다. 현재 자계 예측에 연결된 파라미터는 30,594개입니다.  
근거: [그래프 감사](../performance_diagnosis_20261028/GRAPH_TOPOLOGY_AUDIT.json), [추가 native 감사](NATIVE_TOPOLOGY_PRIOR_AUDIT.json), [자계 출력 경로](../phase2_multigeometry_turns_20261027/dual_solver_mgn.py#L42).

**A2 완료 후에는 기존 기준을 그대로 적용합니다.**

- 각 3시드의 final 88,565-update checkpoint, 1,440개 평가 graph, source·selection·evaluator·checkpoint hash와 완료 receipt를 확인합니다.
- `Δ = mean(A0) − mean(A2)`, `spread = max(range(A0), range(A2))`.
- **Δ ≥ 1.0 pp이면서 Δ > 2×spread**, 평균 악화가 **seen9 ≤ 0.5 pp / copper ≤ 1.0 pp / PA≥70 그룹 ≤ 1.0 pp**여야 개선 주장 가능합니다.
- 시드별 paired delta 3개, 각 arm 평균·최솟값·최댓값, B 크기, 구리 B² bias, 위상별 경향을 함께 보고합니다. range는 신뢰구간이 아닙니다.

현재는 숫자가 없는 A2 시드를 추정하거나 중간 checkpoint로 대체하지 않습니다. [사전등록 평가계약](../phase2_multigeometry_turns_20261027/master_plan_artifacts/COMMON_EVALUATION_CONTRACT.json)

**첫 후속 실험 계약안: 요약 정보의 범위만 변경**

A2가 위 기준을 모두 통과하면 전기각 입력을, 통과하지 못하면 동결 A0의 기계각 입력을 양쪽 arm에 공통 적용합니다. A2 미완료 상태에서는 이 선택도 확정하지 않습니다.

| 항목 | C: 연결 성분 내부 요약 | G: 전체 그래프 요약 |
|---|---|---|
| 요약 대상 | 해당 요소가 속한 연결 성분 | 동일 물리 graph의 모든 유효 요소 |
| pooling | 면적 가중 평균 | 면적 가중 평균 |
| 자계 출력 | `field(hᵢ + W·cᵢ)` | 동일 |
| 추가 파라미터 | bias 없는 32×32 행렬 W | 동일 |
| 초기화 | W=0, 시드별 동일 backbone 초기값 | 동일 |

이 설계에서는 두 arm 모두 **1,024개 파라미터가 추가**됩니다. 기존 A0/A2 결과는 참고 기준으로 유지하되, **전역 전달 효과의 직접 대조군 C는 별도로 같은 조건에서 학습**해야 합니다. G와 기존 모델만 비교하면 추가 파라미터와 전역 전달의 효과가 섞입니다.

고정 조건은 width32/depth3, TRAIN328의 phase9 2,952 graph, 기존 split, seed17/29/43, optimizer·학습률·데이터 순서·88,565 updates·final checkpoint·손실·prior 없음입니다. 기존 edge와 a2 제외도 유지합니다. **각도 변경과 전역 전달 변경을 한 비교에서 동시에 적용하지 않습니다.**

실행 전에는 다음을 통과해야 합니다.

- 성분 구분과 pooling이 native connectivity·면적·입력 특징만 사용하고 정답 B를 사용하지 않을 것.
- 서로 다른 batch graph 사이에 pooling이 섞이지 않을 것.
- W=0일 때 기존 backbone과 출력이 일치할 것.
- 별도 비영점 W 검사에서 G에는 상대 성분→자계 출력 경로가 있고 C에는 없을 것.
- 파라미터 수·초기 상태·source와 입력 manifest hash가 두 arm에서 대응할 것.

새 비교의 성공 기준도 사전에 **G−C 비교에 동일한 1 pp·2×spread·no-harm 규칙**을 고정합니다. 통과해도 주장은 “이 구조에서 전역 요약 전달이 dev32 정확도를 개선했다”까지입니다.

**native 연결 복원은 별도 실험으로 둡니다.** 추가 감사가 확인한 것은 reference connectivity입니다. 회전 위치에서 a2의 실제 재메시 연결까지 복원됐다는 증거는 아닙니다. 따라서 a2의 잘못된 면적·edge vector·B target을 되살리는 수정은 허용하지 않습니다. native ID로 추적되는 reference 연결을 중계용으로 사용하는 실험은 후속 후보지만, 실제 물리 연결 복원과 구분하고 depth3에서 정보가 도달하는 경로 길이부터 검사해야 합니다. [이동 공극의 유효성 제한](../../../eval/mesh_regions.py#L142)

**E1 재사용은 다음처럼 분리하는 것이 타당합니다.**

| 재사용 대상 | 판단과 최소 조건 |
|---|---|
| 구조·코드 | from-scratch 재사용은 조건부 가능. 첫 용량 비교는 현재 구조의 width32→64처럼 한 축만 변경. h256/depth15 전체 도입은 별도 profile·smoke·소규모 overfit 후 검토 |
| 공극 가중 손실 | stationary a1의 native ID·공통 마스크 확인 후 별도 축으로 가능. 현재 면적 가중 손실의 a1 계수만 변경하고 전체 가중치 합으로 정규화 |
| spectral 손실 | 별도 축. band·각도·기저·정규화가 확인되기 전에는 E1의 계수를 그대로 복사하지 않음 |
| prior 캐시 | **HOLD.** 파일명·shape 일치만으로 source/native ID/geometry·OP/material/current scale 결합을 인정하지 않음 |
| E1 weights·teacher·distillation | **HOLD.** donor reserved 제외와 feature/normalizer 호환성이 미확정이며 dev32는 E1 oldTRAIN 노출 |

특히 기존 prior는 “현재 sample의 field를 읽지 않는다”는 사실만으로 무라벨 prior가 되지 않습니다. 코드상 `mu_r=20`은 과거 사례 4·7의 B를 이용한 sweep에서 선택됐고, winding template도 reference J로 보정됐습니다. 그 출처와 조정 비용을 기록해야 합니다. `nodal_prior_features()`는 캐시 누락 시 계산·덮어쓰기를 수행하므로, 감사에서는 호출하지 않고 기존 archive만 읽어야 합니다.  
근거: [prior 상수와 캐시 동작](../../../fem_warmstart/prior.py#L71), [winding 보정 출처](../../../fem_warmstart/winding.py#L30), [전이 admission](../phase2_multigeometry_turns_20261027/TRANSFER_ADMISSION.json).

**자원과 실행 경계는 아래 수준으로 제한하는 안입니다.**

| 단계 | 제안 상한 |
|---|---|
| 234 CPU: 구조·ID·마스크·hash 검사 | 30분 |
| 133 GPU: 독립된 소규모 smoke/profile | 15분 |
| C/G × 3시드 학습 | 6실행, 실행당 최대 2시간, 최대 3개 동시 실행은 profile 통과 시 |
| 동결 평가·회수·검증 | 45분 |
| 전체 | 예상 3–5시간, **hard cap 6시간**, retry 0 |

실제 처리량은 아직 측정 전인 추정입니다. smoke에서 남은 시간 내 완료가 어렵다고 판단되면 전체 학습을 시작하지 않습니다. timeout·NaN·hash 불일치·누락은 해당 소유 작업을 STOP하고, 시드를 줄이거나 시간을 연장해 성공으로 바꾸지 않습니다. 실행 소유자는 주 연구 chat의 Sol이며 PID/start-time, STOP 경로, source·입력·checkpoint hash를 기록합니다.

공통 평가 확장은 **별도 평가계약을 고정한 뒤** 수행해야 합니다. 최소 범위는 dev32의 native ID·공통 마스크, 면적 가중/비가중 B-vector·|B|, stationary a1, 예측과 정답에 동일한 Arkkio 연산자입니다. a1 정보가 없을 때 전체 airgap으로 대체하는 fallback은 허용하지 않습니다. 저장 예측이 있으면 CPU 집계부터 재사용하고, 없으면 필요한 동결 추론만 별도로 계수합니다. 이는 AC 손실이나 독립 최종시험 성과가 아닙니다.

현재 후속 실험은 **A2 완료·과학 판정·실행 전 검사까지 HOLD**입니다. 이 채팅에서는 설계 검토만 수행했으며, A2 완료확인 heartbeat는 아직 완료 조건에 도달하지 않아 유지합니다.


