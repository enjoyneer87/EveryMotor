# MotorAI 개선 요청·아이디어 통합 대장

갱신일: 2026-10-28 15:20 KST. 원본은 `IMPROVEMENT_REGISTER.json`이며 이 문서는 `python update_improvement_register.py`로 갱신한다.

사용자 요청, Astra 판정, 실행 계획과 실제 결과를 같은 항목 ID로 추적한다. 제안·판정·실행 완료는 서로 다른 상태다. 새 계산은 이 대장만으로 시작하지 않고 별도 bounded 실행 계약과 원자료/분할 gate를 통과해야 한다.

Astra 검토 채팅: `01a17d9e-8a81-7262-9eae-e086c1957892`. 전체 형상·OP·모달리티는 geometry/alias group을 상속하고, reserved 및 기존 동결시험 계약을 유지한다.

| ID | 우선순위 | 개선 요청 / 아이디어 | 상태 |
|---|---|---|---|
| [MI-001](#mi-001) | P0 | 기존 다형상 MES/NPZ/TXT/HDF5 회수와 실행 환경 | 완료·보류 자료 별도 |
| [MI-002](#mi-002) | P0 | 충분한 형상 수로 133 확대 학습 | 실행 완료 |
| [MI-003](#mi-003) | P0 | 이전 MotorAI 5% 이하 성과의 지표 감사 | 원점수·target 경로 확인 |
| [MI-004](#mi-004) | P0 | 전체 자계 정확도 개선 | Astra 판정 중 |
| [MI-005](#mi-005) | P0 | 이전 MotorAI 가중치의 전이학습 | 조건부사용·선행 감사 필요 |
| [MI-006](#mi-006) | P0 | 두 솔버·다중 정밀도·멀티모달 네트워크 | 부분 구현·연결 전략 Astra 판정 중 |
| [MI-007](#mi-007) | P0 | 슬롯당 4·6·8도체 Field 데이터 확보 | 기하 완료·Field 확보 전 |
| [MI-008](#mi-008) | P1 | 추가 형상 수 확대와 보류 원자료 복구 | 계보 복구 필요 |
| [MI-009](#mi-009) | P0 | 구리 B²·전류·AC 손실 연결 | 물리량 정의·적분 검증 필요 |
| [MI-010](#mi-010) | P1 | 미사용 위상36개에서 시간 일반화 진단 | L12 완료·A2 계약 승인 대기 |
| [MI-011](#mi-011) | P1 | 국소3-hop 한계와 global context/모델 용량 | 제안·우선순위 재검토 |
| [MI-012](#mi-012) | P0 | 공정 비교·사전학습 누출·Full-FEA 비용 | 항상 적용 |
| [MI-013](#mi-013) | P0 | 동결 새OP 손실 비교 반환 | 6건 회수·집계 완료,2건 미실행 보존 |
| [MI-014](#mi-014) | P0 | 실행 자원·마감·보존 관리 | 항상 적용 |
| [MI-015](#mi-015) | P0 | Astra 전체계획·주요판정 / Sol 구현·실행 운영 | Astra 계획 v1 복원·Claude 대행 실행 중 |

<a id="mi-001"></a>
## MI-001 · 기존 다형상 MES/NPZ/TXT/HDF5 회수와 실행 환경

**P0 · 완료·보류 자료 별도**

**요청 출처:** 사용자 요청

**요청과 목표:** 새 해석에 앞서 이미 생성한 다형상 Field를 찾아 학습에 재사용한다. HDF5도 사용 가능한 실행 환경을 확보한다.

**확인된 근거와 현재 결과:** 480조건/360선언 형상그룹 중 원본·사본 hash연결440. 초기40 source미연결, native/manifest 불일치80은 첫 학습에서 보류. native h5py3.16 환경으로360조건 all45step reader PASS,3240graph 준비. Docker는 HDF5의 필수조건이 아니다.

**Astra 판정:** B-only field reuse 조건부사용; J/Je 및 AC 정답으로 확대 해석하지 않는다.

**다음 행동:** 보류 자료의 source/input 계보를 복구해 입수·검증 수를 갱신한다.

근거 파일: [DATA_REUSE_KO.md](DATA_REUSE_KO.md) · [NATIVE_DOE_CATALOG.json](NATIVE_DOE_CATALOG.json) · [SCALE360_PREP_ATTEMPTS.json](SCALE360_PREP_ATTEMPTS.json) · [SCALE360_BUNDLE_MANIFEST.json](SCALE360_BUNDLE_MANIFEST.json)

<a id="mi-002"></a>
## MI-002 · 충분한 형상 수로 133 확대 학습

**P0 · 실행 완료**

**요청 출처:** 사용자 요청

**요청과 목표:** 20형상 pilot만으로 결론내지 않고 충분한 샘플을 학습하며 큰 학습은133에 우선 배치한다.

**확인된 근거와 현재 결과:** 320형상그룹/360조건 전체 중 학습288그룹328조건,내부검증32그룹32조건.133 L40S에서30epoch/88565updates 완료. dev32 B벡터 nRMSE18.0765%,기존pilot dev4에서16.6035%.32형상은학습수가아닌내부검증수다.

**Astra 판정:** 가용 형상 범위 확대 우선; 288/32그룹 및 기존pilot16/4역할 보존 확인.

**다음 행동:** 완료 checkpoint와 같은 분할·지표로 후속 기법을 비교한다. 표본/update/LR가 함께 바뀌어 순수 표본 효과로 주장하지 않는다.

근거 파일: [SCALE360_PLAN.json](SCALE360_PLAN.json) · [SCALE360_SELECTION_FROZEN.json](SCALE360_SELECTION_FROZEN.json) · [SCALE360_TRAIN_STATUS.json](SCALE360_TRAIN_STATUS.json) · [SCALE360_REPORT_KO.md](SCALE360_REPORT_KO.md)

<a id="mi-003"></a>
## MI-003 · 이전 MotorAI 5% 이하 성과의 지표 감사

**P0 · 원점수·target 경로 확인**

**요청 출처:** 사용자 지적

**요청과 목표:** 이전 MotorAI 학습 성과와 현재 결과를 같은 의미의 지표로 비교한다.

**확인된 근거와 현재 결과:** E1 legacy6: 전체 |B|11.096697%,공극4.956319%,토크3.214573%. all18:전체 |B|10.859%,공극4.071%,토크2.982%. 현재18.0765%는다른분할의면적가중B벡터오차로직접비교불가.

**Astra 판정:** E1 target=B, nodal Bx/By→element 평균을 확인. 초기 nodal-A/curl 단정은 철회한다.

**다음 행동:** 실제 checkpoint hash·정규화·노출데이터 감사와 공통 평가 가중치를 추가 확인한다.

근거 파일: [PRIOR_MOTORAI_COMPARISON_KO.md](PRIOR_MOTORAI_COMPARISON_KO.md)

<a id="mi-004"></a>
## MI-004 · 전체 자계 정확도 개선

**P0 · Astra 판정 중**

**요청 출처:** 사용자 요청

**요청과 목표:** 이전 모델의 전체 |B| 약11.1%도 개선한다. 전체·공극·구리 영역 성능을 함께 본다.

**확인된 근거와 현재 결과:** 현 모델 dev32 전체 B벡터18.08%,구리 영역34.61%. 공극/토크5% 이하를 전체 자계 성과로 바꾸지 않는다.

**Astra 판정:** 새 평가·모델 전략 검토 중; 목표치와 PASS gate는 아직 미확정.

**다음 행동:** 공통 데이터·형상분할·위치·단위·가중치에서 |B|와B벡터를모두평가하고,영역별성능 및 B²/AC 연결을확인한다.

근거 파일: [SCALE360_TRAIN_RESULT.json](SCALE360_TRAIN_RESULT.json) · [ASTRA_REVIEW_KO.md](ASTRA_REVIEW_KO.md) · [PRIOR_MOTORAI_COMPARISON_KO.md](PRIOR_MOTORAI_COMPARISON_KO.md)

<a id="mi-005"></a>
## MI-005 · 이전 MotorAI 가중치의 전이학습

**P0 · 조건부사용·선행 감사 필요**

**요청 출처:** 사용자 명시 요청

**요청과 목표:** 검증된 이전 모델의 가중치·물리 표현을 활용해 새 솔버/형상/권선 모델로 전이한다.

**확인된 근거와 현재 결과:** E1checkpoint f5863662...와r3/priorcode를발견/해시확인했다. prior-native binding/전체donorreserved제외/normalizer호환은UNKNOWN이라weights/teacher는HOLD.

**Astra 판정:** 원 nodal branch/normalizer를 보존한 frozen backbone+작은 residual head 우선. 현재element모델에직접삽입FAIL. 준비30분/조건통과후133비교60분제안,아직새실행계약은아니다.

**실행·해석 관문:** pretraining에 노출된 geometry를 새 독립형상시험으로 쓰지 않는다. 기존 checkpoint와 구조·recipe 재사용을 구분한다.

**다음 행동:** Astra가 재사용 모듈,freeze/unfreeze,adapter/새head,학습예산을정한다. 이전모델이현재dev32를pretrain에서보았는지감사한다.

근거 파일: [PRIOR_MOTORAI_COMPARISON_KO.md](PRIOR_MOTORAI_COMPARISON_KO.md) · [ASTRA_TRANSFER_MULTIMODAL_KO.md](ASTRA_TRANSFER_MULTIMODAL_KO.md) · [TRANSFER_ADMISSION.json](TRANSFER_ADMISSION.json)

<a id="mi-006"></a>
## MI-006 · 두 솔버·다중 정밀도·멀티모달 네트워크

**P0 · 부분 구현·연결 전략 Astra 판정 중**

**요청 출처:** 사용자 명시 요청

**요청과 목표:** Motor-CAD와Elmer의이산화차이를관리하도록솔버전용네트워크를연결하고형상/OP/Field/scalar 정보를통합한다.

**확인된 근거와 현재 결과:** 두 domain encoder/shared design-OP 표현 및 누락modality/분모/bridge gate 코드가있다. 실제추가학습은Motor-CAD B전용이며연결학습성과는아직없다. exact bridge0.

**Astra 판정:** B-field 중심으로조건부사용. solver전용encoder/normalizer/decoder와sharedgeometryOP/fidelity를연결하고missingmodality는mask. J/Je/AC는정의확인후활성화.

**실행·해석 관문:** nominal/near match를paired truth로사용하지않고solver와numerical fidelity를분리한다.

**다음 행동:** solver전용normalizer/adapter/decoder,공유표현,field/scalar손실,누락modality마스크를정하고최소crosspoint를제안한다.

근거 파일: [MULTISOLVER_DESIGN_KO.md](MULTISOLVER_DESIGN_KO.md) · [dual_solver_mgn.py](dual_solver_mgn.py) · [DUAL_NETWORK_CONTRACT_TESTS.json](DUAL_NETWORK_CONTRACT_TESTS.json)

<a id="mi-007"></a>
## MI-007 · 슬롯당 4·6·8도체 Field 데이터 확보

**P0 · 기하 완료·Field 확보 전**

**요청 출처:** 사용자 명시 요청

**요청과 목표:** R1 272형상에슬롯당도체4/6/8을다양화한다. 새과도해석은각history총128step상한이다.

**확인된 근거와 현재 결과:** 816개geometry-only QA완료,mesh/FEA/제조검증은별도UNKNOWN. 현재EveryMotor다형상field는6도체중심. Motor-CAD probe02의4/6설정사본은실제drawncount검증을대체하지않는다.

**Astra 판정:** 전이·멀티모달 전략과 함께 최소 추가점 선정 검토 요청.

**실행·해석 관문:** native권선/회로/치수검증,물성·대칭·stack·전류관례확인후해석진입. 숨은자동fit과128자동연장금지.

**다음 행동:** 대표base/OP/4·8도체native geometry·전류·mesh ID를먼저확인한뒤bounded MS/Full후보를case_queue에사전선언한다.

근거 파일: [configuration_queue.csv](configuration_queue.csv) · [GEOMETRY_QA.json](GEOMETRY_QA.json) · [MOTORCAD_INTERACTIVE_READBACK.json](MOTORCAD_INTERACTIVE_READBACK.json)

<a id="mi-008"></a>
## MI-008 · 추가 형상 수 확대와 보류 원자료 복구

**P1 · 계보 복구 필요**

**요청 출처:** 사용자 요청

**요청과 목표:** 더많은형상을학습하되동일형상여러OP와중첩DOE를독립형상으로세지않는다.

**확인된 근거와 현재 결과:** 가용320그룹은현재학습/내부검증에전체사용중.초기40source미연결/입력불일치80의수는추가독립형상수와동일하지않다.

**Astra 판정:** 분할누출·중복을먼저검사.

**다음 행동:** 보존root·원본hash·native입력을대조해복구하고alias group과기존split을상속한다.

근거 파일: [NATIVE_DOE_CATALOG.json](NATIVE_DOE_CATALOG.json) · [DATA_REUSE_KO.md](DATA_REUSE_KO.md)

<a id="mi-009"></a>
## MI-009 · 구리 B²·전류·AC 손실 연결

**P0 · 물리량 정의·적분 검증 필요**

**요청 출처:** 최종 연구 목표

**요청과 목표:** Field예측을전체모터의AC권선손실로연결해최신AF보다정확하고효율적인모델을만든다.

**확인된 근거와 현재 결과:** Motor-CAD J/Je의total/source/induced의미는UNKNOWN. element A를임의nodalA로curl하지않는다. 평균B제곱과B제곱평균을구분한다. 추가동결B²진단의전체45위상상대MAE14.4803%/bias-9.4580%,truth9/45차이0.2619%.이값은AC손실이아니다.

**Astra 판정:** 멀티모달손실/최소물리검증전략검토중.

**실행·해석 관문:** 보정과적용의분모동일. Ref AF계수를다른Motor-CAD/GetDP분모에곱하지않는다.

**다음 행동:** 재료경계·도체/circuit ID·단위·공간가중치·평균구간을확인하고저장손실/전류와재적분대조한다.

근거 파일: [BASELINE_CONTRACT_KO.md](../BASELINE_CONTRACT_KO.md) · [DATA_REUSE_KO.md](DATA_REUSE_KO.md) · [quality_summary.json](../new_op_validation_20261027/quality_summary.json) · [FROZEN_FIELD_DIAGNOSTIC_REPORT_KO.md](FROZEN_FIELD_DIAGNOSTIC_REPORT_KO.md)

<a id="mi-010"></a>
## MI-010 · 미사용 위상36개에서 시간 일반화 진단

**P1 · L12 완료·A2 계약 승인 대기**

**요청 출처:** Astra 제안

**요청과 목표:** 45개저장시각중학습에쓴9개외36개에서위상오차를확인한다.

**확인된 근거와 현재 결과:** L12 phase45 equal-update(2 seed): 전체 B벡터 +0.81 pp 개선(seed 일치)이나 1.0 pp 기준 미달, seen9 +1.54 pp 악화로 NO_IMPROVEMENT_CLAIM. 41–44 외삽 오차는 −8.8 pp로 해소되어 위상 범위 기제는 확인됨.

**Astra 판정:** D01-v3(Claude 대행): 다음 단일축은 phase9 데이터의 전기각 주기 인코딩(A2). 전기 주기 각도 폭 확인 후 계약, 사용자 승인 후 실행.

**다음 행동:** A2_ELECTRICAL_ANGLE_RUN_CONTRACT.json 사용자 승인 후 게이트(평가기·prefetch·변환) → A2 seed 병렬 실행.

근거 파일: [SCALE360_PLAN.json](SCALE360_PLAN.json) · [native_element_graph.py](native_element_graph.py) · [FROZEN_FIELD_DIAGNOSTIC_REPORT_KO.md](FROZEN_FIELD_DIAGNOSTIC_REPORT_KO.md) · [FROZEN_FIELD_DIAGNOSTIC_REVIEW_KO.md](master_plan_artifacts/FROZEN_FIELD_DIAGNOSTIC_REVIEW_KO.md) · [L12_PHASE45_RUN_CONTRACT.json](master_plan_artifacts/L12_PHASE45_RUN_CONTRACT.json) · [L12_PHASE45_REPORT_KO.md](master_plan_artifacts/L12_PHASE45_REPORT_KO.md) · [L12_PHASE45_RESULT.json](master_plan_artifacts/l12/L12_PHASE45_RESULT.json) · [A2_ELECTRICAL_PERIOD_AUDIT.json](master_plan_artifacts/A2_ELECTRICAL_PERIOD_AUDIT.json) · [A2_ELECTRICAL_ANGLE_RUN_CONTRACT.json](master_plan_artifacts/A2_ELECTRICAL_ANGLE_RUN_CONTRACT.json)

<a id="mi-011"></a>
## MI-011 · 국소3-hop 한계와 global context/모델 용량

**P1 · 제안·우선순위 재검토**

**요청 출처:** Astra 검토와 실행자 제안

**요청과 목표:** 원거리회전자/고정자정보전달과구리field표현의한계를확인한다.

**확인된 근거와 현재 결과:** 현field head는국소3-hop+design/OP조건을사용하고pooled graph정보를각element에재주입하지않는다.용량부족이나이구조가오차원인이라고확정하지않았다.

**Astra 판정:** 기존E1재사용/공통평가를우선하고generic global-context는후순위로둔다.

**다음 행동:** 기존E1재사용경로의prior/노출/메모리gate를먼저확인한다.

근거 파일: [dual_solver_mgn.py](dual_solver_mgn.py) · [ASTRA_REVIEW_KO.md](ASTRA_REVIEW_KO.md)

<a id="mi-012"></a>
## MI-012 · 공정 비교·사전학습 누출·Full-FEA 비용

**P0 · 항상 적용**

**요청 출처:** 사용자 연구 계약

**요청과 목표:** 최신AF/scalar잔차/MS사전학습/다중소스를같은Full예산으로비교하고독립시험을보존한다.

**확인된 근거와 현재 결과:** 기존AC보정예산63과새OP예측은동결.기존E1의Full450train donor를MS교사로표현하거나총예산63안에숨길수없다.

**Astra 판정:** 전이학습누출및예산설계검토중.

**실행·해석 관문:** R1_0199/R1_0076및alias는pretrain/train/tune/eval전면제외;pretrained노출geometry는독립형상시험이아니다;HalfSC32/4분리.

**다음 행동:** 총historical donor비용과새증분Full예산을별도ledger로기록하고동일split/ID/평가량으로비교한다.

근거 파일: [BASELINE_CONTRACT_KO.md](../BASELINE_CONTRACT_KO.md) · [SCALE360_SELECTION_FROZEN.json](SCALE360_SELECTION_FROZEN.json) · [PREDICTIONS_FROZEN.json](../new_op_validation_20261027/PREDICTIONS_FROZEN.json)

<a id="mi-013"></a>
## MI-013 · 동결 새OP 손실 비교 반환

**P0 · 6건 회수·집계 완료,2건 미실행 보존**

**요청 출처:** 기존 승인 실험

**요청과 목표:** 동결8방법을새Ref/SC속도8조건에서평가하고원시A/회로파형으로재적분한다.

**확인된 근거와 현재 결과:** 6완료중current/nonlinear PASS5,whole-current FAIL1,SC후속2미실행.6건포함wMAE AF3.2645%,scalar1.7820%,MS-MGN3seed평균0.9658%.2MSgroup의known-template시험이며formal수렴UNKNOWN,완전한8조건최종시험으로주장하지않는다.

**Astra 판정:** W04(Claude 대행): 품질 PASS 5건만 보면 AF 0.748%로 순위가 뒤집힌다. 6건 결과는 품질 FAIL인 SC_I460.1_N12000(Full 가중 51.4%)이 좌우하므로 어느 쪽 우월도 주장하지 않는다. L18 benchmark의 동결 reference로만 보존한다.

**다음 행동:** 실패·미실행을포함한보고서와hash/품질층을게시한다.라벨로모델을재튜닝하지않는다.

근거 파일: [EVALUATION.json](../new_op_validation_20261027/EVALUATION.json) · [quality_summary.json](../new_op_validation_20261027/quality_summary.json) · [FROZEN_NEWOP_FINAL_SCOPE_KO.md](master_plan_artifacts/FROZEN_NEWOP_FINAL_SCOPE_KO.md)

<a id="mi-014"></a>
## MI-014 · 실행 자원·마감·보존 관리

**P0 · 항상 적용**

**요청 출처:** 사용자 운영 지시

**요청과 목표:** 큰학습은133,234는검증/작은비교,필요시NAS보존,2026-10-28 08:00KST까지만실행한다.

**확인된 근거와 현재 결과:** 확대133작업은00:46KST경완료(exit0).새실험은별도case/label/시간/메모리/disk/횟수/소유PID/수렴·중단계획을먼저등록한다.

**Astra 판정:** 계산마감과소유권규칙유지.

**다음 행동:** Astra판정후구체적새실험만사전선언하고deadline/STOP/소유신원guard를적용한다.원본·사용자앱·이전PAUSED캠페인을보존한다.

근거 파일: [CONTINUATION_KO.md](CONTINUATION_KO.md) · [SCALE360_PLAN.json](SCALE360_PLAN.json)

<a id="mi-015"></a>
## MI-015 · Astra 전체계획·주요판정 / Sol 구현·실행 운영

**P0 · Astra 계획 v1 복원·Claude 대행 실행 중**

**요청 출처:** 사용자 명시 요청

**요청과 목표:** Astra가전체개발계획을수립하고Sol은그계획에따라실행한다.판단필요사항은Astra에게되돌린다.

**확인된 근거와 현재 결과:** Astra 계획 턴은 Codex 사용량 한도로 파일 기록 전에 멈췄다. Claude가 Astra의 JS 호출을 그대로 재생해 v1(19단계·10 gate·9 판정)을 복원했다. 08시 창은 b46a4d7로 종료했다. 이후 사용자 결정(D07)에 따라 Claude가 Astra 판정과 Sol 실행을 겸한다.

**Astra 판정:** v1 D01: phase9 대 phase45 equal-update 비교 계약을 먼저 작성한다. 학습은 계약 승인 전 HOLD. E1 weights는 HOLD.

**다음 행동:** SOL_EXECUTION_LOG.json에 단계별 receipt를 기록한다. W04→W03→W07 순으로 문서·계약을 작성하고, W05/W06은 읽기전용 감사로 진행한다.

근거 파일: [ASTRA_TRANSFER_MULTIMODAL_KO.md](ASTRA_TRANSFER_MULTIMODAL_KO.md) · [CONTINUATION_KO.md](CONTINUATION_KO.md) · [MASTER_DEVELOPMENT_PLAN_KO.md](MASTER_DEVELOPMENT_PLAN_KO.md) · [SOL_EXECUTION_LOG.json](SOL_EXECUTION_LOG.json) · [SESSION_CLOSURE_20261028_KO.md](SESSION_CLOSURE_20261028_KO.md)

## 갱신 규칙

새 요청은 기존 항목을 찾아 내용·근거·다음 행동을 갱신한다. 별도 연구 축이면 새 MI 번호를 추가한다. 판정이 바뀌면 변경 이유와 근거를 보존하고, 코드 존재·reader PASS·학습 완료를 물리 정확도 PASS로 확대하지 않는다. 결과에는 실험 ID, 분할, 지표, 시드, Full donor 비용과 남은 UNKNOWN을 연결한다.
