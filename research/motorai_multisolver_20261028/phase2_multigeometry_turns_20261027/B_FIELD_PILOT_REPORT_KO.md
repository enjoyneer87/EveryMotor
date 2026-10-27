# Motor-CAD 다형상 FE 자계 pilot 결과

실제 native FE 요소 인접 그래프에서 Bx/By 사전학습을 실행했다. 새 FEA0건, 기존 Full donor16형상 학습+4형상 내부 검증의 비용을 별도 계산한다. 기존 Full63/새OP8의 모델·예측은 동결 유지한다.

| 서버 / seed | 상태 | epoch / updates | train nRMSE | 내부 검증 nRMSE | PyTorch peak MiB |
|---|---|---:|---:|---:|---:|
| 133 / 17 | COMPLETED | 10 / 1445 | 62.074% | 58.186% | 311.1 |
| 234 / 29 | COMPLETED | 10 / 1445 | 63.070% | 60.317% | 311.1 |

수치는 a2 제외 native 요소 면적 가중 B벡터 nRMSE다. 4형상 내부 검증이며 최종시험이 아니다. 계획한 마지막 epoch를 보고하고 best epoch/seed를 선택하지 않는다. GPU/Torch 환경도 다르므로 차이를 순수seed 효과로 해석하지 않는다.

이 작은 pilot은 데이터→FE graph→GPU 학습의 실행 증거다. 큰 자계 오차를 물리정확도 PASS나 AC 손실 우위로 표현하지 않는다. J/Je 전류성분 정의, Joule/DC/AC 적분, 공통 AF 분모와 native MS/Full 대응은 별도 검증이 필요하다.

480행은360개 선언 형상 그룹이다.440행 원본/사본HDF5 hash일치,80행 native/manifest 입력불일치와초기40 source미연결을첫pilot에서제외했다. 남은360은기존TRAIN뿐이므로입력만으로20그룹을선택해16/4동결했다. 기존val/test와과거checkpoint는사용하지않았다.

45스텝 전체 ID·기하·B 산술 검사 후9스텝씩180graph를저장했다. 시간축소의AC손실보존은검증하지않았다.45step transient field이며MS교사값이아니다.

10GiB는PyTorch allocated peak의스텝간소프트검사다.전체GPU하드한도보장이아니다.각run은3600초,UTC22:59,STOP과소유PID/시작시각또는containerID로제한한다.

재현: B_FIELD_PILOT_PLAN.json → audit_native_doe_catalog.py → prepare_existing_field_pilot.py → bundle hash 검증 → train_existing_field_pilot.py. raw/graph/checkpoint경로는private mapping,공유물은logical key/hash로연결한다.

다음과제: 입력불일치80행계보해결,도체·영역별B및J/Je손실적분검증,공통형상의Motor-CAD/Elmer교차점확보. mesh/time fidelity를분리하며그전paired학습이나AF대비AC우위를주장하지않는다.
