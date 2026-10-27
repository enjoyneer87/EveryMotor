# MGN 및 다중 소스 비교를 위한 입력 계약

이번 결과는 기존 Elmer 자료를 실제 FE 연결과 검증된 수치 label로 읽는 기반이다. MGN 학습 또는 AF 대비 우월성 검증은 아직 수행하지 않았다. 사용자는 후속 새 FEA·학습을 허용했지만, 이번 결손은 기존 자료 회수로 해결했다.

## 지금 만들 수 있는 그래프

- MS: `load_ms_graph.load_ms_slots(private_case_directory)`는 6개 stationary 균질 슬롯의 원래 node/element ID와 triangle edge를 반환한다. 공개 예제는 Ref_P097/Ref_P117의 1,377 node, 2,191 triangle, 7,124 directed edge, 128위상이다. 회전자·antiperiodic 경계를 포함한 전체 메시 reader로 확대하지 않는다.
- Full: 상위 `load_shard.load_copper_graph()`는 실제 구리 submesh와 element-node J, P1 A/B, 적분 가중치를 읽는다. 원래 Ref/SC 6개 및 이번 HalfSC 3개 공간 샤드가 있다. 36개 도체 중심 kNN과 구분한다.
- 노드 ID는 case 내부 ID다. MS의 균질 슬롯과 Full의 개별 도체는 다른 영역·mesh이므로 같은 번호나 최근접 좌표로 label을 연결하지 않는다. FE 요소 containment와 선언된 좌표/스케일 변환을 검증한 projection이 후속 작업이다.
- 구리 submesh는 도체 사이에 FE edge가 없다. 자기적 상호작용을 전달할 전체 도메인 graph 또는 별도의 검증된 계층 구조 없이 독립 도체 graph를 완전한 모터 MGN이라고 부르지 않는다.

## 입력과 label 분리

설계 시 알려진 geometry, 물질·conductivity, active length, 회로 연결, prescribed current와 phase, speed, 허용된 MS 자계만 입력으로 둔다. Full solver의 component voltage, native A/J, 적분 손실은 target 또는 검증용이다. 이를 inference 입력으로 넣으면 Full solver를 대체하는 비교가 아니다.

MS A/B는 diffusion-free 저충실도 교사다. 개별 도체 J, Joule 손실 또는 Full AC 정답을 제공하지 않는다. MS pretraining loss와 최종 Full AC 예측 오차를 분리한다. 원시 B 표본과 second moment를 유지하고, 도체 평균 B의 제곱으로 mean-square 정보를 대체하지 않는다.

Full J는 `sigma*(V/L-dA/dt)`인 total axial current density다. residual J를 학습할 때 도체 적분 전류 제약을 적용할 수 있지만, 저장 자료의 적분 전류와 prescribed current 차이 및 cross term을 임의 삭제하지 않는다. 해당 투영을 모델에 도입하면 전류·손실 두 지표를 함께 검증해야 한다.

8위상 Field 샤드는 공간 reader 검증용이다. 이 위상의 평균을 P2 loss label로 쓰지 않는다. 원래 256개 P2 적분 이력과 512개 전류 이력을 사용한다. 전체 시간 Field가 필요한 학습은 원시 iso 자료에서 같은 reader로 별도 추출하며 시간 스텝을 바꾸지 않는다.

## 공정한 비교와 누출

같은 target Full-FEA 보정 예산과 평가 조건에서 frozen AF × canonical mean-square Hybrid, scalar residual, 다중 소스, MS 사전학습/Field 전이를 비교한다. raw Hybrid만을 경쟁 기준으로 쓰지 않는다. Full field를 supervision으로 사용하면 그 acquisition을 Full budget에 포함한다. 추가 MS 취득·전처리·추론 비용도 별도 보고한다.

Ref/SC/HalfSC는 한 기준 template의 파생 형상이다. geometry group에 더해 `MS_Full_pairs.csv`의 `shared_MS_history_group`으로 공통 원천을 추적한다. 동일 MS history를 여러 형상 fold에 재사용하는 알려진-template/transductive 실험과, 새 template에 대한 inductive 실험을 구분한다. 후자의 pretraining/normalization/tuning에는 held-out 원천이 들어가면 안 된다.

HalfSC 사전 선언32는 Field까지 보호된 평가 자료다. 전처리 코드의 수치 검증 이력을 보존하고, 모델 학습·사전학습·튜닝에는 사용하지 않는다. exploratory4는 별도다. R1_0199/R1_0076과 alias는 모든 모델 단계에서 제외한다. 과거 QC 이력 때문에 globally never-opened라고 표현하지 않는다.

## 다음 실행 전에 고정할 것

1. 독립 기준 template와 geometry/OP 범위, 원천별 ID 대응, 전도도·온도·회로·파형 계약.
2. 같은 Full 예산의 보정·개발·최종시험 목록과 MS 원천 그룹, HalfSC32 보호 규칙.
3. period/time-step/mesh/current 품질 gate와 실패 cohort 처리. 현재 period/mesh는 UNKNOWN이다.
4. 모델 입력·label, 시간 이력 필요량, graph 범위, normalization, loss, seed, optimizer와 평가 지표.
5. 실행 case/model, 예상 시간·RAM/VRAM·디스크, 1차 pilot 후 확대 한도, 소유 PID, 재시도 한도와 중단 방법.

새 해석·학습 권한은 있으나 중단된 이전 캠페인이나 자동화의 재개 권한으로 해석하지 않는다. 현재 부족한 독립 template와 수렴 근거를 한 template의 반복 OP 수로 대체하지 않는다.
