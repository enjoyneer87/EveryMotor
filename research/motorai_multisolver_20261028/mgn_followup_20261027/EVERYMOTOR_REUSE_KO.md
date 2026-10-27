# 기존 EveryMotor 자산의 직접 확인과 재사용 경계

iso의 실제 EveryMotor HEAD는 `65dda206e28770b242fa0dd888fd394f7a155ca0`였다. 기존 작업은 idle이었으며, 학습 캠페인·Docker·GPU 배턴을 변경하지 않고 코드와 저장 자료를 읽었다. 원본 파일 해시는 `EVERYMOTOR_REUSE_AUDIT.json`에 있다.

**재사용한 것은 P1 삼각형의 node-A → element-B curl 감독 원리다.** `train_doe_curl_mgn.py`는 A와 B 출력 모드를 모두 지원한다. 실제 E1 ep47 체크포인트는 111,835,629bytes, SHA256 `f58636620d5601f202d0203bff06d3f29c601e2885811c271edded28123bcba7`로 registry와 일치했다. 실행하지 않는 pickle opcode 확인에서 저장 인자는 `target=B`였다. 따라서 E1 자체가 A/curl 챔피언이라는 인계 설명은 채택하지 않는다. 코드의 명칭만으로 체크포인트 출력을 단정하지 않았다.

기록된 E1 legacy6 pooled 공극4.9563%, 토크3.2146%는 이번에 재채점한 값이 아니다. 공극<5%와 토크<3%를 모두 요구하는 G1′는 전체 PASS가 아니다. 과거 new12 토크 G2 PASS 정정과 spectral48/96/144의 비식별·발산 기록을 유지한다. `phase2_dynamic`에는 sample/collation/sliding-band 기초 코드가 있으나 완성된 시간 Field GNN이나 학습 결과로 보지 않는다.

HDF5의 `fields/je` 경로를 숫자 배열 로드 없이 다시 확인했다.

| manifest | 요청/발견 | Je 있음 | Je 없음 |
|---|---:|---:|---:|
| 초기 doe_data | 40/40 | 0 | 40 |
| doe_data_120 | 120/120 | 80 | 40 |
| doe_data_240 | 240/240 | 200 | 40 |
| doe_data_v2 | 320/320 | 280 | 40 |
| doe_data_v3 | 480/480 | 440 | 40 |

이들은 중첩 manifest이므로 더하지 않는다. 모두 초기 index0–39에서 Je가 없으며 legacy6도 이 구간에 속한다. Je dataset은 확인한 예에서 float32 `[45,18506]`이고 dataset attribute는 비어 있었다. 존재·shape 검사는 단위, total/source/induced 전류의 정의, Joule 적분, AC excess 정답 검증을 대신하지 못한다. HDF5 전체 파일 hash와 원시 txt의 현재 비트 일치는 이번 범위에서 재검증하지 않았다. 10월21일의 변환 비트 일치 기록은 과거 근거로만 보존한다.

현재 Elmer의 원시 노드 A는 직접 확보되어 있다. Motor-CAD의 요소 평균 A를 노드로 복원하는 문제와 다르므로, Elmer 원소 curl과 gauge를 직접 검증한 뒤 A 감독을 사용한다. Motor-CAD에서 보고한 17%/59.5% 오차를 Elmer 수치로 옮기지 않고 Elmer24조건에서 새로 측정했다.

E1 가중치·Motor-CAD J/Je·기존 평가 정답은 이번 학습에 넣지 않았다. 형상/운전점 exact bridge는0개이며 reserved alias와 출처 전체의 그룹 계약이 확인되기 전에는 다중 소스 정답으로 병합할 수 없다. 현재 재사용 분류는 **curl 원리·코드 설계: 가능**, **Motor-CAD Je 감독: 조건부**, **E1 가중치의 직접 Elmer 이식·완성 Phase2 주장·실패 spectral 재실행: 채택 안 함**이다.
