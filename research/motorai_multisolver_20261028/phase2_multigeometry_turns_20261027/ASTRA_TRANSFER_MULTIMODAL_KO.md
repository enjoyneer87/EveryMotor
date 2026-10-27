# Astra 판정: 기존 MotorAI 전이학습과 멀티모달 전략

사용자 요청으로 Astra 채팅 `01a17d9e-8a81-7262-9eae-e086c1957892`에서 원본 recipe·평가기·source hash·현재 split을 읽기 전용 검토했다. 실행·가중치 적용을 수행한 결과가 아닌 과학적 판정이다.

## 확인과 정정

E1은 원본 recipe의 `--target B` 및 저장 checkpoint 감사의 `target=B` 경로다. **nodal Bx/By를 요소로 평균**해 평가하며, E1 자체를 nodal A→P1 curl 모델이라고 설명한 초기 의견은 철회한다. trainer 파일명 `train_doe_curl_mgn.py`는 실제 E1 출력 head의 증거가 아니다. E1의 물리적 요소는 linear-solve prior, 공극 가중 및 band spectral supervision 등이다.

이전 legacy6 성적은 전체 |B|11.0967%,공극4.9563%,토크3.2146%다. 전체 |B|도 개선 대상이며 현재 다른 분할·가중치·위치에서의 B벡터18.0765%와 직접 비교하지 않는다.

## 전이학습 판정

| 항목 | 판정 | 조건 |
|---|---|---|
| E1 node/edge encoder와 processor 재사용 | 조건부사용 | 원래 native nodal graph의 feature 순서·단위·normalizer·출력 scale 복원. 첫 시험은 backbone 동결 |
| 현재 width32 element graph에 E1 h256 가중치 직접 삽입 | FAIL | shape뿐 아니라 graph 지지점·feature 의미가 다름. 원 nodal branch 또는 명시적 native node–element adapter 필요 |
| 새 Elmer branch | 조건부사용 | solver 전용 input adapter/normalizer/field decoder. processor 초기화 전이는 별도 가설 |
| 기존 E1 weights로 dev32 독립형상시험 | FAIL | 현재 admitted360조건은 모두 oldTRAIN 출처. source-domain 노출 재적합 진단과 독립 일반화를 구분 |
| teacher/distillation | 조건부사용 | donor·reserved 노출·teacher 오차·비용을 모두 상속. reserved 제외가 해소되지 않으면 보류 |
| 기존 architecture/recipe from-scratch | 조건부사용 | admitted manifest부터 제한하고 새 split의 weights/정규화로 시작. prior 신원과 비용 별도 기록 |

원본 E1 decoder를 보존한 동결 대조군, 새 residual adapter/head, 제한 fine-tuning은 각각 별도 arm으로 둔다. 원 normalizer만 임의 교체하면서 가중치를 재사용하지 않는다.

## 멀티모달 구성

솔버별 native graph encoder + 공유 geometry/OP 표현 + 별도 numerical-fidelity 조건 → solver별 field decoder → 예측field/latent의 물리 가중 요약 + geometry/OP → solver별 AC scalar head를 권고한다.

현재 활성화 가능한 확정 감독량은 Bx/By다. 모달리티별 관측 mask와 유효 표본 수로 손실을 정규화하고 결측을0 정답으로 만들지 않는다. actual Full B를 입력으로 요구하면 해석 보조 모델이므로 솔버 대체 모델과 추론 계약을 구분한다.

전체 B벡터와 |B|를 주 개선 지표로 두고 공극·구리 영역과 도체별 공간 평균 B²를 함께 진단한다. B²는 B의 파생 감독량이며 새 독립 label이 아니다. Motor-CAD transient B²를 기존 JEET canonical Hybrid 분모로 자동 대입할 수 없다. J/Je·고조파·AC head는 정의·시간샘플·회로·단위·평균구간·분모가 확인된 범위만 활성화한다.

exact bridge0에서 솔버별 사전학습·공유 geometry/OP 표현은 가능하지만 nominal/near match의 field/latent 정렬·discrepancy 정답 생성은 FAIL이다. 첫 exact bridge 후보는 reserved가 아닌 한 native 형상·6도체·한 OP의 계약 검증점이다. 한 점으로 교차 솔버 일반화를 주장하지 않는다.4/8도체는 우선 기존 Field를 탐색하며 geometry-only816구성을label수에더하지않는다.

## 분할과 비용

현재dev32는from-scratch SCALE360의내부검증집단이다. E1 warm-start의미노출geometry집단이아니다. 기존test의legacy6는초기40 source미연결,나머지12는CurrentAxis입력불일치에속해출처·입력복구와ancestor checkpoint/normalizer/prior/튜닝노출감사가필요하다. 과거반복평가이력도보존한다.

E1은450train/12val/18test의480조건source corpus로기록한다. 현재328train과중첩되므로canonical source/geometry키로중복을표시한다. 총Full데이터접근비용과이번증분Full비용을분리한다. 새solve0인E1 warm-start를Full63-only방법이라고표현하지않는다. 기존Full63/newOP8의동결모델을새결과로재튜닝하지않는다.

## 08시 이전 실행 후보와 관문

우선 후보는 train-only2형상·고정9시각에서 E1원본예측,동결E1 feature+작은 B residual head,무작위동결feature+같은head의재적합비교다. 같은head초기값/데이터순서/LR/update수로전체B벡터/|B|/구리B/B²를본다. 독립형상일반화나AC개선주장은하지않는다.

checkpoint hash/실제head/feature-normalizer/전체donor reserved제외/prior cache신원/native element대응이선행gate다. 현재복사본에weights·r3_models·fem_warmstart/prior.py가없어실행준비완료는UNKNOWN이다. prior cache가없어도새solve를자동발사하지않는다.

준비·호환성확인30분상한,통과후133 한작업의비교전체60분상한을권고한다. train-only속도측정후공통update한도를고정한다.10GiB Torch allocated soft검사·소유PID시작시각·STOP·07:59KST종료를유지한다. h256/p15전체재현을08시까지보장하지않는다.

gate가충족되지않으면가중치실험은보류하고완료SCALE360의전체|B|·구리B²·미사용36시각진단을완료한다. generic global-context추가는기존E1재사용경로평가후후순위로둔다.
