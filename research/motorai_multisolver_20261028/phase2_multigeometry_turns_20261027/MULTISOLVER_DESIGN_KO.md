# 두 솔버 네트워크를 연결하는 다중 소스·정밀도·물리량 모델

사용자 방향: Motor-CAD와 Elmer의 이산화·결과 차이를 없다고 가정하지 않고, 각각의 네트워크를 연결해 관리한다. 한 솔버를 일률적인 정답 또는 상위 fidelity로 지정하지 않는다.

```
Motor-CAD 원래 FE ID/메시 → Motor-CAD 전용 인코더 → 공유 형상·OP 표현 → Motor-CAD 전용 field/loss head
                                                   ↕ 검증된 bridge만
Elmer 원래 FE ID/메시    → Elmer 전용 인코더    → 공유 형상·OP 표현 → Elmer 전용 field/loss head
                              + solver와 별도의 mesh/time/초기화 fidelity 조건
```

형상·슬롯당 도체수·전류·속도·온도 등 공통 물리 조건을 공유한다. 각각의 native 메시 연결과 field 위치를 보존하고 좌표 최근접으로 도체·재료 ID를 맞추지 않는다. Motor-CAD element-node 값은 재료 경계에서 중복될 수 있으므로 무조건 하나의 node 값으로 평균하지 않는다. Elmer A의 curl과 Motor-CAD B/J/Je 이름도 자동으로 같은 물리량이라 보지 않는다.

`dual_solver_mgn.py`는 별도 graph encoder/field head, 공유 design/OP network, solver별 alignment adapter와 residual loss head를 구현한다. numerical fidelity token은 solver 이름과 별개다. 초기 scalar head는 입력된 AF×canonical mean-square Hybrid를 그대로 재현한다. 정상적인 기준선이 없는 새로운 형상에는 임의 계수를 대입하지 않고 계약 확인 후 해당 Full 예산으로 AF 함수족을 다시 보정한다. 현재 최신 frozen AF 비교는 별도 Ref/SC 시험에서 유지한다.

학습은 (1) 각 solver의 실제 MS/field로 각각 사전학습, (2) 같은 고정 Full budget에서 scalar residual과 MGN head 비교, (3) 근거가 있는 exact 교차점으로만 shared representation/discrepancy 연결 순서다. 동일 native 메시에서 계산 해상도만 다른 쌍은 solver 내부 fidelity 전이로 평가한다. mesh/time refinement와 솔버 변경을 한 개의 fidelity 등급에 섞지 않는다. modality가 없으면 mask로 제외하며 예측값을 독립 Full 정답으로 세지 않는다.

현재 exact cross-solver bridge는0이므로 paired alignment loss는 비활성이고 코드도 빈/nominal-only pair 입력을 거부한다. near_match 비교는 형상·재료·평균구간 차이까지 포함한 진단이며 순수 solver 편향 추정용 정답 쌍이 아니다. 최소 bridge는 먼저 한 형상·한 턴·한 OP에서 슬롯/모서리/절연/회로/재료/온도/파형/평균구간을 확인하고 확장한다. native 원자료와 explicit ID mapping을 함께 남겨야 한다.

272개는 단일 R1 템플릿이다. 4·6·8 도체 변형 모두 원래 geometry/alias split을 상속한다. 256개 train,16개 development이며 두 reserved 형상은 포함하지 않는다. 새 템플릿 일반화는 추가 템플릿이 있어야 평가할 수 있다. 같은 63개 독립 Full label을 모든 비교 arm이 공유하고 donor 보정·사전학습·추론의 전체 시간도 집계한다. 기존8개 동결 평가점은 새 네트워크 튜닝에 쓰지 않는다.

Motor-CAD R1 실행 코드에서 `TorquePointsPerCycle=128`, `TorqueNumberCycles=1`인 원본 템플릿을 확인했다. 과거 `fieldexport.py --steps 65:128`은 출력 범위를 고르는 옵션으로 계산 스텝 수 설정 자체가 아니다. 새 runner는 두 변수를 명시하고 실제 저장 step 수를 검사해야 한다. 첫 주기/후반 반주기를 자동으로 정상상태로 부르지 않는다. 기존 manager의 자동 재시도와 도체가 안 맞으면 절연·간격·높이를 몰래 줄이는 fit 기능은 새 계약에서 사용하지 않는다.

133과234 모두 Motor-CAD v261 설치 파일이 확인됐다. 설치 존재와 license/API/solver 검증은 별개다. 234에는 사용자 소유 MotorCAD.exe가 있으므로 기존 인스턴스에 attach하지 않는다. 새 전용 사본/인스턴스의 PID·시작시각·timeout을 확보한 뒤 무해석 geometry/readback pilot부터 한다. GPU 학습은 독립 seed/ablation을 두 서버에 분배하되, 실제 메시와 정의가 확인된 target이 준비되기 전에는 synthetic smoke를 학습 성과로 보고하지 않는다.

Thesis_SKKU 최신 확인본 `6a597523d0ebcc9d9cd367565d4d2a76d41418e2`의7장 전략을 따른다. 최종 목적은 형상·운전점 의존 AC 손실 및 희소 FEA 보정이 동일 선택 규칙에서 설계 선택을 바꾸는지 검증하는 것이다. 모델 오차와 전체 FEA 예산을 먼저 검증하고 시스템 에너지–질량 탐색으로 연결한다. 현재 시스템 최적설계 완료를 주장하지 않는다.
