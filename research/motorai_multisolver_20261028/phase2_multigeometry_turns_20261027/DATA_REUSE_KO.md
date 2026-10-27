# 기존 다형상 Field 재사용 감사

2026-10-27 iso 원본에서 최신 EveryMotor v3 manifest 480행의 OnLoadTorque HDF5를 모두 확인했다. 선언된 형상 파라미터와 source_geometry_index 조합은360개다. R1 272형상과 동일한 집합으로 세지 않는다. 중첩된40/120/240/320/480 목록을 합산하지 않는다.

440행은 원본 HDF5와 학습 사본의 전체 SHA-256이 일치한다. 이440행의 native .mot와 MES·TXT도 연결했다. 최초40행의 원래 native source는 이 경로에서 연결하지 못했고, 다른 메시의 DOE4TrainingData로 대체하지 않았다. J/Je가 없는 최초40행과 Je가 존재하는440행을 구분한다. 원본은 변경하지 않았다.

native 입력과 manifest를 비교하니 CurrentAxis80행의 형상 파라미터가 달랐다. 예를 들어 v3 index240은 manifest bore/slot-depth 비율0.762933/0.456527, 저장 native 입력은0.767272/0.424808이다. 해당80행은 첫 학습 준비에서 제외하고, 기존 source_geometry_index 연결은 보수적 분할 그룹으로 보존한다. 입력 파일과 solver 당시 geometry의 추가 대조 없이 과거 입력값을 정답으로 확정하지 않는다.

440행의 원래 split은420 train/8 val/12 test다. source index 또는 선언 형상 또는 native 형상 hash의 전이적 연결로360개 그룹을 구성했으며 기존 split 충돌은0이다. native 자석 V각112/169.8도는 두 R1 reserved 입력과 다르다. 이는 서로 다른 기준 형상을 식별하는 명목 입력 근거이며 exact 교차 솔버 일치의 근거는 아니다. reserved의 field payload를 열거나 모델에 사용하지 않았다.

TXT의 ElementsTable과 exporter 구현은 B/A/J/Je가 native element index에 저장됨을 보여준다. 단위는 B[T], A[Wb/m], J·Je[A/mm²]다. HDF5에는 native node/triangle/region ID, 연결, moving-node ID와45스텝 좌표가 있다. 따라서 도체 중심kNN을 만들 필요 없이 실제 FE 요소의 인접 graph를 구성할 수 있다. 요소 A를 node A로 취급해 curl하지 않는다. J와Je의 total/source/induced 의미 및 손실 재적분은 아직 UNKNOWN이다.

첫 pilot은 기존 transient B만 사용한다. 45스텝 자료를 MS 자료로 부르지 않으며, formal 수렴UNKNOWN인 해당 numerical fidelity의 자계 supervision이다. a2 sliding band는 명시적으로 제외하고 나머지 실제 triangle의 native 공유 node ID로 연결한다. 변하는 좌표를 스텝마다 복원하고 모든45스텝의 유한값·기하·B벡터/크기 산술을 검사한 뒤9스텝을 균일 선택한다. 이 샘플 축소가 AC 손실을 보존한다고 주장하지 않는다.

원래 val8은 모두 보류된 CurrentAxis80에 속한다. 따라서 기존TRAIN 내부에서 입력만으로20그룹을 선택하고, seed17로16 train/4 inner-development를 동결했다. 기존 val/test는 이 pilot에 사용하지 않는다. 과거 checkpoint를 재사용하지 않는 새 학습이며 inner-development는 최종시험이 아니다. 실패한 첫 metadata 선택 시도는 field를 읽기 전에 종료했고 별도 receipt로 보존했다.

이 pilot의 목적은 두 솔버 중 Motor-CAD 전용 FE field 인코더를 실제 다형상 자료로 사전학습할 수 있는지 확인하는 것이다. 정확한 연결점0개에서 Elmer와 paired alignment를 하지 않는다. AF 분모가 없는 Motor-CAD 데이터에 frozen Ref AF를 붙이지 않는다. `forward_fields`는 AC 기준값을 만들지 않고 field만 예측한다. 원래Full63/새OP8 비교는 별도로 동결 유지한다. 기존 Full-FEA donor 형상 수도 비용에 명시하며 새 FEA0건을 전체 데이터 비용0으로 표현하지 않는다.

관련 파일: `MULTIGEOMETRY_FIELD_INVENTORY.json`, `NATIVE_DOE_CATALOG.json`, `B_FIELD_PILOT_PLAN.json`, `native_element_graph.py`, `prepare_existing_field_pilot.py`. 대용량 source와 graph NPZ는 iso의 private source-key 경로에 보존한다.

확대 split 검사: 첫 pilot의 eligible360조건은320개 conservative 형상 그룹이다.20형상은3개OP를가진다. 전체480행의360개그룹과혼동하지않는다. 확대는288 train그룹(328조건),32 inner-development그룹(32조건),기존pilot16/4역할보존으로동결했다.
