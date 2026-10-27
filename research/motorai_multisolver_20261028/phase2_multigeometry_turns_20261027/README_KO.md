# Phase2: 272형상 × 슬롯당 도체4·6·8

개선 요청·전이학습·멀티모달 아이디어와 후속 판정은 [개선 요청·아이디어 통합 대장](IMPROVEMENT_REGISTER_KO.md)에 기록한다. 원본 상태는 `IMPROVEMENT_REGISTER.json`, Astra 검토 근거는 `ASTRA_REVIEW_KO.md`다. 제안·판정·실행 완료를 구분하며 새 요청은 기존 MI 항목에 연결하거나 새 번호로 추가한다.

816개 구성을 등록했다. 768개 train구성과48개 development구성이 원래256/16형상그룹을상속한다. 두reserved형상은기존272목록에서이미제외되어있다. 추가두솔버네트워크설계는 MULTISOLVER_DESIGN_KO.md,현재상태는 STATE.json과 STATUS_KO.html을확인한다.

새과도해석총128스텝은상한이다.정착부족을해결하려고스텝을자동으로늘리지않는다.기존N256/N512의3주기결과는시간간격·기동과도진단에재사용한다. 128×1주기/64×2주기진단은모두총128이며형상다양화성과와구별한다.

`python build_contract.py`는queue를초기계약으로재생성하므로결과이후무조건재실행하지않는다. `build_geometry.py --source <private_pinned_ipmfea> --out <new_private_directory>`는solver-free기하생성/검사다. 실제FEA/학습을하지않는다. `python test_dual_solver_contract.py`는CPU합성구조검사로학습성과가아니다. `python update_dashboard.py`는saved상태로HTML을갱신한다.

현재원본기하와해석기자료는private저장소/iso에보존하고작은queue/hash/코드/검증만게시한다. 실제mesh/field와MS/Full대응,Motor-CAD J/Je의정의,exact교차점이남은핵심결손이다.0개exactbridge에서paired학습을시작하지않는다.08시이후소유계산중단과부분자료보존을유지한다.

현재816개기하생성/해석적QA는완료했고mesh/solver검증은별도다. 총128진단2건의기존P3대비AC차이는+19.403%/+3.752%,공식수렴UNKNOWN이다. Motor-CAD무해석probe01은기동실패,02는API기동후180초timeout으로4/6설정사본만회수했다. 실제drawncount/8은미완료,새FEA0건이다.

기존다형상HDF5는480행/360선언형상그룹,원본/사본SHA일치440행이다. 입력불일치80행과원천미연결40행을제외한기존TRAIN에서20형상을선택해16/4내부분할,180실제FE요소graph를구축했다. 133/234에서고정10epoch모델2개를완료했다. 내부검증B nRMSE58.186/60.317%로큰오차이며AC손실성과가아니다. DATA_REUSE_KO.md, B_FIELD_PILOT_REPORT_KO.md, ASTRA_REVIEW_KO.md에근거를남겼다.

사용자후속지시는충분한샘플로확대하고큰학습을133에우선배치하는것이다. 확대는별도계획/분할/실행receipt로추적하고기존pilot·Full63/새OP8동결평가를덮지않는다. 최신후속실행은CONTINUATION_KO.md와STATE.json을따른다.

확대133학습은320형상그룹360조건(학습288그룹328조건/내부검증32그룹32조건)에서30epoch·88565updates를완료했다. dev32 B벡터오차18.0765%,동일pilot dev4는16.6035%,구리34.6131%다. [확대 결과](SCALE360_REPORT_KO.md)와[이전 MotorAI 지표 비교](PRIOR_MOTORAI_COMPARISON_KO.md)를함께본다. 후속은[아스트라 전이·멀티모달 판정](ASTRA_TRANSFER_MULTIMODAL_KO.md)과개선대장의선행gate를따르며이전학습을몰래연장하지않는다.
