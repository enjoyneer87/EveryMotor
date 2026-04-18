# Motor-CAD optiSLang 폴더 검토 보고서

대상 폴더:
- C:/Program Files/ANSYS Inc/v261/motorcad/Motor-CAD Data/optiSLang

## 검토한 파일과 역할

1. optimisationScript_Header.py
- 런타임 상수와 유틸리티 함수를 정의합니다.
- Logger 및 결과 파일 기록 헬퍼를 구현합니다.
- `RunOptimisation(aRunMode)`의 시작부를 포함하며, Motor-CAD 연결, 파라미터 적용, 형상/권선 검증, 계산 시작을 수행합니다.
- 설계가 유효하지 않을 때 NaN 출력으로 반환하는 검증/오류 경로를 제공합니다.

2. optimisationScript_Footer.py
- 계산 이후 솔버 흐름을 마무리합니다.
- 최종 Motor-CAD 설계를 저장하고 인스턴스를 해제/종료하며 `MotorCAD_Outputs.txt`를 기록합니다.
- 실행 모드(Python node / IDE / internal test) 판별, 입출력 초기화, `RunOptimisation` 호출을 담당하는 런타임 엔트리 블록을 포함합니다.

3. optislangSetupScript.py
- optiSLang 프로젝트를 코드로 생성합니다.
- Motor-CAD 통합 노드를 생성하고 Python 실행 파일 설정을 적용하며, 파라미터/응답을 로드한 뒤 위저드를 실행합니다.
- `MotorCAD_Summary.txt`를 읽어 최적화 파라미터와 응답 기준(목적함수/제약)을 구성합니다.

4. Integration/MotorCAD_ci.py
- optiSLang용 커스텀 통합 플러그인 API 계층입니다.
- 필수 콜백을 노출합니다:
  - `DefaultSettings`
  - `ExtractInputContainerForDesigns`
  - `SetParametersForDesigns`
  - `RunSolverForDesigns`
  - `ExtractOutputContainerForDesigns`
- 실제 로직은 `MotorCAD_integration.py`, `MotorCAD_settings.py`로 위임합니다.

5. Integration/MotorCAD/MotorCAD_integration.py
- 핵심 브리지 로직입니다.
- `scan_input`: 입력 섹션 마커를 스캔해 `i_*` 파라미터를 읽습니다.
- `set_parameters`: 설계별 스크립트를 생성하면서 파라미터 값과 참조 디렉터리를 반영합니다.
- `scan_output`: 결과 파일 선언을 찾아 `MotorCAD_Outputs.txt`를 파싱합니다.
- `run_Python_Script`: 실행 환경을 준비하고 서브프로세스로 `RunOptimisation(0)`를 호출합니다.

6. Integration/MotorCAD/MotorCAD_settings.py
- Input/Execution/Output 탭용 통합 설정 객체와 기본값을 정의합니다.
- 사용자 지정 Python 실행 파일 선택을 지원합니다 (`Execution.use_other_python`, `Execution.otherpy_exe`).

7. Integration/MotorCAD/MotorCAD_helpers.py
- 이름 중복 검출, 환경변수 export, 서브프로세스 실행, 디버그 출력 등의 유틸리티를 제공합니다.

8. Integration/MotorCAD_ci.cfg
- 통합 런타임 동작을 설정합니다:
  - `EnableParallel true`
  - `ScriptInterfaceVersion 2`
  - `EnableMultiDesignMode false`
  - `SupportsHPCLicenseContext true`

## End-to-End 실행 흐름

1. Export/Setup 단계
- `optislangSetupScript.py`가 `.opf` 프로젝트와 Motor-CAD 통합 노드를 생성합니다.
- 요약 정의를 파싱해 optiSLang에 파라미터/응답을 등록합니다.

2. 설계별 실행 단계
- optiSLang이 `MotorCAD_ci.py` 콜백을 호출합니다.
- 입력 추출: `scan_input`이 솔버 스크립트 내 수정 가능한 `i_*` 변수를 식별합니다.
- 파라미터 반영: `set_parameters`가 설계별 스크립트를 생성합니다.
- 솔버 실행: `run_Python_Script`가 Python 서브프로세스를 실행하고 `RunOptimisation(0)`를 호출합니다.

3. 솔버 내부 단계
- 헤더/푸터 스크립트의 `RunOptimisation`이 Motor-CAD 설정, 파라미터 적용, 검증, 계산 실행, 결과 파일 기록을 수행합니다.

4. 출력 추출 단계
- `scan_output`이 `MotorCAD_Outputs.txt`를 파싱해 optiSLang으로 결과를 반환합니다.

## 생성된 다이어그램

- UML 스타일 활동 다이어그램(PNG):
  - `optiSLang_analysis/MotorCAD_optiSLang_UML_flow.png`
