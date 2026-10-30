# MotorAI W64 → W128 본학습과 native 컨투어

판정: **IMPROVEMENT_CLAIM_WITHIN_INNERDEV**. 고정3시드17·29·43 / 각88565갱신 / 각1440개평가. 원천GPU실행: HPC_133 / 192.168.0.133, NVIDIA L40S.

[HTML 보고서](index.html)를 열어 같은단위·색상범위의 실제저장FEM 기준/예측/오차를 비교합니다. 노출이력있는innerdev32의 비교이며 독립test·AC손실·물리토크검증이 아닙니다. native BC/벡터basis/회전자–고정자pair는UNKNOWN입니다.

공개파일 무결성과703개저장진단/고정판정 산술 재현: `python verify_published_report.py` (Python3.10이상, 표준라이브러리만 사용). 이는private원본을다시읽거나FEM/모델추론을하는명령이아닙니다.

원본HDF·모델·큰예측배열·private매핑·license header는포함하지않습니다. repro/는실행당시소스의byte보존감사본이며 전체재학습/field재생성에는별도private입력과해당runtime이필요합니다. source_manifest.json은원본SHA와공개복사본SHA를구분합니다. 브라우저HTML렌더링은보안정책차단으로미검증이며standalonePNG검토와정적링크검증은별도기록했습니다.
