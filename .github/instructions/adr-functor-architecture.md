---
description: "Architecture Decision Record: Functor Directionality, Parsing Boundaries, and Visualization Sharing"
applyTo: "postproc_interop/**/*.py, eMach/**/*.py"
---

# ADR: 범주론적 함자(Functor) 방향성에 따른 모듈 분리 및 공유 원칙

본 문서는 `eMach` 서브모듈과 `EveryMotor` 상위 패키지 간의 의존성 및 코드 중복(DRY) 딜레마를 범주론(Category Theory)의 함자(Functor) 개념으로 해결한 아키텍처 결정 기록(Architecture Decision Record)입니다. 신규 에이전트나 개발자는 코드를 수정하기 전 본 맥락을 최우선으로 숙지해야 합니다.

## 1. 읽기/파싱 (Parsing) - 의도적 중복 허용
- **결정:** 파싱 로직(`read()`, 텍스트 처리 등)은 절대로 위아래 패키지 간에 Import 하여 공유하지 않는다. '우연한 중복(Accidental Duplication)'을 의도적으로 허용하여 각자 폴더 내부에 독립 구현한다.
- **이유 (Functor의 방향성):** 파싱은 **"외부 포맷 ➡️ 내부 도메인"**으로 쏘아지는 수렴형 Functor이다. 
  - `eMach` 파서는 레거시/뮤터블 객체로 향하고, `EveryMotor/postproc` 파서는 순수/불변 ML 객체로 향한다.
  - 목적지(Target Category)가 다른 두 Functor를 하나로 합치면 상위 패키지가 하위 레거시 구조에 강결합(Tight Coupling)되는 안티 코럽션(Anti-corruption) 위반이 발생한다.

## 2. 시각화 (Visualization) - 순수 텐서 변환 및 공유
- **결정:** 시각화 함수(`plot_mesh`, `plot_field` 등)는 도메인 객체(`MeshSolution`, `MagneticRegions` 등)를 절대로 인자로 받지 않는다. 오직 `numpy.ndarray` 등 순수 원시 수학/텐서 타입만 받도록 공통 유틸리티에 작성하여 양쪽에서 공유(Import)한다. (혹은 `.vtu` 공통 포맷으로 export 하여 ParaView 등을 사용한다)
- **이유 (Functor의 방향성):** 시각화는 **"내부 도메인 ➡️ 공통 수학 포맷(Numpy)"**으로 쏘아지는 발산형 Functor이다. 
  - 서로 다른 도메인 객체를 갖고 있더라도 도착지(Target Category)가 완전한 순수 수학 모델(Numpy)로 동일하다.
  - 따라서 공유 함수는 순수 영역에서만 동작하므로, 서로의 도메인을 침범하지 않고 안전하게 재사용이 가능하다.

## 3. 부수 효과 및 예외 처리 (Side-Effect & Result Wrapping)
- **결정:** Morphism(순수 변환 함수) 내부에서 경로 이탈이나 파싱 누락이 발생해도 `raise Exception`을 던져 파이프라인을 붕괴시키지 않는다.
- **해결 방안:** 예외를 값으로 래핑하여 `(None, ErrorCode)` 튜플이나 Result 모나드 형태로 반환한다. 이때 ErrorCode는 레포지토리 내의 `Action09_Execution_Failure_Taxonomy_KO.md` (예: `E-IO-001`, `E-PARSE-001`)를 참조한다.
- **이유:** ML 파이프라인의 스테이지 경계(`raw -> canonical -> graph -> batch`) 사이에서 발생하는 오류를 데이터 흐름의 일부로 취급하여 우회(Fallback/Routing)하기 위함이다.
