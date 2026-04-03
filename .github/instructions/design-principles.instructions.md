---
description: "Use when: creating or reviewing any class, adapter, model, or transformation function. Enforce category-theory-aligned OOP: immutable data objects, pure morphisms, composition over inheritance."
applyTo: "**/*.py"
---

# Design Principles: Category-Theory-Aligned OOP

## Core Mental Model

```
Category Theory  →  "어떻게 변환되는가" (변환, 합성 가능성)
OOP              →  "무엇이 존재하는가" (구조, 타입 정의)
```

이 둘은 충돌하지 않는다. OOP는 도구, 범주론은 설계 원칙이다.

```
Object   = 불변 데이터 클래스 (MeshMat, SolutionMat)
Morphism = 순수 변환 함수/메서드 (read(), to_dataframes())
Functor  = 포맷 어댑터 (TxtReader, H5Reader) — 구조를 보존하며 변환
Composition = 파이프라인 (TXT → MeshSolution → DataFrame → VTU)
```

---

## 규칙 1: 데이터 클래스는 불변으로

```python
# ✓ 올바름 — @dataclass는 상태만, 초기화 후 변경 없음
@dataclass
class MeshMat(Mesh):
    node_id: np.ndarray
    x_mm: np.ndarray

# ✗ 금지 — 초기화 후 속성 추가/변경
mesh.new_attr = []          # 불변 원칙 위반
mesh.x_mm = other_array     # 사이드이펙트
```

**Why:** 불변 객체는 순수 변환(morphism)의 입출력으로 안전하게 사용 가능.

---

## 규칙 2: 변환은 순수 함수로

```python
# ✓ 올바름 — 입력만 보고 출력 생성, 사이드이펙트 없음
def meshsolution_to_dataframes(meshsol: MeshSolution) -> tuple:
    ...

# ✗ 금지 — 전역 상태 변경, 숨은 의존성
def process(meshsol):
    GLOBAL_CACHE[id(meshsol)] = meshsol   # 사이드이펙트
    meshsol.processed = True               # 입력 변경
```

**Why:** 순수 함수는 합성 가능(f∘g). 테스트가 입출력만으로 완결됨.

---

## 규칙 3: 상속보다 컴포지션

```python
# ✓ 올바름 — MeshSolution은 Mesh를 상속하지 않고 포함
@dataclass
class MeshSolution:
    mesh: Mesh                        # composition
    solution_dict: dict[str, Solution]

# ✗ 금지 — 기하와 해석결과를 상속으로 합침
class MeshSolution(MeshMat):          # 상속으로 합치면 교체 불가
    fields: dict
```

**Why:** 컴포지션은 각 부분을 독립적으로 교체/테스트 가능.

---

## 규칙 4: 추상 클래스는 역할명으로, 구체 클래스는 명시적 이름으로

```python
# 추상 (ABC) — 역할만
class Mesh(ABC): ...
class Solution(ABC): ...
class MeshReader(ABC): ...

# 구체 — {소스}{데이터}{포맷}{역할}
class MeshMat(Mesh): ...                        # numpy 기반 Mesh
class MotorCADMeshSolutionH5Reader(MeshReader): # MotorCAD의 MeshSolution을 H5에서 읽음
class MotorCADMeshSolutionTxtReader(MeshReader):
```

**Why:** 이름만으로 추상/구체, 데이터 종류, 포맷, 역할이 파악됨 (pyleecan 컨벤션).

---

## 규칙 5: 파일명 = 클래스명 (pyleecan 컨벤션)

```
MeshMat.py                        → class MeshMat
MotorCADMeshSolutionH5Reader.py   → class MotorCADMeshSolutionH5Reader
```

`h5_motorcad.py → MotorCADH5Adapter` 같은 불일치 금지.

---

## 규칙 6: 계층별 테스트 전략

각 morphism은 독립적으로 테스트:

```python
# Layer 1: Model (Object 테스트)
def test_meshmat_shape():
    m = MeshMat(node_id=..., x_mm=..., ...)
    assert m.tri_index.shape == (n_elem,)

# Layer 2: Adapter (Morphism 테스트)
def test_txt_reader_returns_meshsolution():
    reader = MotorCADMeshSolutionTxtReader()
    sol = reader.read("data/MagTransient_20260130_152245.txt")
    assert isinstance(sol, MeshSolution)
    assert isinstance(sol.mesh, MeshMat)
    assert "bx" in sol.solution_dict

# Layer 3: Composition (Pipeline 테스트)
def test_txt_to_dataframe_pipeline():
    sol = MotorCADMeshSolutionTxtReader().read("data/...")
    nodetable, *_ = meshsolution_to_dataframes(sol)
    assert "NodeIndex" in nodetable.columns
```

실제 샘플 파일 사용: `data/MagTransient_20260130_152245.txt`

---

## 적용 대상 파일 구조

```
postproc_interop/
  model/
    Mesh.py          ← Object (ABC)
    MeshMat.py       ← Object (concrete, immutable dataclass)
    Solution.py      ← Object (ABC)
    SolutionMat.py   ← Object (concrete, immutable dataclass)
    MeshSolution.py  ← Composition (Mesh + solution_dict)
  adapters/
    MeshReader.py                       ← Morphism (ABC)
    MotorCADMeshSolutionH5Reader.py     ← Functor (H5 → MeshSolution)
    MotorCADMeshSolutionTxtReader.py    ← Functor (TXT → MeshSolution)
    VTUMeshReader.py                    ← Functor (VTU → Mesh, scaffold)
  tabular.py         ← Morphism (MeshSolution → DataFrames)
  header_mapping.py  ← 공유 매핑 레지스트리 (변경 금지 원칙)
```

---

## eMach 서브모듈 주의사항

- `eMach/`는 독립 git submodule이므로 `postproc_interop` import 금지
- 동일 로직(헤더 파싱 등)은 eMach 내부에서 독립적으로 구현
- eMach의 `MagneticRegions`는 뮤터블(레거시) — 신규 코드에서 모방 금지
