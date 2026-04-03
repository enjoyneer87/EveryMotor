# postproc_interop Refactor Plan
# pyleecan-style MeshSolution architecture

## 진행 상태 (체크리스트)

- [ ] 1. `postproc_interop/model/` 패키지 생성 (5개 파일)
- [ ] 2. `postproc_interop/adapters/` 신규 파일 생성 (4개)
- [ ] 3. `postproc_interop/adapters/__init__.py` 업데이트
- [ ] 4. `postproc_interop/__init__.py` 업데이트
- [ ] 5. `postproc_interop/tabular.py` 업데이트
- [ ] 6. 구 파일 삭제
- [ ] 7. `eMach/tools/motorCAD/pyMCAD/magnetic.py` 업데이트

---

## 배경

### 문제
- `MeshFrame` 하나에 기하(Mesh)와 해석결과(Fields)가 혼재
- 어댑터 파일명(`h5_motorcad.py`)과 클래스명(`MotorCADH5Adapter`) 불일치
- 추상 클래스인지 구체 클래스인지 이름으로 구분 불가
- eMach `_parse_first_block_magnetic_file`: 헤더 4줄 통째로 skip → 컬럼 인덱스 하드코딩

### 참조 아키텍처: pyleecan
```
Mesh (ABC)          ← 기하만 (topology + nodes + regions)
├── MeshMat         ← numpy 기반 구체
└── MeshVTK         ← VTK 기반 구체

Solution (ABC)      ← 필드값만 (geometry 없음)
├── SolutionMat     ← numpy 기반 구체
└── SolutionVector  ← SciDataTool 기반 구체

MeshSolution        ← Mesh + solution_dict 컴포지션 (상속 아님)
  .mesh             → Mesh 인스턴스
  .solution_dict    → {"Bx": SolutionMat, "By": SolutionMat, ...}
```

### 네이밍 규칙 (pyleecan 스타일)
- **파일명 = 클래스명** (완전 일치, PascalCase)
- 추상: 역할명만 (`MeshReader`, `Mesh`, `Solution`)
- 구체: `{소스}{데이터타입}{포맷}{역할}` 순서
  - `MotorCADMeshSolutionH5Reader`
  - `MotorCADMeshSolutionTxtReader`
  - `VTUMeshReader`

---

## 파일 변경 맵

### 삭제 대상
```
postproc_interop/model.py                → model/ 패키지로 분리
postproc_interop/adapters/base.py        → MeshReader.py로 대체
postproc_interop/adapters/h5_motorcad.py → MotorCADMeshSolutionH5Reader.py로 대체
postproc_interop/adapters/vtu.py         → VTUMeshReader.py로 대체
postproc_interop/adapters/txt_motorcad.py → MotorCADMeshSolutionTxtReader.py로 대체
```

### 신규 생성
```
postproc_interop/model/__init__.py
postproc_interop/model/Mesh.py
postproc_interop/model/MeshMat.py
postproc_interop/model/Solution.py
postproc_interop/model/SolutionMat.py
postproc_interop/model/MeshSolution.py

postproc_interop/adapters/MeshReader.py
postproc_interop/adapters/MotorCADMeshSolutionH5Reader.py
postproc_interop/adapters/MotorCADMeshSolutionTxtReader.py
postproc_interop/adapters/VTUMeshReader.py
```

### 수정 대상
```
postproc_interop/__init__.py
postproc_interop/adapters/__init__.py
postproc_interop/tabular.py
eMach/tools/motorCAD/pyMCAD/magnetic.py
```

---

## 새 모델 클래스 스펙

### `postproc_interop/model/Mesh.py`
```python
from __future__ import annotations
from abc import ABC

class Mesh(ABC):
    """Abstract base: mesh geometry only (topology + nodes + regions)."""
    pass
```

### `postproc_interop/model/MeshMat.py`
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from postproc_interop.model.Mesh import Mesh

@dataclass
class MeshMat(Mesh):
    # nodes
    node_id: np.ndarray       # shape (N,)
    x_mm: np.ndarray          # shape (N,)
    y_mm: np.ndarray          # shape (N,)
    # topology (triangular elements)
    tri_index: np.ndarray     # shape (E,)
    node_1: np.ndarray        # shape (E,)
    node_2: np.ndarray        # shape (E,)
    node_3: np.ndarray        # shape (E,)
    reg_code: np.ndarray      # shape (E,)
    # regions (optional)
    region_code: np.ndarray | None = None   # shape (R,)
    region_name: np.ndarray | None = None   # shape (R,), dtype=object
    # metadata
    attrs: dict[str, Any] = field(default_factory=dict)
```

### `postproc_interop/model/Solution.py`
```python
from __future__ import annotations
from abc import ABC

class Solution(ABC):
    """Abstract base: field data only (no geometry)."""
    pass
```

### `postproc_interop/model/SolutionMat.py`
```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from postproc_interop.model.Solution import Solution

@dataclass
class SolutionMat(Solution):
    label: str            # e.g. "Bx", "By", "A", "J"
    field: np.ndarray     # values per element shape (E,) or per node shape (N,)
    unit: str = ""        # e.g. "T", "Wb/m", "A/mm2"
    type_element: str = "triangle"  # "triangle" | "node"
```

### `postproc_interop/model/MeshSolution.py`
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from postproc_interop.model.Mesh import Mesh
from postproc_interop.model.Solution import Solution

@dataclass
class MeshSolution:
    """Composition of Mesh geometry + Solution field dict. (cf. pyleecan MeshSolution)"""
    mesh: Mesh
    solution_dict: dict[str, Solution] = field(default_factory=dict)
    label: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        from postproc_interop.model.MeshMat import MeshMat
        m = self.mesh
        return {
            "n_elements": int(m.tri_index.shape[0]) if isinstance(m, MeshMat) else None,
            "n_nodes": int(m.node_id.shape[0]) if isinstance(m, MeshMat) else None,
            "solution_keys": sorted(self.solution_dict.keys()),
        }
```

### `postproc_interop/model/__init__.py`
```python
from .Mesh import Mesh
from .MeshMat import MeshMat
from .Solution import Solution
from .SolutionMat import SolutionMat
from .MeshSolution import MeshSolution

__all__ = ["Mesh", "MeshMat", "Solution", "SolutionMat", "MeshSolution"]
```

---

## 새 어댑터 클래스 스펙

### `postproc_interop/adapters/MeshReader.py`
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from postproc_interop.model.Mesh import Mesh
from postproc_interop.model.MeshSolution import MeshSolution

class MeshReader(ABC):
    """Abstract base: reads a file and returns Mesh or MeshSolution."""

    @abstractmethod
    def can_read(self, path: Path) -> bool: ...

    @abstractmethod
    def read(self, path: Path) -> Mesh | MeshSolution: ...
```

### `postproc_interop/adapters/MotorCADMeshSolutionH5Reader.py`
- 구 `h5_motorcad.py`의 로직을 이식
- 반환: `MeshSolution(mesh=MeshMat(...), solution_dict={"bx": SolutionMat(...), ...})`
- H5 키 → 필드 매핑:
  ```
  mesh/tri_index → MeshMat.tri_index
  mesh/node_1    → MeshMat.node_1
  mesh/node_2    → MeshMat.node_2
  mesh/node_3    → MeshMat.node_3
  mesh/reg_code  → MeshMat.reg_code
  mesh/node_id   → MeshMat.node_id
  mesh/node_x_mm → MeshMat.x_mm
  mesh/node_y_mm → MeshMat.y_mm
  regions/reg_code → MeshMat.region_code
  regions/name     → MeshMat.region_name
  fields/bx → SolutionMat(label="Bx", unit="T")
  fields/by → SolutionMat(label="By", unit="T")
  fields/a  → SolutionMat(label="A",  unit="Wb/m")
  fields/j  → SolutionMat(label="J",  unit="A/mm2")
  ```

### `postproc_interop/adapters/MotorCADMeshSolutionTxtReader.py`
핵심 함수:
```python
def _open_mcad_text(path):
    """UTF-16 BOM / cp949 자동감지 컨텍스트 매니저."""

def _read_col_indices(in_file, expected_keys: frozenset) -> dict[str, int]:
    """4줄 preamble 처리: blank skip → 컬럼명 READ → units skip → separator skip."""
    in_file.readline()            # line 1: blank
    col_line = in_file.readline() # line 2: 컬럼명 ← 핵심 (eMach는 이 줄을 skip함)
    in_file.readline()            # line 3: units
    in_file.readline()            # line 4: separator ---
    tokens = [t.strip() for t in col_line.split(",")]
    return {t: i for i, t in enumerate(tokens) if t in expected_keys}

def _is_table_header(line: str, table_name: str) -> bool:
    tokens = line.strip().split()
    return len(tokens) >= 3 and tokens[1].isdigit() and tokens[2].strip() == table_name

def _scan_to_table(in_file, table_name: str) -> str | None:
    """ElementsTable / NodesTable / RegionsTable 헤더 라인 탐색."""
```

TXT 포맷 구조 (실제 파일: `data/MagTransient_20260130_152245.txt`):
```
10 Solution 1 Rotate Step 0.0000
1 11330 ElementsTable
                                              ← blank (line 1 of preamble)
TriIndex, Node1, Node2, Node3, RegCode, Bx, By, A, J   ← 컬럼명 (line 2)
  [-]  ,  [-] ,  [-] ,  [-] ,       , [T], [T], [Wb/m], [A/mm2]   ← units
-------, -----, -----, -----, -------, ---, ---, ------, -------   ← separator
     1,   1853,  1854,  1855,    109, -0.409, 0.670, ...           ← 데이터 시작
```

반환: `MeshSolution`

### `postproc_interop/adapters/VTUMeshReader.py`
- 구 `vtu.py` 내용 그대로 이식 (scaffold, NotImplementedError 유지)
- 반환 타입 힌트: `Mesh`

---

## eMach `magnetic.py` 변경 상세

파일: `eMach/tools/motorCAD/pyMCAD/magnetic.py`

### 추가할 함수
```python
_ELEM_COL_KEYS = frozenset({"TriIndex", "Node1", "Node2", "Node3", "RegCode", "Bx", "By", "A", "J"})
_NODE_COL_KEYS = frozenset({"NodeIndex", "X", "Y"})
_REGION_COL_KEYS = frozenset({"RegionCode", "RegionName"})

def _read_col_indices(in_file, expected_keys: frozenset) -> dict[str, int]:
    """4줄 preamble에서 컬럼명 줄을 읽어 {컬럼명: 인덱스} 반환."""
    in_file.readline()            # blank
    col_line = in_file.readline() # 컬럼명
    in_file.readline()            # units
    in_file.readline()            # separator
    tokens = [t.strip() for t in col_line.split(",")]
    return {t: i for i, t in enumerate(tokens) if t in expected_keys}
```

### `_parse_first_block_magnetic_file` 변경
```python
# 기존
_skip_header_lines(in_file, 4)
row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]  # 하드코딩

# 변경 후
ci = _read_col_indices(in_file, _ELEM_COL_KEYS)
ti_i = ci.get("TriIndex", 0)
n1_i = ci.get("Node1", 1)
n2_i = ci.get("Node2", 2)
n3_i = ci.get("Node3", 3)
rc_i = ci.get("RegCode", 4)
bx_i = ci.get("Bx", 5)
by_i = ci.get("By", 6)
a_i  = ci.get("A",  7)
j_i  = ci.get("J",  8)
# row[ti_i], row[n1_i], ...
```

동일 패턴을 `get_magnetic_timeseries_from_file` 내부에도 적용.
**주의: `postproc_interop` import 금지** (eMach는 독립 submodule)

---

## 검증 스크립트

작업 완료 후 아래로 검증:

```python
# 1. 모델 import
from postproc_interop.model import Mesh, MeshMat, Solution, SolutionMat, MeshSolution

# 2. TXT reader (실제 파일 있음)
from postproc_interop.adapters.MotorCADMeshSolutionTxtReader import MotorCADMeshSolutionTxtReader
reader = MotorCADMeshSolutionTxtReader()
sol = reader.read("data/MagTransient_20260130_152245.txt")
assert isinstance(sol, MeshSolution)
assert isinstance(sol.mesh, MeshMat)
assert "bx" in sol.solution_dict
print(sol.summary())

# 3. H5 reader (파일 있을 경우)
from postproc_interop.adapters.MotorCADMeshSolutionH5Reader import MotorCADMeshSolutionH5Reader
reader = MotorCADMeshSolutionH5Reader()
# sol = reader.read("data/some.h5")

# 4. tabular
from postproc_interop.tabular import load_as_tables
# frame, nodetable, regiontable, elementtable, elementtable_with_xy, stats = load_as_tables("data/...")
```

---

## 참고 파일
- 실제 TXT 샘플: `data/MagTransient_20260130_152245.txt`
- 구 모델: `postproc_interop/model.py` (MeshFrame, MeshTopology, NodeTable, RegionTable)
- 구 H5 어댑터: `postproc_interop/adapters/h5_motorcad.py`
- 헤더 매핑: `postproc_interop/header_mapping.py` (MOTORCAD_TXT_HEADER_TO_H5_KEY_MAP, 변경 없음)
- pyleecan 참조: `d:/KDH/gitPyleecan/pyleecan/pyleecan/Classes/MeshSolution.py` 등
