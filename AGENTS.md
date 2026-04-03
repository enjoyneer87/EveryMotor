# EveryMotor — Agent Instructions

> **For GitHub Copilot:** see `.github/instructions/` and `.github/copilot-instructions.md`
> **For Codex / OpenAI agents:** read this file first, then `.github/plans/` for task-specific plans.
> **For Antigravity:** this file + `.github/instructions/design-principles.instructions.md`

---

## Repository Overview

Motor electromagnetic FEA postprocessing pipeline.
Primary language: Python 3.10+. Docker runtime: PhysicsNeMo container.

Key packages:
- `postproc_interop/` — format-agnostic FEA mesh postprocessing library
- `eMach/` — git submodule (independent, do NOT import postproc_interop inside it)
- `phase1_static/` — PhysicsNeMo GNN training scaffolding

---

## Design Principles (mandatory — read before editing any .py)

### Category-Theory-Aligned OOP

```
Object   = immutable dataclass  (MeshMat, SolutionMat)
Morphism = pure function        (MeshReader.read, meshsolution_to_dataframes)
Functor  = format adapter       (MotorCADMeshSolutionTxtReader, H5Reader)
Pipeline = composition          (TXT → MeshSolution → DataFrame → VTU)
```

**Rules:**
1. Data classes are immutable after `__init__` (no post-init attribute mutation)
2. Transformation functions are pure (no side effects, no global state writes)
3. Composition over inheritance — `MeshSolution` contains `Mesh`, does not extend it
4. Abstract classes named by role only (`Mesh`, `MeshReader`)
5. Concrete classes named `{Source}{DataType}{Format}{Role}` (`MotorCADMeshSolutionH5Reader`)
6. **File name = Class name** (pyleecan convention, PascalCase)

Full rationale: `.github/instructions/design-principles.instructions.md`
Phase execution playbook: `.github/instructions/phase-dev-context-harness.instructions.md`

---

## Naming Convention

| Type | Example |
|------|---------|
| Abstract base | `Mesh`, `Solution`, `MeshReader` |
| Numpy concrete | `MeshMat`, `SolutionMat` |
| Format reader (geo only) | `VTUMeshReader` |
| Format reader (geo+fields) | `MotorCADMeshSolutionH5Reader` |

---

## Architecture: postproc_interop

```
model/
  Mesh(ABC) ← MeshMat          geometry only
  Solution(ABC) ← SolutionMat  field data only
  MeshSolution                  Mesh + solution_dict (composition)

adapters/
  MeshReader(ABC)
  MotorCADMeshSolutionH5Reader  .h5  → MeshSolution
  MotorCADMeshSolutionTxtReader .txt → MeshSolution
  VTUMeshReader                 .vtu → Mesh (scaffold)

tabular.py   MeshSolution → pandas DataFrames
header_mapping.py  MOTORCAD_TXT_HEADER_TO_H5_KEY_MAP (do not modify keys)
```

---

## Test Strategy

Test each layer independently with real sample data:

```python
# Layer 1 — model (object shape)
from postproc_interop.model import MeshMat, MeshSolution

# Layer 2 — adapter (morphism correctness)
from postproc_interop.adapters.MotorCADMeshSolutionTxtReader import MotorCADMeshSolutionTxtReader
reader = MotorCADMeshSolutionTxtReader()
sol = reader.read("data/MagTransient_20260130_152245.txt")
assert isinstance(sol, MeshSolution)
assert "bx" in sol.solution_dict

# Layer 3 — pipeline (composition)
from postproc_interop.tabular import meshsolution_to_dataframes
nodetable, *_ = meshsolution_to_dataframes(sol)
assert "NodeIndex" in nodetable.columns
```

Real sample file: `data/MagTransient_20260130_152245.txt`

---

## Active Work Plans

See `.github/plans/` for ongoing refactoring tasks:
- [`postproc_interop_refactor.md`](.github/plans/postproc_interop_refactor.md) — pyleecan-style MeshSolution architecture (in progress, checklist inside)
- [`phase1_phase2_phase3_motorgnn_roadmap.md`](.github/plans/phase1_phase2_phase3_motorgnn_roadmap.md) — shared Phase 1-3 execution roadmap for Codex/Claude/Copilot/Antigravity

---

## Environment Rules

- Run inference scripts in Docker (PhysicsNeMo container), not host Python
- `eMach/` is a git submodule — do NOT add `postproc_interop` as a dependency inside it
- Commit format: `[TASKKEY] short action summary`
- Do not run long GPU training unless task explicitly requires it
- Full agent cycle policy: `.github/copilot-instructions.md`
