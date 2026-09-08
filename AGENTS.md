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
- [`handoff_20260908.md`](.github/plans/handoff_20260908.md) — **latest session handoff (2026-09-08). Start here to resume.** Machine topology (134 unreachable; all GPU on the isolated PC), current champion (v3, |B| 10.934 / torque 2.336, torque-G2 passed), data-axis saturation, and the open 2군 decisions. Supersedes the 07-22 status below.
- [`handoff_20260722.md`](.github/plans/handoff_20260722.md) — historical session handoff (2026-07-22). The 07-22 commits and the then-current gate status (champion `mgn_nodeB_long`, "both 2–3× off G2"); superseded by the 09-08 handoff and `methodology_review §35`.
- [`methodology_review_20260720.md`](.github/plans/methodology_review_20260720.md) — **amends the roadmap below; read first.** Critical findings (split leak, target design, eval protocol) and the R0–R4 priority order
- [`postproc_interop_refactor.md`](.github/plans/postproc_interop_refactor.md) — pyleecan-style MeshSolution architecture (in progress, checklist inside)
- [`phase1_phase2_phase3_motorgnn_roadmap.md`](.github/plans/phase1_phase2_phase3_motorgnn_roadmap.md) — shared Phase 1-3 execution roadmap for Codex/Claude/Copilot/Antigravity

---

## Benchmark Rules (R0, non-negotiable)

All model comparison goes through `eval/` — never through per-model eval scripts.

```bash
python -m eval.benchmark --data-dir backup/doe_data --out results/benchmark_v2_baseline.json
python -m eval.benchmark --data-dir backup/doe_data --mgn-ckpt <ckpt.pt>   # needs torch
```

1. **Case-level holdout.** Splits come from `eval/splits/doe40_case_split.json`
   (30 train / 4 val / 6 test, cut by geometry). Training and eval share it via
   `eval.case_split.resolve_case_split`. Never split the flat record list —
   timesteps of one geometry are near-duplicates.
2. **No solution-derived inputs.** `A`, `J`, `B*`, `Je` are targets. Input feature
   lists are declared in `eval.feature_guard` and asserted at graph-build time.
3. **Physical units.** Report tesla / Wb/m, not normalized MSE.
4. **Torque is the acceptance metric.** `nrmse_torque_pct` from `eval.torque`
   (Arkkio band integral on the stationary airgap layer). Do not gate on
   instantaneous relative torque error — it diverges near zero crossings.
5. **Read node-support models against `node_resampling_floor`, not against zero.**
   The element→node→element round trip alone costs 17% |B| nRMSE and 60% torque
   nRMSE (it overestimates torque by ~57%), because the airgap band is one
   element thick and its nodes pick up stator-iron flux density. Any node-support
   model inherits that before it predicts anything.
6. **Pre-R0 numbers are void.** `model_comparison_summary.json`,
   `comparison_results.json` and `mgn_fem_comparison/test_summary.json` used a
   record-level shuffle and are not comparable to `benchmark_v2_baseline.json`.
   `doe_meshgraphnet_ckpt.pt` and `doe_meshgraphnet_ckpt_4d.pt` take 11 node
   inputs — they were fed A and J — and the harness refuses to score them.

`eval/` imports no torch (adapters in `eval/predictors.py` do), so the harness and
its tests run on the host: `python -m pytest tests/test_eval_*.py`

Its dependencies are **numpy, h5py and scipy**. scipy is used in exactly one place —
`GridResamplingFloorPredictor`, which needs a Delaunay triangulation to reproduce the
mesh→grid round trip — and that row is on by default, so it is a hard requirement of a
plain `python -m eval.benchmark`. The predictor probes for scipy when it is constructed
(before the multi-minute H5 load) rather than failing deep inside scoring. Pass
`--skip-grid-floor` to score without it.

---

## Field Representation: predict A, derive B

**Phase 1 is a *static* (memoryless) fit, not an angle-free one.** The model learns
`(rotor angle θ, operating point, geometry) → field` as a pointwise map: each
`(case, step)` is an independent shuffled sample, with **no `field(t-1)` input and no
rollout**, and connectivity is **read from the H5 per step, not generated from θ**.
Rotor angle is a condition coordinate, not a time axis — so learning angle-dependence
is *not* what would make it dynamic. That is Phase 2 (connectivity from θ(t) + `field(t-1)`
state). The DOE data is a multi-static rotor sweep (`Mag_OnLoadTorque`, one magnetostatic
solve per position), which is why `time_s` was a redundant shortcut for θ and was removed.

Two trainers exist and they are not equivalent:

| script | predicts | B support | torque floor |
|---|---|---|---|
| `train_doe_meshgraphnet.py` | Bx, By at nodes | node → element average | **59.5%** |
| `train_doe_curl_mgn.py` | **A at nodes** | P1 element curl, no round trip | **1.6%** |

Motor-CAD stores B per element. Averaging a node-predicted B onto elements
destroys the airgap/iron interface — the band is one element thick, so its nodes
carry stator-iron flux density. Gate G2 (torque < 3%) is unreachable through the
node-support path no matter how good the model is.

`phase1_static/discrete_curl.py` is the operator: exact for linear A, precomputed,
differentiable (numpy and torch). Use it instead of `autograd.grad(A, pos)`, which
is not a spatial derivative for a message-passing model (review F4).

**Do not supervise A against the exported potential.** The H5 stores A already
averaged onto elements; differentiating that reconstruction scores 38.7% |B|
nRMSE — worse than the round trip it replaces. Supervise element B *through* the
curl and pin only the gauge constant. `--w-a` exists to re-test this, not to use.

### The sliding band is invalid at rotated steps

The moving-mesh export writes one reference connectivity plus per-step node
coordinates, but the solver re-meshes the airgap sliding band every step.
Combining them inverts 180–246 elements from step 3 onward, all in layer `a2`.

- Exclude `a2`/`a3`/`a4` from any per-step geometry (element gradients, areas,
  the curl, the loss) via `eval.mesh_regions.sliding_band_mask`.
- **Keep them in the message-passing graph** — they are the only rotor↔stator
  connection, and dropping them splits the graph in two.
- The stationary stator-side layer `a1` is never inverted, which is why the
  torque band is built on it.
- The benchmark masks these elements for every model alike and reports
  `element_coverage` (~0.865) in the artifact.

## Sector symmetry: the model is 1/8 anti-periodic

45-degree sector, 8-fold, **anti**-periodic. Measured on the export: rotating one
sector negates the field (corr -0.994), two sectors restores it (+0.981), and
across the cut planes `||A(0)+A(-45)||/||A|| = 0.017` while the periodic
hypothesis gives 2.007. `phase1_static/sector_symmetry.py` owns this.

Four things the DOE path got wrong before, all now handled by
`train_doe_curl_mgn.py`:

1. **`rotate_step` is the per-step increment**, so it only ever holds {0, -2} and
   cannot identify a timestep. Use `rotor_angle_features` — cumulative angle as
   sin/cos at a two-sector period, so one sector shifts the phase by pi and the
   sign flip is carried by the encoding.
2. **The export reports the rotor unwrapped** (out to -153 deg while the stator
   stays at [-45, 0]). 47% of the training samples sat outside the solved
   domain. `wrap_rotor_coordinates` folds the rigid rotor back and returns the
   anti-periodic `sign` — **multiply the target by it**, or the wrapped samples
   are mislabeled.
3. **The two cut planes had zero connecting edges.** `cut_plane_pairs` +
   `anti_periodic_edges` add sign -1 edges so message passing can cross.
4. **Nothing asked the model to be anti-periodic.** `--w-pbc` penalizes
   `A(0) + A(-45)`.

Wrapping does **not** fix the `a2` inversions — that layer is re-meshed by the
solver every step, and wrapping neither causes nor cures it. What matters is
that wrapping introduces zero inversions *outside* the already-excluded sliding
band, which is asserted in `tests/test_sector_symmetry.py`.

Only the rigid rotor regions are rotated; `a2`'s own nodes are never moved.

### Running it

```bash
# in the PhysicsNeMo container, repo mounted at /workspace/app
python train_doe_curl_mgn.py --data-dir backup/doe_data --epochs 30 \
    --batch-size 4 --step-stride 3 --ckpt results/mgn_curl_caseholdout.pt
python -m eval.benchmark --data-dir backup/doe_data \
    --curl-ckpt results/mgn_curl_caseholdout.pt --out results/benchmark_v2_curl.json
```

- `--batch-size 4`, not 8: a 15-layer MeshGraphNet on ~7k-node graphs fills a
  24 GB card at batch 8 and the resulting allocator thrashing made an epoch
  **12x** slower (50 s -> 10 min+). Check `nvidia-smi` before blaming the model.
- `--step-stride N` subsamples the rotor sweep evenly. `--max-steps-per-case`
  truncates instead, keeping only the start of the electrical cycle.
- Batched curl operators need `CurlData.__inc__` to offset `curl_node_index` per
  graph. Get that wrong and every graph reads the first graph's nodes with no
  error and no NaN — `tests/test_curl_trainer_batching.py` guards it (container
  only; needs torch).

---

## Environment Rules

- **Execution path is machine-dependent — check which machine you are on first.**
  - Machines with Docker + WDDM GPU: run in the PhysicsNeMo container, not host Python.
  - `HPC_134` / `192.168.0.134` (L40S): **Docker is not available.** The GPU is in
    TCC driver mode and WSL2 GPU passthrough requires WDDM, so container+GPU cannot
    work. Use the native `.venv` (`.venv\Scripts\python.exe`). This path is
    validated — it reproduces the reference scorecard exactly (13.324% |B| /
    9.207% torque on `mgn_nodeB_long.pt`). The `.sh` files in `results/logs/`
    hardcode the container path `/workspace/app` and will not run as-is.
  - Whichever path you use, the acceptance contract is the same: score
    `mgn_nodeB_long.pt` and confirm **13.324% |B| / 9.207% torque** before
    trusting any new result from that machine.
- **Score through `eval/determinism.py`; an unpinned run is not repeatable.**
  Before it existed, the same checkpoint scored on the same machine with the
  same code moved the ~8th significant digit every run (measured: |B|
  13.32386180647098 / 13.323861819126696 / 13.32386179404891). The numpy floors
  were bit-identical throughout, so the drift was entirely in the GPU path.
  With TF32, cuDNN autotuning and `use_deterministic_algorithms` pinned, three
  runs produced **byte-identical scorecards**. The flag that did it is
  `use_deterministic_algorithms`: `torch_scatter` delegates its sum to torch's
  `scatter_add_`, which has a deterministic implementation. `run_benchmark`
  calls the pinning itself, so this is automatic — but it runs `warn_only`, so
  the guarantee is *measured, not enforced*. Re-measure after changing the model.
- **Bitwise equality is a same-machine claim. Across machines the contract is
  still three decimals** — **13.324% |B| / 9.207% torque** — which is what the
  WSL2 container and the `HPC_134` native venv agree on. Do not read a
  cross-machine difference in the 8th digit as a regression.
- **`physicsnemo`'s MeshGraphNet requires `torch_scatter`.** Without it the
  benchmark used to load the checkpoint and silently score **zero samples** — no
  exception, exit code 0, just `no samples`. That is now guarded twice:
  `eval/runtime_guard.py` fails at checkpoint load with the fix in the message,
  and `eval.benchmark` exits **1** if any model scored nothing, whatever the
  cause. `check_runtime_contract.py` still does not check for it.
- **`.venv` is `--system-site-packages`: `torch` and `numpy` come from the
  *global* interpreter, the graph stack from the venv.** `torch_scatter` is
  compiled against one exact torch release (`2.1.2+pt211cu128` ↔ torch 2.11.0),
  so a global torch upgrade breaks the acceptance path without touching `.venv`.
  Full table and rationale: MIGRATION.md, **Environment pinning**.
- **When adding a section to any `.md`, record which machine produced the
  result** — hostname + IP (e.g. `HPC_134 / 192.168.0.134`). Stacks differ per
  machine (container vs native), so results are only comparable when the origin
  is known. See `.github/plans/methodology_review_20260720.md` §10 for the format.
  The JSON scorecards now record this themselves: `run_benchmark` writes a
  `provenance` block with the commit (and whether the tree was dirty), the
  hostname and IP, the **interpreter path** — which is what separates a `.venv`
  run from a bare global-interpreter one — and the command. Quote it rather than
  reconstructing it by hand. Scorecards written before `eval/provenance.py` keep
  their thin `environment` block; do not backfill one, since a provenance field
  invented after the fact is a guess wearing a record's clothes.
- Data (`backup/doe_data`, `*.pt`) travels via the department share
  `\\192.168.0.165\디지털융합사업본부\01_EM사업부\강도현\EveryMotor_migration`,
  never git. Code travels via git only. Checkpoints with the same filename may
  hold different weights — check the sha256 before overwriting.
- `eMach/` is a git submodule — do NOT add `postproc_interop` as a dependency inside it
- Commit format: `[TASKKEY] short action summary`
- Do not run long GPU training unless task explicitly requires it
- Full agent cycle policy: `.github/copilot-instructions.md`
