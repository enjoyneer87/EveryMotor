# Motor GNN Roadmap (Phase 1-3)

This document is the shared execution contract for Codex, Claude, GitHub Copilot, and Antigravity.

## Why This Plan Exists
- A prior Gemini prompt outlined a practical roadmap for static-to-dynamic motor GNN development.
- This plan validates that roadmap, applies repository constraints, and defines an implementation-ready sequence.
- Use this file as the single source of truth for multi-agent handoff.

## Prompt Review (Gemini Proposal)
The proposal is largely valid and aligned with this repo.

Accepted as-is:
- KDTree-based master/slave matching for 1/8 periodic boundary mapping
- Anti-periodic message sign handling on boundary edges
- Physics-informed constraint from curl(A) to Bx/By
- Overfit-single sanity test and boundary continuity visualization
- Lambda annealing for physics terms

Required adjustments for this repo:
- Training target contract is fixed to channel order: `[Bx, By, A, J]`
- Primary loss strategy is hybrid: supervised A + supervised B + curl consistency
- Inference/training runtime should be validated in Docker PhysicsNeMo environment
- Keep `eMach/` independent (no `postproc_interop` dependency injection)

## Phase 1 (Now): Static 1/8 + PBC + Hybrid Loss
### Objective
Build a stable static training baseline that can overfit one sample and satisfy boundary symmetry.

### Copilot scaffold prompts
Use these comments directly in code files to drive GitHub Copilot generation:

```python
# TODO: Use scipy.spatial.KDTree to match master/slave boundary nodes for 1/8 sector PBC.
# Rotate slave nodes by -45 deg, find nearest master nodes within tolerance, and create bidirectional edge_index.
# Set interior edge_attr=+1.0 and anti-periodic PBC edge_attr=-1.0.
```

```python
# TODO: Build a StaticMotorDataset that returns PyG Data(x, pos, edge_index, edge_attr, y).
# x must include coordinates and physical/context features, with coords autograd-ready for curl loss.
# y channel order must be [Bx, By, A, J].
```

```python
# TODO: Implement hybrid physics loss:
# total = wA*MSE(A_pred,A_gt) + wB*MSE([Bx_pred,By_pred],[Bx_gt,By_gt]) + wCurl*MSE(curl(A_pred),[Bx_gt,By_gt]).
# Use autograd.grad on A_pred wrt coordinates to compute Bx=dA/dy and By=-dA/dx.
```

### Implementation tasks
- Data preprocessing:
  - Identify master/slave boundary node pairs (1/8 sector rotation basis)
  - Build bidirectional PBC edges
  - Assign edge sign flags: interior `+1`, anti-periodic PBC `-1`
- Dataset:
  - Build PyG `Data` objects with node features, edge indices, edge attributes, and targets
  - Ensure coordinate tensors are autograd-compatible for curl loss
- Model:
  - Use anti-periodic message handling (`message * edge_attr`)
  - Preserve compatibility with PhysicsNeMo MeshGraphNet forward signatures
- Loss:
  - `L = wA * L_A + wB * L_B + wCurl * L_curl`
  - `L_A`: MSE(A_pred, A_gt)
  - `L_B`: MSE([Bx_pred, By_pred], [Bx_gt, By_gt])
  - `L_curl`: MSE(curl(A_pred), [Bx_gt, By_gt])
  - Anneal `wCurl` by epoch
- Training checks:
  - Overfit-single loss should drop near `1e-4` target range
  - Boundary continuity and sign behavior must be visually verified

### Verification checklist (mandatory before Phase 2)
- No isolated nodes after PBC edge insertion
- `edge_index` shape is valid `[2, E]`
- PBC sign handling is correctly applied on boundary edges
- Overfit-single converges without divergence
- Boundary continuity plots are archived (Matplotlib/Paraview evidence)

## Phase 2: Dynamic Sliding-Band Time-Series Pipeline
### Objective
Support time-varying topology/conditions without memory instability.

### Implementation tasks
- Build dynamic edge refresh for gap/sliding regions per timestep
- Reuse static region topology to reduce overhead
- Build time-aware dataloader returning `(x_t, edge_t, y_t)`
- Apply normalization/standardization policy for mixed-scale physical features

### Verification checklist
- No memory leak under long dataloader iteration
- Stable GPU/CPU usage during multi-step epoch execution
- Interpolated rotor angles do not produce boundary discontinuities

## Phase 3: Autoregressive Rollout Stability
### Objective
Enable robust long-horizon rollout with controlled error accumulation.

### Implementation tasks
- Transition from absolute prediction to residual prediction (`delta A_t` or equivalent)
- Introduce teacher forcing schedule
- Add noise injection for rollout robustness
- Add multi-step training loss (2-5+ rollout steps)

### Verification checklist
- 10/50/100-step rollout error trends are quantified
- No exponential blow-up in rollout MSE
- Torque/energy trend consistency against FEM baseline is tracked
- Inference speed-up ratio versus FEM is measured and logged

## Agent Handoff Protocol
All agents should follow these rules when continuing work:
- Read `AGENTS.md` first
- Read this roadmap before editing phase code
- Keep commit scope focused to one subtask
- Keep commit message format: `[TASKKEY] short action summary`
- Do not rewrite history and do not revert unrelated local changes

## Suggested Next Task Slice
Complete Phase 1 to "verification complete":
- Stabilize dataset+model+loss integration
- Pass overfit-single target
- Produce boundary continuity artifact
- Record run config and metrics in commit notes
