---
description: "Use when: implementing or reviewing phase1/phase2/phase3 motor GNN code. Enforce contract-first context engineering and reproducible harness engineering."
applyTo: "phase1_static/**/*.py,train_*_meshgraphnet*.py,infer_*meshgraphnet*.py,doe_data_utils.py,patched_datasets.py"
---

# Phase Development Context + Harness Instruction

## Purpose
Keep multi-agent development consistent across Codex, Claude, Copilot, and Antigravity.
Prevent drift by freezing contracts first and validating via repeatable harnesses.

## Contract-First Rule (required)
Before coding feature behavior, freeze:
- Canonical target channel contract: `[Bx, By, A, J]`
- Stage boundaries: `raw -> canonical sample -> graph -> batch -> loss`
- Purity at boundaries: stage inputs are not mutated in-place
- Multi-fidelity observation contract:
  - `step_semantics` must be explicit per sample:
    - `explicit_time_step` (transient)
    - `implicit_single_step` (static single-step)
    - `derived_static_step` (aligned/derived mapping)
  - `fidelity_type` and `coupling_policy` must be attached at sample build time
  - Do not force static-only targets into transient step loss directly; route by coupling policy

If any contract changes, update plan docs and validation checks in the same commit.

## Context Engineering Workflow
For each non-trivial task, produce a short context pack with:
- Goal
- Hard constraints and invariants
- Inputs/outputs and expected tensor shapes
- Non-goals
- Acceptance checks

Prefer linking source-of-truth files over copying long explanations.

## Harness Engineering Workflow
Every training/inference change must have:
- Contract harness:
  - Shape/order checks at loader, graph, and loss boundaries
- Overfit harness:
  - Single sample or single batch convergence check
- Smoke harness:
  - 1-epoch run with deterministic seed and small data slice
- Regression harness:
  - Baseline metric snapshot and tolerance-based comparison

Multi-fidelity harness additions:
- Alignment harness:
  - Verify static observations can be represented as `T=1` sequence entries
  - Verify multi-static alignment mapping to transient indices is deterministic and pure
- Routing harness:
  - Verify loss routing by `coupling_policy` (transient/weak/decoupled) matches expectation

Use Docker PhysicsNeMo runtime for final validation consistency.

## Anti-Patterns to Avoid
- Adding dynamic phase logic before contract freeze gates pass
- Silent target channel reorder without explicit mapping and tests
- Hidden global state or in-place mutation across stage boundaries
- Long training runs before overfit and smoke harnesses pass

## Minimum Acceptance Gate
Do not mark a task complete unless:
- Contract checks pass
- Overfit-single passes for the changed path
- One deterministic smoke run passes
- Metrics/logs are captured in notes
