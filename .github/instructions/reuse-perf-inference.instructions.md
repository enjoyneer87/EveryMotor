---
description: "Use when: editing or reviewing model inference code. Enforce reusable structure, performance logging, and A/J compatibility rules."
applyTo: "**/*infer*.py"
---
# Reuse and Performance Instruction

Apply these rules to inference code changes.

## Required structure
- Separate logic into dedicated functions for: data loading/parsing, sampling, per-model inference, and result serialization.
- Do not embed model-specific logic directly in main flow unless unavoidable.
- Prefer dynamic model I/O channel detection from checkpoint tensor shapes.

## Reuse rules
- If identical sampling logic appears in 2+ model paths, extract a shared helper.
- If identical checkpoint normalization logic appears in 2+ model paths, extract a shared helper.
- Preserve backward-compatible output keys when renaming keys used by notebooks.
- Prefer shared utilities over copy-paste blocks.

## Performance protocol
For model updates, record on same case/environment:
- end-to-end wall time (s)
- per-step average inference time (ms)
- output file size (MB)
- peak GPU memory if available

## Execution environment (mandatory)
- Run infer scripts in Docker by default (PhysicsNeMo runtime), not host Python.
- Use host Python for infer only when explicitly requested by user.
- Prefer mounted workspace paths used by the container (for example `/workspace/app`, `/workspace/host_data`).

Use:
- Command template: docker exec <container_name> bash -lc "cd /workspace/app; python infer_all_steps_nodes.py --case-idx <idx> --out <path>"
- Log template: logs/infer_<model>_<yyyymmdd_hhmm>.log

## Pre-merge checklist
- Functions are separated by responsibility.
- Duplicate sampling blocks removed.
- Output channel count is not hardcoded unless required.
- A/J fallback behavior is explicit when checkpoint is 2-channel.
- Runtime stats are reported and comparable to baseline.

## Baseline and regression rules
Track in PR/commit notes:
- model name, checkpoint path, output channels, case index, wall time, regression notes

Any slowdown over 10% vs baseline must include reason and mitigation plan.
Prefer at least 30% shared code path across model inference flows.
