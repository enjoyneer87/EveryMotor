# Reuse and Performance Guidelines
> Migrated: Active rule source is [.github/instructions/reuse-perf-inference.instructions.md](.github/instructions/reuse-perf-inference.instructions.md).

This repository should prefer reusable model-inference pipelines and measurable performance checks over one-off script logic.

## 1. Mandatory Structure for Inference Scripts
- Split into dedicated functions:
  - data loading / parsing
  - grid or node sampling
  - per-model inference
  - result serialization
- Avoid embedding model-specific logic directly in `main()`.
- Keep model I/O channel handling dynamic from checkpoint shapes when possible.

## 2. Reuse Rules
- If the same sampling logic is used by 2 or more models, extract it to a shared function.
- If the same checkpoint normalization pattern is used by 2 or more models, extract a helper.
- Save backward-compatible output keys when renaming keys (for notebook compatibility).
- Prefer utility modules (`doe_data_utils.py` and shared infer helpers) over copy-paste.

## 3. Performance Comparison Protocol
For each model update, record the following on the same case and environment:
- end-to-end inference wall time (seconds)
- per-step average inference time (ms)
- output file size (MB)
- peak GPU memory (if available)

Use a fixed command template and log file naming convention:
- Command: `python infer_all_steps_nodes.py --case-idx <idx> --out <path>`
- Log file: `logs/infer_<model>_<yyyymmdd_hhmm>.log`

## 4. Minimum Review Checklist (Before Merge)
- [ ] Functions are separated by responsibility.
- [ ] No duplicated sampling code blocks remain.
- [ ] Model output channel count is not hardcoded unless required.
- [ ] A/J fallback behavior is explicit when checkpoint has only 2 channels.
- [ ] Runtime stats are reported and comparable to baseline.

## 5. Baseline Tracking
Keep a baseline table in PR description or commit notes:
- model name
- checkpoint path
- output channels
- case index
- wall time
- notes on regressions/improvements

## 6. Practical Target
- Prefer >= 30% shared code path across model inference flows.
- Any > 10% slowdown vs baseline must include a reason and mitigation plan.
