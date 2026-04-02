---
name: xenv-validate-3-cases
description: "Use when: running 3-case DOE inference validation and producing Notion-ready evidence outputs."
---

# XENV Validate 3 Cases

## Purpose
Run 3-case inference validation and produce summary evidence for status sync.

## Inputs
- Optional case indices (default 0 1 2)
- Existing data and environment required by validate_three_cases.py

## Execution
1. Run default validation:
   - python validate_three_cases.py
2. Or run explicit indices:
   - python validate_three_cases.py --case-indices 0 1 2

## Expected outputs
- NPZ files under HOST_DATA_DIR/validation_3cases:
  - case_XXXX_allsteps.npz
- Summary JSON:
  - HOST_DATA_DIR/validation_3cases/summary.json

## Verification checklist
- summary.json exists
- summary.ok is true before marking verification complete
- case-level output files exist for requested indices

## Notion sync template
- 비고: 3-case validation complete
- 완료근거: path to summary.json
- 검증완료: true only when summary.ok is true

## Failure handling
- If summary.ok is false, set status to hold and include blocker details in notes.
