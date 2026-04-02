# XENV-5 Validate 3 Cases
> Migrated: Active workflow source is [.github/skills/xenv-validate-3-cases/SKILL.md](.github/skills/xenv-validate-3-cases/SKILL.md).

## Goal
Run inference validation for three DOE cases and emit summary evidence for Notion.

## Command

```powershell
python validate_three_cases.py
```

Optional explicit cases:

```powershell
python validate_three_cases.py --case-indices 0 1 2
```

## Outputs
- NPZ files: HOST_DATA_DIR/validation_3cases/case_XXXX_allsteps.npz
- Summary JSON: HOST_DATA_DIR/validation_3cases/summary.json

## Notion Evidence Suggestion
- 비고: "3-case validation complete"
- 완료근거: path to summary.json
- 검증완료: true only if summary.ok is true
