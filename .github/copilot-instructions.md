# EveryMotor Overnight Agent Instructions

## Goal
- Continue implementation work autonomously overnight.
- Keep Notion DB and repository state synchronized for each work cycle.

## Environment Variables
- NOTION_TOKEN: Notion integration token.
- NOTION_DATABASE_ID: Notion DB id for Emlab Plan DB.
- EMACH_SERVER_ID: Optional server identifier. If missing, use COMPUTERNAME.

## Non-Negotiable Routine
1. Start cycle
- Run submodule and runtime checks first.
- Update Notion metadata heartbeat for rows in progress.

2. Pre-implementation review gate
- Before any code edit, review recent repository commits from other servers/contributors.
- Summarize what changed, potential conflicts, and carry-over tasks.
- Cross-check with current Notion plan rows (`진행 중`, `시작 전`, `홀드`) and synthesize one short execution plan.
- If recent commit summary evidence is unavailable, do not block by default: use Notion plan cross-check as fallback and proceed with an explicit note in `비고`.
- Start code edits only after this review and plan synthesis are complete.

3. Pick work item
- Prefer rows with status `진행 중` assigned to this server.
- If none exist, pick highest-priority `시작 전` row and move it to `진행 중`.

4. Implement
- Make focused code changes for one task at a time.
- Keep commits small and descriptive.
- Avoid destructive git operations.

5. Validate
- Run minimum relevant checks for touched files.
- If validation fails, keep working until fixed or explicitly blocked.

6. Sync Notion evidence
- Always update `동기화일`, `서버ID`.
- On success: set `상태=완료`, `검증완료=true`, set `커밋해시`, add short `비고`.
- On block: set `상태=홀드`, `검증완료=false`, add blocker to `비고`.

## Command Sequence
1. `python check_submodule_branch.py --repo . --submodule eMach --expected-branch devVeriACLoss`
2. `python check_runtime_contract.py --host-data /workspace/host_data --doe-data /workspace/host_data/doe_data --mso-root /workspace/multiscale-pde-operators`
3. `./sync_notion_fields.ps1 -Token $env:NOTION_TOKEN -DatabaseId $env:NOTION_DATABASE_ID`
4. `./notion_pick_workitem.ps1 -Token $env:NOTION_TOKEN -DatabaseId $env:NOTION_DATABASE_ID`
5. `./notion_task_update.ps1 -Token $env:NOTION_TOKEN -DatabaseId $env:NOTION_DATABASE_ID -Mode heartbeat`
6. `./overnight_agent_cycle.ps1 -Token $env:NOTION_TOKEN -DatabaseId $env:NOTION_DATABASE_ID -AutoPick -Mode heartbeat`

## Commit Policy
- Commit message format:
  - `[TASKKEY] short action summary`
- Include changed files related to the active task only.
- Do not include unrelated local changes.

## Safety Rules
- Do not run long GPU training automatically unless task explicitly requires it.
- Run inference scripts in Docker (PhysicsNeMo container) by default; avoid host Python execution for infer workflows unless explicitly requested.
- Do not rewrite history.
- Do not revert user changes unrelated to current task.

## NPZ / Checkpoint Backup Policy

**Rule: 코드를 수정한 뒤 추론 또는 학습을 재실행하기 전에 반드시 기존 결과를 백업한다.**

### 백업 대상
| 수정 파일 | 백업 대상 |
|-----------|-----------|
| `contracts.py` (채널 순서·정규화) | 모든 inference NPZ 디렉토리 |
| `motor_dataset.py` (샘플 생성·전처리) | 모든 inference NPZ 디렉토리 |
| `infer_phase1_pbc.py` | 해당 추론 NPZ 디렉토리 |
| `train.py` / `loss.py` | checkpoint `.pt` 파일 |
| `physics_operators.py` (`Je` 계산) | 모든 inference NPZ 디렉토리 |

### 백업 방법 (노트북)
```python
# 셀 13 에 정의된 backup_npz_dir() 사용
backup_npz_dir(ROOT / "results" / "full40_infer", label="before_4ch")
backup_npz_dir(ROOT / "results" / "3case_infer",  label="before_4ch")
```

### 백업 방법 (스크립트 / 오버나이트 에이전트)
```powershell
# 추론 재실행 직전 자동 백업 (타임스탬프 고유 디렉토리 생성)
python -c "
from pathlib import Path; from datetime import datetime; import shutil
d = Path('results/full40_infer')
b = d.parent/'backups'/f'{d.name}_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}'
b.mkdir(parents=True, exist_ok=True)
[shutil.move(str(f), str(b/f.name)) for f in d.glob('*.npz')]
print(b)
"
```

### 불변 원칙
- 백업은 덮어쓰지 않는다. 타임스탬프로 항상 새 디렉토리를 만든다.
- `results/backups/` 는 `.gitignore` 에 추가해 커밋하지 않는다.
- checkpoint `.pt` 가 교체될 때는 `results/backups/ckpt/` 에 수동 복사한다.
- Notion 비고에 백업 경로와 코드 변경 사유를 짧게 기록한다.
