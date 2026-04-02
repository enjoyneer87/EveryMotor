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
- Do not rewrite history.
- Do not revert user changes unrelated to current task.
