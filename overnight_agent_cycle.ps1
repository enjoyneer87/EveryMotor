param(
    [string]$Token,
    [string]$DatabaseId = '33507031978c81e5a8eaeaf27372f37d',
    [string]$ServerId = $env:EMACH_SERVER_ID,
    [string]$RepoPath = '.',
    [string]$ExpectedSubmoduleBranch = 'devVeriACLoss',
    [switch]$AutoPick,
    [string]$TaskKey,
    [ValidateSet('start','heartbeat','done','hold')]
    [string]$Mode = 'heartbeat',
    [string]$Note,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ServerId)) {
    $ServerId = $env:COMPUTERNAME
}
if ([string]::IsNullOrWhiteSpace($ServerId)) {
    $ServerId = 'SERVER-UNKNOWN'
}
if ([string]::IsNullOrWhiteSpace($Token)) {
    $Token = $env:NOTION_TOKEN
}
if (-not [string]::IsNullOrWhiteSpace($env:NOTION_DATABASE_ID)) {
    $DatabaseId = $env:NOTION_DATABASE_ID
}

function Try-Run([scriptblock]$Action, [string]$Label) {
    try {
        Write-Output ("[STEP] " + $Label)
        & $Action
        Write-Output ("[OK] " + $Label)
    }
    catch {
        Write-Output ("[WARN] " + $Label + " :: " + $_.Exception.Message)
    }
}

$repo = (Resolve-Path $RepoPath).Path
Write-Output ("Repo: " + $repo)
Write-Output ("ServerId: " + $ServerId)
Write-Output ("Mode: " + $Mode)
Write-Output ("DryRun: " + [bool]$DryRun)

# 1) Local checks (best-effort)
if (Get-Command python -ErrorAction SilentlyContinue) {
    Try-Run {
        & python (Join-Path $repo 'check_submodule_branch.py') --repo $repo --submodule eMach --expected-branch $ExpectedSubmoduleBranch
        if ($LASTEXITCODE -ne 0) {
            throw ("check_submodule_branch exit code=" + $LASTEXITCODE)
        }
    } 'check_submodule_branch'
}
else {
    Write-Output '[WARN] python command not found. skip check_submodule_branch'
}

# 2) Notion metadata sync
if (-not [string]::IsNullOrWhiteSpace($Token)) {
    Try-Run {
        & (Join-Path $repo 'sync_notion_fields.ps1') -Token $Token -DatabaseId $DatabaseId -ServerId $ServerId -DryRun:$DryRun
        if ($LASTEXITCODE -ne 0) {
            throw ("sync_notion_fields exit code=" + $LASTEXITCODE)
        }
    } 'sync_notion_fields'

    if ($AutoPick) {
        Try-Run {
            & (Join-Path $repo 'notion_pick_workitem.ps1') -Token $Token -DatabaseId $DatabaseId -ServerId $ServerId -DryRun:$DryRun
            if ($LASTEXITCODE -ne 0) {
                throw ("notion_pick_workitem exit code=" + $LASTEXITCODE)
            }
        } 'notion_pick_workitem'
    }

    $taskArgs = @{
        Token = $Token
        DatabaseId = $DatabaseId
        ServerId = $ServerId
        Mode = $Mode
        RepoPath = $repo
    }
    if (-not [string]::IsNullOrWhiteSpace($TaskKey)) { $taskArgs.TaskKey = $TaskKey }
    if (-not [string]::IsNullOrWhiteSpace($Note)) { $taskArgs.Note = $Note }
    if ($DryRun) { $taskArgs.DryRun = $true }

    Try-Run {
        & (Join-Path $repo 'notion_task_update.ps1') @taskArgs
        if ($LASTEXITCODE -ne 0) {
            throw ("notion_task_update exit code=" + $LASTEXITCODE)
        }
    } 'notion_task_update'
}
else {
    Write-Output '[WARN] NOTION token not provided. Notion sync steps skipped.'
}

Write-Output '[DONE] overnight_agent_cycle completed.'
