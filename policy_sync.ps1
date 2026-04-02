param(
    [string]$RepoPath = ".",
    [string]$Branch = "",
    [string]$Remote = "origin",
    [string]$LogDir = "logs/policy-sync",
    [string]$NotionToken = $env:NOTION_TOKEN,
    [string]$NotionDatabaseId = $env:NOTION_DATABASE_ID,
    [string]$NotionTaskKey = "",
    [string]$ServerId = $env:EMACH_SERVER_ID
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param(
        [string]$Level,
        [string]$Message,
        [hashtable]$Data
    )

    if (-not (Test-Path -LiteralPath $script:LogDirAbs)) {
        New-Item -ItemType Directory -Path $script:LogDirAbs -Force | Out-Null
    }

    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$timestamp][$Level] $Message"
    Add-Content -Path $script:TextLogPath -Value $line

    $obj = [ordered]@{
        ts = (Get-Date).ToString("o")
        level = $Level
        msg = $Message
        data = $Data
    }
    Add-Content -Path $script:JsonLogPath -Value ($obj | ConvertTo-Json -Compress -Depth 8)
}

function Invoke-Git {
    param([string[]]$GitArgs)
    $output = & git -C $script:RepoAbs @GitArgs 2>&1
    $code = $LASTEXITCODE
    return [PSCustomObject]@{ Output = ($output -join "`n"); ExitCode = $code }
}

function Send-NotionHeartbeat {
    param(
        [ValidateSet("heartbeat", "hold")]
        [string]$Mode,
        [string]$Note
    )

    if ([string]::IsNullOrWhiteSpace($NotionToken) -or [string]::IsNullOrWhiteSpace($NotionDatabaseId)) {
        Write-Log -Level "INFO" -Message "Notion token/db missing. Skip Notion update." -Data @{}
        return
    }

    $scriptPath = Join-Path $script:RepoAbs "notion_task_update.ps1"
    if (-not (Test-Path -LiteralPath $scriptPath)) {
        Write-Log -Level "WARN" -Message "notion_task_update.ps1 not found. Skip Notion update." -Data @{}
        return
    }

    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $scriptPath,
        "-Token", $NotionToken,
        "-DatabaseId", $NotionDatabaseId,
        "-Mode", $Mode,
        "-ServerId", $ServerId,
        "-RepoPath", $script:RepoAbs,
        "-Note", $Note
    )
    if (-not [string]::IsNullOrWhiteSpace($NotionTaskKey)) {
        $args += @("-TaskKey", $NotionTaskKey)
    }

    $p = Start-Process -FilePath "powershell.exe" -ArgumentList $args -NoNewWindow -Wait -PassThru
    Write-Log -Level "INFO" -Message "Notion update executed." -Data @{ mode = $Mode; exitCode = $p.ExitCode }
}

$script:RepoAbs = (Resolve-Path -LiteralPath $RepoPath).Path
$script:LogDirAbs = Join-Path $script:RepoAbs $LogDir
$script:TextLogPath = Join-Path $script:LogDirAbs "policy_sync.log"
$script:JsonLogPath = Join-Path $script:LogDirAbs "policy_sync.jsonl"

$policyPaths = @(
    ".github",
    "REUSE_PERF_GUIDELINES.md",
    "XENV_VALIDATE_3_CASES.md",
    "XENV_SUBPROCESS_BRIDGE.md",
    "eMach/.github"
)

if ([string]::IsNullOrWhiteSpace($Branch)) {
    $branchRes = Invoke-Git -GitArgs @("rev-parse", "--abbrev-ref", "HEAD")
    if ($branchRes.ExitCode -ne 0) {
        Write-Log -Level "ERROR" -Message "Unable to determine current branch." -Data @{ output = $branchRes.Output }
        Send-NotionHeartbeat -Mode "hold" -Note "policy-sync failed: branch resolution error"
        exit 2
    }
    $Branch = $branchRes.Output.Trim()
}

Write-Log -Level "INFO" -Message "Policy sync start." -Data @{ repo = $script:RepoAbs; branch = $Branch; remote = $Remote }

$statusRes = Invoke-Git -GitArgs @("status", "--porcelain")
if ($statusRes.ExitCode -ne 0) {
    Write-Log -Level "ERROR" -Message "Failed to check policy path status." -Data @{ output = $statusRes.Output }
    Send-NotionHeartbeat -Mode "hold" -Note "policy-sync failed: status check error"
    exit 3
}

$changedLines = @()
if (-not [string]::IsNullOrWhiteSpace($statusRes.Output)) {
    $changedLines = $statusRes.Output -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}

$policyChanged = @()
foreach ($line in $changedLines) {
    foreach ($p in $policyPaths) {
        if ($line -match [regex]::Escape($p.Replace('/', '\\'))) {
            $policyChanged += $line
            break
        }
    }
}

if ($policyChanged.Count -gt 0) {
    Write-Log -Level "WARN" -Message "Local policy changes detected. Pull skipped for safety." -Data @{ changed = ($policyChanged -join "`n") }
    Send-NotionHeartbeat -Mode "hold" -Note "policy-sync skipped: local policy changes detected"
    exit 4
}

$fetchRes = Invoke-Git -GitArgs @("fetch", $Remote, $Branch)
if ($fetchRes.ExitCode -ne 0) {
    Write-Log -Level "ERROR" -Message "Fetch failed." -Data @{ output = $fetchRes.Output }
    Send-NotionHeartbeat -Mode "hold" -Note "policy-sync failed: git fetch error"
    exit 5
}

$headRes = Invoke-Git -GitArgs @("rev-parse", "HEAD")
$remoteHeadRes = Invoke-Git -GitArgs @("rev-parse", "$Remote/$Branch")
if ($headRes.ExitCode -ne 0 -or $remoteHeadRes.ExitCode -ne 0) {
    Write-Log -Level "ERROR" -Message "Unable to resolve local/remote heads." -Data @{ local = $headRes.Output; remote = $remoteHeadRes.Output }
    Send-NotionHeartbeat -Mode "hold" -Note "policy-sync failed: head resolution error"
    exit 6
}

$localHead = $headRes.Output.Trim()
$remoteHead = $remoteHeadRes.Output.Trim()
if ($localHead -eq $remoteHead) {
    Write-Log -Level "INFO" -Message "Already up to date." -Data @{ head = $localHead }
    Send-NotionHeartbeat -Mode "heartbeat" -Note "policy-sync heartbeat: already up to date"
    exit 0
}

$pullRes = Invoke-Git -GitArgs @("pull", "--ff-only", $Remote, $Branch)
if ($pullRes.ExitCode -ne 0) {
    $isConflict = $pullRes.Output -match "CONFLICT|cannot fast-forward|Not possible to fast-forward"
    Write-Log -Level "ERROR" -Message "Pull failed." -Data @{ output = $pullRes.Output; conflict = $isConflict }
    if ($isConflict) {
        Send-NotionHeartbeat -Mode "hold" -Note "policy-sync failed: fast-forward conflict/manual intervention required"
    }
    else {
        Send-NotionHeartbeat -Mode "hold" -Note "policy-sync failed: git pull error"
    }
    exit 7
}

$newHeadRes = Invoke-Git -GitArgs @("rev-parse", "--short", "HEAD")
$newHead = if ($newHeadRes.ExitCode -eq 0) { $newHeadRes.Output.Trim() } else { "unknown" }
Write-Log -Level "INFO" -Message "Policy sync success." -Data @{ previous = $localHead; current = $newHead }
Send-NotionHeartbeat -Mode "heartbeat" -Note "policy-sync success: updated to $newHead"
exit 0
