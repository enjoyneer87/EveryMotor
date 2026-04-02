param(
    [string]$TaskName = "EveryMotor-PolicySync-5min",
    [string]$RepoPath = "D:\KDH\NvidiaNemo",
    [string]$Branch = "main",
    [string]$Remote = "origin",
    [string]$RunAsUser = $env:USERNAME,
    [switch]$RunWithHighest
)

$ErrorActionPreference = "Stop"

$syncScript = Join-Path $RepoPath "policy_sync.ps1"
if (-not (Test-Path -LiteralPath $syncScript)) {
    throw "policy_sync.ps1 not found at $syncScript"
}

$tr = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" -RepoPath "{1}" -Branch "{2}" -Remote "{3}"' -f $syncScript, $RepoPath, $Branch, $Remote
$trQuoted = '"{0}"' -f $tr

$argList = @(
    "/Create",
    "/SC", "MINUTE",
    "/MO", "5",
    "/TN", $TaskName,
    "/TR", $trQuoted,
    "/F"
)
if (-not [string]::IsNullOrWhiteSpace($RunAsUser)) {
    $argList += @("/RU", $RunAsUser)
}
if ($RunWithHighest) {
    $argList += @("/RL", "HIGHEST")
}

Write-Output "Register command: schtasks $($argList -join ' ')"
$p = Start-Process -FilePath "schtasks.exe" -ArgumentList $argList -NoNewWindow -Wait -PassThru
if ($p.ExitCode -ne 0) {
    throw "Failed to register task. schtasks exit code: $($p.ExitCode)"
}
Write-Output "Scheduled task '$TaskName' registered (every 5 minutes)."
Write-Output "Run once now: schtasks /Run /TN `"$TaskName`""
Write-Output "Check logs: $RepoPath\logs\policy-sync\policy_sync.log"
