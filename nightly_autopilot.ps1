$ErrorActionPreference = "Continue"

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$outRoot = "D:\KDH\EveryMotorOut\nightly_20260401"
$logPath = Join-Path $outRoot "nightly_autopilot.log"
New-Item -ItemType Directory -Path $outRoot -Force | Out-Null

function Write-Log($msg) {
    $line = "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] $msg"
    Add-Content -Path $logPath -Value $line
}

Write-Log "Autopilot tick started"

# 1) Ensure container is up
$ps = wsl docker ps --format "{{.Names}}" 2>$null
if ($ps -notcontains "motor_compare") {
    Write-Log "motor_compare not running. Attempting compose up..."
    wsl docker compose -f /mnt/d/KDH/NvidiaNemo/docker-compose.yml up -d --build | Out-Null
    Start-Sleep -Seconds 3
}

# 2) Ensure compare_models is running
$proc = wsl docker exec motor_compare bash -lc "ps -ef | grep compare_models.py | grep -v grep || true" 2>$null
if ([string]::IsNullOrWhiteSpace($proc)) {
    Write-Log "compare_models.py not running. Starting nightly run..."
    wsl docker exec -d motor_compare bash -lc "mkdir -p /workspace/out/nightly_20260401; python /workspace/app/compare_models.py --data-dir /workspace/app/doe_data --mgn-ckpt /workspace/app/doe_meshgraphnet_ckpt.pt --fno-ckpt /workspace/app/doe_fno_ckpt.pt --gino-ckpt /workspace/app/doe_gino_ckpt.pt --rnn-ckpt /workspace/app/doe_rnn_ckpt.pt --out-dir /workspace/out/nightly_20260401 > /workspace/out/nightly_20260401/compare_models.log 2>&1"
    Start-Sleep -Seconds 2
} else {
    Write-Log "compare_models.py is already running"
}

# 3) Tail latest log lines to heartbeat log
$tail = wsl docker exec motor_compare bash -lc "tail -n 20 /workspace/out/nightly_20260401/compare_models.log || true" 2>$null
if (-not [string]::IsNullOrWhiteSpace($tail)) {
    Add-Content -Path $logPath -Value "----- compare_models tail -----"
    Add-Content -Path $logPath -Value $tail
    Add-Content -Path $logPath -Value "-------------------------------"
}

Write-Log "Autopilot tick finished"
