# em_status.ps1 -- one-screen status of the EveryMotor track on the isolated PC.
# ASCII-only on purpose: Windows PowerShell 5.1 reads BOM-less .ps1 files in the ANSI
# codepage (cp949 here), so any non-ASCII literal in this file would be corrupted.
# Usage from the relay PC:  bash scripts/em.sh status
param([int]$EpochLines = 2, [int]$Commits = 4)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
Set-Location 'C:\work\integrate\EveryMotor'

Write-Output ('time   ' + (Get-Date -Format 'yyyy-MM-dd HH:mm'))
Write-Output ('git    ' + (git rev-parse --abbrev-ref HEAD) + ' @ ' + (git log -1 --format='%h %ad %s' --date=short))
Write-Output ('tree   ' + @(git status --porcelain).Count + ' untracked/modified')
git log -$Commits --skip=1 --format='       %h %ad %s' --date=short
Write-Output ('gpu    ' + (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null))

$srv = docker version --format '{{.Server.Version}}' 2>$null
if (-not $srv) {
  Write-Output 'docker DOWN -- start Docker Desktop in the moa RDP session (an engine started over ssh dies with the ssh session)'
} else {
  Write-Output ('docker up ' + $srv)
  $running = @(docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}')
  if ($running.Count -eq 0) { Write-Output '       no running containers' }
  foreach ($r in $running) {
    $p = $r.Split('|')
    Write-Output ('  run  ' + $p[0] + '  (' + $p[2] + ')')
    if ($p[1] -like '*physicsnemo*') {
      docker logs $p[0] 2>&1 |
        Select-String -Pattern '^\s+ep \d{4}/|Resumed|TRAIN exit|SCORE |DONE|FAILED|Traceback' |
        Select-Object -Last $EpochLines |
        ForEach-Object { $t = $_.Line.Trim(); '       ' + $t.Substring(0, [Math]::Min(150, $t.Length)) }
    }
  }
}
python tools/ops/em_baton.py
