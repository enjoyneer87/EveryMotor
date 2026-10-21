# em_bundle.ps1 -- write C:\work\sync\em_push.bundle holding every commit after -Base.
# Called by scripts/em.sh push on the relay PC, which knows the GitHub tip.
# ASCII-only on purpose (see em_status.ps1). Exit codes: 0 ok, 2 base missing,
# 3 nothing to push, 4 not fast-forward, 5 bundle verify failed.
param([Parameter(Mandatory = $true)][string]$Base,
      [string]$Branch = 'iso/integrate-cli-native-20260901')
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location 'C:\work\integrate\EveryMotor'

git cat-file -e "$Base^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Output "ERROR base $Base is not in the isolated repo: GitHub has commits this PC lacks. Reconcile first."; exit 2 }
$tip = git rev-parse $Branch
if ($tip -eq (git rev-parse $Base)) { Write-Output ('NOTHING tip == base (' + $tip.Substring(0, 7) + ')'); exit 3 }
git merge-base --is-ancestor $Base $Branch
if ($LASTEXITCODE -ne 0) { Write-Output "ERROR ${Base} is not an ancestor of ${Branch} -- the push would not be a fast-forward."; exit 4 }

$out = 'C:\work\sync\em_push.bundle'
if (Test-Path $out) { Remove-Item $out -Force }
git bundle create $out "$Base..$Branch" 2>$null
git bundle verify $out 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Output 'ERROR bundle verify failed'; exit 5 }
Write-Output ('OK ' + $tip + ' commits=' + (git rev-list --count "$Base..$Branch"))
git log --format='   %h %s' "$Base..$Branch"
