# Accuracy push, experiment 3 — band spectral loss.
#
# LAUNCH AFTER the experiment-2 (proc24) verdict, not alongside it.
#
# Rationale (methodology review §14): the band projection study measured that
# the torque error lives entirely in the coefficients of the admissible band
# harmonics — filtering removed nothing — so this run supervises those
# coefficients directly (--band-spectral-weight). This is NOT the failed
# airgap-weight approach: raw region weighting re-emphasised the same per-element
# errors; the spectral term adds a new supervision signal (the quantity torque
# is actually made of).
#
# Config knobs come from the capacity experiments:
#   HIDDEN defaults to 256 (exp1: width helped torque; params 9.3M)
#   PROC   defaults to 15  — RAISE TO 24 IF exp2 wins before launching this
#   BAND_W defaults to 1.0 (coefficient loss is normalized; 1.0 puts it on the
#          same footing as the field term; sweep 0.3/3 if the first run is
#          inconclusive)

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot\..\..
$py = ".\.venv\Scripts\python.exe"
$EPOCHS = if ($env:EPOCHS) { $env:EPOCHS } else { 100 }
$HIDDEN = if ($env:HIDDEN) { $env:HIDDEN } else { 256 }
$PROC   = if ($env:PROC)   { $env:PROC }   else { 15 }
$BAND_W = if ($env:BAND_W) { $env:BAND_W } else { 1.0 }

& $py -u train_doe_curl_mgn.py `
  --data-dir backup/doe_data `
  --split eval/splits/doe40_case_split.json `
  --target B `
  --no-wrap-rotor --no-anti-periodic-edges `
  --hidden-dim $HIDDEN --processor-size $PROC `
  --band-spectral-weight $BAND_W `
  --epochs $EPOCHS --batch-size 2 --step-stride 1 `
  --ckpt results/mgn_nodeB_spectral.pt `
  *> results/logs/train_spectral.log
Write-Output "TRAIN exit=$LASTEXITCODE"

& $py -u -m eval.benchmark `
  --data-dir backup/doe_data --skip-curl-floor --skip-grid-floor `
  --curl-ckpt results/mgn_nodeB_spectral.pt `
  --out results/benchmark_v2_nodeB_spectral.json `
  *> results/logs/score_spectral.log
Write-Output "SCORING exit=$LASTEXITCODE"
Write-Output "SPECTRAL PIPELINE DONE"
