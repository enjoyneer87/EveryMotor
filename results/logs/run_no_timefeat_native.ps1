# Native (Docker-free) twin of run_no_timefeat.sh.
#
# The .sh assumes the PhysicsNeMo container (`cd /workspace/app`, container
# python). This machine cannot run it: the L40S is in TCC driver mode, and
# WSL2 GPU passthrough needs WDDM, so Docker+GPU is unavailable. The venv at
# .venv reproduces the reference scorecard exactly (13.324% |B| / 9.207%
# torque on results/mgn_nodeB_long.pt), so the native path is equivalent.
#
# Flags below are byte-identical to run_no_timefeat.sh — same target, loss
# support, sector settings, stride, epochs and batch — so the pair still
# isolates the time_s shortcut feature (methodology review section 9).
#
# Keep --batch-size 4. Batch 8 fills the card and allocator thrashing made an
# epoch 12x slower; check nvidia-smi before blaming the model.

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot\..\..
$py = ".\.venv\Scripts\python.exe"
$EPOCHS = if ($env:EPOCHS) { $env:EPOCHS } else { 60 }

& $py -u train_doe_curl_mgn.py `
  --data-dir backup/doe_data `
  --split eval/splits/doe40_case_split.json `
  --target B `
  --no-wrap-rotor --no-anti-periodic-edges `
  --epochs $EPOCHS --batch-size 4 --step-stride 1 `
  --ckpt results/mgn_nodeB_notime.pt `
  *> results/logs/train_nodeB_notime.log
Write-Output "TRAIN exit=$LASTEXITCODE"

& $py -u -m eval.benchmark `
  --data-dir backup/doe_data --skip-curl-floor `
  --curl-ckpt results/mgn_nodeB_notime.pt `
  --out results/benchmark_v2_nodeB_notime.json `
  *> results/logs/score_nodeB_notime.log
Write-Output "SCORING exit=$LASTEXITCODE"
Write-Output "NOTIME PIPELINE DONE"
