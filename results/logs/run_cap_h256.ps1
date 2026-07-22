# Accuracy push, experiment 1 — capacity (width) probe.
#
# Diagnosis (2026-07-22): the no-time_s model scores |B| 10.6% / torque 4.6% on
# the TRAIN split and 13.0% / 8.2% on TEST (same eval.benchmark metric). It cannot
# fit training data below 10.6% |B| — well above the 5% G2 target and far above the
# 1.6% curl representation floor. The binding constraint is underfit (capacity), not
# data. This run tests whether doubling width moves the training ceiling.
#
# Only two knobs change from the winning no-time_s recipe:
#   hidden_dim 128 -> 256   (2x width; the capacity lever)
#   batch_size  4  -> 2     (forced: 2x width would OOM the 46 GB card at batch 4)
#   epochs      60 -> 100   (cosine sized to the longer run so it anneals fully)
# processor_size stays 15 so width is isolated. If width alone does not move the
# train ceiling, experiment 2 raises processor_size (receptive field) instead.
#
# Watch nvidia-smi on epoch 1: if it approaches 46 GB, kill and drop to hidden 192.

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot\..\..
$py = ".\.venv\Scripts\python.exe"
$EPOCHS = if ($env:EPOCHS) { $env:EPOCHS } else { 100 }

& $py -u train_doe_curl_mgn.py `
  --data-dir backup/doe_data `
  --split eval/splits/doe40_case_split.json `
  --target B `
  --no-wrap-rotor --no-anti-periodic-edges `
  --hidden-dim 256 --processor-size 15 `
  --epochs $EPOCHS --batch-size 2 --step-stride 1 `
  --ckpt results/mgn_nodeB_h256.pt `
  *> results/logs/train_cap_h256.log
Write-Output "TRAIN exit=$LASTEXITCODE"

& $py -u -m eval.benchmark `
  --data-dir backup/doe_data --skip-curl-floor --skip-grid-floor `
  --curl-ckpt results/mgn_nodeB_h256.pt `
  --out results/benchmark_v2_nodeB_h256.json `
  *> results/logs/score_cap_h256.log
Write-Output "SCORING exit=$LASTEXITCODE"
Write-Output "CAP H256 PIPELINE DONE"
