# Accuracy push, experiment 2 — receptive-field (depth) probe.
#
# Experiment 1 (hidden 128->256) confirmed capacity helps but the absolute gain
# on the training floor was small (train nRMSE 13.9% -> ~13.7%). The stronger
# untested hypothesis is receptive field: magnetostatics is globally coupled and
# the dominant residual is now torque ripple (methodology_review §7 결론3), both
# of which point at too few message-passing steps rather than too little width.
#
# Isolates processor_size against experiment 1 — SAME width (256) and SAME batch
# (2), only depth changes:
#   processor_size 15 -> 24   (+60% message-passing depth; the receptive-field lever)
#   hidden_dim     256        (held at exp1)
#   batch_size     2          (held at exp1)
#   epochs         100        (cosine sized to the run)
#
# Memory: exp1 (hidden 256, proc 15, batch 2) measured 27.9 GB; the processor
# dominates, so proc x1.6 lands ~35-40 GB, under the 46 GB card. WATCH nvidia-smi
# on epoch 1 — if it approaches ~42 GB (allocator-thrash territory, MIGRATION.md),
# kill and drop batch to 1.
#
# Verdict this run settles: if depth breaks the ~13% train floor toward 5%,
# capacity/receptive-field is the lever and we scale further. If it plateaus like
# exp1, the bottleneck is architectural -> R3 graph transformer (Transolver/GNOT).

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot\..\..
$py = ".\.venv\Scripts\python.exe"
$EPOCHS = if ($env:EPOCHS) { $env:EPOCHS } else { 100 }

& $py -u train_doe_curl_mgn.py `
  --data-dir backup/doe_data `
  --split eval/splits/doe40_case_split.json `
  --target B `
  --no-wrap-rotor --no-anti-periodic-edges `
  --hidden-dim 256 --processor-size 24 `
  --epochs $EPOCHS --batch-size 2 --step-stride 1 `
  --ckpt results/mgn_nodeB_proc24.pt `
  *> results/logs/train_cap_proc24.log
Write-Output "TRAIN exit=$LASTEXITCODE"

& $py -u -m eval.benchmark `
  --data-dir backup/doe_data --skip-curl-floor --skip-grid-floor `
  --curl-ckpt results/mgn_nodeB_proc24.pt `
  --out results/benchmark_v2_nodeB_proc24.json `
  *> results/logs/score_cap_proc24.log
Write-Output "SCORING exit=$LASTEXITCODE"
Write-Output "CAP PROC24 PIPELINE DONE"
