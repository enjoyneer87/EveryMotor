#!/bin/bash
# R7-A verdict scoring on the checkpoint that survived the 2026-08-06 14:49 KST host
# reboot. The run died at ~ep49/50; the best-val checkpoint on disk is epoch 47/50
# (ck["epoch"]==47, len(val_hist)==47, best val B 0.021275 / nRMSE 17.416%).
#
# Flags are byte-identical to the scoring block of run_r7a_docker.sh -- same
# legacy fixed 6-test gate (cases 4, 7, 18, 32, 37, 39) the champion was scored on.
# Only --out differs, tagged _ep47 so the partial-epoch provenance cannot be lost.
set -u
cd /workspace/app
CKPT="results/mgn_nodeB_doe240_bw30_prior.pt"
python -u -m eval.benchmark \
  --data-dir backup/doe_data_240 \
  --split eval/splits/doe240_case_split.json --subset test \
  --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out results/benchmark_v2_nodeB_doe240_bw30_prior_ep47.json
echo "SCORE exit=$?"
echo "R7A EP47 SCORING DONE"
