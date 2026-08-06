#!/bin/bash
# R7-A epochs 48-50: finish the run the 2026-08-06 14:49 KST host reboot cut short.
#
# Flags are identical to run_r7a_docker.sh -- the trainer's --resume guard aborts if any
# run-defining flag differs from the checkpoint, so this cannot silently become a
# different experiment. The resume SOURCE is the preserved epoch-47 copy, while --ckpt
# writes the canonical name, so the ep47 artifact that section 27's interim verdict was
# scored on survives whatever happens here.
#
# --ckpt-every 1: only three epochs remain and each costs ~29 min, so write the
# crash-insurance checkpoint every epoch.
set -u
cd /workspace/app
CKPT="results/mgn_nodeB_doe240_bw30_prior.pt"
RESUME="results/mgn_nodeB_doe240_bw30_prior_ep47.pt"
SPLIT="eval/splits/doe240_case_split.json"
DATA="backup/doe_data_240"

echo "=== R7-A resume: epochs 48-50 of 50 (from ${RESUME}) ==="
python -u train_doe_curl_mgn.py \
  --data-dir "${DATA}" \
  --split "${SPLIT}" \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --prior-features \
  --model mgn --hidden-dim 256 --processor-size 15 \
  --band-spectral-weight 30 \
  --epochs 50 --batch-size 1 --step-stride 1 \
  --ckpt "${CKPT}" --resume "${RESUME}" --ckpt-every 1
echo "TRAIN exit=$?"

echo "=== R7-A final scoring: legacy fixed 6-test gate ==="
python -u -m eval.benchmark \
  --data-dir "${DATA}" \
  --split "${SPLIT}" --subset test \
  --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out results/benchmark_v2_nodeB_doe240_bw30_prior_ep50.json
echo "SCORE exit=$?"
echo "R7A RESUME DONE"
