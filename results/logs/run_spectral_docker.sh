#!/bin/bash
# Experiment 3 — band spectral loss, ONE BAND_W value, in the physicsnemo container.
# RTX 3090 (24 GB): batch 1 is forced — h256/p15/batch2 measured 27.9 GB on the L40S.
# Usage (inside container):  bash results/logs/run_spectral_docker.sh <BAND_W>
set -u
cd /workspace/app
BAND_W="${1:?need BAND_W}"
EPOCHS="${EPOCHS:-100}"
TAG="bw${BAND_W}"
CKPT="results/mgn_nodeB_spectral_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_spectral_${TAG}.json"

echo "=== spectral BAND_W=${BAND_W}  epochs=${EPOCHS}  batch=1  h256/p15 ==="
python -u train_doe_curl_mgn.py \
  --data-dir backup/doe_data \
  --split eval/splits/doe40_case_split.json \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --hidden-dim 256 --processor-size 15 \
  --band-spectral-weight "${BAND_W}" \
  --epochs "${EPOCHS}" --batch-size 1 --step-stride 1 \
  --ckpt "${CKPT}"
echo "TRAIN exit=$?"

python -u -m eval.benchmark \
  --data-dir backup/doe_data --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out "${SCORE}"
echo "SCORE exit=$?"
echo "SPECTRAL bw=${BAND_W} DONE"
