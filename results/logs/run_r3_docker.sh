#!/bin/bash
# R3 -- Transolver (physics-attention graph transformer), in the physicsnemo container.
# Global receptive field to attack the |B| floor the capacity + spectral axes could not
# move (methodology review §16). Sweep winner BAND_W=30 carried for torque supervision.
# RTX 3090 (24 GB): batch 1. Smoke measured ~93 s/epoch, 4.2 GB (Transolver is lighter
# than the h256 MGN). use_te=False is set inside the adapter for the determinism contract.
# Usage (inside container):  bash results/logs/run_r3_docker.sh
set -u
cd /workspace/app
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-100}"
NLAYERS="${NLAYERS:-12}"
SLICE="${SLICE:-32}"
TAG="${TAG:-r3_transolver}"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"

echo "=== R3 Transolver  BAND_W=${BAND_W}  epochs=${EPOCHS}  n_layers=${NLAYERS}  h256/slice${SLICE}  batch=1 ==="
python -u train_doe_curl_mgn.py \
  --data-dir backup/doe_data \
  --split eval/splits/doe40_case_split.json \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --model transolver --hidden-dim 256 --n-layers "${NLAYERS}" --n-head 8 --slice-num "${SLICE}" --emb-dim 64 \
  --band-spectral-weight "${BAND_W}" \
  --epochs "${EPOCHS}" --batch-size 1 --step-stride 1 \
  --ckpt "${CKPT}"
echo "TRAIN exit=$?"

python -u -m eval.benchmark \
  --data-dir backup/doe_data --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out "${SCORE}"
echo "SCORE exit=$?"
echo "R3 DONE"
