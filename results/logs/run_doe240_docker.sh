#!/bin/bash
# R5 retrain on the 240-case DOE (R4a 120 + 120 new geometries). Same bw30 recipe, test/val
# identical to doe40. 230 train cases * 45 = ~10350 graphs (2.09x the 120-case 4950) -> longer
# epochs; ~25 epochs gives gradient-step parity with the 120-model's 50 epochs (both ~250k
# steps, and the 120 best was ep39 so ~25 converges the 2x-data set). Check epoch 1 and adjust
# EPOCHS. batch 1. Usage: bash results/logs/run_doe240_docker.sh
set -u
cd /workspace/app
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-25}"
TAG="${TAG:-doe240_bw30}"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"
DATA="backup/doe_data_240"
SPLIT="eval/splits/doe240_case_split.json"

echo "=== R5 DOE240 retrain  BAND_W=${BAND_W}  epochs=${EPOCHS}  h256/p15  batch=1  data=${DATA} ==="
python -u train_doe_curl_mgn.py \
  --data-dir "${DATA}" \
  --split "${SPLIT}" \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --model mgn --hidden-dim 256 --processor-size 15 \
  --band-spectral-weight "${BAND_W}" \
  --epochs "${EPOCHS}" --batch-size 1 --step-stride 1 \
  --ckpt "${CKPT}"
echo "TRAIN exit=$?"

python -u -m eval.benchmark \
  --data-dir "${DATA}" --split "${SPLIT}" --subset test \
  --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out "${SCORE}"
echo "SCORE exit=$?"
echo "DOE240 DONE"
