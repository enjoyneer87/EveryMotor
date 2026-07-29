#!/bin/bash
# R4(a) retrain on the 120-case DOE (40 original + 80 new geometries). Same winning recipe
# as spectral bw30 (target B, h256/p15, BAND_W=30), only the training-geometry count grows
# 30->110. Test/val held IDENTICAL to doe40 (cases 4,7,18,32,37,39 / 5,16,25,28) so |B|
# generalization is compared on the same geometries. 110 train cases * 45 = ~4950 graphs
# (3.67x the 40-case 1350) -> ~690 s/epoch; 50 epochs = ~1.8x the 40-case's total gradient
# steps, comfortably converged and overnight-scale. batch 1 (h256/p15 batch 2 OOMs on 24 GB).
# Usage: bash results/logs/run_doe120_docker.sh
set -u
cd /workspace/app
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-50}"
TAG="${TAG:-doe120_bw30}"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"
DATA="backup/doe_data_120"
SPLIT="eval/splits/doe120_case_split.json"

echo "=== R4(a) DOE120 retrain  BAND_W=${BAND_W}  epochs=${EPOCHS}  h256/p15  batch=1  data=${DATA} ==="
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
echo "DOE120 DONE"
