#!/bin/bash
# R3 fallback -- HybridMeshGraphNet: MGN local message passing + long-range airgap-band
# "world" edges (slot-pitch angular skip connections). Tests the locality hypothesis while
# keeping the mesh/curl inductive bias that Transolver discarded (§17). h208/p15 = 9.49M
# matches the h256 MGN budget. Sweep winner BAND_W=30 for torque supervision.
# Smoke measured ~186 s/epoch; batch 1. Usage: bash results/logs/run_hybrid_docker.sh
set -u
cd /workspace/app
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-100}"
HIDDEN="${HIDDEN:-208}"
PROC="${PROC:-15}"
TAG="r3_hybrid"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"

echo "=== R3 HybridMGN  BAND_W=${BAND_W}  epochs=${EPOCHS}  h${HIDDEN}/p${PROC}  +world-edges  batch=1 ==="
python -u train_doe_curl_mgn.py \
  --data-dir backup/doe_data \
  --split eval/splits/doe40_case_split.json \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --model hybrid --hidden-dim "${HIDDEN}" --processor-size "${PROC}" \
  --band-spectral-weight "${BAND_W}" \
  --epochs "${EPOCHS}" --batch-size 1 --step-stride 1 \
  --ckpt "${CKPT}"
echo "TRAIN exit=$?"

python -u -m eval.benchmark \
  --data-dir backup/doe_data --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out "${SCORE}"
echo "SCORE exit=$?"
echo "HYBRID DONE"
