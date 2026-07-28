#!/bin/bash
# R4 (c) -- curl representation (target A: predict nodal potential A, derive element B by
# the P1 curl) + band-spectral loss. Untried combination (all spectral/R3 runs were target B).
# Rationale (methodology review R4 design): the curl representation guarantees div B = 0 and
# has a far lower |B| floor (5.31%) than the node round-trip -- headroom the target-B models
# (stuck at ~12.7%) may lack -- while the spectral term is the proven torque lever (bw30 5.22%).
# Sector symmetry (wrap-rotor + anti-periodic edges) is ENABLED: for a potential A the
# anti-periodic sign is correct and helps (the node-B runs disabled it because B is not a
# potential). Gauge + PBC terms auto-activate for target A. RTX 3090: batch 1.
# curl-A is ~2x slower per epoch than node-B -- check epoch 1 and trim EPOCHS for overnight.
# Usage: bash results/logs/run_r4_curl_docker.sh
set -u
cd /workspace/app
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-100}"
HIDDEN="${HIDDEN:-256}"
PROC="${PROC:-15}"
TAG="${TAG:-r4_curl_spectral}"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"

echo "=== R4 curl(target A) + spectral  BAND_W=${BAND_W}  epochs=${EPOCHS}  h${HIDDEN}/p${PROC}  batch=1  (wrap+anti-periodic ON) ==="
python -u train_doe_curl_mgn.py \
  --data-dir backup/doe_data \
  --split eval/splits/doe40_case_split.json \
  --target A \
  --model mgn --hidden-dim "${HIDDEN}" --processor-size "${PROC}" \
  --band-spectral-weight "${BAND_W}" \
  --epochs "${EPOCHS}" --batch-size 1 --step-stride 1 \
  --ckpt "${CKPT}"
echo "TRAIN exit=$?"

python -u -m eval.benchmark \
  --data-dir backup/doe_data --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out "${SCORE}"
echo "SCORE exit=$?"
echo "R4 CURL DONE"
