#!/bin/bash
# Native twin of run_doe120_docker.sh -- same recipe, no container.
#
# DRAFT, UNTESTED on the native box. Differences from the docker twin, and
# nothing else: the hardcoded `cd /workspace/app` becomes the repo root
# resolved from this script's location, and DATA/SPLIT accept environment
# overrides so a native host can point at its own data mounts. The training
# arguments are byte-identical to the docker script -- if you change one,
# change both, or better, factor the argument list before adding a third.
#
# Dual-runtime rule (docs: HANDOFF review 2026-09-01): before trusting native
# numbers, run the cross-runtime parity smoke -- same seed, small case, docker
# vs native, explicit tolerance.
# Usage: bash results/logs/run_doe120_native.sh
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-50}"
TAG="${TAG:-doe120_bw30_native}"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"
DATA="${DATA:-backup/doe_data_120}"
SPLIT="${SPLIT:-eval/splits/doe120_case_split.json}"

echo "=== DOE120 retrain (native)  BAND_W=${BAND_W}  epochs=${EPOCHS}  h256/p15  batch=1  data=${DATA} ==="
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
echo "DOE120 NATIVE DONE"
