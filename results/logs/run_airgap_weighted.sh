#!/bin/bash
# Airgap-weighted variant of the winning recipe. Everything except
# --airgap-weight matches results/logs/run_long_nodeB.sh, so the pair is a
# controlled comparison against 13.32% |B| / 9.20% torque.
#
# Weight 5: the airgap band is 6-7% of the scoreable elements but sets the
# acceptance metric (the Arkkio integral runs on exactly those elements).
# Weight 5 lifts its share of the loss to ~26-29% — a fourfold emphasis, short
# of letting a 6% region dictate the model.
set -u
cd /workspace/app

EPOCHS=${EPOCHS:-60}
AIRGAP_W=${AIRGAP_W:-5}

python -u train_doe_curl_mgn.py \
  --data-dir backup/doe_data \
  --split eval/splits/doe40_case_split.json \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --airgap-weight "$AIRGAP_W" \
  --epochs "$EPOCHS" --batch-size 4 --step-stride 1 \
  --ckpt /workspace/app/results/mgn_nodeB_airgap.pt \
  > results/logs/train_nodeB_airgap.log 2>&1
echo "TRAIN exit=$?"

python -u -m eval.benchmark \
  --data-dir backup/doe_data \
  --skip-curl-floor \
  --curl-ckpt results/mgn_nodeB_airgap.pt \
  --out results/benchmark_v2_nodeB_airgap.json \
  > results/logs/score_nodeB_airgap.log 2>&1
echo "SCORING exit=$?"
echo "AIRGAP PIPELINE DONE"
