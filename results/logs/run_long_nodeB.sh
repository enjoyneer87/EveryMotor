#!/bin/bash
# Long node-B run: element-support loss, sector fixes off, all 45 rotor steps.
#
# Sector handling is disabled deliberately. The controlled A/B (curl pre vs post)
# showed the fixes cost torque accuracy while the anti-periodic sign is only an
# MLP input feature rather than a multiplicative constraint — see the plan doc
# section 7, conclusion 2. Re-enable them together with custom_mgn's signed
# message passing, not before.
set -u
cd /workspace/app

# 60 epochs first, not 150: stride 1 triples the data, so this run alone
# separates "data-limited" from "convergence-limited". The cosine schedule is
# sized to the epoch count, so a 60-epoch run anneals properly — stopping a
# 150-epoch run at 60 would leave the LR mid-decay and understate the result.
EPOCHS=${EPOCHS:-60}

python -u train_doe_curl_mgn.py \
  --data-dir backup/doe_data \
  --split eval/splits/doe40_case_split.json \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --epochs "$EPOCHS" --batch-size 4 --step-stride 1 \
  --ckpt /workspace/app/results/mgn_nodeB_long.pt \
  > results/logs/train_nodeB_long.log 2>&1
echo "LONG RUN exit=$?"

python -u -m eval.benchmark \
  --data-dir backup/doe_data \
  --skip-curl-floor \
  --curl-ckpt results/mgn_nodeB_long.pt \
  --out results/benchmark_v2_nodeB_long.json \
  > results/logs/score_nodeB_long.log 2>&1
echo "SCORING exit=$?"
echo "LONG PIPELINE DONE"
