#!/bin/bash
# Same winning recipe as run_long_nodeB.sh; the only change is that time_s is
# gone from the node features (methodology review section 9). Everything else —
# target, loss support, sector settings, stride, epochs, batch — is identical,
# so the pair isolates the shortcut feature's effect.
set -u
cd /workspace/app
EPOCHS=${EPOCHS:-60}

python -u train_doe_curl_mgn.py \
  --data-dir backup/doe_data \
  --split eval/splits/doe40_case_split.json \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --epochs "$EPOCHS" --batch-size 4 --step-stride 1 \
  --ckpt /workspace/app/results/mgn_nodeB_notime.pt \
  > results/logs/train_nodeB_notime.log 2>&1
echo "TRAIN exit=$?"

python -u -m eval.benchmark \
  --data-dir backup/doe_data --skip-curl-floor \
  --curl-ckpt results/mgn_nodeB_notime.pt \
  --out results/benchmark_v2_nodeB_notime.json \
  > results/logs/score_nodeB_notime.log 2>&1
echo "SCORING exit=$?"
echo "NOTIME PIPELINE DONE"
