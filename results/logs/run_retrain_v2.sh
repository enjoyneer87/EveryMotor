#!/bin/bash
# Post-fix retrains: A-prediction (curl) then B-prediction (node round trip).
# Same features, split, sector handling and hyperparameters — only the output
# representation differs, so the pair is a controlled A/B.
set -u
cd /workspace/app

python -u train_doe_curl_mgn.py \
  --data-dir backup/doe_data \
  --split eval/splits/doe40_case_split.json \
  --epochs 30 --batch-size 4 --step-stride 3 \
  --target A --ckpt /workspace/app/results/mgn_curl_v2.pt \
  > results/logs/train_curl_v2.log 2>&1
echo "curl(A) run exit=$?"

python -u train_doe_curl_mgn.py \
  --data-dir backup/doe_data \
  --split eval/splits/doe40_case_split.json \
  --epochs 30 --batch-size 4 --step-stride 3 \
  --target B --ckpt /workspace/app/results/mgn_nodeB_v2.pt \
  > results/logs/train_nodeB_v2.log 2>&1
echo "node-B run exit=$?"
echo "ALL RETRAINS DONE"
