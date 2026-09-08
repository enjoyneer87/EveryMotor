#!/bin/bash
cd /workspace/app
COMMON="--input-format doe --data-dir backup/doe_data_120 --seed 42"
echo "=== A. step-aware batch-4 quick test $(date -Is) ==="
python -m phase1_static.train $COMMON --case-indices 0 1 2 3 --max-steps-per-case 4 \
  --epochs 1 --batch-size 4 --step-aware-batching > results/_sa_test.log 2>&1
RC=$?; echo "STEPAWARE_TEST_RC=$RC"
grep -E "epoch=|Error|error" results/_sa_test.log | tail -2
if [ $RC -eq 0 ]; then BATCH="--batch-size 4 --step-aware-batching"; else BATCH="--batch-size 1"; fi
echo "BATCHING=$BATCH"
echo "=== B. full 1-epoch timing $(date -Is) ==="
SECONDS=0
python -m phase1_static.train $COMMON --epochs 1 $BATCH \
  --ckpt-out results/_pkg_ep1.pt --log-file results/_pkg_ep1.log > results/_pkg_ep1.container.log 2>&1
echo "PKG_EP1_RC=$? WALL_S=$SECONDS"
grep -E "Loaded|epoch=|checkpoint saved|Error|error" results/_pkg_ep1.container.log | tail -5
echo "=== done $(date -Is) ==="