#!/bin/bash
cd /workspace/app
DATA=backup/doe_data_120
SPLIT=eval/splits/doe120_case_split.json
TRAIN=$(python -c "import json;s=json.load(open('$SPLIT'));print(' '.join(map(str,s['train'])))")
TEST=$(python -c "import json;s=json.load(open('$SPLIT'));print(' '.join(map(str,s['test'])))")
echo "train cases: $(echo $TRAIN | wc -w)   test cases: $TEST"
# --weight-current 0: the Je target is identically zero in this DOE (no fields/je
# in the Motor-CAD exports), so the channel is removed from the loss in BOTH arms.
COMMON="--input-format doe --data-dir $DATA --case-indices $TRAIN --epochs 50 --batch-size 1 --seed 42 --ckpt-interval 10 --weight-current 0"

evaluate () {
  for k in $TEST; do
    python infer_phase1_pbc.py --ckpt "$2" --data-dir $DATA --case-idx $k \
      --out results/gate_$1_case$k.npz > results/gate_$1_infer_case$k.log 2>&1 || echo "INFER_FAIL arm=$1 case=$k"
  done
  python _gate_aggregate.py --arms $1 --cases $TEST --out results/gate_$1_summary.json
}

echo "=== 1. arm A  (control: --target-normalization none --current-focus-alpha 0.0)  $(date -Is) ==="
SECONDS=0
python -m phase1_static.train $COMMON --target-normalization none --current-focus-alpha 0.0 \
  --ckpt-out results/gate_A.pt --log-file results/gate_A.log > results/gate_A.container.log 2>&1
echo "ARM_A_RC=$? WALL_S=$SECONDS"
grep -E "epoch=50/50|checkpoint saved" results/gate_A.container.log | tail -2
echo "=== 1e. arm A eval  $(date -Is) ==="
evaluate A results/gate_A.pt

echo "=== 2. arm B  (treatment: --target-normalization zscore --current-focus-alpha 2.0 [inert: Je target is zero])  $(date -Is) ==="
SECONDS=0
python -m phase1_static.train $COMMON --target-normalization zscore --current-focus-alpha 2.0 --current-focus-gamma 1.0 \
  --ckpt-out results/gate_B.pt --log-file results/gate_B.log > results/gate_B.container.log 2>&1
echo "ARM_B_RC=$? WALL_S=$SECONDS"
grep -E "epoch=50/50|checkpoint saved" results/gate_B.container.log | tail -2
echo "=== 2e. arm B eval  $(date -Is) ==="
evaluate B results/gate_B.pt

echo "=== 3. combined  $(date -Is) ==="
python _gate_aggregate.py --arms A B --cases $TEST --out results/gate_summary.json
echo "=== GATE CHAIN DONE  $(date -Is) ==="