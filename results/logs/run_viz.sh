#!/bin/bash
set -u
cd /workspace/app
python -u tools/make_field_gif.py --case 4 --ckpt results/mgn_nodeB_long.pt \
  --out results/viz/case0004_representative.gif --fps 8 \
  --label "node-B element loss, stride 1 / 60 ep  (|B| 10.3%, torque 9.3%)"
python -u tools/make_field_gif.py --case 32 --ckpt results/mgn_nodeB_long.pt \
  --out results/viz/case0032_worst.gif --fps 8 \
  --label "worst holdout case: phase advance 89 deg, near-zero mean torque"
echo "VIZ DONE"
