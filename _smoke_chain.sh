#!/bin/bash
cd /workspace/app
echo "=== resume smoke start $(date -Is) ==="
tr -d '\r' < results/logs/_resume_smoke.sh > /tmp/rs.sh
SECONDS=0
bash /tmp/rs.sh > results/_resume_smoke.container.log 2>&1
echo "RESUME_SMOKE_RC=$? WALL_S=$SECONDS"
tail -3 results/_resume_smoke.container.log
echo "=== pkg full 1-epoch timing start $(date -Is) ==="
SECONDS=0
python -m phase1_static.train --input-format doe --data-dir backup/doe_data_120 \
  --epochs 1 --seed 42 --ckpt-out results/_pkg_ep1.pt --log-file results/_pkg_ep1.log \
  > results/_pkg_ep1.container.log 2>&1
echo "PKG_EP1_RC=$? WALL_S=$SECONDS"
grep -E "Loaded|epoch=|normalization" results/_pkg_ep1.container.log | tail -4
echo "=== chain done $(date -Is) ==="