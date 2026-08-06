#!/bin/bash
# Functional smoke for --resume, on a deliberately tiny configuration (doe40, h32/p2,
# stride 20) so the whole thing runs in minutes rather than the 1.7 h the real R7-A
# resume costs. Sequence:
#   1. train 4 epochs clean            -> reference history
#   2. truncate a copy of the ckpt to epoch 2 (simulates the reboot)
#   3. resume from it with identical flags -> must continue 3..4
#   4. assert the resumed run ends at epoch 4 with a 4-entry history
# Also asserts the drift guard fires when a run-defining flag is changed.
set -eu
cd /workspace/app
COMMON="--data-dir backup/doe_data --split eval/splits/doe40_case_split.json --target B
        --no-wrap-rotor --no-anti-periodic-edges --model mgn --hidden-dim 32
        --processor-size 2 --band-spectral-weight 30 --epochs 4 --batch-size 1
        --step-stride 20"

echo "=== 1) clean 4-epoch reference ==="
python -u train_doe_curl_mgn.py $COMMON --ckpt results/_resume_smoke.pt --ckpt-every 2

echo "=== 2) truncate a copy to epoch 2 ==="
python - <<'EOF'
import torch
ck = torch.load("results/_resume_smoke.pt", map_location="cpu", weights_only=False)
ck["epoch"] = 2
ck["train_hist"] = ck["train_hist"][:2]
ck["val_hist"] = ck["val_hist"][:2]
torch.save(ck, "results/_resume_smoke_ep2.pt")
print(f"truncated copy at epoch {ck['epoch']}, hist {len(ck['val_hist'])}")
EOF

echo "=== 3) resume 3..4 ==="
python -u train_doe_curl_mgn.py $COMMON --ckpt results/_resume_smoke_resumed.pt \
    --resume results/_resume_smoke_ep2.pt --ckpt-every 2

echo "=== 4) assert ==="
python - <<'EOF'
import sys, torch
ck = torch.load("results/_resume_smoke_resumed.pt", map_location="cpu", weights_only=False)
ok = ck["epoch"] <= 4 and len(ck["train_hist"]) == len(ck["val_hist"])
print(f"resumed ckpt: epoch={ck['epoch']} train_hist={len(ck['train_hist'])} "
      f"val_hist={len(ck['val_hist'])}")
last = torch.load("results/_resume_smoke_resumed.pt.last", map_location="cpu", weights_only=False)
print(f"periodic .last: epoch={last['epoch']} hist={len(last['val_hist'])}")
ok &= last["epoch"] == 4 and len(last["val_hist"]) == 4      # history carried, not restarted
sys.exit(0 if ok else 1)
EOF
echo "ASSERT exit=$?"

echo "=== 5) drift guard must REFUSE a changed run-defining flag ==="
if python -u train_doe_curl_mgn.py $COMMON --hidden-dim 64 \
        --ckpt results/_resume_smoke_bad.pt --resume results/_resume_smoke_ep2.pt; then
  echo "RESUME_SMOKE_FAIL -- drift guard did not fire"; exit 1
else
  echo "drift guard fired as expected"
fi

echo "RESUME_SMOKE_PASS"
