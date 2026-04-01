import torch
import sys

for ckpt_path in ["/workspace/host_data/doe_fno_ckpt.pt",
                  "/workspace/host_data/doe_meshgraphnet_ckpt.pt"]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        name = ckpt.get("model_name", "MGN" if "meshgraphnet" in ckpt_path else "?")
        ep = ckpt.get("epoch", "?")
        train_hist = ckpt.get("train_hist", [])
        val_hist = ckpt.get("val_hist", [])
        print(f"\n=== {name} ({ckpt_path}) ===")
        print(f"  Epoch: {ep}")
        if train_hist:
            print(f"  Final train MSE: {train_hist[-1]:.6f}")
        if val_hist:
            print(f"  Final val MSE:   {val_hist[-1]:.6f}")
            print(f"  Best val MSE:    {min(val_hist):.6f}")
        args = ckpt.get("args", {})
        if args:
            print(f"  Args: {args}")
    except Exception as e:
        print(f"  {ckpt_path}: {e}")
