"""Read R7-A checkpoint metadata after the 2026-08-06 host reboot killed the run.

Answers: which epoch did the surviving best-val checkpoint come from, and does it
carry optimizer state (i.e. is a resume-to-50 possible instead of a 24 h retrain)?
"""
import torch, json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/mgn_nodeB_doe240_bw30_prior.pt"
ck = torch.load(path, map_location="cpu", weights_only=False)

out = {"path": path, "top_keys": sorted(ck.keys()) if isinstance(ck, dict) else type(ck).__name__}
if isinstance(ck, dict):
    for k in ("epoch", "best_val", "best", "val_loss", "b_std", "a_scale", "args", "node_feat_dim"):
        if k in ck:
            v = ck[k]
            out[k] = v if not hasattr(v, "shape") else float(v)
    for k in ("train_hist", "val_hist", "hist"):
        if k in ck and hasattr(ck[k], "__len__"):
            out[f"len({k})"] = len(ck[k])
            out[f"{k}_tail"] = list(ck[k])[-6:]
    opt = ck.get("optimizer_state_dict")
    out["has_optimizer_state"] = opt is not None
    if isinstance(opt, dict) and "state" in opt:
        out["optimizer_entries"] = len(opt["state"])
    sd = ck.get("model_state_dict") or ck.get("state_dict")
    if isinstance(sd, dict):
        out["n_tensors"] = len(sd)
        # first encoder weight shape reveals the node-feature width (V2=9 vs V3=12)
        for k, v in sd.items():
            if hasattr(v, "shape") and v.dim() == 2:
                out["first_2d"] = [k, list(v.shape)]
                break
print(json.dumps(out, indent=2, default=str))
