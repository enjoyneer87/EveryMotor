#!/usr/bin/env python3
"""Quick GINO debug: check single forward pass for NaN."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/host_data")
from doe_data_utils import load_doe_data, build_gino_sample

# Load 1 record
records, conditions = load_doe_data("/workspace/host_data/doe_data", max_steps=2)
print(f"Records: {len(records)}")

s = build_gino_sample(records[0], conditions[0], latent_res=32)
print(f"input_geom: {s['input_geom'].shape}, range [{s['input_geom'].min():.3f}, {s['input_geom'].max():.3f}]")
print(f"features: {s['features'].shape}, range [{s['features'].min():.3f}, {s['features'].max():.3f}]")
print(f"target: {s['target'].shape}, range [{s['target'].min():.3f}, {s['target'].max():.3f}]")
print(f"latent_queries: {s['latent_queries'].shape}")

# Check for NaN/Inf in input
print(f"\ninput_geom NaN: {np.isnan(s['input_geom']).any()}, Inf: {np.isinf(s['input_geom']).any()}")
print(f"features NaN: {np.isnan(s['features']).any()}, Inf: {np.isinf(s['features']).any()}")
print(f"target NaN: {np.isnan(s['target']).any()}, Inf: {np.isinf(s['target']).any()}")

# Normalize like training
feat_mean = s['features'].mean(axis=0, keepdims=True)
feat_std = np.clip(s['features'].std(axis=0, keepdims=True), 1e-6, None)
tgt_mean = s['target'].mean(axis=0, keepdims=True)
tgt_std = np.clip(s['target'].std(axis=0, keepdims=True), 1e-6, None)

feat_norm = (s['features'] - feat_mean) / feat_std
tgt_norm = (s['target'] - tgt_mean) / tgt_std
print(f"\nNormalized features range: [{feat_norm.min():.3f}, {feat_norm.max():.3f}]")
print(f"Normalized target range: [{tgt_norm.min():.3f}, {tgt_norm.max():.3f}]")
print(f"feat_std: {feat_std.flatten()}")

# Try GINO forward
from neuralop.models import GINO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GINO(
    in_channels=9, out_channels=2, gno_coord_dim=2,
    in_gno_radius=0.1, out_gno_radius=0.1,
    fno_in_channels=9,
    fno_n_modes=(16, 16), fno_hidden_channels=64, fno_n_layers=4,
    fno_lifting_channel_ratio=2, projection_channel_ratio=4,
    in_gno_transform_type="linear", out_gno_transform_type="linear",
    in_gno_pos_embed_type="transformer", out_gno_pos_embed_type="transformer",
    gno_embed_channels=16,
    in_gno_channel_mlp_hidden_layers=[80, 80],
    out_gno_channel_mlp_hidden_layers=[256, 128],
    gno_channel_mlp_non_linearity=F.gelu,
    gno_use_open3d=False,
    gno_use_torch_scatter=True,
    fno_use_channel_mlp=True, fno_channel_mlp_expansion=0.5,
    fno_non_linearity=F.gelu, fno_norm=None, fno_skip="linear",
).to(device)

ig = torch.from_numpy(feat_norm).unsqueeze(0).to(device)  # wrong - this is features not geom
input_geom_t = torch.from_numpy(s['input_geom']).unsqueeze(0).to(device)
latent_q = torch.from_numpy(s['latent_queries']).unsqueeze(0).to(device)
output_q = torch.from_numpy(s['input_geom']).unsqueeze(0).to(device)
x_feat = torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0).to(device)
y_true = torch.from_numpy(tgt_norm.astype(np.float32)).unsqueeze(0).to(device)

print(f"\nForward pass shapes:")
print(f"  input_geom: {input_geom_t.shape}")
print(f"  latent_queries: {latent_q.shape}")
print(f"  output_queries: {output_q.shape}")
print(f"  x_feat: {x_feat.shape}")

with torch.no_grad():
    try:
        pred = model(input_geom=input_geom_t, latent_queries=latent_q,
                     output_queries=output_q, x=x_feat)
        print(f"  pred shape: {pred.shape}")
        print(f"  pred range: [{pred.min().item():.6f}, {pred.max().item():.6f}]")
        print(f"  pred has NaN: {torch.isnan(pred).any().item()}")
        print(f"  pred has Inf: {torch.isinf(pred).any().item()}")
        loss = F.mse_loss(pred.squeeze(0), y_true.squeeze(0))
        print(f"  loss: {loss.item()}")
        print(f"  loss is NaN: {torch.isnan(loss).item()}")
    except Exception as e:
        print(f"  ERROR: {e}")
