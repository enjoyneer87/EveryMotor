#!/usr/bin/env python3
"""RNN simple evaluation"""
import torch
import sys
sys.path.insert(0, '/workspace/host_data')
sys.path.insert(0, '/workspace/host_data/physicsnemo')

from doe_data_utils import load_doe_data
from physicsnemo.models.rnn.rnn_seq2seq import Seq2SeqRNN
import numpy as np

print("[1] Loading data...")
train_list, test_list = load_doe_data('/workspace/host_data/doe_data')
print(f"  train samples: {len(train_list)}")
print(f"  test samples: {len(test_list)}")

print("\n[2] Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  device: {device}")

# Load RNN model
ckpt = torch.load('/workspace/host_data/doe_rnn_ckpt.pt', map_location=device, weights_only=False)
rnn_model = Seq2SeqRNN(
    input_channels=7,
    dimension=2,
    nr_latent_channels=64,
    nr_tsteps=4
).to(device)
rnn_model.load_state_dict(ckpt['model_state_dict'], strict=False)
rnn_model.eval()
print("  OK: RNN loaded")

x_mean, x_std = ckpt['inp_mean'].to(device), ckpt['inp_std'].to(device)
y_mean, y_std = ckpt['tgt_mean'].to(device), ckpt['tgt_std'].to(device)

print("\n[3] Running predictions (first 5 test samples)...")

channel_names = ['Bx', 'By', 'A', 'J']
errors = {ch: [] for ch in channel_names}

n_test = min(5, len(test_list))
with torch.no_grad():
    for i in range(n_test):
        sample = test_list[i]
        x_tensor = torch.from_numpy(sample['x']).float().unsqueeze(0).to(device)  # (1, 7, H, W)
        y_truth = torch.from_numpy(sample['y']).float().to(device)  # (4, H, W)
        
        # Predict
        y_pred_norm = rnn_model(x_tensor)  # (1, 4, H, W)
        y_pred = y_pred_norm[0] * (y_std.view(4, 1, 1) + 1e-8) + y_mean.view(4, 1, 1)
        
        # Calculate MAE per channel
        for ch, ch_name in enumerate(channel_names):
            mae = torch.abs(y_pred[ch] - y_truth[ch]).mean().item()
            errors[ch_name].append(mae)
            print(f"  [test {i}] {ch_name}: MAE = {mae:.8f}")

print("\n[4] Summary:")
print("-" * 70)
print("Channel | Avg MAE   | Min MAE   | Max MAE   | Status")
print("-" * 70)
for ch_name in channel_names:
    mae_list = errors[ch_name]
    avg_mae = sum(mae_list) / len(mae_list)
    min_mae = min(mae_list)
    max_mae = max(mae_list)
    status = "OK (nonzero)" if avg_mae > 1e-4 else "FAIL (zero)"
    print(f"{ch_name:7} | {avg_mae:.8f} | {min_mae:.8f} | {max_mae:.8f} | {status}")

print("-" * 70)
print("\nEvaluation complete!")

