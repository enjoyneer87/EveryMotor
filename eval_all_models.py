#!/usr/bin/env python3
"""평가 스크립트: FNO, GINO, RNN 모델의 A, J 채널 확인"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/workspace/host_data')

from doe_data_utils import load_doe_data, mesh_to_grid
from physicsnemo.models.fno import FNO
from physicsnemo.models.rnn import Seq2SeqRNN
from neuralop.models import GINO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("All Models Evaluation: FNO, GINO, RNN (4-channel outputs)")
print("=" * 70)

# Load DOE data
print("\n[1] 데이터 로드 중...")
doe_records, doe_conditions = load_doe_data('/workspace/host_data/doe_data', max_steps_per_case=10)
print(f"✓ {len(doe_records)} 레코드 로드됨")

# Load model checkpoints
print("\n[2] 모델 체크포인트 로드...")
fno_ckpt = torch.load('/workspace/host_data/doe_fno_ckpt.pt', map_location=device, weights_only=False)
gino_ckpt = torch.load('/workspace/host_data/doe_gino_ckpt.pt', map_location=device, weights_only=False)
rnn_ckpt = torch.load('/workspace/host_data/doe_rnn_ckpt.pt', map_location=device, weights_only=False)

# Initialize models
print("\n[3] 모델 초기화...")

# FNO
fno_model = FNO(
    in_channels=7, out_channels=4, dimension=2,
    latent_channels=64, num_fno_layers=4, num_fno_modes=[16, 16],
    padding=8, activation_fn="gelu", coord_features=True
).to(device)
fno_model.load_state_dict(fno_ckpt['model_state_dict'], strict=False)
fno_model.eval()
print("  ✓ FNO 로드 완료")

# GINO
gino_model = GINO(
    in_channels=7, out_channels=4,
    hidden_channels=64, num_layers=2, num_modes=12,
    use_mlp=True, act=torch.nn.GELU()
).to(device)
gino_model.load_state_dict(gino_ckpt['model_state_dict'], strict=False)
gino_model.eval()
print("  ✓ GINO 로드 완료")

# RNN
rnn_model = Seq2SeqRNN(
    input_channels=7, output_channels=4, hidden_size=64,
    num_layers=2, seq_length=4
).to(device)
rnn_model.load_state_dict(rnn_ckpt['model_state_dict'], strict=False)
rnn_model.eval()
print("  ✓ RNN 로드 완료")

# Get normalization stats
x_mean_fno, x_std_fno = fno_ckpt['x_mean'].to(device), fno_ckpt['x_std'].to(device)
y_mean_fno, y_std_fno = fno_ckpt['y_mean'].to(device), fno_ckpt['y_std'].to(device)

x_mean_gino, x_std_gino = gino_ckpt['x_mean'].to(device), gino_ckpt['x_std'].to(device)
y_mean_gino, y_std_gino = gino_ckpt['y_mean'].to(device), gino_ckpt['y_std'].to(device)

x_mean_rnn, x_std_rnn = rnn_ckpt['x_mean'].to(device), rnn_ckpt['x_std'].to(device)
y_mean_rnn, y_std_rnn = rnn_ckpt['y_mean'].to(device), rnn_ckpt['y_std'].to(device)

# Evaluation
print("\n[4] 예측 수행 (5개 샘플)...")

channel_names = ['Bx', 'By', 'A', 'J']
results = {model: {ch: [] for ch in channel_names} for model in ['FNO', 'GINO', 'RNN']}

n_test = min(5, len(doe_records))

for idx in range(n_test):
    sample = mesh_to_grid(doe_records[idx], doe_conditions[idx], grid_res=64)
    if sample is None:
        continue
    
    input_grid = torch.from_numpy(sample['input'][None, :, :, :]).float().to(device)
    target_grid = torch.from_numpy(sample['target'][None, :, :, :]).float().to(device)
    
    # FNO prediction
    input_norm = (input_grid - x_mean_fno) / x_std_fno
    with torch.no_grad():
        fno_pred = fno_model(input_norm) * y_std_fno + y_mean_fno
    
    # GINO prediction
    input_norm = (input_grid - x_mean_gino) / x_std_gino
    with torch.no_grad():
        gino_pred = gino_model(input_norm) * y_std_gino + y_mean_gino
    
    # RNN prediction (需要 (B, T, C, H, W) format)
    input_norm = (input_grid - x_mean_rnn) / x_std_rnn
    input_seq = input_norm.unsqueeze(1)  # (1, 1, 7, 64, 64)
    with torch.no_grad():
        rnn_pred = rnn_model(input_seq)[:, -1]  # Take last timestep (1, 4, 64, 64)
        rnn_pred = rnn_pred * y_std_rnn + y_mean_rnn
    
    true_vals = target_grid
    
    for c, ch_name in enumerate(channel_names):
        fno_mae = torch.abs(fno_pred[0, c] - true_vals[0, c]).mean().item()
        gino_mae = torch.abs(gino_pred[0, c] - true_vals[0, c]).mean().item()
        rnn_mae = torch.abs(rnn_pred[0, c] - true_vals[0, c]).mean().item()
        
        results['FNO'][ch_name].append(fno_mae)
        results['GINO'][ch_name].append(gino_mae)
        results['RNN'][ch_name].append(rnn_mae)

print("✓ 예측 완료")

# Display results table
print("\n[5] 채널별 평균 MAE (5개 샘플):")
print("=" * 70)
print(f"{'Channel':<10} {'FNO':<15} {'GINO':<15} {'RNN':<15}")
print("-" * 70)

for ch_name in channel_names:
    fno_mean = np.mean(results['FNO'][ch_name]) if results['FNO'][ch_name] else 0
    gino_mean = np.mean(results['GINO'][ch_name]) if results['GINO'][ch_name] else 0
    rnn_mean = np.mean(results['RNN'][ch_name]) if results['RNN'][ch_name] else 0
    print(f"{ch_name:<10} {fno_mean:<15.6f} {gino_mean:<15.6f} {rnn_mean:<15.6f}")

print("-" * 70)

# A, J validation
print("\n[6] A와 J 채널 검증 (0이 아닌지 여부):")
print("=" * 70)

all_valid = True
for model_name in ['FNO', 'GINO', 'RNN']:
    a_mae = np.mean(results[model_name]['A']) if results[model_name]['A'] else 0
    j_mae = np.mean(results[model_name]['J']) if results[model_name]['J'] else 0
    a_valid = "✓" if a_mae > 1e-4 else "✗"
    j_valid = "✓" if j_mae > 1e-4 else "✗"
    print(f"{model_name:5s}: A = {a_mae:.6f} {a_valid} | J = {j_mae:.6f} {j_valid}")
    
    if a_mae <= 1e-4 or j_mae <= 1e-4:
        all_valid = False

print("=" * 70)

if all_valid:
    print("\n✓✓✓ 모든 모델의 A와 J 채널이 정상적으로 예측되었습니다!")
else:
    print("\n✗ 일부 모델에서 A 또는 J 채널이 0에 가깝습니다. 검토 필요합니다.")

print("\n" + "=" * 70)
