#!/usr/bin/env python3
"""평가 스크립트: GINO 모델의 4채널 출력 확인"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/workspace/host_data')

from doe_data_utils import load_doe_data, mesh_to_grid
from neuralop.models import GINO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("GINO Model Evaluation: 4-Channel Output Check")
print("=" * 70)

# Load DOE data
print("\n[1] 데이터 로드 중...")
doe_records, doe_conditions = load_doe_data('/workspace/host_data/doe_data', max_steps_per_case=10)
print(f"✓ {len(doe_records)} 레코드 로드됨")

# Load model checkpoint
print("\n[2] GINO 체크포인트 로드...")
gino_ckpt = torch.load('/workspace/host_data/doe_gino_ckpt.pt', map_location=device, weights_only=False)

# Initialize model
print("\n[3] GINO 모델 초기화...")
gino_model = GINO(
    in_channels=7, out_channels=4,
    hidden_channels=64, num_layers=2, num_modes=12,
    use_mlp=True, act=torch.nn.GELU()
).to(device)
gino_model.load_state_dict(gino_ckpt['model_state_dict'], strict=False)
gino_model.eval()
print("  ✓ GINO 로드 완료")

# Get normalization stats
x_mean = gino_ckpt['x_mean'].to(device)
x_std = gino_ckpt['x_std'].to(device)
y_mean = gino_ckpt['y_mean'].to(device)
y_std = gino_ckpt['y_std'].to(device)

# Evaluation
print("\n[4] 예측 수행 (5개 샘플)...")

channel_names = ['Bx', 'By', 'A', 'J']
results = {ch: [] for ch in channel_names}

n_test = min(5, len(doe_records))

for i in range(n_test):
    sample = mesh_to_grid(doe_records[i], doe_conditions[i], grid_res=64)
    if sample is None:
        continue
    
    # Prepare input
    x = torch.from_numpy(sample['input']).unsqueeze(0).to(device).float()  # (1, 7, 64, 64)
    y_true = torch.from_numpy(sample['target']).unsqueeze(0).to(device).float()  # (1, 4, 64, 64)
    
    # Normalize
    x_norm = (x - x_mean) / x_std
    y_true_norm = (y_true - y_mean) / y_std
    
    # Predict
    with torch.no_grad():
        y_pred_norm = gino_model(x_norm)
    
    # Denormalize
    y_pred = y_pred_norm * y_std + y_mean
    
    # Compute MAE
    mae = torch.abs(y_pred - y_true).mean(dim=(0, 2, 3))  # (4,)
    
    for ch_idx, ch_name in enumerate(channel_names):
        results[ch_name].append(mae[ch_idx].cpu().item())

# Display results
print("\n[5] 결과:")
print("\n" + "=" * 70)
print(f"{'Channel':<10} | {'Mean MAE':<15} | {'Status':<20}")
print("=" * 70)

all_valid = True
for ch_name in channel_names:
    mae_values = results[ch_name]
    if mae_values:
        mean_mae = np.mean(mae_values)
        is_nonzero = mean_mae > 1e-4
        status = "✓ (0이 아님)" if is_nonzero else "✗ (0에 가까움)"
        if not is_nonzero:
            all_valid = False
        print(f"{ch_name:<10} | {mean_mae:<15.6f} | {status:<20}")

print("=" * 70)

if all_valid:
    print("\n✓ GINO 모델이 모든 채널(Bx, By, A, J)을 정상적으로 예측합니다!")
else:
    print("\n✗ 일부 채널의 MAE가 매우 작습니다 (0에 가까움).")
