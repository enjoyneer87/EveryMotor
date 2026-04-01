#!/usr/bin/env python3
"""평가 스크립트: FNO 모델의 A, J 채널 확인"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/host_data')

from doe_data_utils import load_doe_data, mesh_to_grid
from physicsnemo.models.fno import FNO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("FNO 모델 평가: A, J 채널 확인")
print("=" * 60)

# DOE 데이터 로드
print("\n[1] DOE 데이터 로드 중...")
doe_records, doe_conditions = load_doe_data('/workspace/host_data/doe_data', max_steps_per_case=10)
print(f"✓ {len(doe_records)} 레코드 로드됨 ({len(doe_conditions)} 케이스)")

# Grid 데이터로 변환
print(f"\n[2] Grid 데이터로 변환 중...")
sample = mesh_to_grid(doe_records[0], doe_conditions[0], grid_res=64)
if sample is None:
    print("ERROR: mesh_to_grid conversion failed")
    sys.exit(1)

input_grid = sample['input']
target_grid = sample['target']

print(f"  - Input shape: {input_grid.shape}")
print(f"  - Target shape: {target_grid.shape}")
print(f"  - Input channels: {input_grid.shape[0]}")
print(f"  - Target channels: {target_grid.shape[0]}")

# 타겟 채널 통계
target_names = ['Bx', 'By', 'A', 'J']
print(f"\n[3] 타겟 채널별 통계 (True):")
for i, name in enumerate(target_names):
    vals = target_grid[i]
    print(f"  {name}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
          f"min={vals.min():.6f}, max={vals.max():.6f}")


# FNO 모델 로드
print(f"\n[4] FNO 모델 로드 중...")
fno_ckpt = torch.load('/workspace/host_data/doe_fno_ckpt.pt', map_location=device)

fno_model = FNO(
    in_channels=7,
    out_channels=4,
    dimension=2,
    latent_channels=64,
    num_fno_layers=4,
    num_fno_modes=[16, 16],
    padding=8,
    activation_fn="gelu",
    coord_features=True,
).to(device)

fno_model.load_state_dict(fno_ckpt['model_state_dict'])
fno_model.eval()
print(f"✓ FNO 모델 로드됨: {sum(p.numel() for p in fno_model.parameters()):,} parameters")

# 정규화 파라미터
x_mean_fno = fno_ckpt['x_mean'].to(device)
x_std_fno = fno_ckpt['x_std'].to(device)
y_mean_fno = fno_ckpt['y_mean'].to(device)
y_std_fno = fno_ckpt['y_std'].to(device)

print(f"\nNormalization shapes:")
print(f"  x_mean: {x_mean_fno.shape}, y_mean: {y_mean_fno.shape}")
print(f"  x_std: {x_std_fno.shape}, y_std: {y_std_fno.shape}")


# 여러 샘플에서 예측 수행
print(f"\n[5] FNO 예측 수행 중 ({min(10, len(doe_records))} 샘플)...")
n_test = min(10, len(doe_records))
all_mae = {ch: [] for ch in target_names}
all_pred_mean = {ch: [] for ch in target_names}
all_pred_max = {ch: [] for ch in target_names}

for idx in range(n_test):
    sample = mesh_to_grid(doe_records[idx], doe_conditions[idx], grid_res=64)
    if sample is None:
        continue
    
    input_grid = torch.from_numpy(sample['input'][None, :, :, :]).float().to(device)
    target_grid = torch.from_numpy(sample['target'][None, :, :, :]).float().to(device)
    
    # 정규화 (공간 차원 포함)
    input_grid_norm = (input_grid - x_mean_fno) / x_std_fno
    
    # 예측
    with torch.no_grad():
        pred_norm = fno_model(input_grid_norm)
    
    # 역정규화
    pred = pred_norm * y_std_fno + y_mean_fno
    target = target_grid
    
    # 채널별 분석
    for c, ch_name in enumerate(target_names):
        pred_vals = pred[0, c].cpu().numpy()
        true_vals = target[0, c].cpu().numpy()
        mae = np.abs(pred_vals - true_vals).mean()
        all_mae[ch_name].append(mae)
        all_pred_mean[ch_name].append(pred_vals.mean())
        all_pred_max[ch_name].append(pred_vals.max())

# 결과 출력
print(f"\n[6] FNO 예측 결과 ({n_test} 샘플):")
print("\n채널별 평균 MAE:")
for ch_name in target_names:
    mae_mean = np.mean(all_mae[ch_name])
    mae_std = np.std(all_mae[ch_name])
    print(f"  {ch_name}: {mae_mean:.6f} ± {mae_std:.6f}")

print("\n채널별 예측값 통계:")
for ch_name in target_names:
    pred_mean_val = np.mean(all_pred_mean[ch_name])
    pred_max_val = np.mean(all_pred_max[ch_name])
    print(f"  {ch_name}: mean={pred_mean_val:.6f}, max={pred_max_val:.6f}")

# A, J 채널 검증
print(f"\n[7] A, J 채널 검증:")
a_mae_mean = np.mean(all_mae['A'])
j_mae_mean = np.mean(all_mae['J'])
a_is_valid = a_mae_mean > 1e-4
j_is_valid = j_mae_mean > 1e-4

print(f"  A 채널: MAE={a_mae_mean:.6f} {'✓ (정상 예측됨)' if a_is_valid else '✗ (거의 0)'}")
print(f"  J 채널: MAE={j_mae_mean:.6f} {'✓ (정상 예측됨)' if j_is_valid else '✗ (거의 0)'}")

if a_is_valid and j_is_valid:
    print("\n✓✓✓ A와 J 채널이 모두 정상적으로 예측되었습니다!")
else:
    print("\n✗✗✗ A 또는 J 채널이 제대로 예측되지 않았습니다.")

print("\n" + "=" * 60)
