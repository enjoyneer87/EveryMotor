import sys
print(f"Python: {sys.version}")
import torch
from runtime_paths import get_runtime_paths

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
try:
    import physicsnemo
    print("PhysicsNeMo: OK")
except Exception as e:
    print(f"PhysicsNeMo: {e}")
try:
    import torch_geometric
    print(f"PyG: {torch_geometric.__version__}")
except Exception as e:
    print(f"PyG: {e}")
try:
    import h5py
    print(f"h5py: {h5py.__version__}")
except Exception as e:
    print(f"h5py: {e}")
try:
    from scipy.interpolate import griddata
    print("scipy: OK")
except Exception as e:
    print(f"scipy: {e}")
import os

paths = get_runtime_paths()
host_data = paths["host_data"]
doe_data = paths["doe_data"]
mso_root = paths["mso_root"]

print(f"\nHOST_DATA_DIR: {host_data}")
print(f"DOE_DATA_DIR:  {doe_data}")
print(f"MSO_ROOT:      {mso_root}")

print(f"\nHost data mount: {host_data.exists()}")
if host_data.exists() and host_data.is_dir():
    items = os.listdir(host_data)
    print(f"  Files: {len(items)}")
    for i in items[:20]:
        print(f"    {i}")
