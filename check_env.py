import sys
print(f"Python: {sys.version}")
import torch
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
print(f"\nHost data mount: {os.path.exists('/workspace/host_data')}")
if os.path.exists('/workspace/host_data'):
    items = os.listdir('/workspace/host_data')
    print(f"  Files: {len(items)}")
    for i in items[:20]:
        print(f"    {i}")
