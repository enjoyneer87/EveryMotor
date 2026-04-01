from doe_data_utils import load_doe_data
data = load_doe_data('/workspace/host_data/doe_data')
print(f"Type: {type(data)}")
if isinstance(data, tuple):
    print(f"Tuple length: {len(data)}")
    for i, item in enumerate(data):
        if hasattr(item, 'shape'):
            print(f"  [{i}] shape: {item.shape}")
        else:
            print(f"  [{i}] type: {type(item)}")
elif isinstance(data, dict):
    print(f"Dict keys: {list(data.keys())}")
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape {v.shape}")
