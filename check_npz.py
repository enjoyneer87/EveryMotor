import subprocess
r = subprocess.run([
    'docker', 'exec', 'motor_compare', 'python', '-c',
    'import numpy as np; d=np.load("/workspace/host_data/field_compare_results.npz", allow_pickle=True); print("Keys:", list(d.keys())); print("Shapes:"); [print(f"  {k}: {d[k].shape if hasattr(d[k],\"shape\") else d[k]}") for k in d.keys()]'
], capture_output=True, text=True, timeout=30)
print(r.stdout)
if r.stderr:
    print("ERR:", r.stderr[:500])
