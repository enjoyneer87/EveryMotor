import numpy as np
from phase1_static.motor_dataset import build_samples_from_doe_manifest
s = build_samples_from_doe_manifest("backup/doe_data_120", None, case_indices=[4])
print("== loader samples for case 4:", len(s), " keys:", sorted(k for k in s[0].keys())[:20])
for smp in list(s[:2]) + list(s[-2:]):
    y = np.asarray(smp["y"]); 
    ch = " ".join(f"c{i}:absmax={np.abs(y[:,i]).max():.3g}/nz={np.mean(y[:,i]!=0):.2f}" for i in range(y.shape[1]))
    print(f"  src={smp.get('source_file_type')} step={smp.get('step_index')} fid={smp.get('fidelity_type')} y={y.shape} {ch}")