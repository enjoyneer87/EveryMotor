import json
import sys
from pathlib import Path

sys.path.insert(0, 'd:/KDH/NvidiaNemo/eMach')
import ansys.motorcad.core as pymotorcad
from tools.pyutils.sweep import DOEAxis, build_doe_lhs
from tools.motorCAD.pyMCAD.doe_batch import doe_batch_run

BASE_MOT = r'd:/KDH/Sim_4SolverX/TestCAD1.mot'
OUT = Path(r'd:/KDH/Sim_4SolverX/DOE4TrainingData_runtimecheck')
OUT.mkdir(parents=True, exist_ok=True)

try:
    mc = pymotorcad.MotorCAD(open_new_instance=False)
except Exception:
    mc = pymotorcad.MotorCAD(open_new_instance=True)

mc.load_from_file(BASE_MOT)
rb = float(mc.get_variable('Ratio_Bore')[1] if isinstance(mc.get_variable('Ratio_Bore'), tuple) else mc.get_variable('Ratio_Bore'))
rsd = float(mc.get_variable('Ratio_SlotDepth_ParallelSlot')[1] if isinstance(mc.get_variable('Ratio_SlotDepth_ParallelSlot'), tuple) else mc.get_variable('Ratio_SlotDepth_ParallelSlot'))

axes = [
    DOEAxis('Ratio_Bore', round(rb * 0.99, 6), round(rb * 1.01, 6), steps=0),
    DOEAxis('Ratio_SlotDepth_ParallelSlot', round(rsd * 0.99, 6), round(rsd * 1.01, 6), steps=0),
    DOEAxis('PeakCurrent', 120.0, 120.0, steps=0),
    DOEAxis('PhaseAdvance', 20.0, 20.0, steps=0),
]
grid = build_doe_lhs(axes=axes, n_samples=1, seed=7, criterion='classic')

manifest = doe_batch_run(
    mc,
    grid,
    base_mot=BASE_MOT,
    doe_out_root=OUT,
    phases=['solve'],
    parallel_workers=2,
    verbose=True,
)

print('=== mini batch result ===')
print(json.dumps({
    'n_cases': manifest.get('n_cases'),
    'parallel_workers': manifest.get('parallel_workers'),
    'failed_count': len(manifest.get('failed', [])),
    'case_count': len(manifest.get('cases', [])),
    'first_failed': manifest.get('failed', [None])[0],
}, indent=2, ensure_ascii=False))
