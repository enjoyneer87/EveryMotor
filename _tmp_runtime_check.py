import sys
sys.path.insert(0, 'd:/KDH/NvidiaNemo/eMach')
import ansys.motorcad.core as pymotorcad
from tools.motorCAD.pyMCAD.melec_req_check import get_mcad_variables, set_mcad_variables
from tools.motorCAD.pyMCAD.winding_auto import auto_resize_copper_after_geometry_update

BASE_MOT = r'd:/KDH/Sim_4SolverX/TestCAD1.mot'

try:
    mc = pymotorcad.MotorCAD(open_new_instance=False)
except Exception:
    mc = pymotorcad.MotorCAD(open_new_instance=True)

mc.load_from_file(BASE_MOT)

keys = [
    'Ratio_Bore', 'Ratio_SlotDepth_ParallelSlot',
    'Copper_Width', 'Copper_Height',
    'Area_Slot', 'Area_Winding_With_Liner', 'Slot_Width', 'Winding_Depth',
]
before = get_mcad_variables(mc, keys, verbose=False)

rb = float(before['Ratio_Bore'])
rsd = float(before['Ratio_SlotDepth_ParallelSlot'])
set_mcad_variables(mc, {
    'Ratio_Bore': rb * 0.97,
    'Ratio_SlotDepth_ParallelSlot': rsd * 1.03,
}, verbose=False)

auto_resize_copper_after_geometry_update(
    mc,
    all_vars={'Ratio_Bore': rb * 0.97, 'Ratio_SlotDepth_ParallelSlot': rsd * 1.03},
    verbose=True,
)

after = get_mcad_variables(mc, keys, verbose=False)

print('=== Motor-CAD runtime numeric check ===')
print(f"Ratio_Bore: {before['Ratio_Bore']} -> {after['Ratio_Bore']}")
print(f"Ratio_SlotDepth_ParallelSlot: {before['Ratio_SlotDepth_ParallelSlot']} -> {after['Ratio_SlotDepth_ParallelSlot']}")
print(f"Copper_Width: {before['Copper_Width']} -> {after['Copper_Width']}")
print(f"Copper_Height: {before['Copper_Height']} -> {after['Copper_Height']}")
print(f"Area_Slot: {before['Area_Slot']} -> {after['Area_Slot']}")
print(f"Area_Winding_With_Liner: {before['Area_Winding_With_Liner']} -> {after['Area_Winding_With_Liner']}")
