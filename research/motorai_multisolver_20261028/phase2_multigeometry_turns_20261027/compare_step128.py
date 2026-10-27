from pathlib import Path
import json
R=Path(__file__).resolve().parent
def main():
 ref=json.loads((R.parent/'mgn_development_20261027/convergence/N512_P3/audit.json').read_text())
 reference=ref['periods'][2]['AC_W'];rows=[]
 for name,idx in [('N128_P1',0),('N64_P2',1)]:
  p=R/'step128_results'/name/'audit.json'
  if not p.exists():continue
  a=json.loads(p.read_text());assert a['steps']==128
  for key in ['mesh/mesh.nodes','mesh/mesh.elements','MODEL.json','INPUT_CONTRACT.json']:
   assert a['input_sha256'][key]==ref['input_sha256'][key]
  assert a['sigma_S_m']==ref['sigma_S_m'] and a['active_length_m']==ref['active_length_m']
  y=a['periods'][idx]['AC_W'];same=ref['periods'][idx]['AC_W']
  rows.append(dict(case=name,total_steps=128,steps_per_period=a['steps_per_period'],adopted_diagnostic_window=f'P{idx+1}',AC_W=y,
   reference_N512_same_period_AC_W=same,difference_same_period_percent=100*(y-same)/same,
   reference_N512P3_AC_W=reference,difference_vs_N512P3_percent=100*(y-reference)/reference,
   current_status=a['whole_current_status'],current_max_percent=a['whole_current_max_percent'],nonlinear_status=a['nonlinear_status'],
   geometry_material_contract='same input mesh/model/material/initial condition hashes verified',
   formal_period_time_mesh='UNKNOWN',eligible_as_converged_Full_training_truth=False))
 out=dict(cases=rows,reference='existing Ref117 N512P3',new_extra_steps_above128=0,
  interpretation='Same-period difference diagnoses timestep plus within-period transient resolution. Difference vs P3 also includes startup/settling. Neither is an approved accuracy PASS.',
  decision='Keep128 cap. Use as low-fidelity/diagnostic until initialization or validated averaging resolves error; no automatic longer solve.')
 (R/'STEP128_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
