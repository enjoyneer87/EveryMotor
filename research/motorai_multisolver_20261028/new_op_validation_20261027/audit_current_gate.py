"""Locate current-gate excursions without changing acceptance or rerunning FEA."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
def main():
 rows=[]
 for p in sorted((R/'results').glob('*/waveforms.npz')):
  a=json.loads((p.parent/'audit.json').read_text());w=np.load(p,allow_pickle=False);s=w['scalar'];peak=int(np.argmax(s[:,4]));gate=a['current_gate_percent'];spp=a['steps_per_period']
  rows.append(dict(case=p.parent.name,whole_current_status=a['whole_current_status'],whole_current_max_percent=float(s[peak,4]),
   peak_step=peak+1,peak_period=peak//spp+1,peak_time_s=float(s[peak,0]),gate_percent_of_phase_peak=gate,
   violating_steps_1based=(np.where(s[:,4]>gate)[0]+1).tolist(),
   P2_current_status='PASS' if a['periods'][1]['current_max_percent']<=gate else 'FAIL',P2_current_max_percent=a['periods'][1]['current_max_percent'],
   status_scope='whole-history launch guard and accepted-P2 current check are separate; P2 PASS does not override whole-history FAIL'))
 (R/'CURRENT_GATE_DIAGNOSTICS.json').write_text(json.dumps(dict(cases=rows,threshold_changed=False,new_solver_runs=0),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
