"""A2 preflight (read-only, train graphs only): mechanical-angle step, electrical-cycle span and field periodicity across the 45 native steps."""
from pathlib import Path
import json,sys,collections
import numpy as np
def main():
 bundle=Path(sys.argv[1]);m=json.loads((bundle/'BUNDLE_MANIFEST.json').read_text())
 train=[r for r in m['records'] if r['split']=='train'];assert all(len(r['files'])==45 for r in train)
 spans=collections.Counter();steps=collections.Counter();offsets=collections.Counter();rows=[]
 for k,r in enumerate(train):
  names=sorted(f['name'] for f in r['files'])
  ang=np.array([float(np.load(bundle/n)['mechanical_angle_deg']) for n in names])
  inc=np.diff(ang);steps[round(float(inc.mean()),6)]+=1;assert np.allclose(inc,inc[0],atol=1e-6),(r['case_index'],inc[:3])
  span=45*inc[0];spans[round(float(span),6)]+=1;offsets[round(float(ang[0]%span),6)]+=1
  if k%14==0:  # field periodicity on a sample: stator-region elements keep their native IDs across steps
   def stator(n):
    with np.load(bundle/n) as z:
     keep=z['node'][:,3]>.5;return dict(zip(z['native_element_id'][keep].tolist(),z['target_B_T'][keep])),z['weights'][keep]
   b={i:stator(names[i])[0] for i in [0,1,22,43,44]}
   def d(i,j):
    ids=sorted(set(b[i])&set(b[j]));x=np.array([b[i][e] for e in ids]);y=np.array([b[j][e] for e in ids])
    return float(100*np.sqrt(((x-y)**2).sum()/(y**2).sum()))
   rows.append({'case':r['case_index'],'d_0_1':d(0,1),'d_43_44':d(43,44),'d_44_0_wrap':d(44,0),'d_0_22_half':d(0,22)})
 dw=np.array([x['d_44_0_wrap'] for x in rows]);ds=np.array([x['d_0_1'] for x in rows]);dh=np.array([x['d_0_22_half'] for x in rows])
 out={'train_cases':len(train),'mechanical_step_deg':dict(steps),'electrical_span_mech_deg':dict(spans),'implied_pole_pairs':{str(round(360/s,6)):n for s,n in spans.items()},'start_angle_mod_span':dict(offsets),
  'periodicity_sample':len(rows),'stator_B_step_change_percent':{'d_0_1_median':float(np.median(ds)),'d_44_0_wrap_median':float(np.median(dw)),'d_44_0_wrap_max':float(dw.max()),'d_0_22_half_cycle_median':float(np.median(dh))},
  'periodic_gate':'PASS' if np.median(dw)<=2*np.median(ds) and np.median(dw)<0.5*np.median(dh) else 'FAIL','rows':rows}
 print(json.dumps(out))
if __name__=='__main__':main()
