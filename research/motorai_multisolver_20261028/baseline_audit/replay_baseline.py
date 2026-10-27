"""Independent stdlib-only replay of frozen coefficients and saved loss metrics.

No fitting, solver, canonical import, or writes outside the audit directory.
Input objects are pinned by source_manifest.json. Run after extract_sources.py.
"""
from pathlib import Path
import json, math, csv, hashlib
ROOT=Path(__file__).resolve().parent
S=ROOT/'sources'
def load(p): return json.loads((S/p).read_text(encoding='utf-8'))
def poly(c,x):
 y=0.
 for a in c:y=y*x+a
 return y
def predict(m,n,I,beta):
 m=m['separable_model'];terms=[]
 for i,b,w in zip(m['base_centers_i'],m['base_centers_p'],m['base_weights']):
  r2=((I-i)/m['ls_i'])**2+((beta-b)/m['ls_p'])**2
  terms.append(w*r2*math.log(math.sqrt(r2)+1e-12))
 k=math.fsum(terms);s=n/1000
 return poly(m['speed_poly_coeffs'],s)*max(k,1e-3)**poly(m['spread_poly_coeffs'],s)
def metrics(rows,field='replay_prediction_W'):
 truth=[r['P2_AC_excess_W'] for r in rows]
 err=[abs(r[field]-t) for r,t in zip(rows,truth)]
 return dict(n=len(rows),wmae_pct=100*math.fsum(err)/math.fsum(truth),
             mean_abs_relative_pct=100*math.fsum(e/t for e,t in zip(err,truth))/len(rows),
             max_abs_relative_pct=100*max(e/t for e,t in zip(err,truth)),truth_loss_sum_W=math.fsum(truth))
models={k:load('handoff/20261019_AF_loaded96/AF_model_'+k+'_loaded96.json') for k in ['Ref','SC']}
rows=load('handoff/20261019_AF_loaded96/AF_POINTS_192.json')
reg=load('handoff/20261019_AF_loaded96/REGRESSION_INPUTS.json')
out=dict(schema='independent-frozen-coefficient-score-replay-v1',new_FEA_runs=0,AF_refit=False,
         evidence_level='independent coefficient evaluation and scalar scoring; saved raw fields not recomputed',cases={})
export=[]
for name in ['Ref','SC']:
 rs=[dict(r) for r in rows if r['model']==name];m=models[name]
 own_coords={tuple(x) for x in m['_meta']['own_train_coordinates']}
 transfers={r['index'] for r in reg[name] if r['kind']=='transferred_predicted_loss'}
 for r in rs:
  r['replay_AF']=predict(m,r['speed_rpm'],r['current_Arms'],r['beta_deg'])
  r['replay_prediction_W']=r['replay_AF']*r['hybrid_W']
  assert r['own_calibration']==((r['speed_rpm'],r['current_Arms'],r['beta_deg']) in own_coords)
  assert abs(r['hybrid_W']-r['hybrid_proximity_W']-r['hybrid_skin_excess_W'])<1e-8
  assert r['P2_current_pass']
  r['transferred_probe']=r['index'] in transfers
  export.append({k:r[k] for k in ['model','index','point_id','case_id','speed_rpm','current_Arms','beta_deg','own_calibration','transferred_probe','P2_AC_excess_W','hybrid_W','replay_AF','replay_prediction_W','prediction_W']})
 withheld=[r for r in rs if not r['own_calibration']]
 extra_excluded=[r for r in withheld if not r['transferred_probe']]
 out['cases'][name]=dict(full=metrics(rs),own_calibration_excluded=metrics(withheld),
  own_and_transferred_excluded=metrics(extra_excluded),uncorrected=metrics(rs,'hybrid_W'),
  no_anchor_speed=metrics([r for r in rs if r['speed_rpm']!=16000]),
  own_calibration_count=len(own_coords),anchor_count=sum(r['own_calibration'] and r['speed_rpm']==16000 for r in rs),
  transferred_count=len(transfers),regression_input_count=len(reg[name]),
  withheld_loss_weight=math.fsum(r['P2_AC_excess_W'] for r in withheld)/math.fsum(r['P2_AC_excess_W'] for r in rs),
  max_prediction_delta_W=max(abs(r['replay_prediction_W']-r['prediction_W']) for r in rs),
  max_AF_observed_ratio_delta=max(abs(r['AF_observed']-r['P2_AC_excess_W']/r['hybrid_W']) for r in rs),
  withheld_speeds=sorted({r['speed_rpm'] for r in withheld}),
  per_speed={str(n):metrics([r for r in rs if r['speed_rpm']==n]) for n in [2000,4000,8000,16000]},
  field_hashes=len({r['field_sha256'] for r in rs}),kernel_hashes=sorted({r['kernel_sha256'] for r in rs}))
half=load('handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.json')
for r in half:
 r['replay_prediction_W']=r['hybrid_W']*predict(models['SC'],r['speed_rpm']*.5625,r['current_Arms']*4/3,r['beta_deg'])
new=[r for r in half if r['role']!='existing_exploratory']
assert len(new)==32
out['HalfSC']=dict(predeclared32=metrics(new),
 uncorrected=metrics(new,'hybrid_W'),in_AF_speed_domain=metrics([r for r in new if not r['primary_AF_speed_extrapolated']]),
 below_AF_speed_domain=metrics([r for r in new if r['primary_AF_speed_extrapolated']]),
 excluded_exploratory4=metrics([r for r in half if r['role']=='existing_exploratory']),
 max_prediction_delta_W=max(abs(r['replay_prediction_W']-r['predictions']['normalized_current_similarity_speed']['prediction_W']) for r in half),
 frozen_SC_AF_sha256=hashlib.sha256((S/'handoff/20261019_AF_loaded96/AF_model_SC_loaded96.json').read_bytes()).hexdigest(),
 original_whole_history_current_failures=sum(not r['original_whole_history_current_pass'] for r in new),
 P2_current_failures=sum(not r['P2_current_pass'] for r in new))
assert out['cases']['Ref']['own_calibration_count']==36 and out['cases']['SC']['own_calibration_count']==27
assert max(c['max_prediction_delta_W'] for c in out['cases'].values())<1e-6
assert out['HalfSC']['max_prediction_delta_W']<1e-6
with (ROOT/'baseline_rows_192.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
(ROOT/'replay_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
