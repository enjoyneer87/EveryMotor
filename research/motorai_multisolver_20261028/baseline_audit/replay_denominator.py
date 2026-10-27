"""Replay published denominator evaluator on 24 saved [128,600] scalar-B caches.

Uses pinned canonical evaluator without fitting. Does not reconstruct B from A.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
R=Path(__file__).resolve().parent;S=R/'sources';sys.path.insert(0,str(S))
from jeet_acloss_rbf.rea_hybrid.e10_sampling import evaluate_active_samples
rows=json.loads((S/'handoff/20261019_AF_loaded96/AF_POINTS_192.json').read_text())
contract=json.loads((S/'jeet_acloss_rbf/rea_hybrid/examples/e10_ref_input.json').read_text())
index=json.loads((S/'handoff/20261019_rea_hybrid_remote_results/INPUT_INDEX_60.json').read_text())['rows']
index={r['source_key']:r for r in index}
cache={};audit=[]
for row in rows:
 key='Ref_'+row['ref_point_id']
 if key not in cache:
  p=S/'handoff/20261019_rea_hybrid_remote_results/samples'/(key+'.npz')
  with np.load(p,allow_pickle=False) as d:cache[key]={k:d[k] for k in d.files}
 d=cache[key];assert d['B_magnitude_T'].shape==(128,600)
 assert row['field_sha256']==index[key]['field_sha256']
 assert row['manifest_sha256']==index[key]['manifest_sha256']
 kr=1 if row['model']=='Ref' else 2
 c=dict(contract,Ref_current_Arms=row['current_Arms']/kr)
 # Fixed current is used for skin; B cache uses the documented bounded match.
 actual=evaluate_active_samples(d['B_magnitude_T'],d,c,row['current_Arms'],row['speed_rpm'],kr)
 audit.append(dict(model=row['model'],point_id=row['point_id'],
  delta_hybrid_W=actual['hybrid_W']-row['hybrid_W'],
  delta_skin_W=actual['hybrid_skin_excess_W']-row['hybrid_skin_excess_W'],
  delta_proximity_W=actual['hybrid_proximity_W']-row['hybrid_proximity_W'],
  source_current_delta_A=row['current_delta_ref_A']))
out=dict(schema='saved-scalar-B-denominator-replay-v1',rows=len(rows),Ref_MS_histories=len(cache),
 shape=[128,600],line_counts=d['line_counts'].tolist(),new_FEA_runs=0,AF_refit=False,
 max_abs_delta_hybrid_W=max(abs(a['delta_hybrid_W']) for a in audit),
 max_abs_delta_skin_W=max(abs(a['delta_skin_W']) for a in audit),
 max_abs_delta_proximity_W=max(abs(a['delta_proximity_W']) for a in audit),
 max_abs_source_current_delta_A=max(abs(a['source_current_delta_A']) for a in audit),
 sigma_S_m=contract['sigma_S_m'],active_length_m=contract['active_length_m'],
 caveat='Cached B samples replayed; raw MS A/P1 curl and Full-FEA Joule/DC integration not recomputed.',
 raw_A_paths_existing_here=sum(Path(index[k]['field_path']).exists() for k in cache))
assert out['max_abs_delta_hybrid_W']<1e-6
(R/'denominator_replay_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
