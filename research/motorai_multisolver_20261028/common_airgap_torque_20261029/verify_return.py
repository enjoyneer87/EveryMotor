"""Portable JSON-only integrity and decision replay for both imported packages."""
from pathlib import Path
import hashlib,json,math,statistics

ROOT=Path(__file__).resolve().parent.parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def numeric_digest(j):
 values=[]
 def walk(v):
  if isinstance(v,dict):
   for x in v.values():walk(x)
  elif isinstance(v,list):
   for x in v:walk(x)
  elif isinstance(v,(int,float,bool)) or v is None:values.append(v)
 walk(j);return sha(json.dumps(values,allow_nan=False,separators=(',',':')).encode())
def same(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)
def run_metric(run,key):
 if key.startswith('field/'):return run['metrics'][key]
 if key.startswith('torque/'):
  _,phase,aggregation,metric=key.split('/')
  if aggregation=='pooled':return run['torque_groups'][phase]['pooled'][metric]
  return run['torque_groups'][phase]['case_distributions'][metric]['mean']
 raise KeyError(key)

def verify_package(folder,col):
 p=ROOT/folder;m=read(p/'source_manifest.json');indexed={x['path']:x for x in m['files']}
 for row in m['files']:
  q=p/row['path'];b=q.read_bytes();assert len(b)==row['published_bytes'] and sha(b)==row['published_sha256'],q
  if row['numeric_fingerprint'] is not None:assert numeric_digest(json.loads(b))==row['numeric_fingerprint'],q
 collection=read(p/col/'COLLECTION_RECEIPT.json')
 for row in collection['files']:
  origin=indexed[col+'/'+row['relative']]
  assert origin['source_sha256']==row['sha256'] and origin['source_bytes']==row['bytes']
 return {'published_files_verified':len(m['files']),'original_collection_pins_verified':len(collection['files'])}

def verify_distributions(j,arms):
 checked=0
 for arm in arms:
  runs=[j['runs'][f'{arm}_s{s}'] for s in (17,29,43)]
  for key,distribution in j['arms'][arm].items():
   if not key.startswith(('field/','torque/')):continue
   values=[run_metric(v,key) for v in runs]
   defined=[v for v in values if v is not None]
   if not defined:assert distribution['mean'] is None;continue
   assert len(defined)==distribution['defined_count']
   same(statistics.mean(defined),distribution['mean']);same(min(defined),distribution['min']);same(max(defined),distribution['max']);same(max(defined)-min(defined),distribution['range'])
   checked+=1
 return checked

def main():
 out={}
 for folder,col in [('common_airgap_torque_20261029','evaluation_return'),('a1_loss_allocation_20261029','a1_results_return_20261029')]:out[folder]=verify_package(folder,col)
 common=read(ROOT/'common_airgap_torque_20261029/COMMON_A1_SUMMARY.json')
 assert common['completed_models']==12 and common['completed_graphs']==17280 and not common['errors']
 out['common_metric_distributions_recomputed']=verify_distributions(common,['A0','A2','C','G'])
 loss=read(ROOT/'a1_loss_allocation_20261029/A1_ALLOCATION_VERDICT.json')
 assert loss['status']=='COMPLETE' and loss['passed_run_count']==6 and not loss['errors']
 out['loss_metric_distributions_recomputed']=verify_distributions(loss,['L0','L05'])
 key='field/all45/a1/area/vector_nrmse_percent';c=loss['arms']['L0'][key];g=loss['arms']['L05'][key]
 delta=c['mean']-g['mean'];spread=max(c['range'],g['range'])
 same(delta,loss['decision']['primary']['delta_L0_minus_L05_pp']);same(2*spread,loss['decision']['primary']['twice_spread_pp'])
 limits={'all45_all_Bvector':('field/all45/all/area/vector_nrmse_percent',.5),
         'seen9_all_Bvector':('field/seen9/all/area/vector_nrmse_percent',.5),
         'copper_Bvector':('field/all45/copper/area/vector_nrmse_percent',1.),
         'common_Arkkio_pooled_torque_nRMSE':('torque/all45/pooled/nrmse_percent',1.)}
 for gate,(metric,limit) in limits.items():
  harm=loss['arms']['L05'][metric]['mean']-loss['arms']['L0'][metric]['mean']
  same(harm,loss['decision']['no_harm'][gate]['change_L05_minus_L0_pp']);assert (harm<=limit)==loss['decision']['no_harm'][gate]['pass_']
 high=lambda a:statistics.mean(loss['runs'][f'{a}_s{s}']['PA_mean_case_metrics']['all45/PA_ge_70/all']['mean_case_Bvector_nrmse_percent'] for s in (17,29,43))
 same(high('L05')-high('L0'),loss['decision']['no_harm']['PA_ge70_mean_case_Bvector']['change_L05_minus_L0_pp'])
 assert loss['verdict']=='NO_IMPROVEMENT_CLAIM' and delta>=1 and delta>2*spread and not loss['decision']['no_harm_pass']
 out.update(status='PASS_INTEGRITY_AND_SAVED_ARITHMETIC_ONLY',new_training=0,new_inference=0,new_FEA=0,physical_qualification='UNKNOWN_not_established_by_replay')
 print(json.dumps(out,indent=2))

if __name__=='__main__':main()
