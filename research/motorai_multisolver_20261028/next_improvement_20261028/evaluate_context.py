"""Frozen C/G evaluation with unchanged L12 area/phase/quantity metrics; no E1/AC gate claim."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time,datetime,hashlib,csv,collections
import numpy as np
import torch
from dual_solver_mgn import DomainContract
from context_field_model import ContextFieldModel
from context_checkpoint import restore_context
from train_context import load_component_map
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace_retry(tmp,dst):
 """Windows refuses os.replace while a reader (status poll) holds dst open; wait briefly instead of failing the run."""
 for _ in range(40):
  try:return tmp.replace(dst)
  except PermissionError:time.sleep(.25)
 return tmp.replace(dst)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--components',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--anchor-result',type=Path,required=True);ap.add_argument('--expected-sha256');a=ap.parse_args();r=Path(__file__).parent
 plan=json.loads((r/'FROZEN_FIELD_DIAGNOSTIC_PLAN.json').read_text());manifest=json.loads((a.bundle/'BUNDLE_MANIFEST.json').read_text());selection=json.loads((r/'SCALE360_SELECTION_FROZEN.json').read_text())
 l12=json.loads((r/'CONTEXT_CG_PLAN.json').read_text());assert (a.expected_sha256 is None or sha(a.checkpoint)==a.expected_sha256) and sha(r/'dual_solver_mgn.py')==plan['model_sha256']
 assert manifest['reader_pass_cases']==32 and manifest['failed_cases']==0 and manifest['plan_sha256']==sha(r/'FROZEN_FIELD_DIAGNOSTIC_PLAN.json')
 assert manifest['selection_sha256']==sha(r/'SCALE360_SELECTION_FROZEN.json')
 admitted={x['case_index']:x for x in selection['records'] if x['split']=='development'};assert len(admitted)==32
 files=[]
 for rec in manifest['records']:
  assert rec['case_index'] in admitted and rec['geometry_group']==admitted[rec['case_index']]['geometry_group'] and rec['source_sha256']==admitted[rec['case_index']]['source_sha256'] and rec['original_split']=='train'
  assert len(rec['files'])==45 and rec['QA']['all_45_steps_checked']
  for f in rec['files']:
   p=a.bundle/f['name'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];files.append((rec['case_index'],p))
 assert len(files)==1440;files.sort(key=lambda x:(x[0],x[1].name))
 a.out.mkdir(exist_ok=False);start=time.monotonic();deadline=datetime.datetime.fromisoformat(l12['execution']['guard_deadline_UTC'])
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);assert torch.cuda.is_available()
 # Hash-pinned checkpoint was produced by this research run; metadata includes
 # TorchVersion, so deserialize its trusted local dictionary explicitly.
 checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);meta=checkpoint['metadata'];assert checkpoint['epoch']==30 and meta['training_updates']==88565 and not meta['old_checkpoint_reused'] and meta.get('status','COMPLETED') in ['COMPLETED','RUNNING']
 groups=sorted({x['geometry_group'] for x in selection['records'] if x['split']=='train'});assert meta['train_groups']==groups and set(meta['development_groups'])=={x['geometry_group'] for x in admitted.values()}
 gh=hashlib.sha256(json.dumps(groups).encode()).hexdigest();contract=DomainContract(tuple(manifest['node_features']),tuple(manifest['edge_features']),('native_element_Bx_T','native_element_By_T'),gh,'native_element_reported','native_explicit_ID')
 enc=meta.get('phase_encoding','mechanical');assert enc in ('mechanical','electrical')
 def design(z):
  """Model input as trained: electrical checkpoints see sin/cos(4 x mechanical angle) in design_op[10:12]."""
  d=z['design_op']
  if enc=='mechanical':return d
  m=np.deg2rad(float(z['mechanical_angle_deg']));assert abs(d[10]-np.sin(m))<1e-5 and abs(d[11]-np.cos(m))<1e-5
  d=d.copy();d[10]=np.sin(4*m);d[11]=np.cos(4*m);return d
 model=restore_context(checkpoint,{d:contract for d in ['elmer','motorcad']},len(manifest['design_features']),l12,
  {'context_model_sha256':sha(r/'context_field_model.py'),'model_sha256':sha(r/'dual_solver_mgn.py'),'plan_sha256':sha(r/'CONTEXT_CG_PLAN.json'),'selection_sha256':sha(r/'SCALE360_SELECTION_FROZEN.json')}).cuda();model.eval()
 comp_manifest=json.loads((a.components/'COMPONENT_MANIFEST.json').read_text());components=load_component_map(a.components,comp_manifest['source_bundle_manifest_sha256'])
 seen=set(plan['seen_phase_steps']);stats={};conductors={};snapshots=[]
 result={'status':'RUNNING','checkpoint_sha256':sha(a.checkpoint),'model_sha256':sha(r/'dual_solver_mgn.py'),'context_model_sha256':sha(r/'context_field_model.py'),'context_scope':meta['context_scope'],'context_plan_sha256':sha(r/'CONTEXT_CG_PLAN.json'),'evaluator_sha256':sha(__file__),'arm':meta.get('arm','A0'),'phase_encoding':enc,'seed':meta['seed'],'bundle_manifest_sha256':sha(a.bundle/'BUNDLE_MANIFEST.json'),'selection_sha256':manifest['selection_sha256'],'plan_sha256':manifest['plan_sha256'],'new_FEA':0,'training_updates':0,'cases':32,'geometry_role':'same inner development, not independent final test','scope':plan['physical_scope'],'graphs_processed':0}
 def save():
  result.update(elapsed_seconds=time.monotonic()-start,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated());tmp=a.out/'RESULT.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');replace_retry(tmp,a.out/'RESULT.json')
 def check():
  if (r/'STOP').exists() or (a.out/'STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline or time.monotonic()-start>=1800:raise InterruptedError('STOP/time')
  if torch.cuda.max_memory_allocated()>10*1024**3:raise InterruptedError('Torch allocated soft cap')
 def accum(key,w,p,t):
  if not len(w):return
  v=stats.setdefault(key,[0.,0.,0.,0.,0]);v[0]+=float((w*np.sum((p-t)**2,axis=1)).sum());v[1]+=float((w*(np.linalg.norm(p,axis=1)-np.linalg.norm(t,axis=1))**2).sum());v[2]+=float((w*np.sum(t*t,axis=1)).sum());v[3]+=float(w.sum());v[4]+=1
 save()
 try:
  with torch.inference_mode():
   for case,p in files:
    check()
    with np.load(p,allow_pickle=False) as z:
     g={k:torch.as_tensor(z[k],device='cuda',dtype=torch.long if k=='edge_index' else torch.float32) for k in ['node','edge_index','edge_attr','weights']};g['batch']=torch.zeros(len(g['node']),device='cuda',dtype=torch.long)
     c=components[case];assert np.array_equal(z['native_element_id'],c['native_element_id']) and hashlib.sha256(z['edge_index'].tobytes()).hexdigest()==c['edge_index_sha256'];g['component_id']=torch.as_tensor(c['component_id'],device='cuda',dtype=torch.long)
     x=torch.as_tensor(design(z),device='cuda')[None];pred=model.forward_fields('motorcad',g,x,torch.zeros(1,device='cuda',dtype=torch.long))['fields'].cpu().numpy().astype(float)
     target=z['target_B_T'].astype(float);w=z['weights'].astype(float);node=z['node'];reg=z['native_reg_code'];step=int(z['step']);assert np.isfinite(pred).all();copper=node[:,6]>.5
    phase='seen9' if step in seen else 'unused36'
    for scope in ['all45',phase]:
     accum(scope+'/all',w,pred,target);accum(scope+'/copper',w[copper],pred[copper],target[copper]);accum(scope+'/case:'+str(case),w,pred,target)
    snapshots.append({'case':case,'step':step,'phase_scope':phase,'vector_nRMSE_percent':100*np.sqrt(np.sum(w[:,None]*(pred-target)**2)/np.sum(w[:,None]*target**2)),'magnitude_nRMSE_percent':100*np.sqrt(np.sum(w*(np.linalg.norm(pred,axis=1)-np.linalg.norm(target,axis=1))**2)/np.sum(w[:,None]*target**2))})
    for code in np.unique(reg[copper]):
     m=copper&(reg==code);area=float(w[m].sum());pb2=float(np.sum(w[m,None]*pred[m]**2)/area);tb2=float(np.sum(w[m,None]*target[m]**2)/area)
     pm2=float(np.sum((np.sum(w[m,None]*pred[m],axis=0)/area)**2));tm2=float(np.sum((np.sum(w[m,None]*target[m],axis=0)/area)**2));assert pm2<=pb2+1e-9 and tm2<=tb2+1e-9
     for scope in ['all45',phase]:
      v=conductors.setdefault((case,int(code),scope),np.zeros(6));v+=np.array([pb2,tb2,pm2,tm2,area,1.])
    result['graphs_processed']+=1
    if result['graphs_processed']%50==0:save()
  metrics={k:{'B_vector_nRMSE_percent':100*np.sqrt(v[0]/v[2]),'B_magnitude_nRMSE_percent':100*np.sqrt(v[1]/v[2]),'B_vector_RMSE_T':np.sqrt(v[0]/v[3]),'snapshots':int(v[4])} for k,v in stats.items()}
  anchor=json.loads(a.anchor_result.read_text())['metrics'][-1]['development']['all']['vector_nRMSE_percent'];anchor_error=abs(metrics['seen9/all']['B_vector_nRMSE_percent']-anchor);assert anchor_error<.0001,'Frozen checkpoint/seen9 arithmetic reproduction failed'
  rows=[];b2={};sampling=[]
  for (case,code,scope),v in sorted(conductors.items()):
   pb2,tb2,pm2,tm2,area,n=v;pb2/=n;tb2/=n;pm2/=n;tm2/=n;area/=n
   rows.append({'case':case,'native_reg_code':code,'phase_scope':scope,'snapshots':int(n),'mean_area_m2':area,'pred_spatial_mean_B2_T2':pb2,'true_spatial_mean_B2_T2':tb2,'pred_square_spatial_mean_B_T2':pm2,'true_square_spatial_mean_B_T2':tm2,'B2_signed_relative_percent':100*(pb2-tb2)/tb2 if tb2 else None})
   s=b2.setdefault(scope,np.zeros(3));s+=np.array([area*abs(pb2-tb2),area*(pb2-tb2),area*tb2])
   if scope=='all45':
    nine=conductors[(case,code,'seen9')];true9=nine[1]/nine[5];sampling.append((area*abs(true9-tb2),area*(true9-tb2),area*tb2))
  total=np.sum(sampling,axis=0)
  for name,values in [('conductor_B2.csv',rows),('snapshot_errors.csv',snapshots)]:
   with (a.out/name).open('w',newline='',encoding='utf-8') as f:writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
  result.update(status='COMPLETED',metrics=metrics,seen9_anchor_difference_percentage_points=anchor_error,seen9_anchor_arithmetic_status='PASS_float32_reduction_tolerance_0.0001pp_not_physics',copper_B2={k:{'area_weighted_relative_MAE_percent':100*v[0]/v[2],'signed_bias_percent':100*v[1]/v[2]} for k,v in b2.items()},truth_seen9_vs_all45_copper_B2_sampling={'area_weighted_relative_MAE_percent':100*total[0]/total[2],'signed_bias_percent':100*total[1]/total[2]},formal_AC_qualification='UNKNOWN_not_evaluated',completed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except InterruptedError as e:result.update(status='STOPPED',stop_reason=str(e))
 except Exception as e:result.update(status='FAILED',error=repr(e));raise
 finally:save()
 print(json.dumps({k:v for k,v in result.items() if k!='metrics'}))
if __name__=='__main__':main()
