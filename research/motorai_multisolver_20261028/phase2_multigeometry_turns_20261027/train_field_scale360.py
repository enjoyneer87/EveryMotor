"""Expanded from-scratch B field experiment on133, frozen group split and budget."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time,datetime,hashlib,math,random
import numpy as np
import torch
from dual_solver_mgn import DomainContract,DualSolverMGN

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=Path(__file__).parent
 plan=json.loads((root/'SCALE360_PLAN.json').read_text());spec=plan['training'];selection=json.loads((root/'SCALE360_SELECTION_FROZEN.json').read_text());manifest=json.loads((a.bundle/'BUNDLE_MANIFEST.json').read_text())
 assert manifest['selection_sha256']==sha(root/'SCALE360_SELECTION_FROZEN.json') and manifest['plan_sha256']==sha(root/'SCALE360_PLAN.json')
 assert manifest['reader_pass_cases']==360 and manifest['failed_cases']==0 and len(manifest['records'])==360
 declared={r['case_index']:r for r in selection['records']};files={s:[] for s in ['train','development']};metadata={};groups={s:set() for s in files}
 for r in manifest['records']:
  d=declared[r['case_index']];assert r['geometry_group']==d['geometry_group'] and r['split']==d['split'] and r['original_split']=='train' and r['source_sha256']==d['source_sha256']
  assert r['status']=='READER_ARITHMETIC_PASS' and len(r['files'])==9
  groups[r['split']].add(r['geometry_group'])
  for f in r['files']:
   p=a.bundle/f['name'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];files[r['split']].append(p);metadata[p.stem]={'case':r['case_index'],'group':r['geometry_group'],'current_Arms':d['native_input']['RMSCurrent']}
 assert len(groups['train'])==288 and len(groups['development'])==32 and not groups['train']&groups['development']
 assert len(files['train'])==2952 and len(files['development'])==288
 pilot=json.loads((root/'B_FIELD_BUNDLE_MANIFEST.json').read_text());pilotdev={r['geometry_group'] for r in pilot['records'] if r['split']=='development'}
 assert len(pilotdev)==4
 for r in pilot['records']:assert r['geometry_group'] in groups[r['split']]
 deadline=datetime.datetime.fromisoformat(plan['guard_deadline_UTC']);assert utc()<deadline
 a.out.mkdir(exist_ok=False);start=time.monotonic();seed=spec['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=False)
 assert torch.cuda.is_available();device=torch.device('cuda:0')
 group_hash=hashlib.sha256(json.dumps(sorted(groups['train'])).encode()).hexdigest()
 contract=DomainContract(tuple(manifest['node_features']),tuple(manifest['edge_features']),('native_element_Bx_T','native_element_By_T'),group_hash,'native_element_reported','native_explicit_ID')
 model=DualSolverMGN({d:contract for d in ['elmer','motorcad']},len(manifest['design_features']),('MotorCAD_N45P1_formal_UNKNOWN',),width=spec['width'],depth=spec['depth']).to(device)
 optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate']['start'])
 result=dict(status='RUNNING',started_UTC=utc().isoformat(),host='133',seed=seed,device=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,
  model_sha256=sha(root/'dual_solver_mgn.py'),trainer_sha256=sha(__file__),plan_sha256=sha(root/'SCALE360_PLAN.json'),selection_sha256=sha(root/'SCALE360_SELECTION_FROZEN.json'),bundle_manifest_sha256=sha(a.bundle/'BUNDLE_MANIFEST.json'),
  train_groups=sorted(groups['train']),development_groups=sorted(groups['development']),fixed_pilot_development_groups=sorted(pilotdev),train_conditions=328,development_conditions=32,original_val_test_used=0,
  new_FEA=0,historical_Full_train_donor_conditions=328,AC_loss_training=False,MS_teacher_used=False,paired_alignment=False,old_checkpoint_reused=False,seed_selection=False,
  scope=plan['quantity'],epochs_requested=30,training_updates=0,epochs_completed=0,metrics=[],memory_scope='PyTorch allocated peak, soft between-graph check, not whole-device hard limit')
 def save():
  result.update(checked_UTC=utc().isoformat(),elapsed_seconds=time.monotonic()-start,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
  temp=a.out/'RESULT.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(a.out/'RESULT.json')
 def check():
  if (root/'STOP').exists() or (a.out/'STOP').exists():raise InterruptedError('STOP')
  if utc()>=deadline or time.monotonic()-start>=spec['wall_seconds_max']:raise InterruptedError('time cap')
  if torch.cuda.max_memory_allocated()>spec['torch_allocated_soft_limit_GiB']*1024**3:raise InterruptedError('torch allocated soft cap')
 def sample(p):
  with np.load(p,allow_pickle=False) as z:
   g={k:torch.as_tensor(z[k],device=device,dtype=torch.long if k=='edge_index' else torch.float32) for k in ['node','edge_index','edge_attr','weights']};g['batch']=torch.zeros(len(g['node']),dtype=torch.long,device=device)
   return g,torch.as_tensor(z['design_op'],device=device)[None],torch.as_tensor(z['target_B_T'],device=device)
 def forward(g,x):return model.forward_fields('motorcad',g,x,torch.zeros(1,dtype=torch.long,device=device))['fields']
 def update(p):
  check();g,x,y=sample(p);optimizer.zero_grad(set_to_none=True);pred=forward(g,x);w=g['weights'];loss=(w[:,None]*(pred-y)**2).sum()/(2*w.sum());assert torch.isfinite(loss)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step();result['training_updates']+=1
 def evaluate(paths):
  model.eval();stats={}
  def add(key,sse,energy,area):
   x=stats.setdefault(key,[0.,0.,0.,0]);x[0]+=float(sse);x[1]+=float(energy);x[2]+=float(area);x[3]+=1
  with torch.no_grad():
   for p in paths:
    check();g,x,y=sample(p);pred=forward(g,x);assert torch.isfinite(pred).all();w=g['weights'];err=((pred-y)**2).sum(1);energy=(y**2).sum(1);m=metadata[p.stem]
    vals=((w*err).sum(),(w*energy).sum(),w.sum())
    for key in ['all','case:'+str(m['case']),'geometry:'+m['group'],'current_Arms:'+str(m['current_Arms'])]:add(key,*vals)
    if m['group'] in pilotdev:add('fixed_pilot_dev4',*vals)
    for i,name in enumerate(['stator','rotor','magnet','copper','air','wedge','shaft','impreg','other']):
     mask=g['node'][:,3+i]>.5
     if mask.any():add('material:'+name,(w[mask]*err[mask]).sum(),(w[mask]*energy[mask]).sum(),w[mask].sum())
  return {k:{'vector_nRMSE_percent':100*(v[0]/v[1])**.5 if v[1]>0 else None,'vector_RMSE_T':(v[0]/v[2])**.5,'snapshots':v[3],'weighted_SSE_T2_m2':v[0],'target_energy_T2_m2':v[1],'area_m2':v[2]} for k,v in stats.items()}
 def assessment(epoch):
  row={'epoch':epoch,'learning_rate':optimizer.param_groups[0]['lr'],'train':evaluate(files['train']),'development':evaluate(files['development'])};result['metrics'].append(row);save()
 save()
 try:
  model.train()
  for p in files['train'][:5]:update(p)
  torch.cuda.synchronize();check();result['GPU_smoke']='PASS';save();assessment(0)
  rng=random.Random(seed)
  for epoch in range(1,31):
   lr=plan['learning_rate']['end']+.5*(plan['learning_rate']['start']-plan['learning_rate']['end'])*(1+math.cos(math.pi*(epoch-1)/29))
   for group in optimizer.param_groups:group['lr']=lr
   model.train();order=files['train'].copy();rng.shuffle(order)
   for p in order:
    update(p)
    if result['training_updates']%100==0:save()
   result['epochs_completed']=epoch
   if epoch in plan['learning_rate']['evaluation_epochs']:assessment(epoch)
   torch.save({'model':model.state_dict(),'epoch':epoch,'metadata':result},a.out/'last.tmp');(a.out/'last.tmp').replace(a.out/'last.pt');save()
  assert result['training_updates']==spec['expected_optimizer_updates'];result.update(status='COMPLETED',completed_UTC=utc().isoformat())
 except InterruptedError as e:result.update(status='STOPPED',stop_reason=str(e))
 except Exception as e:result.update(status='FAILED',error=repr(e));raise
 finally:save()
 print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','train_groups','development_groups']}))
if __name__=='__main__':main()
