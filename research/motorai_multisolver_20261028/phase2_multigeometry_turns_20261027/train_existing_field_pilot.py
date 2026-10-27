"""Bounded from-scratch native element B pretraining; no AC/paired targets."""
from pathlib import Path
import argparse,os,json,time,datetime,hashlib,random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from dual_solver_mgn import DomainContract,DualSolverMGN

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,choices=[17,29],required=True);ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--wall-seconds',type=int,default=3600);ap.add_argument('--smoke-only',action='store_true');args=ap.parse_args()
 assert 0<args.epochs<=10 and 0<args.wall_seconds<=3600
 deadline=datetime.datetime.fromisoformat('2026-10-27T22:59:00+00:00');assert utc()<deadline
 args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 manifest=json.loads((args.bundle/'BUNDLE_MANIFEST.json').read_text());records=manifest['records'];assert manifest['reader_pass_cases']==20 and manifest['failed_cases']==0
 assert all(r['original_split']=='train' for r in records)
 train_groups={r['geometry_group'] for r in records if r['split']=='train'};dev_groups={r['geometry_group'] for r in records if r['split']=='development'}
 assert len(train_groups)==16 and len(dev_groups)==4 and not train_groups&dev_groups
 files={s:[] for s in ['train','development']}
 for r in records:
  assert r['status']=='READER_ARITHMETIC_PASS'
  for f in r['files']:
   p=args.bundle/f['name'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];files[r['split']].append(p)
 assert len(files['train'])==144 and len(files['development'])==36
 group_hash=hashlib.sha256(json.dumps(sorted(train_groups)).encode()).hexdigest()
 torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True,warn_only=False)
 assert torch.cuda.is_available();device=torch.device('cuda:0')
 contract=DomainContract(tuple(manifest['node_features']),tuple(manifest['edge_features']),('native_element_Bx_T','native_element_By_T'),group_hash,'native_element_reported','native_explicit_ID')
 model=DualSolverMGN({d:contract for d in ['elmer','motorcad']},len(manifest['design_features']),('MotorCAD_N45P1_formal_UNKNOWN',),width=32,depth=3).to(device)
 # Only the active Motor-CAD field branch participates. AC heads/paired losses
 # are not called, and no Elmer field supervision is fabricated.
 optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
 result=dict(started_UTC=utc().isoformat(),seed=args.seed,device=torch.cuda.get_device_name(),torch=torch.__version__,model_sha256=sha(Path(__file__).with_name('dual_solver_mgn.py')),
  trainer_sha256=sha(__file__),bundle_manifest_sha256=sha(args.bundle/'BUNDLE_MANIFEST.json'),train_groups=sorted(train_groups),inner_development_groups=sorted(dev_groups),
  original_test_used=0,existing_checkpoint_reused=False,new_FEA=0,historical_Full_donor_train_geometries=16,AC_loss_training=False,MS_teacher_used=False,
  paired_alignment=False,epochs_requested=args.epochs,seed_selection=False,training_updates=0,scope='Motor-CAD element B field pretraining, a2 excluded; not AC-loss/equal-Full-budget benchmark',metrics=[],status='RUNNING')
 def save():
  result.update(elapsed_seconds=time.monotonic()-start,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),checked_UTC=utc().isoformat())
  (args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def check():
  if (args.out/'STOP').exists():raise InterruptedError('STOP')
  if utc()>=deadline:raise InterruptedError('UTC deadline')
  if time.monotonic()-start>args.wall_seconds:raise InterruptedError('wall cap')
  if torch.cuda.max_memory_allocated()>10*1024**3:raise InterruptedError('GPU memory10GiB cap')
 def sample(p):
  with np.load(p,allow_pickle=False) as z:
   g={k:torch.as_tensor(z[k],device=device,dtype=torch.long if k=='edge_index' else torch.float32) for k in ['node','edge_index','edge_attr','weights']}
   g['batch']=torch.zeros(len(g['node']),dtype=torch.long,device=device)
   return g,torch.as_tensor(z['design_op'],device=device)[None],torch.as_tensor(z['target_B_T'],device=device)
 def forward(g,x):return model.forward_fields('motorcad',g,x,torch.zeros(1,dtype=torch.long,device=device))['fields']
 def evaluate(paths):
  model.eval();sse=energy=area=0.
  with torch.no_grad():
   for p in paths:
    check();g,x,y=sample(p);pred=forward(g,x);w=g['weights']
    assert torch.isfinite(pred).all();sse+=float((w[:,None]*(pred-y)**2).sum());energy+=float((w[:,None]*y**2).sum());area+=float(w.sum())
  return dict(vector_nRMSE_percent=100*(sse/energy)**.5,vector_RMSE_T=(sse/area)**.5,graphs=len(paths),area_weighted=True)
 save()
 try:
  # Five real-data updates are the GPU/memory/gradient smoke, retained in the
  # same training trajectory and counted in the total optimizer budget.
  model.train()
  for p in files['train'][:5]:
   check();g,x,y=sample(p);optimizer.zero_grad(set_to_none=True);pred=forward(g,x);w=g['weights'];loss=(w[:,None]*(pred-y)**2).sum()/(2*w.sum())
   assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   optimizer.step();result['training_updates']+=1
  torch.cuda.synchronize();result['GPU_smoke']='PASS_finite_gradients_native_graphs_and_memory';save()
  if not args.smoke_only:
   rng=random.Random(args.seed)
   for epoch in range(1,args.epochs+1):
    model.train();order=files['train'].copy();rng.shuffle(order)
    for p in order:
     check();g,x,y=sample(p);optimizer.zero_grad(set_to_none=True);pred=forward(g,x);w=g['weights'];loss=(w[:,None]*(pred-y)**2).sum()/(2*w.sum())
     assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step();result['training_updates']+=1
    row={'epoch':epoch,'train':evaluate(files['train']),'inner_development':evaluate(files['development'])};result['metrics'].append(row);save()
    torch.save({'model':model.state_dict(),'seed':args.seed,'epoch':epoch,'metadata':result},args.out/'last.pt')
  result['status']='COMPLETED';result['completed_UTC']=utc().isoformat()
 except InterruptedError as e:result.update(status='STOPPED',stop_reason=str(e))
 except Exception as e:result.update(status='FAILED',error=repr(e));raise
 finally:save()
 print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','train_groups','inner_development_groups']}))
if __name__=='__main__':main()
