"""L12 guard: 234 preparation, or the 133 chain (reference evaluation, then train+evaluate per run). Zero retry; owned process tree only; chain stops at the first non-COMPLETED step."""
from pathlib import Path
import json,datetime,time,subprocess,sys,traceback
import psutil
R=Path(__file__).parent
def now():return datetime.datetime.now(datetime.timezone.utc)
def write(path,d):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n')
 for _ in range(40):
  try:return tmp.replace(path)
  except PermissionError:time.sleep(.25)
 tmp.replace(path)
def stop_requested(*dirs):return any((d/'STOP').exists() for d in (R,)+dirs)
def run(name,cmd,cap,deadline,workdir):
 """Run one owned subprocess under STOP/wall/deadline; returns its receipt."""
 workdir.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 with (workdir/(name+'.log')).open('xb') as f:p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 owner=dict(step=name,pid=p.pid,process_started_epoch=psutil.Process(p.pid).create_time(),guard_pid=psutil.Process().pid,guard_started_epoch=psutil.Process().create_time(),started_UTC=now().isoformat(),wall_seconds_max=cap)
 write(workdir/(name+'_OWNER.json'),owner);reason=None;rc=None
 try:
  while p.poll() is None:
   if stop_requested(workdir):reason='STOP'
   elif time.monotonic()-start>=cap:reason='wall cap'
   elif now()>=deadline:reason='UTC deadline'
   if reason:break
   time.sleep(5)
  rc=p.poll()
 except Exception as e:reason=repr(e);(workdir/(name+'_guard_error.txt')).write_text(traceback.format_exc())
 finally:
  if p.poll() is None:
   try:
    parent=psutil.Process(p.pid)
    if abs(parent.create_time()-owner['process_started_epoch'])<.01:
     owned=parent.children(recursive=True)+[parent]
     for child in owned:
      try:child.terminate()
      except psutil.NoSuchProcess:pass
     _,alive=psutil.wait_procs(owned,timeout=10)
     for child in alive:
      try:child.kill()
      except psutil.NoSuchProcess:pass
   except psutil.NoSuchProcess:pass
  receipt=dict(owner,finished_UTC=now().isoformat(),elapsed_seconds=time.monotonic()-start,returncode=rc,stop_reason=reason,retries=0);write(workdir/(name+'_RECEIPT.json'),receipt)
 return receipt
def gpu_free():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();return not out,out
def status_of(path):return json.loads(path.read_text()).get('status') if path.exists() else None
def equivalent(evaluated,frozen,tol=1e-9):
 """Reference evaluation must reproduce the frozen diagnostic metrics."""
 a=json.loads(evaluated.read_text())['metrics'];b=json.loads(frozen.read_text())['metrics'];assert set(a)==set(b)
 return max(abs(a[k][m]-b[k][m]) for k in a for m in ['B_vector_nRMSE_percent','B_magnitude_nRMSE_percent'])<tol
def main():
 cfg=json.loads((R/'RUN_PRIVATE.json').read_text());plan=json.loads((R/'L12_PHASE45_PLAN.json').read_text());deadline=datetime.datetime.fromisoformat(plan['guard_deadline_UTC'])
 assert now()<deadline and not (R/'CHAIN_STATUS.json').exists() and not (R/'PREP_RECEIPT.json').exists()
 py=cfg.get('python_executable',sys.executable)
 if cfg['operation']=='preparation':
  receipt=run('prepare',[py,str(R/'prepare_field_l12.py'),'--base',cfg['source_base'],'--reference',cfg['reference_bundle'],'--out',str(R/'bundle_new')],plan['preparation']['wall_seconds_max'],deadline,R/'logs')
  write(R/'PREP_RECEIPT.json',receipt);return
 assert cfg['operation']=='chain'
 chain=dict(started_UTC=now().isoformat(),steps=[],status='RUNNING');write(R/'CHAIN_STATUS.json',chain)
 def record(step,receipt,status,extra=None):
  chain['steps'].append(dict(step=step,status=status,returncode=receipt.get('returncode'),stop_reason=receipt.get('stop_reason'),elapsed_seconds=receipt.get('elapsed_seconds'),**(extra or {})));write(R/'CHAIN_STATUS.json',chain)
  return status=='COMPLETED'
 def evaluate(run_id,checkpoint,anchor,expected=None):
  out=R/'runs'/run_id/'eval';cmd=[py,str(R/'evaluate_field_l12.py'),'--bundle',cfg['dev_bundle'],'--checkpoint',str(checkpoint),'--out',str(out),'--anchor-result',str(anchor)]
  if expected:cmd+=['--expected-sha256',expected]
  free,apps=gpu_free()
  if not free:return record(run_id+':evaluate',{},'BLOCKED_GPU_BUSY',{'apps':apps})
  receipt=run('evaluate',cmd,plan['evaluation']['wall_seconds_max'],deadline,R/'runs'/run_id)
  return record(run_id+':evaluate',receipt,status_of(out/'RESULT.json') if receipt['returncode']==0 else 'FAILED')
 try:
  for spec in plan['runs']:
   rid=spec['id']
   if stop_requested():chain['status']='STOPPED';break
   if spec['operation']=='evaluate_existing':
    ok=evaluate(rid,cfg['reference_checkpoint'],R/'SCALE360_TRAIN_RESULT.json',spec['checkpoint_sha256'])
    ok=ok and equivalent(R/'runs'/rid/'eval'/'RESULT.json',R/'FROZEN_FIELD_DIAGNOSTIC_RESULT.json')
    if not ok:chain['status']='FAILED_REFERENCE_EQUIVALENCE';break
    chain['steps'][-1]['equivalence']='PASS_1e-9pp';write(R/'CHAIN_STATUS.json',chain);continue
   free,apps=gpu_free()
   if not free:record(rid+':train',{},'BLOCKED_GPU_BUSY',{'apps':apps});chain['status']='BLOCKED';break
   out=R/'runs'/rid/'train'
   receipt=run('train',[py,str(R/'train_field_l12.py'),'--bundle',cfg['bundle'],'--out',str(out),'--arm',spec['arm'],'--seed',str(spec['seed'])],plan['training']['wall_seconds_max'],deadline,R/'runs'/rid)
   if not record(rid+':train',receipt,status_of(out/'RESULT.json') if receipt['returncode']==0 else 'FAILED'):chain['status']='FAILED_'+rid;break
   if not evaluate(rid,out/'last.pt',out/'RESULT.json'):chain['status']='FAILED_'+rid+'_EVAL';break
  else:chain['status']='COMPLETED'
 except Exception:chain['status']='GUARD_ERROR';(R/'guard_error.txt').write_text(traceback.format_exc())
 finally:
  if chain['status']=='RUNNING':chain['status']='STOPPED'
  chain['finished_UTC']=now().isoformat();write(R/'CHAIN_STATUS.json',chain)
if __name__=='__main__':main()
