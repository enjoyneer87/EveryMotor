"""L12 parallel stage on133: A0_s29 and A1_s29 train+evaluate concurrently, and A1_s17 is evaluated here once the R2 chain finishes its training. Reuses guard_l12.run (owned PID tree, STOP, wall/deadline, zero retry)."""
from pathlib import Path
import json,datetime,threading,time,sys,traceback
import guard_l12 as g
R=g.R;lock=threading.Lock()
def main():
 cfg=json.loads((R/'RUN_PRIVATE.json').read_text());plan=json.loads((R/'L12_PHASE45_PLAN.json').read_text());deadline=datetime.datetime.fromisoformat(plan['guard_deadline_UTC'])
 assert g.now()<deadline and not (R/'CHAIN_STATUS.json').exists()
 py=cfg.get('python_executable',sys.executable);status=dict(started_UTC=g.now().isoformat(),mode='parallel',steps=[],status='RUNNING')
 def record(step,receipt,state):
  with lock:status['steps'].append(dict(step=step,status=state,returncode=receipt.get('returncode'),stop_reason=receipt.get('stop_reason'),elapsed_seconds=receipt.get('elapsed_seconds')));g.write(R/'CHAIN_STATUS.json',status)
  return state=='COMPLETED'
 def evaluate(rid,checkpoint,anchor):
  out=R/'runs'/rid/'eval'
  receipt=g.run('evaluate',[py,str(R/'evaluate_field_l12.py'),'--bundle',cfg['dev_bundle'],'--checkpoint',str(checkpoint),'--out',str(out),'--anchor-result',str(anchor)],plan['evaluation']['wall_seconds_max'],deadline,R/'runs'/rid)
  return record(rid+':evaluate',receipt,g.status_of(out/'RESULT.json') if receipt['returncode']==0 else 'FAILED')
 def train_then_eval(spec):
  rid=spec['id'];out=R/'runs'/rid/'train'
  receipt=g.run('train',[py,str(R/'train_field_l12.py'),'--bundle',cfg['bundle'],'--out',str(out),'--arm',spec['arm'],'--seed',str(spec['seed'])],plan['training']['wall_seconds_max'],deadline,R/'runs'/rid)
  if record(rid+':train',receipt,g.status_of(out/'RESULT.json') if receipt['returncode']==0 else 'FAILED'):evaluate(rid,out/'last.pt',out/'RESULT.json')
 def evaluate_chain_A1_s17():
  """Wait for the R2 chain's A1_s17 training receipt; evaluate only a completed run."""
  src=Path(cfg['chain_stage'])/'runs'/'A1_s17'
  while not (src/'train_RECEIPT.json').exists():
   if g.stop_requested() or g.now()>=deadline:return record('A1_s17:evaluate',{},'NOT_STARTED_STOP_OR_DEADLINE')
   time.sleep(30)
  receipt=json.loads((src/'train_RECEIPT.json').read_text())
  if receipt['returncode']!=0 or g.status_of(src/'train'/'RESULT.json')!='COMPLETED':return record('A1_s17:evaluate',receipt,'SOURCE_TRAIN_NOT_COMPLETED')
  record('A1_s17:train(chain R2)',receipt,'COMPLETED');evaluate('A1_s17',src/'train'/'last.pt',src/'train'/'RESULT.json')
 g.write(R/'CHAIN_STATUS.json',status)
 specs=[s for s in plan['runs'] if s['id'] in cfg['parallel_runs']];assert [s['id'] for s in specs]==cfg['parallel_runs']
 threads=[threading.Thread(target=train_then_eval,args=(s,)) for s in specs]+[threading.Thread(target=evaluate_chain_A1_s17)]
 try:
  for t in threads:t.start();time.sleep(5)
  for t in threads:t.join()
  with lock:status['status']='COMPLETED' if all(s['status']=='COMPLETED' for s in status['steps']) and len(status['steps'])==6 else 'INCOMPLETE'
 except Exception:
  status['status']='GUARD_ERROR';(R/'guard_error.txt').write_text(traceback.format_exc())
 finally:
  status['finished_UTC']=g.now().isoformat();g.write(R/'CHAIN_STATUS.json',status)
if __name__=='__main__':main()
