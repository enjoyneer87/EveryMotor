"""L12-A2 guard on133: G-EVAL and G-PREFETCH gates in sequence, then the declared runs train+evaluate in parallel. Reuses guard_l12.run (owned PID tree, STOP, wall/deadline, zero retry)."""
from pathlib import Path
import json,datetime,threading,time,sys,traceback
import guard_l12 as g
R=g.R;lock=threading.Lock()
def main():
 cfg=json.loads((R/'RUN_PRIVATE.json').read_text());plan=json.loads((R/'L12_PHASE45_PLAN.json').read_text());a2=json.loads((R/'L12_A2_PLAN.json').read_text())
 deadline=datetime.datetime.fromisoformat(plan['guard_deadline_UTC']);assert g.now()<deadline and not (R/'CHAIN_STATUS.json').exists()
 py=cfg.get('python_executable',sys.executable);status=dict(started_UTC=g.now().isoformat(),mode='gates_then_parallel',steps=[],status='RUNNING')
 def record(step,receipt,state,extra=None):
  with lock:status['steps'].append(dict(step=step,status=state,returncode=receipt.get('returncode'),stop_reason=receipt.get('stop_reason'),elapsed_seconds=receipt.get('elapsed_seconds'),**(extra or {})));g.write(R/'CHAIN_STATUS.json',status)
  return state=='COMPLETED'
 def train(rid,args):
  out=R/'runs'/rid/'train'
  receipt=g.run('train',[py,str(R/'train_field_l12.py'),'--bundle',cfg['bundle'],'--out',str(out)]+args,plan['training']['wall_seconds_max'],deadline,R/'runs'/rid)
  return out,receipt,(g.status_of(out/'RESULT.json') if receipt['returncode']==0 else 'FAILED')
 def evaluate(rid,checkpoint,anchor,expected=None):
  out=R/'runs'/rid/'eval';cmd=[py,str(R/'evaluate_field_l12.py'),'--bundle',cfg['dev_bundle'],'--checkpoint',str(checkpoint),'--out',str(out),'--anchor-result',str(anchor)]
  if expected:cmd+=['--expected-sha256',expected]
  receipt=g.run('evaluate',cmd,plan['evaluation']['wall_seconds_max'],deadline,R/'runs'/rid)
  return record(rid+':evaluate',receipt,g.status_of(out/'RESULT.json') if receipt['returncode']==0 else 'FAILED')
 def train_then_eval(spec):
  out,receipt,state=train(spec['id'],['--arm',spec['arm'],'--seed',str(spec['seed']),'--phase-encoding',spec['phase_encoding'],'--prefetch',str(a2['prefetch'])])
  if record(spec['id']+':train',receipt,state):evaluate(spec['id'],out/'last.pt',out/'RESULT.json')
 g.write(R/'CHAIN_STATUS.json',status)
 try:
  ge=a2['gates']['G-EVAL']
  ok=evaluate(ge['run'],cfg['reference_checkpoint'],R/'SCALE360_TRAIN_RESULT.json',ge['checkpoint_sha256']) and g.equivalent(R/'runs'/ge['run']/'eval'/'RESULT.json',R/'FROZEN_FIELD_DIAGNOSTIC_RESULT.json',ge['tolerance_pp'])
  if not ok:status['status']='FAILED_G-EVAL';return
  gp=a2['gates']['G-PREFETCH']
  out,receipt,state=train(gp['run'],['--arm',gp['arm'],'--seed',str(gp['seed']),'--prefetch',str(gp['prefetch']),'--max-updates',str(gp['max_updates'])])
  got=None
  if state=='SMOKE_COMPLETED':
   row=json.loads((out/'RESULT.json').read_text())['metrics'][-1];got=row['development']['all']['vector_nRMSE_percent'] if row['epoch']==1 else None
  passed=got is not None and abs(got-gp['expected_epoch1_dev_vector_nRMSE_percent'])<gp['tolerance_pp']
  record(gp['run'],receipt,'COMPLETED' if passed else 'FAILED',{'epoch1_dev':got,'expected':gp['expected_epoch1_dev_vector_nRMSE_percent']})
  if not passed:status['status']='FAILED_G-PREFETCH';return
  threads=[threading.Thread(target=train_then_eval,args=(s,)) for s in a2['runs']]
  for t in threads:t.start();time.sleep(5)
  for t in threads:t.join()
  with lock:status['status']='COMPLETED' if all(s['status']=='COMPLETED' for s in status['steps']) and len(status['steps'])==2+2*len(a2['runs']) else 'INCOMPLETE'
 except Exception:
  status['status']='GUARD_ERROR';(R/'guard_error.txt').write_text(traceback.format_exc())
 finally:
  if status['status']=='RUNNING':status['status']='STOPPED'
  status['finished_UTC']=g.now().isoformat();g.write(R/'CHAIN_STATUS.json',status)
if __name__=='__main__':main()
