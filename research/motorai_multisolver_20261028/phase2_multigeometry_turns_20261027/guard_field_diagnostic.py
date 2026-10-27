"""One-shot preparation/inference guard. No retry or other-owner process access."""
from pathlib import Path
import json,datetime,subprocess,sys,time
import psutil
R=Path(__file__).parent
def utc():return datetime.datetime.now(datetime.timezone.utc)
def dump(n,d):(R/n).write_text(json.dumps(d,indent=2)+'\n')
def main():
 cfg=json.loads((R/'RUN_PRIVATE.json').read_text());plan=json.loads((R/'FROZEN_FIELD_DIAGNOSTIC_PLAN.json').read_text());op=cfg['operation'];assert op in ['preparation','inference'];assert not (R/'OWNER.json').exists()
 deadline=datetime.datetime.fromisoformat(plan['guard_deadline_UTC']);assert utc()<deadline;cap=plan[op]['wall_seconds_max'];start=time.monotonic()
 cmd=[cfg.get('python_executable',sys.executable),str(R/('prepare_frozen_field_diagnostic.py' if op=='preparation' else 'evaluate_frozen_field_diagnostic.py'))]
 cmd+=['--base',cfg['source_base'],'--out',str(R/'bundle')] if op=='preparation' else ['--bundle',cfg['bundle'],'--checkpoint',cfg['checkpoint'],'--out',str(R/'result')]
 with (R/'operation.log').open('xb') as f:p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 owner={'pid':p.pid,'process_started_epoch':psutil.Process(p.pid).create_time(),'guard_pid':psutil.Process().pid,'guard_started_epoch':psutil.Process().create_time(),'started_UTC':utc().isoformat(),'operation':op};dump('OWNER.json',owner);reason=None
 try:
  while p.poll() is None:
   if (R/'STOP').exists() or (R/'result/STOP').exists():reason='STOP'
   elif utc()>=deadline:reason='UTC deadline'
   elif time.monotonic()-start>=cap:reason='wall cap'
   if reason:break
   time.sleep(5)
 finally:
  rc=p.poll()
  if rc is None:
   try:
    parent=psutil.Process(p.pid)
    if abs(parent.create_time()-owner['process_started_epoch'])<.01:
     owned=parent.children(recursive=True)+[parent]
     for x in owned:
      try:x.terminate()
      except psutil.NoSuchProcess:pass
     _,alive=psutil.wait_procs(owned,timeout=10)
     for x in alive:
      try:x.kill()
      except psutil.NoSuchProcess:pass
   except psutil.NoSuchProcess:pass
  dump('GUARD_RECEIPT.json',dict(owner,returncode=rc,stop_reason=reason,finished_UTC=utc().isoformat(),elapsed_seconds=time.monotonic()-start,automatic_retries=0))
if __name__=='__main__':main()
