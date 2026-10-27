"""Native CPU preparation or133 GPU training guard; zero retry, owned tree only."""
from pathlib import Path
import json,datetime,time,subprocess,sys,traceback,zipfile,hashlib
import psutil
R=Path(__file__).parent
def now():return datetime.datetime.now(datetime.timezone.utc)
def write(name,d):(R/name).write_text(json.dumps(d,indent=2)+'\n')
def main():
 cfg=json.loads((R/'RUN_PRIVATE.json').read_text());plan=json.loads((R/'SCALE360_PLAN.json').read_text());deadline=datetime.datetime.fromisoformat(plan['guard_deadline_UTC'])
 assert now()<deadline and not (R/'OWNER.json').exists()
 op=cfg['operation'];assert op in ['preparation','training'];cap=plan[op]['wall_seconds_max'];start=time.monotonic()
 cmd=[cfg.get('python_executable',sys.executable),str(R/('prepare_field_scale360.py' if op=='preparation' else 'train_field_scale360.py'))]
 cmd+=['--base',cfg['source_base'],'--out',str(R/'bundle')] if op=='preparation' else ['--bundle',cfg['bundle'],'--out',str(R/'result')]
 with (R/'operation.log').open('xb') as f:p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 owner=dict(operation=op,pid=p.pid,process_started_epoch=psutil.Process(p.pid).create_time(),guard_pid=psutil.Process().pid,guard_started_epoch=psutil.Process().create_time(),started_UTC=now().isoformat(),wall_seconds_max=cap)
 write('OWNER.json',owner);reason=None;rc=None
 try:
  while p.poll() is None:
   if (R/'STOP').exists() or (R/'result/STOP').exists():reason='STOP'
   elif time.monotonic()-start>=cap:reason='wall cap'
   elif now()>=deadline:reason='UTC deadline'
   if reason:break
   time.sleep(5)
  rc=p.poll()
 except Exception as e:reason=repr(e);(R/'guard_error.txt').write_text(traceback.format_exc())
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
  write('GUARD_RECEIPT.json',dict(owner,finished_UTC=now().isoformat(),elapsed_seconds=time.monotonic()-start,returncode=rc,stop_reason=reason,retries=0))
if __name__=='__main__':main()
