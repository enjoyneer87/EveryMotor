"""Windows task guard for one owned native process or Docker container.

No retries. Identity includes PID creation time or immutable container ID.
The trainer also checks STOP, absolute deadline and wall cap per graph.
"""
from pathlib import Path
import datetime,hashlib,json,subprocess,time,sys,traceback
import psutil

ROOT=Path(__file__).resolve().parent
def utc():return datetime.datetime.now(datetime.timezone.utc)
def dump(name,data):(ROOT/name).write_text(json.dumps(data,indent=2)+'\n')
def main():
 cfg=json.loads((ROOT/'RUN_PRIVATE.json').read_text())
 assert not (ROOT/'OWNER.json').exists(), 'Already launched; no retry'
 deadline=datetime.datetime.fromisoformat('2026-10-27T22:59:00+00:00')
 assert utc()<deadline
 start=time.monotonic();owner={'host':cfg['host'],'seed':cfg['seed'],'started_UTC':utc().isoformat(),'guard_pid':os_pid(),'guard_started_epoch':psutil.Process().create_time(),'mode':cfg['mode']}
 dump('OWNER.json',owner)
 reason=None;code=None;proc=None;cid=None;container_stopped_verified=False
 try:
  if cfg['mode']=='native':
   with (ROOT/'training.log').open('xb') as log:
    proc=subprocess.Popen([sys.executable,str(ROOT/'train_existing_field_pilot.py'),'--bundle',cfg['bundle'],'--out',str(ROOT/'result'),'--seed',str(cfg['seed'])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   owner.update(pid=proc.pid,process_started_epoch=psutil.Process(proc.pid).create_time());dump('OWNER.json',owner)
  else:
   cmd=['docker','run','-d','--name',cfg['container_name'],'--label','owner=JEET-B-field-pilot-20261027','--gpus','all','--ipc=host','--memory','12g','--cpus','4','--ulimit','memlock=-1','--ulimit','stack=67108864','-w','/pilot','-v',str(ROOT)+':/pilot','-v',cfg['bundle']+':/bundle:ro',cfg['image'],'python','train_existing_field_pilot.py','--bundle','/bundle','--out','/pilot/result','--seed',str(cfg['seed'])]
   cid=subprocess.check_output(cmd,text=True,timeout=60).strip();assert len(cid)==64
   owner.update(container_id=cid,container_name=cfg['container_name']);dump('OWNER.json',owner)
  while True:
   if proc:
    code=proc.poll()
    if code is not None:break
   else:
    state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',cid],text=True,timeout=20))
    if not state['Running']:code=state['ExitCode'];break
   if utc()>=deadline:reason='absolute deadline'
   elif time.monotonic()-start>=3600:reason='external wall cap3600'
   elif (ROOT/'STOP').exists() or (ROOT/'result'/'STOP').exists():reason='STOP'
   if reason:break
   time.sleep(5)
 except Exception as e:
  reason=repr(e);(ROOT/'guard_error.txt').write_text(traceback.format_exc())
 finally:
  if proc and proc.poll() is None:
   try:
    p=psutil.Process(proc.pid)
    if abs(p.create_time()-owner['process_started_epoch'])<.01:
     p.terminate()
     try:p.wait(10)
     except psutil.TimeoutExpired:p.kill()
   except psutil.NoSuchProcess:pass
  if cid:
   try:
    state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',cid],text=True,timeout=20))
    if state['Running']:subprocess.run(['docker','stop','--time','10',cid],timeout=30,check=True)
    final_state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',cid],text=True,timeout=20))
    container_stopped_verified=not final_state['Running']
    with (ROOT/'training.log').open('xb') as log:subprocess.run(['docker','logs',cid],stdout=log,stderr=subprocess.STDOUT,timeout=30)
   except Exception as e:owner['cleanup_error']=repr(e)
  owner['container_stopped_verified']=container_stopped_verified if cid else None
  if cfg['mode']=='docker' and container_stopped_verified:
   repo=Path(cfg['baton_repo']);baton=repo/'results/logs/hb_control.json'
   try:
    b=json.loads(baton.read_text(encoding='utf-8-sig'))
    if b.get('owner')==cfg['baton_owner']:
     subprocess.run([sys.executable,str(repo/'tools/ops/em_baton.py'),'set','--owner',cfg['baton_owner'],'--status','idle','--experiment','JEET bounded B pilot finished; inspect result and receipt'],cwd=repo,timeout=20,check=True,capture_output=True)
   except Exception as e:owner['baton_release_error']=repr(e)
  dump('GUARD_RECEIPT.json',dict(owner,finished_UTC=utc().isoformat(),elapsed_seconds=time.monotonic()-start,returncode=code,stop_reason=reason,automatic_retries=0))
def os_pid():return psutil.Process().pid
if __name__=='__main__':main()
