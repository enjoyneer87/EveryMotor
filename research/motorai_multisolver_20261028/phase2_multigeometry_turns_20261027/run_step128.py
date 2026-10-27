"""Two isolated diagnostic solves; exactly128 steps per job, no retry/extension."""
from pathlib import Path
import argparse,json,hashlib,shutil,subprocess,os,time,datetime,signal,re
from audit_convergence import audit
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
DEADLINE=datetime.datetime.fromisoformat('2026-10-27T22:59:00+00:00')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);a=ap.parse_args();c=json.loads(a.config.read_text())
 assert c['steps']==128 and c['steps_per_period'] in [64,128] and c['automatic_retry']==0
 if datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=1800)>=DEADLINE:raise SystemExit('Insufficient deadline margin')
 root=Path(c['work']);root.mkdir(parents=True,exist_ok=False);raw=root/'raw';shutil.copytree(c['inputs'],raw)
 source=Path(c['inputs']);manifest=json.loads(Path(c['input_manifest']).read_text());assert len(manifest)==12
 assert all(sha(source/f)==h and sha(raw/f)==h for f,h in manifest.items())
 pkg=Path(c['package']);assert sha(pkg/'runtime/bin/ElmerSolver')=='a2a953b877b9472380186f3e30cf43cd523bb038e1badc69b0ec365a1d3fbb84'
 assert sha(raw/'CircuitsBDF2Consistent.so')=='3c6c227967e31b2f901331211029130cca5ce9d5f43f37fdb078cccf4918e4c6'
 p=json.loads((raw/'PARAMETERS.json').read_text());assert p['Irms']==460 and p['rpm']==16000
 p.update(NbSteps=c['steps_per_period'],NbPeriods=128/c['steps_per_period']);write(raw/'PARAMETERS.json',p)
 text=(raw/'case.sif').read_text();freq=float(re.search(r'\$ freq = ([\d.eE+-]+)',text)[1])
 text,n=re.subn(r'Timestep Sizes = [\d.eE+-]+',f'Timestep Sizes = {1/freq/c["steps_per_period"]:.17g}',text);assert n==1
 text,n=re.subn(r'Timestep Intervals = \d+','Timestep Intervals = 128',text);assert n==1;(raw/'case.sif').write_text(text)
 assert set(c['cpus'])<=os.sched_getaffinity(0) and len(c['cpus'])==8
 env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',OMP_WAIT_POLICY='PASSIVE',OMPI_MCA_ess_singleton_isolated='1')
 start=time.monotonic();reason=None;peak=0
 with (raw/'solver.log').open('xb') as log:
  proc=subprocess.Popen([str(pkg/'launch_solver.sh'),'case.sif'],cwd=raw,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,c['cpus']))
  stat=Path(f'/proc/{proc.pid}/stat');ticks=stat.read_text().rsplit(') ',1)[1].split()[19]
  write(root/'OWNER.json',dict(pid=proc.pid,start_ticks=ticks,group=proc.pid,case=root.name,started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  while proc.poll() is None:
   elapsed=time.monotonic()-start;s=(raw/'solver.log').read_text(errors='replace')
   try:peak=max(peak,int(next(x for x in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024)
   except (OSError,StopIteration):pass
   size=sum(x.stat().st_size for x in raw.rglob('*') if x.is_file())
   write(root/'progress.json',dict(case=root.name,pid=proc.pid,steps_started=s.count('MAIN: Time:'),steps_limit=128,elapsed_seconds=elapsed,output_bytes=size))
   if (root/'STOP').exists():reason='owned_STOP'
   elif datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:reason='deadline'
   elif elapsed>1800:reason='wall_limit'
   elif peak>16*1024**3:reason='RAM_limit'
   elif size>3*1024**3 or shutil.disk_usage(root).free<32*1024**3:reason='disk_limit'
   elif any(x in s for x in ['FATAL','ERROR::','ERROR:','NOT converge']):reason='solver_error'
   if reason:
    if stat.exists() and stat.read_text().rsplit(') ',1)[1].split()[19]==ticks:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:
      if stat.exists() and stat.read_text().rsplit(') ',1)[1].split()[19]==ticks:os.killpg(proc.pid,signal.SIGKILL)
      proc.wait()
    break
   time.sleep(5)
 result=dict(case=root.name,returncode=proc.returncode,stop_reason=reason,wall_seconds=time.monotonic()-start,peak_RSS_bytes=peak,
  input_source_hashes=manifest,steps_limit=128,steps_per_period=c['steps_per_period'],formal_qualification='UNKNOWN',automatic_retries=0)
 write(root/'execution.json',result)
 if proc.returncode==0 and not reason:
  try:
   result['audit']=audit(raw,root/'derived');assert result['audit']['steps']==128
  except Exception as e:result['audit_error']=repr(e)
 assert all(sha(source/f)==h for f,h in manifest.items());result['source_unchanged']=True;write(root/'execution.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['audit','input_source_hashes']}),flush=True)
if __name__=='__main__':main()
