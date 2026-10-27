"""Amendment03 single owned numerical variant; physical audit identical to run_fea.

Linux source host only. Private config supplies runtime and new work directories.
No source writes, restarts, deletions, shared locks or campaign imports.
"""
from pathlib import Path
import argparse,os,json,shutil,subprocess,time,signal,re,hashlib
import numpy as np
from field_reader import read_json,write_json,sha
from audit_convergence import audit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();c=read_json(a.config)
    root=Path(c['work']);root.mkdir(parents=True,exist_ok=False);pkg=Path(c['package']);source=Path(c['inputs']);fine=Path(c['fine_mesh'])
    assert sha(pkg/'runtime/bin/ElmerSolver')=='a2a953b877b9472380186f3e30cf43cd523bb038e1badc69b0ec365a1d3fbb84'
    source_hash={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
    assert sha(source/'CircuitsBDF2Consistent.so')=='3c6c227967e31b2f901331211029130cca5ce9d5f43f37fdb078cccf4918e4c6'
    coarse_e=np.loadtxt(source/'mesh/mesh.elements',dtype=int);fine_e=np.loadtxt(fine/'mesh.elements',dtype=int)
    assert set(coarse_e[:,1])==set(fine_e[:,1]) and len(fine_e)>len(coarse_e)
    def copper_count(e):return int(((e[:,1]>=2301)&(e[:,1]<=2336)).sum())
    meshcheck=dict(coarse_elements=len(coarse_e),fine_elements=len(fine_e),coarse_copper=copper_count(coarse_e),fine_copper=copper_count(fine_e),same_physical_tags=True,
        scope='pre-existing finer mesh; geometric region areas compared after solve; not an assumed uniform refinement')
    write_json(root/'mesh_preflight.json',meshcheck)
    pilot=read_json(Path(c['qualified_pilot']))
    assert pilot['steps']==8 and pilot['whole_current_status']=='PASS' and pilot['nonlinear_status']=='PASS'
    jobs=c['jobs'];assert len(jobs)==1 and jobs[0][0] in ['N512_P3','fine_N256_P3']
    outcomes=[];startall=time.monotonic()
    for name,spp,steps,isfine in jobs:
        if (root/'STOP').exists() or time.monotonic()-startall>=16200:break
        raw=root/name/'raw';raw.mkdir(parents=True);out=raw.parent/'derived';out.mkdir()
        for rel,h in source_hash.items():
            assert sha(source/rel)==h;target=raw/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/rel,target)
        if isfine:
            for n in ['mesh.nodes','mesh.elements','mesh.boundary','mesh.header']:shutil.copyfile(fine/n,raw/'mesh'/n)
        sif=(raw/'case.sif').read_text();freq=float(re.search(r'\$ freq = ([\d.eE+-]+)',sif)[1]);dt=1/freq/spp
        sif,n=re.subn(r'Timestep Sizes = [\d.eE+-]+',f'Timestep Sizes = {dt:.17g}',sif);assert n==1
        sif,n=re.subn(r'Timestep Intervals = \d+',f'Timestep Intervals = {steps}',sif);assert n==1
        (raw/'case.sif').write_text(sif);params=read_json(raw/'PARAMETERS.json');params.update(NbSteps=spp,NbPeriods=steps/spp);write_json(raw/'PARAMETERS.json',params)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',OMPI_MCA_ess_singleton_isolated='1',OMP_WAIT_POLICY='PASSIVE')
        cpus=c['cpus'];assert len(cpus)==8 and set(cpus)<=os.sched_getaffinity(0);start=time.monotonic();reason=None;peak=0
        with (raw/'solver.log').open('xb') as log:
            proc=subprocess.Popen([str(pkg/'launch_solver.sh'),'case.sif'],cwd=raw,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                preexec_fn=lambda:os.sched_setaffinity(0,cpus))
            ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
            write_json(raw.parent/'OWNER.json',dict(pid=proc.pid,start_ticks=ticks,group=proc.pid,case=name,source_untouched=True))
            while proc.poll() is None:
                elapsed=time.monotonic()-start;logtext=(raw/'solver.log').read_text(errors='replace')
                rss=0
                try:rss=int(next(x for x in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
                except (OSError,StopIteration):pass
                peak=max(peak,rss);size=sum(p.stat().st_size for p in raw.rglob('*') if p.is_file())
                write_json(root/'progress.json',dict(case=name,steps_started=logtext.count('MAIN: Time:'),expected_steps=steps,elapsed_seconds=elapsed,pid=proc.pid,peak_RSS_bytes=peak,output_bytes=size,completed=outcomes))
                if (root/'STOP').exists():reason='user_stop_file'
                elif elapsed>c['wall_seconds_per_run']:reason='case_wall_limit'
                elif time.monotonic()-startall>16200:reason='aggregate_wall_limit'
                elif peak>16*1024**3:reason='RAM_limit'
                elif size>40*1024**3 or shutil.disk_usage(root).free<32*1024**3:reason='disk_limit'
                elif any(x in logtext for x in ['FATAL','ERROR::','ERROR:','NOT converge']):reason='solver_error'
                if reason:
                    try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                time.sleep(5)
        result=dict(case=name,returncode=proc.returncode,stop_reason=reason,wall_seconds=time.monotonic()-start,peak_RSS_bytes=peak,source_hashes=source_hash,solver_binary_sha256=sha(pkg/'runtime/bin/ElmerSolver'))
        write_json(raw.parent/'execution.json',result)
        if proc.returncode==0 and not reason:
            try:result['audit']=audit(raw,out)
            except Exception as exc:result['audit_error']=repr(exc)
        write_json(raw.parent/'execution.json',result);outcomes.append(dict(case=name,returncode=result['returncode'],stop_reason=reason,audit_error=result.get('audit_error')))
        print(json.dumps(dict(case=name,wall_seconds=result['wall_seconds'],returncode=proc.returncode,period_status=result.get('audit',{}).get('period_status'))),flush=True)
        if name=='sanity8' and (proc.returncode or result.get('audit_error') or result['audit']['whole_current_status']!='PASS' or result['audit']['nonlinear_status']!='PASS'):break
        if proc.returncode or result.get('audit_error'):break
    assert all(sha(source/rel)==h for rel,h in source_hash.items())
    write_json(root/'progress.json',dict(phase='completed',outcomes=outcomes,source_unchanged=True,total_wall_seconds=time.monotonic()-startall))

if __name__=='__main__':main()
