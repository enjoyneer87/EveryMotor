"""One owned Linux worker, bounded four-case queue with no automatic retry."""
from pathlib import Path
import argparse,datetime,json,os,shutil,subprocess,time,signal,re
from field_reader import sha,read_json,write_json
from audit_convergence import audit
DEADLINE=datetime.datetime.fromisoformat('2026-10-27T23:00:00+00:00').timestamp()

def group_rss(group):
    total=0
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdigit():continue
        try:
            fields=(procdir/'stat').read_text().rsplit(') ',1)[1].split()
            if int(fields[2])==group:total+=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
        except (OSError,ValueError,IndexError):pass
    return total
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read_json(a.config);root=Path(c['work']);root.mkdir(parents=True,exist_ok=False);pkg=Path(c['package']);inputs=Path(c['inputs'])
    assert len(c['cases'])==4 and len(c['cpus'])==8 and len(set(c['cases']))==4
    assert sha(pkg/'runtime/bin/ElmerSolver')=='a2a953b877b9472380186f3e30cf43cd523bb038e1badc69b0ec365a1d3fbb84'
    sourcehash={p.relative_to(inputs).as_posix():sha(p) for p in inputs.rglob('*') if p.is_file()};startall=time.monotonic();outcomes=[];failed=False
    jobs=[('pilot8',c['cases'][0],8,180)]+[(x,x,768,5400) for x in c['cases']]
    for name,key,steps,limit in jobs:
        if time.time()+limit>=DEADLINE or (root/'STOP').exists() or time.monotonic()-startall+limit>21780:break
        source=inputs/key;raw=root/name/'raw';raw.mkdir(parents=True);derived=raw.parent/'derived';derived.mkdir()
        shutil.copytree(source,raw,dirs_exist_ok=True)
        if steps==8:
            sif=(raw/'case.sif').read_text();sif,n=re.subn(r'Timestep Intervals = 768','Timestep Intervals = 8',sif);assert n==1;(raw/'case.sif').write_text(sif)
            pars=read_json(raw/'PARAMETERS.json');pars['NbPeriods']=8/256;write_json(raw/'PARAMETERS.json',pars)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',OMP_WAIT_POLICY='PASSIVE',OMPI_MCA_ess_singleton_isolated='1')
        assert set(c['cpus'])<=os.sched_getaffinity(0);start=time.monotonic();reason=None;peak=0
        with (raw/'solver.log').open('xb') as log:
            proc=subprocess.Popen([str(pkg/'launch_solver.sh'),'case.sif'],cwd=raw,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,c['cpus']))
            ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(') ',1)[1].split()[19];write_json(raw.parent/'OWNER.json',dict(pid=proc.pid,start_ticks=ticks,group=proc.pid,case=name,deadline_UTC='2026-10-27T23:00:00Z',automatic_retry=False))
            while proc.poll() is None:
                elapsed=time.monotonic()-start;logtext=(raw/'solver.log').read_text(errors='replace');rss=group_rss(proc.pid)
                peak=max(peak,rss);size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                write_json(root/'progress.json',dict(case=name,condition_case=key,steps_started=logtext.count('MAIN: Time:'),expected_steps=steps,elapsed_seconds=elapsed,pid=proc.pid,peak_RSS_bytes=peak,output_bytes=size,completed=outcomes))
                if time.time()>=DEADLINE:reason='08:00KST_deadline'
                elif (root/'STOP').exists():reason='owned_STOP'
                elif elapsed>limit:reason='case_wall_limit'
                elif time.monotonic()-startall>21780:reason='worker_aggregate_limit'
                elif peak>16*1024**3:reason='RAM_limit'
                elif size>20*1024**3 or shutil.disk_usage(root).free<32*1024**3:reason='disk_limit'
                elif any(x in logtext for x in ['FATAL','ERROR::','ERROR:','NOT converge']):reason='solver_error'
                if reason:
                    try:current=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
                    except OSError:current=None
                    if current==ticks:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                time.sleep(5)
        result=dict(case=name,condition_case=key,returncode=proc.returncode,stop_reason=reason,wall_seconds=time.monotonic()-start,peak_RSS_bytes=peak,source_hashes={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()},solver_binary_sha256=sha(pkg/'runtime/bin/ElmerSolver'))
        if proc.returncode==0 and not reason:
            try:result['audit']=audit(raw,derived)
            except Exception as exc:result['audit_error']=repr(exc)
        write_json(raw.parent/'execution.json',result);outcomes.append(dict(case=name,condition_case=key,returncode=result['returncode'],stop_reason=reason,audit_error=result.get('audit_error')))
        valid=not result['returncode'] and not reason and not result.get('audit_error') and result['audit']['whole_current_status']=='PASS' and result['audit']['nonlinear_status']=='PASS'
        print(json.dumps(dict(case=name,valid_current_nonlinear=valid,wall_seconds=result['wall_seconds'])),flush=True)
        if not valid:failed=True;break
    assert all(sha(inputs/rel)==h for rel,h in sourcehash.items())
    write_json(root/'progress.json',dict(phase='finished',outcomes=outcomes,source_unchanged=True,total_wall_seconds=time.monotonic()-startall,stopped_on_failure=failed,deadline_UTC='2026-10-27T23:00:00Z',automatic_retry=False))
if __name__=='__main__':main()
