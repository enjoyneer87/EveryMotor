"""One-worker, retry0 guard for two engineering profiles then six fixed train/evals.

No scheduler registration, solver, checkpoint/model deserialization or scientific
judgment. Only newly owned outputs and PID/create-time process trees are touched.
Atomic status-file replacement retries are I/O retries, never job retries.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ARMS={'L0':0.0,'L05':0.5}
RUN_IDS=tuple(f'{arm}_s{seed}' for seed in (17,29,43) for arm in ARMS)
HASH_KEYS=('model_sha256','trainer_sha256','loss_module_sha256','plan_sha256','selection_sha256',
           'bundle_manifest_sha256','mask_manifest_sha256','initial_tensor_state_sha256','data_order_LR_trace_sha256')


def require(condition,message):
    if not condition:raise ValueError(message)


def utc():return dt.datetime.now(dt.timezone.utc)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4*1024**2):digest.update(block)
    return digest.hexdigest()


def pin(spec):
    path=Path(spec['path']).resolve(strict=True)
    require(path.is_file() and sha(path)==spec['sha256'],'pinned SHA differs: '+path.name)
    if 'bytes' in spec:require(path.stat().st_size==spec['bytes'],'pinned size differs: '+path.name)
    return path


def pinned_json(spec):return json.loads(pin(spec).read_text(encoding='utf-8-sig'))


def file_pin(path):
    path=Path(path)
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}


def atomic_json(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for _ in range(40):
        try:temporary.replace(path);return
        except PermissionError:time.sleep(.25)
    temporary.replace(path)


def finite_positive(value,label,zero=False):
    require(type(value) in (float,int) and math.isfinite(value) and (value>=0 if zero else value>0), 'invalid '+label)
    return float(value)


def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate_plan(plan):
    require(plan['status']=='RELEASED_ENGINEERING_THEN_FIXED_SIX_RUNS','chain not released')
    exact={'workers':1,'retry_limit':0,'global_max_seconds':57600,'engineering_max_seconds':900,
           'engineering_updates':2957,'train_max_seconds':7200,'eval_max_seconds':1800,
           'projection_safety_factor':1.25,'collection_reserve_seconds':300}
    for key,value in exact.items():require(type(plan[key]) in (int,float) and plan[key]==value,'chain bound differs: '+key)
    require(plan['disk_free_min_bytes']>=10*1024**3,'disk bound below10GiB')
    require(plan['run_order']==list(RUN_IDS),'fixed pairwise run order differs')
    deadline=dt.datetime.fromisoformat(plan['deadline_UTC'])
    require(deadline.tzinfo is not None and deadline.utcoffset()==dt.timedelta(0),'deadline must be UTC')
    stage=Path(plan['stage_root']).resolve(); runs=Path(plan['runs_root']).resolve()
    require(runs==stage/'runs','runs root must be owned stage/runs')
    require(Path(plan['train_bundle_dir']).is_absolute(),'train bundle must be absolute')
    require(plan['stop_files'] and str(stage/'STOP') in [str(Path(p).resolve()) for p in plan['stop_files']], 'stage STOP missing')
    require(len({Path(s['path']).name for s in plan['source_files']})==len(plan['source_files']),'duplicate source names')
    return deadline


def parse_gpu_pids(output):
    rows=[row.strip() for row in output.splitlines() if row.strip()]
    if rows==['No running processes found']:return []
    require(all(row.isdigit() and int(row)>0 for row in rows),'unknown GPU ownership response')
    return sorted(set(map(int,rows)))


def gpu_compute_pids():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=15,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),check=False)
    require(result.returncode==0,'GPU ownership probe failed')
    return parse_gpu_pids(result.stdout)


def owner_identity(process):
    owner={'pid':int(process.pid),'create_time':float(process.create_time())}
    require(owner['pid']>0 and math.isfinite(owner['create_time']) and owner['create_time']>0,'invalid process identity')
    return owner


def identity_matches(psutil,identity):
    require(type(identity['pid']) is int and identity['pid']>0,'invalid owner PID')
    finite_positive(identity['create_time'],'owner create_time')
    try:return psutil.Process(identity['pid']).create_time()==identity['create_time']
    except psutil.NoSuchProcess:return False


def capture_owned_tree(psutil,identity):
    require(identity_matches(psutil,identity),'owned root PID changed')
    root=psutil.Process(identity['pid']); owners=[identity]
    for child in root.children(recursive=True):
        try:owners.append(owner_identity(child))
        except psutil.NoSuchProcess:pass
    require(identity_matches(psutil,identity),'owned root changed during tree capture')
    return owners


def terminate_owned(psutil,owners):
    matched=[]
    for owner in reversed(owners):
        if identity_matches(psutil,owner):
            try:
                process=psutil.Process(owner['pid'])
                if process.create_time()==owner['create_time']:process.terminate();matched.append((owner,process))
            except psutil.NoSuchProcess:pass
    if matched:psutil.wait_procs([p for _,p in matched],timeout=5)
    for owner,process in matched:
        if identity_matches(psutil,owner):
            try:
                if process.is_running():process.kill()
            except psutil.NoSuchProcess:pass


def limit_reason(now,deadline,elapsed,worker_elapsed=None,worker_limit=None,engineering_elapsed=None,stop=False):
    if stop:return 'STOP_FILE'
    if now>=deadline:return 'UTC_DEADLINE'
    if elapsed>=57600:return 'GLOBAL_57600_SECONDS'
    if engineering_elapsed is not None and engineering_elapsed>=900:return 'ENGINEERING_900_SECONDS'
    if worker_elapsed is not None and worker_elapsed>=worker_limit:return 'WORKER_WALL_LIMIT'
    return None


def worker_environment(base,pythonpath):
    result=dict(base)
    if result.get('CUDA_VISIBLE_DEVICES','').strip()=='':result.pop('CUDA_VISIBLE_DEVICES',None)
    result.update(PYTHONPATH=pythonpath,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return result


def job_rows(runs_root):
    root=Path(runs_root); rows=[]
    for arm in ARMS:
        run=f'{arm}_s17'
        rows.append({'id':'engineering_'+run,'run_id':run,'kind':'engineering','arm':arm,'seed':17,'alpha':ARMS[arm],
                     'status':'NOT_STARTED','out':str(root/'engineering'/run)})
    for run in RUN_IDS:
        arm,seed=run.split('_s')
        for kind in ('train','eval'):
            rows.append({'id':run+'_'+kind,'run_id':run,'kind':kind,'arm':arm,'seed':int(seed),'alpha':ARMS[arm],
                         'status':'NOT_STARTED','out':str(root/run/kind)})
    return rows


def expected_hashes(plan,training_plan,cpu,arm,seed,engineering=False):
    byname={Path(s['path']).name:s for s in plan['source_files']}
    return {'model_sha256':byname['dual_solver_mgn.py']['sha256'],'trainer_sha256':plan['trainer']['sha256'],
            'loss_module_sha256':byname['a1_allocation_loss.py']['sha256'],
            'plan_sha256':plan['engineering_plan' if engineering else 'training_plan']['sha256'],
            'selection_sha256':training_plan['selection']['sha256'],'bundle_manifest_sha256':training_plan['bundle_manifest_sha256'],
            'mask_manifest_sha256':plan['mask_manifest']['sha256'],
            'initial_tensor_state_sha256':cpu['initial_state_sha256_by_seed'][str(seed)],
            'data_order_LR_trace_sha256':cpu['expected_data_order_LR_trace_sha256']['2957' if engineering else '88565'][str(seed)]}


def validate_training_result(result,entry,expected,checkpoint_path):
    engineering=entry['kind']=='engineering'; updates=2957 if engineering else 88565
    require(result['status']==('SMOKE_COMPLETED' if engineering else 'COMPLETED'),'training result incomplete')
    require(result['training_updates']==updates and result['arm']==entry['arm'] and result['seed']==entry['seed']
            and result['alpha']==entry['alpha'],'training identity/update differs')
    require(result['parameters']==70182 and result['phase_encoding']=='mechanical' and result['data_arm']=='A0'
            and result['old_checkpoint_reused'] is False,'training architecture/encoding/reuse differs')
    require(result['new_FEA']==result['new_prior_solves']==result['original_val_test_used']==0,'training source scope differs')
    require(result['GPU_smoke']=='PASS' and finite_positive(result['peak_torch_allocated_bytes'],'torch peak',True)<=10*1024**3,'training memory/smoke gate failed')
    for key,digest in expected.items():require(valid_sha(digest) and result.get(key)==digest,'training metadata differs: '+key)
    require(result['max_updates']==(2957 if engineering else None),'engineering/full split differs')
    expected_epochs=[0,1] if engineering else [0,1,5,10,20,30]
    require([m['epoch'] for m in result['metrics']]==expected_epochs,'training assessment schedule incomplete')
    require(result['metrics'][-1]['training_updates']==updates and result['metrics'][-1]['development']['all']['snapshots']==288,'last development assessment incomplete')
    for metric in result['metrics']:
        finite_positive(metric['assessment_wall_seconds'],'assessment wall')
        finite_positive(metric['development']['all']['vector_nRMSE_percent'],'development arithmetic',True)
    checkpoint=file_pin(checkpoint_path)
    require(checkpoint['sha256']==result['checkpoint_sha256'],'final checkpoint byte SHA differs')
    return checkpoint


def profile_projection(profiles,remaining_seconds):
    require(set(profiles)==set(ARMS),'both engineering profiles required')
    projections={}
    for arm,(result,guard_wall) in profiles.items():
        require(result['status']=='SMOKE_COMPLETED' and result['training_updates']==2957,'engineering profile incomplete')
        elapsed=finite_positive(result['elapsed_seconds'],'profile result wall')
        assessment=finite_positive(result['assessment_wall_seconds_total'],'assessment total')
        require(math.isclose(assessment,sum(m['assessment_wall_seconds'] for m in result['metrics']),rel_tol=1e-9,abs_tol=1e-8),'assessment timings do not sum')
        train_time=elapsed-assessment
        require(train_time>0 and guard_wall>=elapsed-.05,'invalid guard/result timing')
        startup=max(0.,guard_wall-elapsed)
        projected=1.25*(startup+train_time/2957*88565+6*max(m['assessment_wall_seconds'] for m in result['metrics']))
        require(projected<=7200,'projected full training exceeds7200s')
        projections[arm]={'projected_train_seconds':projected,'profile_guard_seconds':guard_wall,'startup_seconds':startup,
                          'profile_update_seconds':train_time,'assessment_seconds':assessment,'safety_factor':1.25}
    expected=3*sum(row['projected_train_seconds'] for row in projections.values())+6*1800+300
    require(expected<=remaining_seconds,'projected six-run workload exceeds remaining deadline')
    return {'status':'PASS_RESOURCE_PROJECTION_ONLY','arms':projections,'projected_remaining_seconds':expected,
            'available_remaining_seconds':remaining_seconds,'evaluation_reserve_seconds':10800,'collection_reserve_seconds':300}


def make_eval_job(template,entry,train_result,checkpoint,train_path,expected,deadline):
    job=copy.deepcopy(template)
    job.update(id=entry['run_id'],arm=entry['arm'],seed=entry['seed'],alpha=entry['alpha'],checkpoint=checkpoint,
               original_train=file_pin(train_path),metadata_hashes=dict(expected),deadline_UTC=deadline)
    require(job['release_status']=='RELEASED_FROZEN_EVALUATION' and job['max_seconds']==1800 and job['workers']==1 and job['retry_limit']==0,'eval template limits differ')
    require(job['training_updates']==job['new_FEA']==job['new_prior_solves']==0 and job['phase_encoding']=='mechanical','eval template scope differs')
    require(job['selection']['sha256']==expected['selection_sha256'] and job['training_mask_manifest']['sha256']==expected['mask_manifest_sha256'],'eval template input binding differs')
    require(train_result['checkpoint_sha256']==checkpoint['sha256'],'training checkpoint differs')
    return job


def admit_inputs(plan):
    for source in plan['source_files']:pin(source)
    for key in ('guard','trainer','evaluator'):pin(plan[key])
    train=pinned_json(plan['training_plan']); engineering=pinned_json(plan['engineering_plan'])
    masks=pinned_json(plan['mask_manifest']); cpu=pinned_json(plan['cpu_admission']); template=pinned_json(plan['evaluation_template'])
    require(train['status']=='RELEASED_FIXED_SIX_RUNS' and engineering['status']=='RELEASED_ENGINEERING_ONLY','training release states differ')
    require(cpu['status']=='PASS' and masks['status']=='ADMITTED' and len(masks['records'])==328 and len(masks['development_records'])==32,'CPU/native-mask admission incomplete')
    require(set(cpu['initial_state_sha256_by_seed'])=={'17','29','43'},'initial-state CPU seed set differs')
    for updates,seeds in (('2957',{'17'}),('88565',{'17','29','43'})):
        traces=cpu['expected_data_order_LR_trace_sha256'][updates]
        require(seeds<=set(traces)<={'17','29','43'} and all(valid_sha(v) for v in traces.values()),'expected CPU trace pins incomplete')
    require(all(valid_sha(v) for v in cpu['initial_state_sha256_by_seed'].values()),'invalid initial-state pins')
    sources={Path(s['path']).name:s['sha256'] for s in plan['source_files']}
    for source in cpu['source_files']:
        require(Path(source['path']).name in sources and source['sha256']==sources[Path(source['path']).name],'CPU source differs from release')
        pin(source)
    for current in (train,engineering):
        require(current['mask_manifest_sha256']==plan['mask_manifest']['sha256'] and current['selection']['sha256']==masks['selection_sha256']
                and current['bundle_manifest_sha256']==masks['bundle_manifest_sha256'],'training plan/input pins differ')
        require(current['guard_deadline_UTC']==plan['deadline_UTC'] and current['initial_state_sha256_by_seed']==cpu['initial_state_sha256_by_seed'],'training deadline/initial pins differ')
        require(current['source_files']==plan['source_files'],'training source declarations differ')
        pin(current['selection']);pin(current['pilot_manifest'])
    require(template['source_files']==plan['source_files'],'evaluator source declarations differ')
    require(template['deadline_UTC']==plan['deadline_UTC'],'evaluator deadline differs')
    pin(template['selection']);pin(template['geometry_manifest']);pin(template['training_mask_manifest'])
    return train,engineering,masks,cpu,template


def run_guard(plan,plan_sha,plan_path):
    deadline=validate_plan(plan)
    require(Path(sys.executable).resolve()==Path(plan['python_executable']).resolve(),'wrong pinned Python')
    require(Path(__file__).resolve()==Path(plan['guard']['path']).resolve(),'guard loaded from wrong path')
    require(sha(plan_path)==plan_sha and utc()<deadline,'chain plan changed/expired')
    train,engineering,masks,cpu,template=admit_inputs(plan)
    root=Path(plan['runs_root']);require(not root.exists(),'runs already exist; retry0')
    require(not gpu_compute_pids(),'foreign GPU process present; no job launched')
    require(shutil.disk_usage(root.parent).free>=plan['disk_free_min_bytes'],'disk admission failed')
    root.mkdir(exist_ok=False);(root/'engineering').mkdir()
    import psutil
    start=time.monotonic();owner=owner_identity(psutil.Process(os.getpid()))
    status={'status':'RUNNING','chain_plan_sha256':plan_sha,'plan_sha256':plan_sha,'owner':owner,'started_UTC':utc().isoformat(),
            'new_FEA':0,'new_prior_solves':0,'automatic_retries':0,'jobs':job_rows(root),'engineering_projection':'PENDING',
            'full_training_updates':0,'engineering_updates':0,'evaluation_graphs':0}
    atomic_json(root/'GUARD_OWNER.json',owner|{'chain_plan_sha256':plan_sha})
    current=None;active_owners={};engineering_start=time.monotonic();profiles={};completed_trains={};full_pair_traces={}
    def save():
        status['elapsed_seconds']=time.monotonic()-start
        atomic_json(root/'CHAIN_STATUS.json',status)
    def check(entry=None,worker_start=None):
        stop_paths=[Path(p) for p in plan['stop_files']]+[root/'STOP']
        if entry:stop_paths += [Path(entry['out'])/'STOP',Path(entry['out']).parent/'STOP']
        engineering_elapsed=time.monotonic()-engineering_start if entry and entry['kind']=='engineering' else None
        limit=900 if entry and entry['kind']=='engineering' else (7200 if entry and entry['kind']=='train' else 1800)
        reason=limit_reason(utc(),deadline,time.monotonic()-start,None if worker_start is None else time.monotonic()-worker_start,limit,engineering_elapsed,any(p.exists() for p in stop_paths))
        if reason:raise InterruptedError(reason)
    def verify_immutable():
        require(sha(plan_path)==plan_sha,'chain plan changed during execution')
        admit_inputs(plan)
    save()
    try:
        for entry in status['jobs']:
            check(entry)
            if entry['kind']=='train' and status['engineering_projection']=='PENDING':
                require(len(profiles)==2,'engineering profiles absent')
                remaining=min(57600-(time.monotonic()-start),(deadline-utc()).total_seconds())
                projection=profile_projection(profiles,remaining)
                status['engineering_projection']=projection
                atomic_json(root/'ENGINEERING_PROJECTION_RECEIPT.json',projection)
                save()
            verify_immutable()
            require(shutil.disk_usage(root).free>=plan['disk_free_min_bytes'],'disk bound failed before job')
            require(not gpu_compute_pids(),'foreign GPU process present; no foreign process will be stopped')
            out=Path(entry['out']);require(not out.exists(),'job output exists; retry0')
            out.parent.mkdir(parents=True,exist_ok=True)
            expected=expected_hashes(plan,engineering if entry['kind']=='engineering' else train,cpu,entry['arm'],entry['seed'],entry['kind']=='engineering')
            command=[plan['python_executable'],'-B','-u']
            if entry['kind'] in ('engineering','train'):
                spec=plan['engineering_plan' if entry['kind']=='engineering' else 'training_plan']
                command += [plan['trainer']['path'],'--plan',spec['path'],'--plan-sha256',spec['sha256'],
                            '--bundle',plan['train_bundle_dir'],'--masks',plan['mask_manifest']['path'],'--masks-sha256',plan['mask_manifest']['sha256'],
                            '--out',str(out),'--arm',entry['arm'],'--seed',str(entry['seed']),'--prefetch','8']
                if entry['kind']=='engineering':command += ['--max-updates','2957']
                entry['plan_sha256']=spec['sha256']
            else:
                training_result,checkpoint,train_path=completed_trains[entry['run_id']]
                job=make_eval_job(template,entry,training_result,checkpoint,train_path,expected,plan['deadline_UTC'])
                job_path=out.parent/'eval_job.json';require(not job_path.exists(),'dynamic evaluation job already exists; retry0')
                atomic_json(job_path,job);entry['job_sha256']=sha(job_path)
                command += [plan['evaluator']['path'],'--job',str(job_path),'--job-sha256',entry['job_sha256'],'--out',str(out)]
            entry.update(status='STARTING',started_UTC=utc().isoformat(),source_sha256_by_name={Path(s['path']).name:s['sha256'] for s in plan['source_files']})
            save();worker_start=time.monotonic();next_deep_check=worker_start+10
            active_owners={}
            with (root/(entry['id']+'.log')).open('xb') as log:
                current=subprocess.Popen(command,cwd=plan['stage_root'],env=worker_environment(os.environ,plan['pythonpath']),
                    stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                identity=owner_identity(psutil.Process(current.pid));active_owners[(identity['pid'],identity['create_time'])]=identity
                entry.update(status='RUNNING',owner=identity);save()
                while current.poll() is None:
                    check(entry,worker_start)
                    try:
                        for child in capture_owned_tree(psutil,identity):active_owners[(child['pid'],child['create_time'])]=child
                    except (psutil.NoSuchProcess,ValueError):
                        if current.poll() is None:raise
                    if time.monotonic()>=next_deep_check:
                        verify_immutable()
                        allowed={o['pid'] for o in active_owners.values() if identity_matches(psutil,o)}
                        require(set(gpu_compute_pids())<=allowed,'foreign GPU process appeared; stopping owned tree only')
                        require(shutil.disk_usage(root).free>=plan['disk_free_min_bytes'],'disk bound failed during job')
                        next_deep_check=time.monotonic()+10
                    time.sleep(.5)
                returncode=current.wait()
            check(entry,worker_start)
            remaining=[o for o in active_owners.values() if identity_matches(psutil,o)]
            require(not remaining,'worker left live descendants; guard will stop only recorded owners')
            current=None;active_owners={}
            entry.update(returncode=returncode,elapsed_seconds=time.monotonic()-worker_start,completed_UTC=utc().isoformat())
            result_path=out/'RESULT.json'
            require(result_path.is_file(),'worker result missing')
            result=json.loads(result_path.read_text(encoding='utf-8'))
            entry.update(result_sha256=sha(result_path),worker_status=result.get('status'))
            require(returncode==0,'worker nonzero exit')
            if entry['kind'] in ('engineering','train'):
                checkpoint=validate_training_result(result,entry,expected,out/'last.pt')
                entry.update(checkpoint_sha256=checkpoint['sha256'],training_updates=result['training_updates'],
                             initial_tensor_state_sha256=result['initial_tensor_state_sha256'],data_order_LR_trace_sha256=result['data_order_LR_trace_sha256'])
                if entry['kind']=='engineering':
                    profiles[entry['arm']]=(result,entry['elapsed_seconds']);status['engineering_updates']+=result['training_updates']
                    if len(profiles)==2:
                        require(profiles['L0'][0]['initial_tensor_state_sha256']==profiles['L05'][0]['initial_tensor_state_sha256']
                                and profiles['L0'][0]['data_order_LR_trace_sha256']==profiles['L05'][0]['data_order_LR_trace_sha256'],'engineering pair initialization/order differs')
                else:
                    completed_trains[entry['run_id']]=(result,checkpoint,result_path)
                    pair=full_pair_traces.setdefault(entry['seed'],[]);pair.append((result['initial_tensor_state_sha256'],result['data_order_LR_trace_sha256']))
                    require(len(pair)==1 or pair[0]==pair[1],'full training pair initialization/order differs')
                    status['full_training_updates']+=result['training_updates']
            else:
                require(result['status']=='COMPLETED' and result['graphs_processed']==1440 and result['training_updates']==0
                        and result['job_sha256']==entry['job_sha256'] and result['checkpoint_sha256']==completed_trains[entry['run_id']][1]['sha256']
                        and result['reproduction_gate']['status']=='PASS_ARITHMETIC_ONLY','frozen evaluation incomplete/mismatched')
                entry.update(checkpoint_sha256=result['checkpoint_sha256'],graphs_processed=1440)
                status['evaluation_graphs']+=1440
            entry['status']='COMPLETED'
            atomic_json(root/(entry['id']+'_GUARD_RECEIPT.json'),entry);save()
        check()
        require(status['full_training_updates']==531390 and status['engineering_updates']==5914 and status['evaluation_graphs']==8640,'fixed work totals incomplete')
        status.update(status='COMPLETED',completed_UTC=utc().isoformat(),scientific_verdict='PENDING_PREREGISTERED_JSON_JUDGMENT')
    except BaseException as exc:
        status.update(status='STOPPED' if isinstance(exc,InterruptedError) else 'INCOMPLETE_NO_CLAIM',error=repr(exc))
        for row in status['jobs']:
            if row['status'] in ('STARTING','RUNNING'):row.update(status='INCOMPLETE_NO_CLAIM',error=repr(exc))
        raise
    finally:
        if active_owners:terminate_owned(psutil,list(active_owners.values()))
        if current is not None:
            try:current.wait(timeout=10)
            except subprocess.TimeoutExpired:status['owned_process_exit_unconfirmed']=True
        status.update(completed_UTC=utc().isoformat(),completed_train_runs=sum(r['kind']=='train' and r['status']=='COMPLETED' for r in status['jobs']),
                      completed_eval_runs=sum(r['kind']=='eval' and r['status']=='COMPLETED' for r in status['jobs']),
                      completed_engineering_runs=sum(r['kind']=='engineering' and r['status']=='COMPLETED' for r in status['jobs']))
        for row in status['jobs']:
            if 'started_UTC' in row:atomic_json(root/(row['id']+'_GUARD_RECEIPT.json'),row)
        save();atomic_json(root/'CHAIN_RECEIPT.json',{key:value for key,value in status.items() if key!='jobs'})
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();plan=pinned_json({'path':str(args.plan),'sha256':args.plan_sha256})
    run_guard(plan,args.plan_sha256,args.plan)


if __name__=='__main__':main()
