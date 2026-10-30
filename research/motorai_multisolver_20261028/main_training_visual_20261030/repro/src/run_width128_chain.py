"""Single-worker retry0 phase45 guard; engineering then separately released science.

Same absolute pins, STOP paths, PID/create-time ownership,28800s global budget,
900s engineering,10800s pertrain,1800s pereval,10GiB torch/disk guards as capacity.
Only W128_s17 engineering2957 is run then discarded; science requires a distinct
--scientific-release SHA and performs W128_s17/s29/s43 final88565+eval1440 only.
Plan mappings engineering_plans/training_plans contain W128 only; status is
RELEASED_ENGINEERING_ONLY or RELEASED_FIXED_WIDTH128_RUNS. CPU admission widths W128
has init3seeds and phase45 order traces2957/88565 plus frozen source_files.
Scientific release: status RELEASED_FIXED_WIDTH128_RUNS,chain_plan_sha256,
cpu_admission_sha256,training_plan_sha256_by_arm,engineering_receipt pin,
phase45_controls_by_seed eval JSON pins,phase45_cpu_admission pin and
phase45_control_train_results_by_seed final training JSON pins are mandatory.
CPU reference_W64.bridge must pass the explicit CPU-only baseline bridge gates.
checkpoint deserialization, remote execution, scheduler, or science judgment.
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

ARMS={'W128':128}
RUN_IDS=tuple(f'W128_s{seed}' for seed in (17,29,43))
PARAMETERS={'W128':(1079430,466434)}
HASH_KEYS=('model_sha256','trainer_sha256','loss_module_sha256','plan_sha256','selection_sha256',
           'bundle_manifest_sha256','mask_manifest_sha256','initial_tensor_state_sha256','data_order_LR_trace_sha256','capacity_contract_sha256')


def require(condition,message):
    if not condition:raise ValueError(message)


def utc():return dt.datetime.now(dt.timezone.utc)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4*1024**2):digest.update(block)
    return digest.hexdigest()


def pin(spec):
    require(Path(spec['path']).is_absolute() and valid_sha(spec['sha256']),'absolute path and SHA pin required')
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


def worker_environment(base,pythonpath):
    result=dict(base)
    if result.get('CUDA_VISIBLE_DEVICES','').strip()=='':result.pop('CUDA_VISIBLE_DEVICES',None)
    result.update(PYTHONPATH=pythonpath,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return result



def validate_plan(plan):
    require(plan['status'] in ('RELEASED_ENGINEERING_ONLY','RELEASED_FIXED_WIDTH128_RUNS'),'chain not released')
    exact={'workers':1,'retry_limit':0,'global_max_seconds':28800,'engineering_max_seconds':900,
           'engineering_updates':2957,'train_max_seconds':10800,'eval_max_seconds':1800,
           'projection_safety_factor':1.25,'collection_reserve_seconds':300,
           'torch_allocated_max_bytes':10*1024**3}
    for key,value in exact.items():require(type(plan[key]) in (int,float) and plan[key]==value,'chain bound differs: '+key)
    require(type(plan['disk_free_min_bytes']) is int and plan['disk_free_min_bytes']>=10*1024**3,'disk bound below10GiB')
    require(plan['run_order']==list(RUN_IDS),'fixed W128 run order differs')
    deadline=dt.datetime.fromisoformat(plan['deadline_UTC'])
    require(deadline.tzinfo is not None and deadline.utcoffset()==dt.timedelta(0),'deadline must be UTC')
    for key in ('stage_root','runs_root','train_bundle_dir','python_executable'):
        require(Path(plan[key]).is_absolute(),'absolute path required: '+key)
    stage=Path(plan['stage_root']).resolve();runs=Path(plan['runs_root']).resolve()
    require(runs==stage/'runs','runs root must be owned stage/runs')
    stops=[Path(p) for p in plan['stop_files']]
    require(all(p.is_absolute() for p in stops),'STOP paths must be absolute')
    require({stage/'STOP',runs/'STOP'}<=set(p.resolve() for p in stops),'stage/run STOP missing')
    require(len({Path(s['path']).name for s in plan['source_files']})==len(plan['source_files']),'duplicate source names')
    return deadline


def limit_reason(now,deadline,elapsed,worker_elapsed=None,worker_limit=None,engineering_elapsed=None,stop=False):
    if stop:return 'STOP_FILE'
    if now>=deadline:return 'UTC_DEADLINE'
    if elapsed>=28800:return 'GLOBAL_28800_SECONDS'
    if engineering_elapsed is not None and engineering_elapsed>=900:return 'ENGINEERING_900_SECONDS'
    if worker_elapsed is not None and worker_elapsed>=worker_limit:return 'WORKER_WALL_LIMIT'
    return None


def job_rows(runs_root,phase):
    root=Path(runs_root);rows=[]
    identities=[(arm,17,'engineering') for arm in ARMS] if phase=='engineering' else [(arm,int(seed),kind) for run in RUN_IDS for arm,seed in [run.split('_s')] for kind in ('train','eval')]
    for arm,seed,kind in identities:
        run=f'{arm}_s{seed}';out=root/'engineering'/run if kind=='engineering' else root/run/kind
        rows.append(dict(id=('engineering_'+run if kind=='engineering' else run+'_'+kind),run_id=run,kind=kind,
                         arm=arm,width=ARMS[arm],seed=seed,alpha=0.0,status='NOT_STARTED',out=str(out)))
    return rows


def expected_hashes(plan,training_plan,cpu,arm,seed,engineering=False):
    sources={Path(s['path']).name:s['sha256'] for s in plan['source_files']}
    width_cpu=cpu['widths'][arm]
    return dict(model_sha256=sources['dual_solver_mgn.py'],trainer_sha256=plan['trainer']['sha256'],
                loss_module_sha256=sources['a1_allocation_loss.py'],capacity_contract_sha256=sources['capacity_contract.py'],
                plan_sha256=plan['engineering_plans' if engineering else 'training_plans'][arm]['sha256'],
                selection_sha256=training_plan['selection']['sha256'],bundle_manifest_sha256=training_plan['bundle_manifest_sha256'],
                mask_manifest_sha256=plan['mask_manifest']['sha256'],initial_tensor_state_sha256=width_cpu['initial_state_sha256_by_seed'][str(seed)],
                data_order_LR_trace_sha256=width_cpu['expected_data_order_LR_trace_sha256']['2957' if engineering else '88565'][str(seed)])


def validate_training_result(result,entry,expected,checkpoint_path):
    engineering=entry['kind']=='engineering';updates=2957 if engineering else 88565
    require(result['status']==('SMOKE_COMPLETED' if engineering else 'COMPLETED'),'training result incomplete')
    require(result['training_updates']==updates and result['arm']==entry['arm'] and result['seed']==entry['seed']
            and type(result['alpha']) in (float,int) and result['alpha']==0,'training identity/update differs')
    require(result['width']==entry['width'] and result['depth']==3 and result['execution_device']=='cuda','training width/depth/device differs')
    require(result['torch']=='2.5.1+cu124' and result['numpy']=='1.26.4','training runtime differs from saved controls')
    require((result['parameters'],result['field_path_parameters'])==PARAMETERS[entry['arm']],'training parameter count differs')
    require(result['phase_encoding']=='mechanical' and result['data_arm']=='A45' and result['experiment_id']=='W128_PHASE45_EQUAL_UPDATE_20261030' and result['old_checkpoint_reused'] is False,'training encoding/reuse differs')
    require(result['train_steps']==list(range(45)) and result['train_graphs']==14760
            and result['train_phase9_diagnostic_graphs']==2952,'phase45 training support differs')
    require(result['new_FEA']==result['new_prior_solves']==result['original_val_test_used']==0,'training source scope differs')
    require(result['GPU_smoke']=='PASS' and finite_positive(result['peak_torch_allocated_bytes'],'torch peak',True)<=10*1024**3,'training memory/smoke gate failed')
    for key,digest in expected.items():require(valid_sha(digest) and result.get(key)==digest,'training metadata differs: '+key)
    require(result['max_updates']==(2957 if engineering else None),'engineering/full split differs')
    require([m['epoch'] for m in result['metrics']]==([0,1] if engineering else [0,1,5,10,20,30]),'training assessment schedule incomplete')
    require(result['metrics'][-1]['training_updates']==updates and result['metrics'][-1]['development']['all']['snapshots']==288,'last development assessment incomplete')
    for metric in result['metrics']:
        finite_positive(metric['assessment_wall_seconds'],'assessment wall')
        finite_positive(metric['development']['all']['vector_nRMSE_percent'],'development arithmetic',True)
    checkpoint=file_pin(checkpoint_path)
    require(checkpoint['sha256']==result['checkpoint_sha256'],'final checkpoint byte SHA differs')
    return checkpoint


def profile_projection(profiles,remaining_seconds):
    require(set(profiles)=={'W128'},'W128 phase45 engineering profile required')
    projections={}
    for arm,(result,guard_wall) in profiles.items():
        require(result['status']=='SMOKE_COMPLETED' and result['training_updates']==2957,'engineering profile incomplete')
        require(result['arm']==arm and result['width']==ARMS[arm] and result['execution_device']=='cuda','profile architecture/device differs')
        elapsed=finite_positive(result['elapsed_seconds'],'profile result wall')
        assessment=finite_positive(result['assessment_wall_seconds_total'],'assessment total')
        require(math.isclose(assessment,sum(m['assessment_wall_seconds'] for m in result['metrics']),rel_tol=1e-9,abs_tol=1e-8),'assessment timings do not sum')
        train_time=elapsed-assessment
        require(train_time>0 and finite_positive(guard_wall,'guard wall')>=elapsed-.05,'invalid guard/result timing')
        startup=max(0.,guard_wall-elapsed)
        projected=1.25*(startup+train_time/2957*88565+6*max(m['assessment_wall_seconds'] for m in result['metrics']))
        require(projected<=10800,'projected full training exceeds10800s')
        projections[arm]=dict(projected_train_seconds=projected,profile_guard_seconds=guard_wall,startup_seconds=startup,
                              profile_update_seconds=train_time,assessment_seconds=assessment,safety_factor=1.25)
    expected=3*projections['W128']['projected_train_seconds']+3*1800+300
    require(expected<=finite_positive(remaining_seconds,'remaining budget'),'projected W128 workload exceeds remaining deadline')
    return dict(status='PASS_RESOURCE_PROJECTION_ONLY',arms=projections,projected_remaining_seconds=expected,
                available_remaining_seconds=remaining_seconds,evaluation_reserve_seconds=5400,collection_reserve_seconds=300)


def make_eval_job(template,entry,train_result,checkpoint,train_path,expected,deadline):
    job=copy.deepcopy(template)
    job.update(id=entry['run_id'],arm=entry['arm'],width=entry['width'],depth=3,seed=entry['seed'],alpha=0.,checkpoint=checkpoint,
               original_train=file_pin(train_path),metadata_hashes=dict(expected),deadline_UTC=deadline)
    require(job['release_status']=='RELEASED_FROZEN_EVALUATION' and job['max_seconds']==1800 and job['workers']==1 and job['retry_limit']==0,'eval template limits differ')
    require(job['training_updates']==job['new_FEA']==job['new_prior_solves']==0 and job['phase_encoding']=='mechanical','eval template scope differs')
    require(job['selection']['sha256']==expected['selection_sha256'] and job['training_mask_manifest']['sha256']==expected['mask_manifest_sha256'],'eval template input binding differs')
    require(train_result['checkpoint_sha256']==checkpoint['sha256'],'training checkpoint differs')
    return job


def admit_inputs(plan,phase):
    for source in plan['source_files']:pin(source)
    for key in ('guard','trainer','evaluator'):pin(plan[key])
    masks=pinned_json(plan['mask_manifest']);cpu=pinned_json(plan['cpu_admission'])
    require(cpu['status']=='PASS' and masks['status']=='ADMITTED' and len(masks['records'])==328 and len(masks['development_records'])==32,'CPU/native-mask admission incomplete')
    require(cpu['torch']=='2.5.1+cu124' and cpu['numpy']=='1.26.4','CPU runtime differs from saved controls')
    require(set(cpu['widths'])==set(ARMS),'CPU width set differs')
    require(cpu['data_arm']=='A45' and cpu['experiment_id']=='W128_PHASE45_EQUAL_UPDATE_20261030','CPU experiment identity differs')
    bridge=cpu['reference_W64']['bridge']
    require(bridge['status']=='PASS_BASELINE_BRIDGE_CPU_ONLY' and bridge['synthetic_W64']['status']=='PASS_BIT_EXACT_SYNTHETIC_CPU','W64 CPU baseline bridge incomplete')
    require(all(bridge[key] is True for key in ('trainer_full_module_AST_equivalent','model_source_identical','numerical_function_AST_equivalent',
                'outer_main_AST_equivalent','initial_states_match_historical','all45_order_LR_traces_match_historical')),'W64 CPU baseline bridge gate failed')
    require(cpu['reference_W64']['parameters']==273478 and cpu['reference_W64']['field_path_parameters']==118530,'reference W64 capacity differs')
    sources={Path(s['path']).name:s['sha256'] for s in plan['source_files']}
    require({Path(s['path']).name:s['sha256'] for s in cpu['source_files']}==sources,'CPU source pins differ')
    plans={}
    for arm,width in ARMS.items():
        admission=cpu['widths'][arm];initials=admission['initial_state_sha256_by_seed']
        require(set(initials)=={'17','29','43'} and all(valid_sha(v) for v in initials.values()),'CPU initial pins incomplete')
        for count,seeds in (('2957',{'17'}),('88565',{'17','29','43'})):
            traces=admission['expected_data_order_LR_trace_sha256'][count]
            require(seeds<=set(traces)<={'17','29','43'} and all(valid_sha(v) for v in traces.values()),'CPU trace pins incomplete')
        for other in ARMS:
            require(admission['expected_data_order_LR_trace_sha256']==cpu['widths'][other]['expected_data_order_LR_trace_sha256'],'width-only order/LR must match')
        require(cpu['reference_W64']['expected_data_order_LR_trace_sha256']==admission['expected_data_order_LR_trace_sha256'],'width-only order/LR differs')
        spec=plan['engineering_plans' if phase=='engineering' else 'training_plans'][arm]
        current=pinned_json(spec);plans[arm]=current
        require(current['status']==('RELEASED_ENGINEERING_ONLY' if phase=='engineering' else 'RELEASED_FIXED_WIDTH128_RUNS'),'training release state differs')
        require(current['training']['width']==width and current['training']['depth']==3 and current['training']['device']=='cuda','explicit width/depth/device differs')
        require(current['data_arm']=='A45' and current['experiment_id']=='W128_PHASE45_EQUAL_UPDATE_20261030'
                and current['train_steps']==list(range(45)) and current['seen_steps']==list(range(0,45,5)),'phase45 experiment identity/support differs')
        require(current['arms']=={'W128':{'alpha':0.,'width':128}} and current['fixed_seeds']==[17,29,43],'arm/seed contract differs')
        require(current['mask_manifest_sha256']==plan['mask_manifest']['sha256'] and current['selection']['sha256']==masks['selection_sha256']
                and current['bundle_manifest_sha256']==masks['bundle_manifest_sha256'],'training plan/input pins differ')
        require(current['guard_deadline_UTC']==plan['deadline_UTC'] and current['initial_state_sha256_by_seed']==initials,'training deadline/initial pins differ')
        require(current['source_files']==plan['source_files'],'training source declarations differ')

        for key in ('selection','pilot_manifest'):pin(current[key])
    template=None
    if phase=='scientific':
        template=pinned_json(plan['evaluation_template'])
        require(template['source_files']==plan['source_files'] and template['deadline_UTC']==plan['deadline_UTC'],'evaluator source/deadline differs')
        for key in ('selection','geometry_manifest','training_mask_manifest'):pin(template[key])
    return plans,masks,cpu,template


def validate_scientific_release(release,plan,plan_sha,receipt_path):
    require(release['status']=='RELEASED_FIXED_WIDTH128_RUNS' and release['chain_plan_sha256']==plan_sha,'separate scientific release missing/mismatched')
    require(release['cpu_admission_sha256']==plan['cpu_admission']['sha256'],'release CPU admission differs')
    require(release['training_plan_sha256_by_arm']=={'W128':plan['training_plans']['W128']['sha256']},'release training plan differs')
    require(Path(release['engineering_receipt']['path']).resolve()==Path(receipt_path).resolve(),'engineering receipt path differs')
    receipt=pinned_json(release['engineering_receipt'])
    require(receipt['status']=='ENGINEERING_COMPLETED_AWAITING_RELEASE' and receipt['chain_plan_sha256']==plan_sha
            and receipt['completed_engineering_runs']==1 and receipt['engineering_updates']==2957,'engineering not complete')
    if release.get('baseline_bridge'):
        bridge=pinned_json(release['baseline_bridge'])
        require(bridge['status']=='PASS_W64_PHASE45_BASELINE_BRIDGE' and bridge['cpu_admission_sha256']==plan['cpu_admission']['sha256'],
                'saved W64 baseline bridge differs from current CPU admission')
        require(set(bridge['control_results_by_seed'])=={'17','29','43'}
                and all(bridge['control_results_by_seed'][seed]['sha256']==release['phase45_controls_by_seed'][seed]['sha256'] for seed in ('17','29','43')),
                'saved W64 baseline bridge control pins differ')
    cpu=pinned_json(plan['cpu_admission']);old=pinned_json(release['phase45_cpu_admission'])
    require(old['status']=='PASS','phase45 CPU receipt incomplete')
    require(old['widths']['W64']['initial_state_sha256_by_seed']==cpu['reference_W64']['initial_state_sha256_by_seed'],'reference W64 initial states differ')
    require(old['widths']['W64']['expected_data_order_LR_trace_sha256']==cpu['reference_W64']['expected_data_order_LR_trace_sha256']
            ==cpu['widths']['W128']['expected_data_order_LR_trace_sha256'],'reference/candidate data order differs')
    require(set(release['phase45_controls_by_seed'])=={'17','29','43'},'phase45 controls incomplete')
    require(set(release['phase45_control_train_results_by_seed'])=={'17','29','43'},'phase45 training controls incomplete')
    for seed,spec in release['phase45_controls_by_seed'].items():
        control=pinned_json(spec)
        require(control['status']=='COMPLETED' and control['arm']=='W64' and control['seed']==int(seed)
                and control['width']==64 and control['depth']==3 and control['graphs_processed']==1440 and control['data_arm']=='A45'
                and control['experiment_id']=='W64_PHASE45_EQUAL_UPDATE_20261030' and control['training_updates']==0,'phase45 control identity/coverage differs')
        meta=control['metadata_hashes']
        require(meta['initial_tensor_state_sha256']==old['widths']['W64']['initial_state_sha256_by_seed'][seed]
                and meta['data_order_LR_trace_sha256']==old['widths']['W64']['expected_data_order_LR_trace_sha256']['88565'][seed],'phase45 CPU binding differs')
        train=pinned_json(release['phase45_control_train_results_by_seed'][seed])
        require(train['status']=='COMPLETED' and train['arm']=='W64' and train['seed']==int(seed)
                and train['width']==64 and train['depth']==3 and train['data_arm']=='A45' and train['alpha']==0.
                and train['experiment_id']=='W64_PHASE45_EQUAL_UPDATE_20261030' and train['training_updates']==88565 and train['train_steps']==list(range(45))
                and train['parameters']==273478 and train['field_path_parameters']==118530
                and train['checkpoint_sha256']==control['checkpoint_sha256'],'phase45 train identity/checkpoint differs')
        require(train['initial_tensor_state_sha256']==meta['initial_tensor_state_sha256']
                and train['data_order_LR_trace_sha256']==meta['data_order_LR_trace_sha256'],'phase45 train CPU binding differs')
    return receipt,old


def run_guard(plan,plan_sha,plan_path,phase='engineering',release_spec=None):
    require(phase in ('engineering','scientific'),'unknown phase')
    deadline=validate_plan(plan)
    require(Path(sys.executable).resolve()==Path(plan['python_executable']).resolve(),'wrong pinned Python')
    require(Path(__file__).resolve()==Path(plan['guard']['path']).resolve(),'guard loaded from wrong path')
    require(sha(plan_path)==plan_sha and utc()<deadline,'chain plan changed/expired')
    plans,masks,cpu,template=admit_inputs(plan,phase)
    root=Path(plan['runs_root']);stage=Path(plan['stage_root'])
    import psutil
    owner=owner_identity(psutil.Process(os.getpid()))
    lock=stage/'WIDTH128_GUARD_LOCK.json'
    # Exclusive creation prevents overlapping invocations. A stale lock requires
    # review; PID reuse never authorizes deleting it or terminating its new owner.
    with lock.open('x',encoding='utf-8') as stream:json.dump(owner|{'chain_plan_sha256':plan_sha,'phase':phase},stream)
    current=None;active_owners={};status=None;release=None
    try:
        require(not gpu_compute_pids(),'foreign GPU process present; no job launched')
        require(shutil.disk_usage(stage).free>=plan['disk_free_min_bytes'],'disk admission failed')
        started=utc();profiles={}
        if phase=='engineering':
            require(not root.exists(),'runs already exist; retry0')
            root.mkdir(exist_ok=False)
            origin=started
        else:
            require(release_spec is not None,'scientific phase requires separate release file and SHA')
            release=pinned_json(release_spec)
            prior,parity=validate_scientific_release(release,plan,plan_sha,root/'ENGINEERING_RECEIPT.json')
            origin=dt.datetime.fromisoformat(prior['started_UTC'])
            require(origin.tzinfo is not None and origin<=started,'invalid engineering origin')
            for row in prior['jobs']:
                require(row['kind']=='engineering' and row['status']=='COMPLETED','incomplete engineering row')
                result=pinned_json(row['result'])
                expected=expected_hashes(plan,pinned_json(plan['engineering_plans'][row['arm']]),cpu,row['arm'],17,True)
                validate_training_result(result,row,expected,Path(row['out'])/'last.pt')
                profiles[row['arm']]=(result,row['elapsed_seconds'])
            require(not (root/'SCIENTIFIC_STATUS.json').exists(),'scientific stage already attempted; retry0')
        start=time.monotonic();prior_elapsed=(started-origin).total_seconds();engineering_start=start
        status=dict(status='RUNNING',phase=phase,chain_plan_sha256=plan_sha,owner=owner,started_UTC=origin.isoformat(),
                    phase_started_UTC=started.isoformat(),new_FEA=0,new_prior_solves=0,automatic_retries=0,
                    jobs=job_rows(root,phase),full_training_updates=0,engineering_updates=0,evaluation_graphs=0)
        if release:status['scientific_release']=release_spec
        def elapsed():return prior_elapsed+time.monotonic()-start
        def remaining():return min(28800-elapsed(),(deadline-utc()).total_seconds())
        def save():
            status['elapsed_seconds']=elapsed()
            atomic_json(root/('ENGINEERING_STATUS.json' if phase=='engineering' else 'SCIENTIFIC_STATUS.json'),status)
        def check(entry=None,worker_start=None):
            paths=[Path(p) for p in plan['stop_files']]
            if entry:paths += [Path(entry['out'])/'STOP',Path(entry['out']).parent/'STOP']
            reason=limit_reason(utc(),deadline,elapsed(),None if worker_start is None else time.monotonic()-worker_start,
                                900 if entry and entry['kind']=='engineering' else (10800 if entry and entry['kind']=='train' else 1800),
                                time.monotonic()-engineering_start if entry and entry['kind']=='engineering' else None,
                                any(p.exists() for p in paths))
            if reason:raise InterruptedError(reason)
        def verify_immutable():
            require(sha(plan_path)==plan_sha,'chain plan changed during execution')
            admit_inputs(plan,phase)
            if release:
                require(pinned_json(release_spec)==release,'scientific release changed')
                validate_scientific_release(release,plan,plan_sha,root/'ENGINEERING_RECEIPT.json')
        save();check()
        if phase=='scientific':
            projection=profile_projection(profiles,remaining())
            status['engineering_projection']=projection
            atomic_json(root/'SCIENTIFIC_PROJECTION_RECEIPT.json',projection);save()
        completed_trains={}
        for entry in status['jobs']:
            check(entry);verify_immutable()
            require(shutil.disk_usage(root).free>=plan['disk_free_min_bytes'],'disk bound failed before job')
            require(not gpu_compute_pids(),'foreign GPU process present; no foreign process will be stopped')
            out=Path(entry['out']);require(not out.exists(),'job output exists; retry0')
            out.parent.mkdir(parents=True,exist_ok=True)
            if phase=='scientific' and entry['kind']=='train':
                unfinished=sum(row['kind']=='train' and row['status']=='NOT_STARTED' for row in status['jobs'])
                unevaluated=sum(row['kind']=='eval' and row['status']=='NOT_STARTED' for row in status['jobs'])
                require(unfinished*status['engineering_projection']['arms']['W128']['projected_train_seconds']+unevaluated*1800+300<=remaining(),'remaining projected budget exhausted')
            expected=expected_hashes(plan,plans[entry['arm']],cpu,entry['arm'],entry['seed'],entry['kind']=='engineering')
            command=[plan['python_executable'],'-B','-u']
            if entry['kind'] in ('engineering','train'):
                spec=plan['engineering_plans' if phase=='engineering' else 'training_plans'][entry['arm']]
                command += [plan['trainer']['path'],'--plan',spec['path'],'--plan-sha256',spec['sha256'],
                            '--bundle',plan['train_bundle_dir'],'--masks',plan['mask_manifest']['path'],'--masks-sha256',plan['mask_manifest']['sha256'],
                            '--out',str(out),'--arm',entry['arm'],'--width',str(entry['width']),'--seed',str(entry['seed']),'--prefetch','8','--device','cuda']
                if phase=='engineering':command += ['--max-updates','2957']
                entry['plan_sha256']=spec['sha256']
            else:
                train_result,checkpoint,train_path=completed_trains[entry['run_id']]
                job=make_eval_job(template,entry,train_result,checkpoint,train_path,expected,plan['deadline_UTC'])
                job_path=out.parent/'eval_job.json';require(not job_path.exists(),'evaluation job exists; retry0')
                atomic_json(job_path,job);entry['job_sha256']=sha(job_path)
                command += [plan['evaluator']['path'],'--job',str(job_path),'--job-sha256',entry['job_sha256'],'--out',str(out)]
            entry.update(status='STARTING',started_UTC=utc().isoformat());save()
            worker_start=time.monotonic();next_deep_check=worker_start
            active_owners={}
            with (root/(entry['id']+'.log')).open('xb') as log:
                current=subprocess.Popen(command,cwd=stage,env=worker_environment(os.environ,plan['pythonpath']),stdout=log,stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                # Keep the handle even if the worker exits before psutil admission;
                # no PID-only fallback is allowed to terminate an unrelated process.
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
                        live_result=out/'RESULT.json'
                        if live_result.is_file():
                            observed=json.loads(live_result.read_text(encoding='utf-8'))
                            if 'peak_torch_allocated_bytes' in observed:require(finite_positive(observed['peak_torch_allocated_bytes'],'live torch peak',True)<=10*1024**3,'live torch memory limit')
                        next_deep_check=time.monotonic()+10
                    time.sleep(.5)
                returncode=current.wait()
            check(entry,worker_start)
            require(not [o for o in active_owners.values() if identity_matches(psutil,o)],'worker left live descendants; only recorded owners will stop')
            current=None;active_owners={}
            entry.update(returncode=returncode,elapsed_seconds=time.monotonic()-worker_start,completed_UTC=utc().isoformat())
            result_path=out/'RESULT.json';require(result_path.is_file(),'worker result missing')
            result=json.loads(result_path.read_text(encoding='utf-8'))
            entry.update(result=file_pin(result_path),worker_status=result.get('status'))
            require(returncode==0,'worker nonzero exit')
            if entry['kind'] in ('engineering','train'):
                checkpoint=validate_training_result(result,entry,expected,out/'last.pt')
                entry.update(checkpoint_sha256=checkpoint['sha256'],training_updates=result['training_updates'])
                if phase=='engineering':
                    profiles[entry['arm']]=(result,entry['elapsed_seconds']);status['engineering_updates']+=2957
                else:
                    completed_trains[entry['run_id']]=(result,checkpoint,result_path);status['full_training_updates']+=88565
            else:
                require(result['status']=='COMPLETED' and result['graphs_processed']==1440 and result['training_updates']==0
                        and result['job_sha256']==entry['job_sha256'] and result['checkpoint_sha256']==completed_trains[entry['run_id']][1]['sha256']
                        and result['reproduction_gate']['status']=='PASS_ARITHMETIC_ONLY','frozen evaluation incomplete/mismatched')
                require(result['arm']==entry['arm'] and result['seed']==entry['seed'] and result['width']==entry['width'] and result['depth']==3,'evaluation identity differs')
                require(result['data_arm']=='A45' and result['experiment_id']=='W128_PHASE45_EQUAL_UPDATE_20261030','evaluation phase45 identity differs')
                entry.update(checkpoint_sha256=result['checkpoint_sha256'],graphs_processed=1440);status['evaluation_graphs']+=1440
            entry['status']='COMPLETED';atomic_json(root/(entry['id']+'_GUARD_RECEIPT.json'),entry);save()
        check()
        if phase=='engineering':
            require(status['engineering_updates']==2957,'engineering total incomplete')
            status['engineering_projection']=profile_projection(profiles,remaining())
            status['status']='ENGINEERING_COMPLETED_AWAITING_RELEASE'
        else:
            require(status['full_training_updates']==265695 and status['evaluation_graphs']==4320,'scientific work totals incomplete')
            status.update(status='COMPLETED',scientific_verdict='PENDING_PREREGISTERED_JSON_JUDGMENT')
    except BaseException as exc:
        if status is not None:
            status.update(status='STOPPED' if isinstance(exc,InterruptedError) else 'INCOMPLETE_NO_CLAIM',error=repr(exc))
            for row in status['jobs']:
                if row['status'] in ('STARTING','RUNNING'):row.update(status='INCOMPLETE_NO_CLAIM',error=repr(exc))
        raise
    finally:
        if active_owners:terminate_owned(psutil,list(active_owners.values()))
        if current is not None:
            try:current.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if status is not None:status['owned_process_exit_unconfirmed']=True
        if status is not None:
            status.update(completed_UTC=utc().isoformat(),completed_train_runs=sum(r['kind']=='train' and r['status']=='COMPLETED' for r in status['jobs']),
                          completed_eval_runs=sum(r['kind']=='eval' and r['status']=='COMPLETED' for r in status['jobs']),
                          completed_engineering_runs=sum(r['kind']=='engineering' and r['status']=='COMPLETED' for r in status['jobs']))
            for row in status['jobs']:
                if 'started_UTC' in row:atomic_json(root/(row['id']+'_GUARD_RECEIPT.json'),row)
            save();atomic_json(root/('ENGINEERING_RECEIPT.json' if phase=='engineering' else 'SCIENTIFIC_RECEIPT.json'),status)
        # Never clear another identity's lock.
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8')).get('pid')==owner['pid'] and json.loads(lock.read_text(encoding='utf-8')).get('create_time')==owner['create_time']:
            lock.unlink()
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--phase',choices=('engineering','scientific'),required=True)
    parser.add_argument('--scientific-release',type=Path);parser.add_argument('--scientific-release-sha256')
    args=parser.parse_args()
    require(args.phase!='scientific' or (args.scientific_release is not None and valid_sha(args.scientific_release_sha256)),'scientific release path and SHA required')
    require(args.phase!='engineering' or (args.scientific_release is None and args.scientific_release_sha256 is None),'engineering cannot accept a scientific release')
    plan=pinned_json({'path':str(args.plan),'sha256':args.plan_sha256})
    release=None if args.scientific_release is None else {'path':str(args.scientific_release),'sha256':args.scientific_release_sha256}
    run_guard(plan,args.plan_sha256,args.plan,args.phase,release)


if __name__=='__main__':main()
