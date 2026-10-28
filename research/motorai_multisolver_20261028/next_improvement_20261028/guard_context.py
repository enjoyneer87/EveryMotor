"""One owned C/G stage: gates, then fixed six runs in two waves; zero retries."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import threading
import time
import traceback
import psutil

import guard_l12 as owned

ROOT=Path(__file__).parent


def main():
    plan=json.loads((ROOT/'CONTEXT_CG_PLAN.json').read_text())
    cfg=json.loads((ROOT/'RUN_PRIVATE.json').read_text())
    deployment=json.loads((ROOT/'DEPLOY_RECEIPT.json').read_text())
    assert not (ROOT/'CONTEXT_CHAIN_STATUS.json').exists(), 'No duplicate stage or retry'
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in deployment['code_hashes'].items())
    deadline=datetime.datetime.fromisoformat(plan['execution']['guard_deadline_UTC'])
    assert owned.now()<deadline and plan['phase_encoding']=='mechanical'
    assert plan['fixed_seeds']==[17,29,43] and plan['execution']['full_runs']==6
    assert hashlib.sha256((ROOT/'components/COMPONENT_MANIFEST.json').read_bytes()).hexdigest()==deployment['component_manifest_sha256']
    owner={'pid':psutil.Process().pid,'process_started_epoch':psutil.Process().create_time(),
           'started_UTC':owned.now().isoformat(),'deadline_UTC':deadline.isoformat(),'automatic_retries':0}
    owned.write(ROOT/'GUARD_OWNER.json',owner)
    # No unrelated GPU user is interrupted; resource contention blocks this stage.
    free,apps=owned.gpu_free()
    memory_free_MiB=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    import shutil
    free_disk_bytes=shutil.disk_usage(ROOT).free
    status={'status':'RUNNING','started_UTC':owned.now().isoformat(),'steps':[],
            'plan_sha256':hashlib.sha256((ROOT/'CONTEXT_CG_PLAN.json').read_bytes()).hexdigest(),
            'scope':'C/G context range; not repaired native FE topology','automatic_retries':0}
    lock=threading.Lock()
    def save():owned.write(ROOT/'CONTEXT_CHAIN_STATUS.json',status)
    def record(name,receipt,state):
        with lock:
            status['steps'].append({'step':name,'status':state,'returncode':receipt.get('returncode'),
                                    'stop_reason':receipt.get('stop_reason'),'elapsed_seconds':receipt.get('elapsed_seconds')})
            save()
        return state in ('COMPLETED','SMOKE_COMPLETED','PASS_ENGINEERING_ONLY')
    save()
    if not free:
        status.update(status='BLOCKED_GPU_BUSY',reason=apps);save();return
    if memory_free_MiB<32768 or free_disk_bytes<5*1024**3:
        status.update(status='BLOCKED_RESOURCES',memory_free_MiB=memory_free_MiB,disk_free_bytes=free_disk_bytes);save();return
    py=cfg['python_executable']
    try:
        engineering_deadline=min(deadline,owned.now()+datetime.timedelta(seconds=900))
        receipt=owned.run('eval_loader_CPU',[py,str(ROOT/'test_context_checkpoint.py')],120,engineering_deadline,ROOT/'gates/eval_loader')
        if receipt['returncode']!=0:
            record('eval_loader_CPU',receipt,'FAILED');status['status']='FAILED_EVALUATOR_LOADER';return
        # CPU restoration is checked before any new GPU operation. It does not
        # replace the frozen seen9 native-data arithmetic gate per trained run.
        owned.write(ROOT/'EVAL_LOADER_GATE.json',{'status':'PASS','returncode':0,
                     'source_sha256':hashlib.sha256((ROOT/'context_checkpoint.py').read_bytes()).hexdigest()})
        receipt=owned.run('overfit',[py,str(ROOT/'overfit_context.py')],900,engineering_deadline,ROOT/'gates/overfit')
        if not record('overfit',receipt,owned.status_of(ROOT/'OVERFIT_RESULT.json') if receipt['returncode']==0 else 'FAILED'):
            status['status']='FAILED_OVERFIT';return
        profile=[]
        for arm in ('C','G'):
            if owned.now()>=engineering_deadline:
                status['status']='HOLD_ENGINEERING_TIME_CAP';return
            rid='profile_'+arm;out=ROOT/'runs'/rid/'train'
            cmd=[py,str(ROOT/'train_context.py'),'--bundle',cfg['bundle'],'--components',str(ROOT/'components'),
                 '--out',str(out),'--arm',arm,'--seed','17','--phase-encoding',plan['phase_encoding'],'--prefetch','8','--max-updates','2957']
            receipt=owned.run('train',cmd,900,engineering_deadline,ROOT/'runs'/rid)
            if not record(rid,receipt,owned.status_of(out/'RESULT.json') if receipt['returncode']==0 else 'FAILED'):
                status['status']='FAILED_PROFILE';return
            profile.append(json.loads((out/'RESULT.json').read_text()))
        # Measured single-worker update time with a conservative multiplier for
        # three workers sharing one device. No learning/projection weights reused.
        per_update=max(r['optimizer_update_wall_seconds']/r['training_updates'] for r in profile)
        projected_each=per_update*88565*3
        projected_remaining=projected_each*2+plan['evaluation']['wall_seconds_max']*2+180
        remaining=(deadline-owned.now()).total_seconds()
        gate={'per_update_single_worker_seconds':per_update,'concurrency_multiplier':3,
              'projected_training_seconds_each':projected_each,'projected_remaining_stage_seconds':projected_remaining,
              'remaining_guard_seconds':remaining,'pass':projected_each<=7200 and projected_remaining<=remaining}
        owned.write(ROOT/'RESOURCE_PROJECTION.json',gate)
        if not gate['pass']:
            status['status']='HOLD_RESOURCE_PROJECTION';return
        def train_then_evaluate(rid):
            arm,seed=rid.split('_s');out=ROOT/'runs'/rid/'train'
            receipt=owned.run('train',[py,str(ROOT/'train_context.py'),'--bundle',cfg['bundle'],'--components',str(ROOT/'components'),
                '--out',str(out),'--arm',arm,'--seed',seed,'--phase-encoding',plan['phase_encoding'],'--prefetch','8'],7200,deadline,ROOT/'runs'/rid)
            if not record(rid+':train',receipt,owned.status_of(out/'RESULT.json') if receipt['returncode']==0 else 'FAILED'):return
            checkpoint=out/'last.pt';expected=hashlib.sha256(checkpoint.read_bytes()).hexdigest();ev=ROOT/'runs'/rid/'eval'
            receipt=owned.run('evaluate',[py,str(ROOT/'evaluate_context.py'),'--bundle',cfg['dev_bundle'],'--components',str(ROOT/'components'),
                '--checkpoint',str(checkpoint),'--expected-sha256',expected,'--out',str(ev),'--anchor-result',str(out/'RESULT.json')],1800,deadline,ROOT/'runs'/rid)
            record(rid+':evaluate',receipt,owned.status_of(ev/'RESULT.json') if receipt['returncode']==0 else 'FAILED')
        def thread_entry(rid):
            try:
                train_then_evaluate(rid)
            except Exception:
                (ROOT/'runs'/rid).mkdir(exist_ok=True,parents=True)
                (ROOT/'runs'/rid/'worker_error.txt').write_text(traceback.format_exc())
                record(rid+':worker',{},'FAILED_WORKER')
        for wave in (('C_s17','G_s17','C_s29'),('G_s29','C_s43','G_s43')):
            if owned.stop_requested() or any(s['status'] not in ('COMPLETED','SMOKE_COMPLETED','PASS_ENGINEERING_ONLY') for s in status['steps']):
                status['status']='STOPPED_BEFORE_NEXT_WAVE';return
            threads=[threading.Thread(target=thread_entry,args=(rid,)) for rid in wave]
            for thread in threads:thread.start();time.sleep(3)
            for thread in threads:thread.join()
        status['status']='COMPLETED' if len(status['steps'])==15 and all(s['status'] in ('COMPLETED','SMOKE_COMPLETED','PASS_ENGINEERING_ONLY') for s in status['steps']) else 'INCOMPLETE_NO_CLAIM'
    except Exception:
        status['status']='GUARD_ERROR';(ROOT/'guard_error.txt').write_text(traceback.format_exc())
    finally:
        status['finished_UTC']=owned.now().isoformat()
        save()
        owned.write(ROOT/'GUARD_RECEIPT.json',{**owner,'finished_UTC':status['finished_UTC'],'status':status['status']})


if __name__=='__main__':
    main()
