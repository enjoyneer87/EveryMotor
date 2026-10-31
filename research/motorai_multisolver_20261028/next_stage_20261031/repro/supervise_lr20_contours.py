"""One retained-handle inference child, bounded by the immutable campaign clock."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import shutil
import sys
import time

BASE=Path('C:/work/EveryMotor_NextLR20_20261031_01a18cac')
CHAIN_SHA='3fc6975ade9d535728a770210f72f1e413032ea7fbe1943e7540e1fb7558ca32'


def sha(p):
    p=Path(p);assert p.stat().st_size<=2*1024**2
    with p.open('rb') as f:raw=f.read(2*1024**2+1)
    assert len(raw)<=2*1024**2
    return hashlib.sha256(raw).hexdigest()


def main():
    import psutil
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);a=p.parse_args()
    assert a.release.resolve()==(BASE/'CONTOUR_LR20_RELEASE.json').resolve() and sha(a.release)==a.release_sha256
    release=json.loads(a.release.read_text(encoding='utf-8-sig'))
    assert release['status']=='RELEASED_FINAL_LR20_SEED17_FIXED12_ONLY'
    assert release['runner_max_seconds']==360 and release['supervisor_max_seconds']==420 and release['contour_transport_max_seconds']==600
    assert release['retry']==0 and release['GPU_workers']==1 and release['new_optimizer_updates']==0
    assert release['supervisor_sha256']==sha(Path(__file__))
    chain_path=BASE/'CHAIN_LR20.json';assert sha(chain_path)==release['chain']['sha256']==CHAIN_SHA
    chain=json.loads(chain_path.read_text(encoding='utf-8-sig'))
    runner=(BASE/'postprocess/infer_lr20_contours.py').resolve();assert sha(runner)==release['runner_sha256']
    child_guard=BASE/'postprocess/contour_child_guard.py';assert sha(child_guard)==release['child_guard_sha256']
    sys.path.insert(0,str(child_guard.parent))
    from contour_child_guard import run_owned_child
    clock_path=Path(chain['stage_clock']['path']);assert clock_path.resolve()==(BASE/'STAGE_CLOCK.json').resolve()
    assert sha(clock_path)==chain['stage_clock']['sha256']
    clock=json.loads(clock_path.read_text())
    source=BASE/'src/stage_clock.py';expected=next(x['sha256'] for x in chain['source_files'] if Path(x['path']).resolve()==source.resolve())
    assert sha(source)==expected;sys.path.insert(0,str(BASE/'src'))
    from stage_clock import elapsed_stage_seconds
    owner=dict(pid=os.getpid(),create_time=psutil.Process().create_time())
    started=time.monotonic()
    receipt=dict(status='STARTING',owner=owner,release_sha256=a.release_sha256,started_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),retry=0,child_exit_verified=True)
    def check(minimum_remaining=0):
        now=dt.datetime.now(dt.timezone.utc)
        spent=elapsed_stage_seconds(clock,time.monotonic(),now,platform.node(),psutil.boot_time())
        assert min(28800-spent,(dt.datetime.fromisoformat(chain['deadline_UTC'])-now).total_seconds())>minimum_remaining
        assert time.monotonic()-started<420
        assert shutil.disk_usage(BASE).free>=10*1024**3
        assert all(not Path(x).exists() for x in chain['stop_files']) and not (BASE/'CONTOUR_STOP').exists()
    def pin(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    def admit():
        check(600)
        for name in ('scientific_receipt','bootstrap_receipt'):
            p=release[name];path=Path(p['path']);assert sha(path)==p['sha256'] and path.stat().st_size==p['bytes']
            saved=json.loads(path.read_text());assert saved['status']=='COMPLETED'
            identities=[saved['owner']]
            if name=='scientific_receipt':
                assert len(saved['jobs'])==6 and {(j['seed'],j['kind']) for j in saved['jobs']}=={(s,k) for s in (17,29,43) for k in ('train','eval')}
                assert all(j['status']=='COMPLETED' for j in saved['jobs']);identities.extend(j['owner'] for j in saved['jobs'])
            else:identities.append(saved['guard'])
            for o in identities:
                try:assert psutil.Process(o['pid']).create_time()!=o['create_time'],'training identity still live'
                except psutil.NoSuchProcess:pass
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5).strip();assert not gpu
        assert not (BASE/'contour_LR20_R1').exists()
    def identity_matches(o):
        try:return psutil.Process(o['pid']).create_time()==o['create_time']
        except psutil.NoSuchProcess:return False
    with (BASE/'CONTOUR_SUPERVISOR_OWNER.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    try:
        with (BASE/'contour_stdout.log').open('x',encoding='utf-8') as out,(BASE/'contour_stderr.log').open('x',encoding='utf-8') as err:
            guarded=run_owned_child([sys.executable,'-B','-u',str(runner),'--release',str(a.release),'--release-sha256',a.release_sha256],admit=admit,check=check,
                                    identify=lambda child:dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time()),identity_matches=identity_matches,
                                    cwd=BASE,stdout=out,stderr=err,runner_seconds=release['runner_max_seconds'])
            receipt.update(guarded)
        assert guarded['status']=='CHILD_COMPLETED' and guarded['child_exit_verified'],guarded.get('error')
        result_path=BASE/'contour_LR20_R1/INFERENCE_RECEIPT.json';result=json.loads(result_path.read_text())
        assert result['status']=='COMPLETED_FIXED12_LR20_CANDIDATE_ONLY' and len(result['files'])==12
        assert result['owner']==receipt['child'] and result['release_sha256']==a.release_sha256
        receipt.update(status='COMPLETED_FIXED12_LR20_INFERENCE',inference_receipt=pin(result_path),archive=result['archive'])
    except BaseException as exc:
        receipt.update(status='INCOMPLETE_NO_CLAIM',error=repr(exc))
    finally:
        receipt.update(finished_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        with (BASE/'CONTOUR_SUPERVISOR_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))
    return 0 if receipt['status']=='COMPLETED_FIXED12_LR20_INFERENCE' and receipt['child_exit_verified'] else 1


if __name__=='__main__':raise SystemExit(main())
