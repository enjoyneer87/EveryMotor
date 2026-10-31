"""One read-only stopped-run archive audit, never an inference retry or success alias."""
from pathlib import Path, PureWindowsPath
import ctypes
import datetime as dt
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'main_training_visual_20261030_01a18cac'
NAS=Path('N:/MotorAI_NextLR20_20261031_01a18cac').resolve()
BASE='C:/work/EveryMotor_NextLR20_20261031_01a18cac'
CHAIN='3fc6975ade9d535728a770210f72f1e413032ea7fbe1943e7540e1fb7558ca32'
sys.path.insert(0,str(OLD))
from remote_transport import python,OPTIONS


def sha(p):
    p=Path(p);assert p.stat().st_size<=32*1024**2
    b=p.read_bytes();assert len(b)<=32*1024**2;return hashlib.sha256(b).hexdigest()
CAPTURED={}
def captured(p,expected=None):
    p=Path(p).resolve();assert p.is_relative_to(ROOT) or p.is_relative_to(OLD) or p.is_relative_to(NAS)
    assert p.stat().st_size<=32*1024**2
    with p.open('rb') as f:b=f.read(32*1024**2+1)
    assert len(b)<=32*1024**2
    if expected is not None:assert hashlib.sha256(b).hexdigest()==expected
    assert p not in CAPTURED or CAPTURED[p]==b,'source changed across captured reads'
    CAPTURED[p]=b;return b
def read(p):return json.loads(captured(p).decode('utf-8-sig'))
def pin(p):
    p=Path(p);return dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
def rss():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    v=Counters();v.cb=ctypes.sizeof(v)
    current=ctypes.windll.kernel32.GetCurrentProcess;current.restype=ctypes.c_void_p
    query=ctypes.windll.psapi.GetProcessMemoryInfo;query.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    assert query(current(),ctypes.byref(v),v.cb)
    return v.PeakWorkingSetSize


PROBE=r'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import datetime as dt,hashlib,json,platform,psutil,subprocess,sys,time
sys.dont_write_bytecode=True
r=Path('C:/work/EveryMotor_NextLR20_20261031_01a18cac')
def raw(p,limit=1048576):
 assert p.is_file() and p.stat().st_size<=limit
 b=p.read_bytes();assert len(b)<=limit;return b
def digest(p,limit=1048576):return hashlib.sha256(raw(p,limit)).hexdigest()
chain=json.loads(raw(r/'CHAIN_LR20.json'));assert digest(r/'CHAIN_LR20.json')==spec['chain_sha256']
for row in chain['source_files']:
 p=Path(row['path']);assert p.resolve().is_relative_to((r/'src').resolve()) and digest(p)==row['sha256']
for name,value in spec['postprocess'].items():assert digest(r/'postprocess'/name)==value
assert digest(r/'STAGE_CLOCK.json')==chain['stage_clock']['sha256']
sys.path.insert(0,str(r/'src'));from stage_clock import elapsed_stage_seconds
now=dt.datetime.now(dt.timezone.utc);clock=json.loads(raw(r/'STAGE_CLOCK.json'))
spent=elapsed_stage_seconds(clock,time.monotonic(),now,platform.node(),psutil.boot_time())
remaining=min(28800-spent,(dt.datetime.fromisoformat(chain['deadline_UTC'])-now).total_seconds());assert remaining>0
metadata={}
names=['CONTOUR_LR20_RELEASE.json','CONTOUR_SUPERVISOR_OWNER.json','CONTOUR_SUPERVISOR_RECEIPT.json','CONTOUR_STOP','contour_LR20_R1/OWNER.json','contour_LR20_R1/INFERENCE_RECEIPT.json']
for name in names:
 b=raw(r/name);metadata[name]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),text=b.decode('utf-8-sig'))
for name,expected in spec['original_text_pins'].items():
 assert metadata[name]['sha256']==expected['sha256'] and metadata[name]['bytes']==expected['bytes']
release=json.loads(metadata['CONTOUR_LR20_RELEASE.json']['text']);assert metadata['CONTOUR_LR20_RELEASE.json']['sha256']==spec['release_sha256']
result=json.loads(metadata['contour_LR20_R1/INFERENCE_RECEIPT.json']['text']);owner=json.loads(metadata['contour_LR20_R1/OWNER.json']['text'])
assert owner['owner']==result['owner']==spec['identities'][2] and owner['release_sha256']==result['release_sha256']==spec['release_sha256']
assert owner['status']=='RUNNING' and owner['files']==[] and owner['retry']==0
identities=[]
for o in spec['identities']:
 try:state='MATCH_LIVE' if psutil.Process(o['pid']).create_time()==o['create_time'] else 'REUSED_PID'
 except psutil.NoSuchProcess:state='EXITED'
 assert state in ('EXITED','REUSED_PID');identities.append(dict(**o,state=state))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5).strip();assert not gpu
assert result['status']=='COMPLETED_FIXED12_LR20_CANDIDATE_ONLY' and result['files']==spec['files']
archive=r/'contour_LR20_R1/candidate_fixed12.zip';assert archive.stat().st_size==spec['archive']['bytes']
assert digest(archive,33554432)==spec['archive']['sha256'] and result['archive']==spec['archive']
for row in result['files']:
 p=Path(row['path']);assert p.parent.resolve()==(r/'contour_LR20_R1').resolve() and p.suffix=='.npz'
 assert p.stat().st_size==row['bytes'] and digest(p,8388608)==row['sha256']
print(json.dumps(dict(UTC=now.isoformat(),metadata=metadata,identities=identities,GPU_compute_processes=[],STOP_preserved=True,archive=spec['archive'],remaining_seconds=remaining)))
'''


def main():
    started=time.monotonic()
    source_sha=sha(Path(__file__));contract_path=ROOT/'CONTOUR_LR20_RECOVERY_EXECUTION_CONTRACT.json';contract=read(contract_path);contract_sha=sha(contract_path)
    assert contract['status']=='RELEASED_ONE_SHOT_READ_ONLY_FAILED_CONTOUR_RECOVERY' and contract['recovery_source_sha256']==source_sha
    assert contract['max_seconds']==180 and contract['retry']==0 and contract['new_inference']==contract['remote_mutations']==0
    review_path=ROOT/'science/CONTOUR_LR20_FAILURE_RECOVERY_REVIEW_R2.json';assert sha(review_path)==contract['review_sha256'];review=read(review_path)
    assert review['status']=='COMPLETE_OPERATIONAL_FAILURE_REVIEW_WITH_CONDITIONAL_READ_ONLY_RECOVERY' and review['operational_verdict']=='INCOMPLETE_NO_CLAIM'
    assert review['recovery_contract']['attempts']==1 and review['recovery_contract']['time']['postmortem_recovery_cap_seconds']==180
    policy_path=ROOT/'science/CONTOUR_LR20_RECOVERY_SOURCE_REVIEW_20261031_R1.json'
    assert sha(policy_path)==contract['scalar_policy_sha256']
    policy=read(policy_path);assert policy['scalar_anchor_policy']['status']=='CLARIFIED_BEFORE_ANY_LOCAL_RECOVERY_ATTEMPT'
    assert sha(ROOT/'CHAIN_LR20.json')==CHAIN
    required={(ROOT/n).resolve() for n in ('CHAIN_LR20.json','CONTOUR_LR20_RELEASE.json','CONTOUR_LR20_FAILURE_RECEIPT.json','CONTOUR_LR20_FAILURE_DIAGNOSIS.json','CONTOUR_LR20_RUNNER_EXIT_DIAGNOSIS.json','science/FINAL_LR20_SCIENTIFIC_REVIEW.json','science/CONTOUR_LR20_FAILURE_RECOVERY_REVIEW_R2.json')}
    required.update((OLD/n).resolve() for n in ('CONTOUR_SELECTION.json','CONTOUR_GEOMETRY_BINDING_RESULT.json','contour_inference/INFERENCE_MANIFEST.json'))
    required.add(policy_path)
    def reviewed_pins(value):
        if isinstance(value,dict):
            if {'path','sha256','bytes'}<=set(value):yield value
            else:
                for child in value.values():yield from reviewed_pins(child)
        elif isinstance(value,list):
            for child in value:yield from reviewed_pins(child)
    authoritative=list(reviewed_pins(review['evidence']))
    required.update(Path(p['path']).resolve() for p in authoritative)
    for p in read(ROOT/'CHAIN_LR20.json')['source_files']:
        relative=PureWindowsPath(p['path']).relative_to(PureWindowsPath(BASE+'/src'));required.add((ROOT/'src'/relative.as_posix()).resolve())
    assert required<={Path(p['path']).resolve() for p in contract['input_pins']},'mandatory frozen input inventory omitted'
    for p in contract['input_pins']:
        assert Path(p['path']).suffix in ('.json','.py')
        assert len(captured(p['path'],p['sha256']))==p['bytes']
    for p in authoritative:assert len(captured(p['path'],p['sha256']))==p['bytes']
    output=NAS/'contour_readonly_recovery_R1';assert not output.exists()
    attempt=ROOT/'CONTOUR_LR20_RECOVERY_ATTEMPT.json';remote_anchor=None;remote_left=None;peak=0
    def check(cap=180):
        nonlocal peak
        peak=max(peak,rss());assert peak<1024**3
        left=180-(time.monotonic()-started)
        if remote_anchor is not None:left=min(left,remote_left-(time.monotonic()-remote_anchor))
        assert left>0;return min(cap,left)
    check()
    with attempt.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),started_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=source_sha,contract_sha256=contract_sha,retry=0),f,indent=2)
    try:
        failure=review['failure'];artifact=review['artifact_evidence'];release=read(ROOT/'CONTOUR_LR20_RELEASE.json')
        spec=dict(chain_sha256=CHAIN,postprocess={n:release[k] for n,k in [('infer_lr20_contours.py','runner_sha256'),('supervise_lr20_contours.py','supervisor_sha256'),('contour_child_guard.py','child_guard_sha256')]},
                  original_text_pins=review['evidence']['embedded_text_pins_verified'],release_sha256=sha(ROOT/'CONTOUR_LR20_RELEASE.json'),identities=[failure['supervisor_owner'],failure['popen_child'],failure['artifact_receipt_owner']],archive=artifact['archive'],files=artifact['files'])
        def snapshot():
            nonlocal remote_anchor,remote_left
            before=time.monotonic();v=json.loads(python('spec='+repr(spec)+'\n'+PROBE,timeout=check(35)));duration=time.monotonic()-before
            assert v['remaining_seconds']>duration;remote_left=v['remaining_seconds']-duration;remote_anchor=time.monotonic();check();return v
        assert artifact['archive']['bytes']+sum(p['bytes'] for p in artifact['files'])+4*1024**2<=64*1024**2
        before=snapshot();output.mkdir();archive=output/'candidate_fixed12.zip'
        subprocess.run(['scp','-q',*OPTIONS,'motorai133:'+artifact['archive']['path'],str(archive)],check=True,timeout=check(90),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert archive.stat().st_size==artifact['archive']['bytes'] and sha(archive)==artifact['archive']['sha256']
        with zipfile.ZipFile(archive) as z:
            members=z.infolist();expected={PureWindowsPath(p['path']).name for p in artifact['files']}|{'INFERENCE_MANIFEST.json'}
            assert len(members)==len(expected)==13 and {p.filename for p in members}==expected
            assert sum(p.file_size for p in members)<=64*1024**2
            for p in members:
                check();assert p.file_size<=8*1024**2 and not p.flag_bits&1 and not stat.S_ISLNK(p.external_attr>>16)
                destination=(output/p.filename).resolve();assert destination.parent==output
                with z.open(p) as a,destination.open('xb') as b:
                    payload=a.read(8*1024**2+1);assert len(payload)==p.file_size;b.write(payload)
        for p in artifact['files']:
            q=output/PureWindowsPath(p['path']).name;assert q.stat().st_size==p['bytes'] and sha(q)==p['sha256']
        manifest_path=output/'INFERENCE_MANIFEST.json';manifest=read(manifest_path)
        original=json.loads(before['metadata']['contour_LR20_R1/INFERENCE_RECEIPT.json']['text'])
        assert set(original)==set(manifest)|{'archive','finished_UTC'}
        for k in manifest:
            if k!='elapsed_seconds':assert manifest[k]==original[k],k
        assert 0<=manifest['elapsed_seconds']<=original['elapsed_seconds']<=360
        for p in read(ROOT/'CHAIN_LR20.json')['source_files']:
            relative=PureWindowsPath(p['path']).relative_to(PureWindowsPath(BASE+'/src'));q=ROOT/'src'/relative.as_posix();assert sha(q)==p['sha256']
        sys.path.insert(0,str(OLD/'runtime_lib'));sys.path.insert(0,str(ROOT/'src'))
        import numpy as np
        from strict_airgap import finish_statistics,field_sufficient_statistics
        from saved_metrics import close
        def arrays(p,expected_sha):
            check()
            payload=captured(p,expected_sha)
            with zipfile.ZipFile(io.BytesIO(payload)) as z:assert sum(m.file_size for m in z.infolist())<=32*1024**2
            with np.load(io.BytesIO(payload),allow_pickle=False) as z:return {k:z[k] for k in z.files}
        binding=read(OLD/'CONTOUR_GEOMETRY_BINDING_RESULT.json');controls=read(OLD/'contour_inference/INFERENCE_MANIFEST.json');rows=[]
        expected_pairs={(c,s) for c in (208,408,49) for s in (0,11,22,33)}
        assert {(p['case_index'],p['step']) for p in manifest['files']}==expected_pairs
        for p in manifest['files']:
            case,step=p['case_index'],p['step'];check();gpin=next(x for x in binding['rows'] if (x['case_index'],x['step'])==(case,step));gp=Path(gpin['output']['path']).resolve()
            assert gp.parent==(OLD/'contour_joined').resolve() and sha(gp)==gpin['output']['sha256']
            cp=next(x for x in controls['files'] if x['model']=='W128_w1' and (x['case_index'],x['step'])==(case,step));control_path=OLD/'contour_inference'/PureWindowsPath(cp['path']).name
            assert control_path.stat().st_size==cp['bytes'] and sha(control_path)==cp['sha256']
            candidate_path=output/PureWindowsPath(p['path']).name;a,b,g=arrays(candidate_path,p['sha256']),arrays(control_path,cp['sha256']),arrays(gp,gpin['output']['sha256'])
            assert p['model']=='W128_LR20' and p['checkpoint_sha256']==release['candidate']['checkpoint']['sha256']
            assert cp['graph_sha256']==p['graph_sha256']==gpin['graph_sha256']
            assert int(a['case_index'])==int(b['case_index'])==case and int(a['step'])==int(b['step'])==step
            ids=g['native_element_id'];assert len(np.unique(ids))==len(ids) and np.array_equal(a['native_element_id'],ids) and np.array_equal(b['native_element_id'],ids)
            for x in (a,b):assert x['pred_B_T'].shape==x['reference_B_T'].shape==(len(ids),2) and np.isfinite(x['pred_B_T']).all() and np.isfinite(x['reference_B_T']).all()
            assert np.array_equal(a['reference_B_T'],b['reference_B_T'])
            assert g['vertex_xy_mm'].shape==(len(ids),3,2) and g['a1_mask'].shape==g['weights'].shape==(len(ids),)
            assert np.isfinite(g['vertex_xy_mm']).all() and np.isfinite(g['weights']).all() and (g['weights']>0).all()
            anchor=next(x for x in review['readonly_reference_inventory']['saved_eval_snapshot_anchors'] if (x['case'],x['step'])==(case,step))
            actual=finish_statistics(field_sufficient_statistics(a['pred_B_T'],a['reference_B_T'],g['weights']))
            scalar_comparison={}
            for k in ('vector_nrmse_percent','magnitude_nrmse_percent'):
                value=getattr(actual,k);saved=anchor[k];assert math.isfinite(value) and math.isfinite(saved)
                try:close(value,saved,'recovered snapshot '+k);within=True
                except ValueError:within=False
                scalar_comparison[k]=dict(saved=saved,recovered=value,signed_delta_pp=value-saved,absolute_delta_pp=abs(value-saved),within_saved_arithmetic_tolerance=within)
            scalar_status='MATCH_EXISTING_SAVED_ARITHMETIC_TOLERANCE' if all(x['within_saved_arithmetic_tolerance'] for x in scalar_comparison.values()) else 'SCALAR_DIFFERENCE_UNEXPLAINED'
            rows.append(dict(case_index=case,step=step,candidate=pin(candidate_path),control=pin(control_path),joined=pin(gp),graph_sha256=p['graph_sha256'],native_elements=len(ids),reference_exact=True,native_ID_order_exact=True,scalar_status=scalar_status,scalar_comparison=scalar_comparison))
        assert len(rows)==12;after=snapshot()
        for k in ('metadata','archive'):assert before[k]==after[k],'stopped-run bytes changed'
        assert sha(Path(__file__))==source_sha and sha(contract_path)==contract_sha and sha(review_path)==contract['review_sha256']
        for p in contract['input_pins']:assert sha(p['path'])==p['sha256']
        for path,payload in CAPTURED.items():assert path.read_bytes()==payload,'captured source changed during recovery'
        scalar_status='ALL12_MATCH_EXISTING_SAVED_ARITHMETIC_TOLERANCE' if all(x['scalar_status']=='MATCH_EXISTING_SAVED_ARITHMETIC_TOLERANCE' for x in rows) else 'SCALAR_DIFFERENCE_UNEXPLAINED'
        check();binding_receipt=dict(status='PASS_ALL12_EXACT_RECOVERED_FIELD_BINDINGS',rows=rows,scalar_status=scalar_status,scalar_policy=pin(policy_path),scalar_scope='Consistency diagnostic only; no new tolerance or CUDA determinism claim; original703 saved arithmetic unchanged',source_vector_basis='UNKNOWN',original_operational_status='INCOMPLETE_NO_CLAIM',process_ancestry='UNKNOWN')
        binding_path=output/'EXACT_FIELD_BINDING_RECOVERY_RECEIPT.json'
        with binding_path.open('x',encoding='utf-8') as f:json.dump(binding_receipt,f,indent=2)
        check()
        result=dict(status='PASS_RECOVERED_FIXED12_ARTIFACT_HASH_AND_FIELD_BINDING_ONLY',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),release_sha256=spec['release_sha256'],original_operational_status='INCOMPLETE_NO_CLAIM',process_ancestry='UNKNOWN',auxiliary_visualization_only=True,new_inference=0,remote_mutations=0,retry=0,source_sha256=source_sha,execution_contract_sha256=contract_sha,review=pin(review_path),manifest=pin(manifest_path),archive=pin(archive),field_binding=pin(binding_path),files=manifest['files'],before=before,after=after,elapsed_seconds=time.monotonic()-started,peak_local_RSS_bytes=peak)
        path=output/'READ_ONLY_RECOVERY_RECEIPT.json'
        with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
        check();assert sum(p.stat().st_size for p in output.iterdir() if p.is_file())<=64*1024**2
        pending=dict(**pin(path),worker_source_sha256=source_sha,execution_contract_sha256=contract_sha)
        with (ROOT/'CONTOUR_LR20_RECOVERY_PENDING_POINTER.json').open('x',encoding='utf-8') as f:json.dump(pending,f,indent=2)
        check()
        print(json.dumps(dict(status=result['status'],files=12,elapsed_seconds=result['elapsed_seconds'],original_operational_status=result['original_operational_status'])))
    except BaseException as e:
        with (ROOT/'CONTOUR_LR20_RECOVERY_FAILURE.json').open('x',encoding='utf-8') as f:json.dump(dict(status='INCOMPLETE_NO_CLAIM',error=repr(e),elapsed_seconds=time.monotonic()-started,retry=0,automatic_retry=False,source_sha256=source_sha,execution_contract_sha256=contract_sha),f,indent=2)
        raise


if __name__=='__main__':main()
