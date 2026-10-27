"""Read-only evidence export after an owned solver variant has finished."""
from pathlib import Path
import argparse,json,hashlib,shutil,subprocess,sys


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--case-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.case_root;raw=root/'raw';derived=root/'derived';execution=json.loads((root/'execution.json').read_text())
    assert execution['returncode']==0 and execution['stop_reason'] is None and 'audit_error' not in execution
    assert execution['audit']['steps']==3*execution['audit']['steps_per_period']
    proc=subprocess.run([sys.executable,str(Path(__file__).with_name('export_fea_shard.py')),'--raw',str(raw),'--derived',str(derived),'--out',str(a.out)],capture_output=True)
    if proc.returncode:raise RuntimeError(proc.stderr.decode())
    print(proc.stdout.decode(),flush=True)
    audit=json.loads((derived/'audit.json').read_text());audit['period_development_threshold_status']=audit.pop('period_status')
    audit['formal_period_qualification']='UNKNOWN';audit['formal_mesh_qualification']='UNKNOWN';audit['formal_time_step_qualification']='UNKNOWN'
    audit['threshold_scope']='See QUALIFICATION_POLICY.json: period/mesh/time thresholds are proposed development checks, not an approved qualification contract'
    audit['source_derived_audit_sha256']=sha(derived/'audit.json')
    (a.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    shutil.copyfile(derived/'waveforms.npz',a.out/'waveforms.npz')
    execution.pop('audit');execution['raw_execution_sha256']=sha(root/'execution.json');execution['new_independent_Full_label']=False
    (a.out/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    files=[]
    for source in sorted(raw.rglob('*')):
        if source.is_file():files.append(dict(logical_path=source.relative_to(raw).as_posix(),bytes=source.stat().st_size,sha256=sha(source)))
    out=dict(logical_source_key='jeet_mgn_20261027:'+execution['case'],condition='Ref_P117;16000rpm;460Arms;beta36',base_input_case='ISO64_ACQ_Ref_P117_N256_P2_Q01',
        source_code_base_commit='4c6ef9325dd0539b0220fb2817782562b9932fb7',raw_files=files,total_raw_bytes=sum(x['bytes'] for x in files),
        large_fields_location='private isolated-host storage, logical path map retained locally',new_independent_label=False,
        source_inputs_unchanged='verified by owned runner before and after; original_hashes in execution.json',
        average='uniform post-step period means; P2 and P3 reported separately',solver_version='Elmer26.2 revision11c118114, pinned executable hash')
    (a.out/'raw_source_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(case=execution['case'],raw_files=len(files),raw_bytes=out['total_raw_bytes'],export_bytes=sum(x.stat().st_size for x in a.out.iterdir() if x.is_file()))),flush=True)


if __name__=='__main__':main()
