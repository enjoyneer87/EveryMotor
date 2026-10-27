"""Read-only source recovery and bounded native reintegration; no solver calls.

Private request JSON supplies paths. All writes are beneath a new --out folder.
Run on the source host with field_reader.py on PYTHONPATH. Stop between cases
by creating STOP in --out, or terminate only the PID in process.json.
"""
import argparse, json, os, shutil, sys, time, traceback, zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from field_reader import audit, read_json, sha, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=read_json(a.request);out=a.out;out.mkdir(exist_ok=False)
    write_json(out/'process.json',dict(pid=os.getpid(),purpose='existing-data read-only recovery',new_solver_calls=0,training_calls=0,
        maximum_cases=60,max_retries=0,wall_limit_seconds=5400,estimated_memory_MiB=1024,estimated_output_MiB=800,
        stop='create STOP for between-case stop; terminate this PID for immediate stop; originals are read only'))
    start=time.monotonic();records=[];locations={};failures=[];ms=[];half=[]
    private=out/'recovered';private.mkdir()
    def record(path,logical,expected=None,dest=None):
        h=sha(path);item=dict(logical_source_key=logical,sha256=h,bytes=path.stat().st_size,expected_sha256=expected,
            status='PASS' if h==expected else 'UNKNOWN' if expected is None else 'FAIL',source_commit='0034d7c24b1cc0cf41499d028733792b14815bc7')
        records.append(item);locations[logical]=str(path)
        if expected is not None and h!=expected:raise ValueError('hash mismatch: '+logical)
        if dest:
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
            assert sha(dest)==h
        return item
    def status(phase,cid=None):
        write_json(out/'progress.json',dict(phase=phase,current_case=cid,MS_completed=len(ms),HalfSC_completed=len(half),
            failures=failures,elapsed_s=time.monotonic()-start,pid=os.getpid()))
        write_json(out/'source_records.json',records);write_json(out/'private_paths.json',locations)
    def bounded():
        if (out/'STOP').exists() or time.monotonic()-start>5400:raise RuntimeError('bounded run stopped')
    for row in r['ms']:
        bounded();key=row['source_key'];status('MS_recovery',key)
        try:
            base=Path(row['field_path']).parent;dest=private/'ms'/key
            checks=[(Path(row['field_path']),'ALL_A_HISTORY.npz',row['field_sha256']),
                (Path(row['manifest_path']),'MS_MANIFEST.json',row['manifest_sha256']),
                (Path(row['solver_receipt_path']),'MS_RECEIPT.json',row['solver_receipt_sha256'])]
            receipt=read_json(row['solver_receipt_path']);arc=read_json(base/'ARCHIVE_SHA256.json')
            checks.append((base/'ARCHIVE_SHA256.json','ARCHIVE_SHA256.json',receipt['output_hashes']['ARCHIVE_SHA256.json']['sha256']))
            by={x['path']:x['sha256'] for x in arc['files']}
            for rel in ['INPUT_CONTRACT.json','MODEL.json','PARAMETERS.json','case.sif','solver.log','mesh/mesh.nodes','mesh/mesh.elements','mesh/mesh.boundary','mesh/mesh.header']:
                checks.append((base/'raw'/rel,'raw/'+rel,by[rel]))
            for path,rel,h in checks:record(path,'ms/'+key+'/'+rel,h,dest/rel)
            assert sha(base/'raw/mesh/mesh.nodes')==row['mesh_nodes_sha256']
            assert sha(base/'raw/solver.log')==row['solver_log_sha256']
            ms.append(dict(source_key=key,case_id=row['case_id'],status='PASS'))
        except Exception as e:failures.append(dict(case=key,error=repr(e),trace=traceback.format_exc()))
    bundle=out/'MS24_recovered.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_STORED) as z:
        for f in sorted((private/'ms').rglob('*')):
            if f.is_file():z.write(f,f.relative_to(private).as_posix())
    write_json(out/'MS_bundle.json',dict(file=bundle.name,sha256=sha(bundle),bytes=bundle.stat().st_size,records=ms))
    status('MS_ready')
    for row in r['halfsc']:
        bounded();cid=row['case_id'];status('HalfSC_native_reintegration',cid)
        try:
            candidates=[Path(rt)/cid for rt in r['half_roots']]+[Path(rt)/cid/'cases'/cid for rt in r['half_roots']]
            found=[q for q in candidates if (q/'raw/mesh/native.result').exists()]
            assert len(found)==1, f'Expected one physical acquisition, found {len(found)}'
            base=found[0];locations['halfsc/'+cid]=str(base)
            arc=read_json(base/'ARCHIVE_SHA256.json');pres=base/'PRESERVATION_MANIFEST.json'
            wanted={}
            if pres.exists():
                receipt=read_json(base/'PRESERVATION_RECEIPT.json')
                record(pres,'halfsc/'+cid+'/PRESERVATION_MANIFEST.json',receipt['manifest_sha256'],private/'halfsc'/cid/pres.name)
                wanted={x['path']:x['sha256'] for x in read_json(pres)['files']}
            rels=['ARCHIVE_SHA256.json','PRESERVATION_RECEIPT.json','RESULT_SUMMARY.json','FIGURE14_RESULT.json','RESULT.json','derived/CURRENT_LOSS_AUDIT.json','derived/NONLINEAR_AUDIT.json']
            for rel in rels:
                if (base/rel).exists():record(base/rel,'halfsc/'+cid+'/'+rel,wanted.get(rel),private/'halfsc'/cid/rel)
            old=read_json(base/'derived/CURRENT_LOSS_AUDIT.json');p2=old['periods'][1]
            assert abs(p2['AC_excess_W']-row['P2_AC_excess_W'])<1e-6
            expected=dict(point_id=cid,case_id=cid,archive_files=arc['files'],
                P2_Cu_total_W=p2['total_active_W'],P2_DC_transport_W=p2['DC_transport_W'],P2_AC_excess_W=p2['AC_excess_W'],
                # Per-step audit rows are ratios; summary/gate fields are percent.
                P2_max_current_error_percent=100*max(x['max_field_current_error_per_phasepeak'] for x in old['rows'][256:]))
            result=audit(base/'raw',expected,out/'half_shards')
            result.update(role=row['role'],canonical_scalar_delta_W=p2['AC_excess_W']-row['P2_AC_excess_W'],
                preservation_receipt_available=pres.exists(),old_P2_current_pass=row.get('P2_current_pass'),
                actual_condition={k:row[k] for k in ['speed_rpm','current_Arms','beta_deg']})
            write_json(out/'half_shards'/(cid+'_audit.json'),result)
            half.append(result)
        except Exception as e:failures.append(dict(case=cid,error=repr(e),trace=traceback.format_exc()))
    with zipfile.ZipFile(out/'HalfSC_metadata.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in sorted((private/'halfsc').rglob('*')):
            if f.is_file():z.write(f,f.relative_to(private).as_posix())
    # All 512-step integral histories retained on source host; small audit reports portable.
    write_json(out/'half_results.json',half)
    status('completed')
    print(json.dumps(dict(ms=len(ms),half=len(half),failures=len(failures))),flush=True)


if __name__=='__main__':main()
