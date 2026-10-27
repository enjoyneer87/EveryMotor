"""Read-only collection after native worker/audit completion; no solver launch."""
from pathlib import Path
import argparse,hashlib,json,shutil,datetime
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--case-dir',type=Path,required=True);p.add_argument('--inputs-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    c=a.case_dir;raw=c/'raw';derived=c/'derived';key=c.name
    expected={r['case_id']:r for r in read(a.inputs_manifest)['cases']};assert key in expected,'Not a registered production case'
    execution=read(c/'execution.json');audit=read(derived/'audit.json');owner=read(c/'OWNER.json')
    stat=Path(f"/proc/{owner['pid']}/stat")
    if stat.exists():assert stat.read_text().rsplit(') ',1)[1].split()[19]!=owner['start_ticks'],'Owned solver is still running'
    assert execution['returncode']==0 and not execution['stop_reason'] and not execution.get('audit_error')
    assert audit['steps']==768 and audit['steps_per_period']==256 and len(audit['periods'])==3
    actual={name:sha(raw/name) for name in expected[key]['input_hashes']}
    assert actual==expected[key]['input_hashes'],'Input mismatch; preserve and quarantine separately'
    wave=dict(np.load(derived/'waveforms.npz',allow_pickle=False));rows=wave['scalar'];assert rows.shape==(768,5)
    assert wave['current_A'].shape==wave['prescribed_A'].shape==(768,36)
    maximum=0.
    for k,period in enumerate(audit['periods']):
        recomputed=rows[k*256:(k+1)*256,1:4].mean(axis=0)
        saved=np.array([period[x] for x in ['Joule_W','DC_W','AC_W']]);maximum=max(maximum,float(np.max(abs(recomputed-saved))))
        np.testing.assert_allclose(recomputed,saved,rtol=1e-12,atol=1e-9)
    a.out.mkdir(parents=True,exist_ok=False)
    for src in [c/'execution.json',derived/'audit.json',derived/'waveforms.npz']:shutil.copy2(src,a.out/src.name)
    files=[dict(path=f.relative_to(raw).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(raw.rglob('*')) if f.is_file()]
    write(a.out/'raw_source_manifest.json',dict(logical_source_key='Elmer/new_OP_20261027/'+key,case_id=key,files=files,total_bytes=sum(x['bytes'] for x in files),absolute_path_mapping='private only; raw retained on iso',input_hashes_match_frozen=True))
    receipt=dict(case_id=key,collected_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_status='PASS',input_files=len(actual),native_steps=768,
        waveform_scalar_roundtrip_max_W=maximum,roundtrip_tolerance=dict(rtol=1e-12,atol_W=1e-9,scope='float64 saved-waveform arithmetic consistency only'),
        current_status=audit['whole_current_status'],nonlinear_status=audit['nonlinear_status'],formal_period_mesh_time='UNKNOWN',
        native_method='BE first step then BDF2 from native A plus circuit voltage; P1 triangle J integration; prescribed transport DC',
        independent_solver_loss_channel=False,accepted_average='P2 post-step arithmetic mean steps257..512',P3_role='period diagnostic only',
        quantity='AC excess = full-motor active Joule minus prescribed transport DC',unit='W',symmetry_factor=8,active_length_m=audit['active_length_m'],sigma_S_m=audit['sigma_S_m'],temperature_C=None,
        waveforms_sha256=sha(a.out/'waveforms.npz'),raw_manifest_sha256=sha(a.out/'raw_source_manifest.json'))
    write(a.out/'READBACK_RECEIPT.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
