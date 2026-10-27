"""CPU-only hash, frozen contract, split, and residual-head replay checks."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mgn_development_20261027'))
from residual import feature_matrix,predict

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=ROOT
    files=json.loads((r/'ARTIFACT_MANIFEST.json').read_text())['files']
    for f in files:assert sha(r/f['path'])==f['sha256'] and (r/f['path']).stat().st_size==f['bytes'],f['path']
    freeze=json.loads((r/'artifacts/TRAINING_COMPLETE.json').read_text());assert freeze['Full_budget']==63 and not freeze['Full_eval_performed']
    for name,h in freeze['frozen_files'].items():assert sha(r/'artifacts'/name)==h,name
    code=json.loads((r/'artifacts/INPUT_FREEZE.json').read_text())
    for name,h in code['code'].items():assert sha(r/name)==h,name
    for name,h in code['inherited_code'].items():assert sha(r.parent/'mgn_development_20261027'/name)==h,name
    assert sha(r/'PROTOCOL.json')==code['protocol_sha256']
    train=json.loads((r.parent/'mgn_development_20261027/artifacts/inputs/calibration63.json').read_text());s=json.loads((r/'split_audit.json').read_text())
    assert len(train)==63 and set(s['calibration_case_ids'])=={x['case_id'] for x in train}
    assert set(x['model'] for x in train)=={'Ref','SC'} and s['HalfSC_training']==0 and s['reserved_observations_in_training']==0
    head_receipts=[]
    for seed in [17,29,43]:
        z=np.load(r/f'artifacts/embeddings_seed{seed}.npz');h=dict(np.load(r/f'artifacts/A_curl_seed{seed}_head.npz'));x=feature_matrix(train,z['A_curl'],z['source_keys'])
        assert np.allclose((x-h['mean'])/h['sd'],h['z'],rtol=1e-12,atol=1e-12)
        pred=np.array([row['hybrid'] for row in train])*np.exp(predict(h,x));assert np.isfinite(pred).all() and (pred>0).all()
        head_receipts.append(dict(seed=seed,calibration_rows=63,finite_positive=True,fit_feature_normalization_reproduced=True))
    v=json.loads((r/'FOLLOWUP_VALIDATION.json').read_text());assert v['status']=='PASS'
    report=dict(status='PASS',artifact_hashes=len(files),frozen_models=3,Full_budget=63,HalfSC_training=0,reserved_training=0,
        head_readback=head_receipts,scope='integrity and arithmetic contract only; Full AC superiority remains UNKNOWN for this followup')
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
