"""Verify the released small package; never runs a solver, training or refit."""
from pathlib import Path
import argparse,hashlib,json,csv
import numpy as np
from reproduce_heads import verify


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();r=a.root
    manifest=json.loads((r/'ARTIFACT_MANIFEST.json').read_text())
    for item in manifest['files']:
        p=r/item['path'];assert p.stat().st_size==item['bytes'];assert sha(p)==item['sha256'],item['path']
    split=json.loads((r/'SPLIT_MANIFEST.json').read_text());train=json.loads((r/'artifacts/inputs/calibration63.json').read_text())
    ids={x['case_id'] for x in train};assert len(ids)==63 and ids==set(split['calibration_case_ids'])
    assert all(not ids.intersection(g) for g in split['groups'].values())
    assert all(not any(x in s for x in ['R1_0199','R1_0076']) for s in ids)
    infer=json.loads((r/'artifacts/inputs/inference228.json').read_text());replay=verify(r/'artifacts',infer)
    obs=list(csv.DictReader((r/'observations.csv').open(encoding='utf-8')))
    assert len(obs)==2280 and all(x['new_independent_truth']=='False' for x in obs)
    for stage in ['stage1','stage2']:
        folder=r/'artifacts'/stage;freeze=json.loads((folder/'TRAINING_COMPLETE.json').read_text())
        for name,h in freeze['frozen_files'].items():assert sha(folder/name)==h
    report=dict(status='PASS',artifact_hashes=len(manifest['files']),head_models=replay['models'],predictions_per_model=228,
        unique_Full_calibration=63,protected_sets_disjoint=True,reserved_count=0,model_prediction_rows=2280,
        scope='package integrity, exclusion masks and frozen arithmetic replay; physical/model superiority gates separate')
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
