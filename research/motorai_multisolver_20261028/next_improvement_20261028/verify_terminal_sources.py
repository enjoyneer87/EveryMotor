"""Post-completion read-only source/checkpoint audit; no model deserialization."""
from pathlib import Path
import argparse,datetime,hashlib,json
import psutil

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);a=ap.parse_args()
    r=a.stage;output=r/'TERMINAL_SOURCE_AUDIT.json'
    assert not output.exists(),'This source audit already completed; preserve its receipt'
    dep=json.loads((r/'DEPLOY_RECEIPT.json').read_text())
    guard=json.loads((r/'GUARD_RECEIPT.json').read_text())
    assert guard['status']=='COMPLETED'
    result={'checked_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'verifier_sha256':sha(Path(__file__)),'source_commit':dep['source_commit'],
            'source_code':[],'component_sidecars':0,'checkpoints':[],'seed_pairs':[],
            'GPU_operations':0,'new_FEA':0,'model_deserialization':0,'raw_graphs_opened':0}
    for n,value in dep['code_hashes'].items():
        assert sha(r/n)==value,n
        result['source_code'].append({'file':n,'sha256':value,'status':'PASS'})
    assert sha(r/'components/COMPONENT_MANIFEST.json')==dep['component_manifest_sha256']
    components=json.loads((r/'components/COMPONENT_MANIFEST.json').read_text())
    for row in components['records']:
        p=r/'components'/row['file']
        assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        result['component_sidecars']+=1
    train={}
    for arm in ('C','G'):
        for seed in (17,29,43):
            rid=f'{arm}_s{seed}';d=r/'runs'/rid
            t=json.loads((d/'train/RESULT.json').read_text());ev=json.loads((d/'eval/RESULT.json').read_text())
            assert t['status']==ev['status']=='COMPLETED' and t['training_updates']==88565 and ev['graphs_processed']==1440
            for name in ('train','evaluate'):
                receipt=json.loads((d/(name+'_RECEIPT.json')).read_text())
                assert receipt['returncode']==0 and receipt['stop_reason'] is None
                try:
                    p=psutil.Process(receipt['pid'])
                    assert abs(p.create_time()-receipt['process_started_epoch'])>=.01,'Owned process still live'
                except psutil.NoSuchProcess:pass
            checkpoint=d/'train/last.pt';value=sha(checkpoint)
            assert value==ev['checkpoint_sha256']
            result['checkpoints'].append({'logical_source_key':'EveryMotor/ContextCG20261028/'+rid+'/final88565',
                                         'file':'runs/'+rid+'/train/last.pt','sha256':value,
                                         'bytes':checkpoint.stat().st_size,'seed':seed,'arm':arm,'status':'PASS'})
            train[rid]=t
    for seed in (17,29,43):
        c,g=train[f'C_s{seed}'],train[f'G_s{seed}']
        assert c['initial_tensor_state_sha256']==g['initial_tensor_state_sha256']
        assert c['train_groups']==g['train_groups'] and c['development_groups']==g['development_groups']
        assert c['parameter_count']==g['parameter_count']==71206
        assert c['phase_encoding']==g['phase_encoding']=='mechanical'
        result['seed_pairs'].append({'seed':seed,'initial_tensor_state_sha256':c['initial_tensor_state_sha256'],
                                    'split_and_encoding':'PASS','parameter_count_each':71206})
    result.update(status='PASS_NUMERICAL_PROVENANCE_ONLY',owned_training_and_evaluation_live=0,
                  formal_mesh_period_physics_qualification='UNKNOWN_not_tested_here')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
