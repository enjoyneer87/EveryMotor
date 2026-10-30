"""Inference-only replay of twelve preselected graphs after all registered runs finish.

Requires a separate hash-frozen release containing all final checkpoint/result pins.
This source is preparation only until that release exists and the GPU is free.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from pathlib import Path
import argparse
import datetime as dt
import gc
import hashlib
import json
import sys
import time
import zipfile
import psutil
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    start=time.monotonic()
    stage=args.release.parent.resolve()
    own=dict(pid=os.getpid(),create_time=psutil.Process().create_time())

    def check():
        assert time.monotonic()-start<1200 and not (stage/'STOP').exists()
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat('2026-10-30T20:05:52.104794+00:00')

    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while block:=f.read(1024**2):
                check();h.update(block)
        return h.hexdigest()

    def pinned(spec):
        p=Path(spec['path']);assert sha(p)==spec['sha256'] and p.stat().st_size==spec['bytes']
        return json.loads(p.read_text(encoding='utf-8-sig'))

    def same_pin(actual,expected):
        assert Path(actual['path']).resolve()==Path(expected['path']).resolve()
        assert actual['sha256']==expected['sha256']
        if 'bytes' in actual and 'bytes' in expected:
            assert actual['bytes']==expected['bytes']

    assert sha(args.release)==args.release_sha256
    release=json.loads(args.release.read_text())
    assert release['status']=='RELEASED_FINAL_SEED17_CONTOUR_INFERENCE_ONLY'
    assert release['retry']==0 and release['GPU_workers']==1 and release['new_optimizer_updates']==0
    assert sha(Path(__file__).resolve())==release['runner_sha256']
    assert not (stage/'INFERENCE_OWNER.json').exists() and not (stage/'outputs').exists()
    (stage/'INFERENCE_OWNER.json').write_text(json.dumps(dict(owner=own,started_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),release_sha256=args.release_sha256),indent=2))
    chain=pinned(release['chain_plan'])
    for spec in chain['source_files']:
        assert sha(spec['path'])==spec['sha256']
    sys.path.insert(0,chain['pythonpath'])
    import run_width128_chain as guard
    import evaluate_common_airgap as common
    from capacity_contract import restore_capacity
    assert not guard.gpu_compute_pids(), 'GPU must be free after the complete registered chain'
    receipt=pinned(release['scientific_receipt'])
    assert receipt['status']=='COMPLETED' and len(receipt['jobs'])==6
    assert receipt['chain_plan_sha256']==release['chain_plan']['sha256']
    main_release=pinned(release['main_scientific_release'])
    assert main_release['chain_plan_sha256']==release['chain_plan']['sha256']
    same_pin(receipt['scientific_release'],release['main_scientific_release'])
    expected_jobs={f'W128_s{s}_{kind}' for s in (17,29,43) for kind in ('train','eval')}
    assert {x['id'] for x in receipt['jobs']}==expected_jobs
    for row in receipt['jobs']:
        assert row['status']=='COMPLETED'
        data=pinned(row['result'])
        assert data['status']=='COMPLETED'
        if row['id'].endswith('_train'):
            assert data['training_updates']==88565
        else:
            assert data['graphs_processed']==1440
    for identity in [receipt['owner']]+[x['owner'] for x in receipt['jobs'] if x.get('owner')]:
        try:
            assert psutil.Process(identity['pid']).create_time()!=identity['create_time'], 'training owner still alive'
        except psutil.NoSuchProcess:
            pass
    verdict=pinned(release['final_verdict']);assert verdict['status']=='COMPLETE'
    same_pin(verdict['evidence_pins']['scientific_receipt'],release['scientific_receipt'])
    binding=pinned(release['geometry_binding_result'])
    assert binding['status']=='PASS_ALL12_STORED_NATIVE_GEOMETRY_BINDINGS'
    selected=pinned(release['contour_selection'])
    assert selected['seed']==17 and selected['steps']==[0,11,22,33]
    assert [x['case_index'] for x in selected['conditions']]==[208,408,49]
    template=pinned(chain['evaluation_template'])
    selection=pinned(template['selection'])
    bundle_path=Path(template['bundle_dir'])/'BUNDLE_MANIFEST.json'
    assert sha(bundle_path)==template['bundle_manifest_sha256']
    bundle=json.loads(bundle_path.read_text())
    source_records={x['case_index']:x for x in selection['records'] if x['split']=='development'}
    records={x['case_index']:x for x in bundle['records']}
    geom=pinned(template['geometry_manifest'])
    operators={}
    for item in selected['conditions']:
        case=item['case_index']
        assert source_records[case]['source_sha256']==records[case]['source_sha256']==item['source_sha256']
        row=next(x for x in geom['records'] if x['case_index']==case)
        op_root=Path(template['geometry_manifest']['path']).parent.resolve()
        op_path=(op_root/row['operator_file']).resolve();assert op_path.parent==op_root
        assert sha(op_path)==row['operator_sha256'] and row['status']=='ADMITTED'
        operators[case]=common.load_operator(op_path)
    assert set(release['models'])=={'W64_w1','W64_w3','W128_w1'}
    import torch
    from dual_solver_mgn import DomainContract
    assert torch.__version__=='2.5.1+cu124' and np.__version__=='1.26.4'
    torch.set_num_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    output=stage/'outputs';output.mkdir()
    result=dict(status='RUNNING',owner=own,started_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                qualification='STORED_NATIVE_HDF_REPLAY_ONLY',source_vector_basis='UNKNOWN',
                seed=17,new_optimizer_updates=0,new_FEA=0,retry=0,files=[],models={})
    for label in ('W64_w1','W64_w3','W128_w1'):
        check();spec=release['models'][label]
        assert spec['width'] in (64,128) and spec['arm']==f"W{spec['width']}"
        assert spec['width']=={'W64_w1':64,'W64_w3':64,'W128_w1':128}[label]
        train=pinned(spec['train_result']);evaluation=pinned(spec['evaluation_result'])
        if label=='W128_w1':
            for kind,key,verdict_key in (('train','train_result','training_results'),('eval','evaluation_result','results')):
                source_row=next(x for x in receipt['jobs'] if x['id']==f'W128_s17_{kind}')
                same_pin(spec[key],source_row['result'])
                same_pin(spec[key],verdict['evidence_pins'][verdict_key]['candidate']['17'])
        assert train['status']==evaluation['status']=='COMPLETED' and train['training_updates']==88565
        assert evaluation['graphs_processed']==1440 and train['seed']==evaluation['seed']==17
        assert evaluation['reproduction_gate']['status']=='PASS_ARITHMETIC_ONLY'
        assert evaluation['original_train_sha256']==spec['train_result']['sha256']
        checkpoint_path=Path(spec['checkpoint']['path'])
        assert sha(checkpoint_path)==spec['checkpoint']['sha256']==train['checkpoint_sha256']==evaluation['checkpoint_sha256']
        assert checkpoint_path.stat().st_size==spec['checkpoint']['bytes']
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        meta=checkpoint['metadata'];assert checkpoint['epoch']==30
        for name,actual in (('checkpoint_metadata',meta),('training_result',train),('evaluation_result',evaluation)):
            expected=spec['expected_metadata'][name]
            required=({'selection_sha256','bundle_manifest_sha256','geometry_manifest_sha256',
                       'training_mask_manifest_sha256','checkpoint_sha256','model_sha256','phase_encoding',
                       'arm','width','depth','seed','experiment_id','metadata_hashes'} if name=='evaluation_result' else
                      {'selection_sha256','bundle_manifest_sha256','mask_manifest_sha256','plan_sha256',
                       'model_sha256','trainer_sha256','initial_tensor_state_sha256','data_order_LR_trace_sha256',
                       'phase_encoding','data_arm','train_steps','train_graphs','arm','seed','width','depth','experiment_id'})
            assert required<=set(expected), 'per-model frozen data/plan/operator metadata must be explicit'
            for key,value in expected.items():
                assert actual.get(key)==value, f'{label}/{name}/{key} differs from its registered lineage'
        expected_experiment={'W64_w1':'W64_PHASE45_EQUAL_UPDATE_20261030',
                             'W64_w3':'W64_PHASE45_A1W3_20261030',
                             'W128_w1':'W128_PHASE45_EQUAL_UPDATE_20261030'}[label]
        assert meta['experiment_id']==train['experiment_id']==evaluation['experiment_id']==expected_experiment
        hashes=evaluation['metadata_hashes']
        required_hashes={'model_sha256','trainer_sha256','loss_module_sha256','plan_sha256',
                         'selection_sha256','bundle_manifest_sha256','mask_manifest_sha256',
                         'initial_tensor_state_sha256','data_order_LR_trace_sha256','capacity_contract_sha256'}
        if label=='W64_w3':
            required_hashes.add('weighted_loss_module_sha256')
        assert set(hashes)==required_hashes
        for key,value in hashes.items():
            assert meta[key]==train[key]==value, f'{label} checkpoint/train/evaluation lineage hash differs: {key}'
        assert meta['training_updates']==88565 and meta['seed']==17 and meta['status']=='COMPLETED'
        assert meta['model_sha256']==train['model_sha256']==evaluation['model_sha256']==sha(Path(chain['pythonpath'])/'dual_solver_mgn.py')
        assert meta['train_steps']==list(range(45)) and meta['train_graphs']==14760
        assert meta['phase_encoding']=='mechanical' and meta['alpha']==0
        assert meta.get('airgap_weight',1.)==(3. if label=='W64_w3' else 1.)
        groups=sorted({x['geometry_group'] for x in selection['records'] if x['split']=='train'})
        dev_groups=sorted({x['geometry_group'] for x in selection['records'] if x['split']=='development'})
        assert sorted(meta['train_groups'])==groups and sorted(meta['development_groups'])==dev_groups
        group_hash=hashlib.sha256(json.dumps(meta['train_groups']).encode()).hexdigest()
        domain=DomainContract(tuple(bundle['node_features']),tuple(bundle['edge_features']),
                              ('native_element_Bx_T','native_element_By_T'),group_hash,
                              'native_element_reported','native_explicit_ID')
        model=restore_capacity(checkpoint,{d:domain for d in ('elmer','motorcad')},len(bundle['design_features']),spec['arm'],spec['width'])
        assert set(guard.gpu_compute_pids()) <= {own['pid']}, 'GPU ownership changed before model.cuda()'
        model=model.cuda().eval();model.requires_grad_(False)
        del checkpoint
        with torch.inference_mode():
            for item in selected['conditions']:
                case=item['case_index']
                for step in selected['steps']:
                    check();assert set(guard.gpu_compute_pids())<= {own['pid']}
                    graph_row=next(x for x in records[case]['files'] if x['name']==f'case{case:04d}_step{step:02d}.npz')
                    source=(bundle_path.parent/graph_row['name']).resolve();assert source.parent==bundle_path.parent.resolve()
                    assert sha(source)==graph_row['sha256'] and source.stat().st_size==graph_row['bytes']
                    with np.load(source,allow_pickle=False) as z:
                        arrays={k:z[k] for k in ('node','edge_index','edge_attr','weights','design_op','target_B_T',
                                                'native_element_id','native_tri_node_ids','native_reg_code','step','mechanical_angle_deg')}
                    assert sha(source)==graph_row['sha256']
                    common.validate_graph(arrays,operators[case],step)
                    support=next(x for x in binding['rows'] if (x['case_index'],x['step'])==(case,step))
                    assert support['graph_sha256']==graph_row['sha256']
                    graph={k:torch.as_tensor(arrays[k],device='cuda',dtype=torch.long if k=='edge_index' else torch.float32)
                           for k in ('node','edge_index','edge_attr','weights')}
                    graph['batch']=torch.zeros(len(arrays['node']),device='cuda',dtype=torch.long)
                    design=common.encoded_design(arrays['design_op'],arrays['mechanical_angle_deg'],'mechanical')
                    pred=model.forward_fields('motorcad',graph,torch.as_tensor(design,device='cuda')[None],
                                              torch.zeros(1,device='cuda',dtype=torch.long))['fields'].cpu().numpy()
                    assert pred.shape==arrays['target_B_T'].shape and np.isfinite(pred).all()
                    assert torch.cuda.max_memory_allocated()<10*1024**3
                    path=output/f'{label}_case{case:04d}_step{step:02d}.npz'
                    with path.open('xb') as f:
                        np.savez_compressed(f,native_element_id=arrays['native_element_id'],pred_B_T=pred,
                                            reference_B_T=arrays['target_B_T'],case_index=np.array(case),step=np.array(step))
                    result['files'].append(dict(model=label,case_index=case,step=step,path=str(path),sha256=sha(path),bytes=path.stat().st_size,
                                                graph_sha256=graph_row['sha256'],checkpoint_sha256=spec['checkpoint']['sha256']))
                    del graph,pred
        result['models'][label]=dict(checkpoint=spec['checkpoint'],graphs=12,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(result['files'])==36
    assert {(x['model'],x['case_index'],x['step']) for x in result['files']}=={(m,c,s) for m in release['models'] for c in (208,408,49) for s in (0,11,22,33)}
    result.update(status='COMPLETED_ALL36_FIXED_CONTOUR_INFERENCES',elapsed_seconds=time.monotonic()-start,
                  completed_UTC=dt.datetime.now(dt.timezone.utc).isoformat())
    archive=stage/'contour_inference.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for row in result['files']:
            p=Path(row['path']);assert sha(p)==row['sha256'];z.write(p,p.name)
        z.writestr('INFERENCE_MANIFEST.json',json.dumps(result,indent=2))
    result['archive']=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)
    check()
    (stage/'INFERENCE_RECEIPT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
