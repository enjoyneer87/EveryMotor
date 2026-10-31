"""Candidate-only twelve native snapshots; no optimizer, control rerun or repair."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import platform
import shutil
import zipfile
import psutil
import numpy as np

BASE=Path('C:/work/EveryMotor_NextLR20_20261031_01a18cac')
CHAIN_SHA='3fc6975ade9d535728a770210f72f1e413032ea7fbe1943e7540e1fb7558ca32'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True);args=parser.parse_args()
    started=time.monotonic();own=dict(pid=os.getpid(),create_time=psutil.Process().create_time())
    assert args.release.resolve()==(BASE/'CONTOUR_LR20_RELEASE.json').resolve()
    def digest(path,limit=32*1024**2):
        p=Path(path);assert p.is_file() and p.stat().st_size<=limit
        with p.open('rb') as f:b=f.read(limit+1)
        assert len(b)<=limit
        return hashlib.sha256(b).hexdigest()
    def pinned(p):
        path=Path(p['path']);assert path.suffix.lower()=='.json' and path.stat().st_size==p['bytes']
        assert digest(path)==p['sha256'];return json.loads(path.read_text(encoding='utf-8-sig'))
    def same(a,b):
        assert Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256'] and a['bytes']==b['bytes']
    assert digest(args.release)==args.release_sha256
    release=json.loads(args.release.read_text(encoding='utf-8-sig'))
    assert release['status']=='RELEASED_FINAL_LR20_SEED17_FIXED12_ONLY'
    assert release['retry']==0 and release['GPU_workers']==1 and release['new_optimizer_updates']==0
    assert release['runner_max_seconds']==360 and release['contour_transport_max_seconds']==600
    assert digest(Path(__file__))==release['runner_sha256']
    chain=pinned(release['chain']);assert release['chain']['sha256']==CHAIN_SHA
    for p in chain['source_files']:
        path=Path(p['path']).resolve();assert path.is_relative_to((BASE/'src').resolve()) and path.suffix=='.py'
        assert path.stat().st_size==p['bytes'] and digest(path)==p['sha256']
    sys.path.insert(0,str(BASE/'src'))
    from stage_clock import elapsed_stage_seconds
    import run_width128_chain as safety
    clock=pinned(chain['stage_clock'])
    def check(minimum_remaining=0):
        assert time.monotonic()-started<360
        now=dt.datetime.now(dt.timezone.utc)
        spent=elapsed_stage_seconds(clock,time.monotonic(),now,platform.node(),psutil.boot_time())
        assert min(28800-spent,(dt.datetime.fromisoformat(chain['deadline_UTC'])-now).total_seconds())>minimum_remaining
        assert shutil.disk_usage(BASE).free>=10*1024**3
        assert all(not Path(p).exists() for p in chain['stop_files']) and not (BASE/'CONTOUR_STOP').exists()
        assert set(safety.gpu_compute_pids())<={own['pid']}
    check(600);assert not safety.gpu_compute_pids()
    receipt=pinned(release['scientific_receipt']);bootstrap=pinned(release['bootstrap_receipt']);verdict=pinned(release['verdict'])
    assert receipt['status']==bootstrap['status']=='COMPLETED' and receipt['phase']=='scientific'
    assert receipt['chain_sha256']==bootstrap['chain_sha256']==verdict['chain_sha256']==CHAIN_SHA
    assert receipt['child_exit_verified'] and bootstrap['child_exit_verified'] and bootstrap['guard_returncode']==bootstrap['judge_returncode']==0
    same(bootstrap['guard_receipt'],release['scientific_receipt']);same(bootstrap['verdict'],release['verdict'])
    main_release=pinned(release['main_release']);assert main_release['chain_sha256']==CHAIN_SHA
    assert main_release['status']=='RELEASED_FIXED_LR20_THREE_SEED_RUNS'
    assert verdict['status']=='COMPLETE' and verdict['fixed_seeds']==[17,29,43] and verdict['judge_sha256']==chain['judge_sha256']
    assert verdict['scientific_receipt_sha256']==release['scientific_receipt']['sha256']
    assert len(receipt['jobs'])==6 and {(j['seed'],j['kind']) for j in receipt['jobs']}=={(s,k) for s in (17,29,43) for k in ('train','eval')}
    for row in receipt['jobs']:
        assert row['status']=='COMPLETED' and row['returncode']==0
        result=pinned(row['result']);assert result['status']=='COMPLETED'
        assert result['seed']==row['seed'] and result['experiment_id']=='W128_LR20_PHASE45_20261031'
        if row['kind']=='train':assert result['training_updates']==88565
        else:assert result['graphs_processed']==1440
    for identity in [receipt['owner'],bootstrap['owner'],bootstrap['guard']]+[row['owner'] for row in receipt['jobs']]:
        try:assert psutil.Process(identity['pid']).create_time()!=identity['create_time'],'training identity remains live'
        except psutil.NoSuchProcess:pass
    selected=pinned(release['selection']);binding=pinned(release['binding'])
    assert selected['seed']==17 and selected['steps']==[0,11,22,33] and [x['case_index'] for x in selected['conditions']]==[208,408,49]
    assert binding['status']=='PASS_ALL12_STORED_NATIVE_GEOMETRY_BINDINGS'
    spec=release['candidate'];train=pinned(spec['train']);ev=pinned(spec['eval'])
    for kind in ('train','eval'):
        same(spec[kind],next(row['result'] for row in receipt['jobs'] if row['seed']==17 and row['kind']==kind))
        same(spec[kind],verdict['evidence_pins']['candidate']['17'][kind])
    assert ev['original_train_sha256']==spec['train']['sha256'] and ev['reproduction_gate']['status']=='PASS_ARITHMETIC_ONLY'
    assert train['checkpoint_sha256']==ev['checkpoint_sha256']==spec['checkpoint']['sha256']
    cp=Path(spec['checkpoint']['path']).resolve()
    assert cp==(BASE/'runs/W128_s17/train/last.pt').resolve() and cp.stat().st_size==spec['checkpoint']['bytes']
    assert digest(cp)==spec['checkpoint']['sha256']
    template=pinned(chain['evaluation_template']);selection=pinned(template['selection'])
    bundle_path=Path(template['bundle_dir'])/'BUNDLE_MANIFEST.json'
    assert digest(bundle_path)==template['bundle_manifest_sha256']
    bundle=json.loads(bundle_path.read_text());records={r['case_index']:r for r in bundle['records']}
    development={r['case_index']:r for r in selection['records'] if r['split']=='development'}
    geometry=pinned(template['geometry_manifest'])
    import torch
    import evaluate_common_airgap as common
    from capacity_contract import restore_capacity
    from dual_solver_mgn import DomainContract
    assert torch.__version__=='2.5.1+cu124' and np.__version__=='1.26.4'
    torch.set_num_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=torch.load(cp,map_location='cpu',weights_only=False);meta=checkpoint['metadata']
    assert checkpoint['epoch']==30 and meta['training_updates']==88565 and meta['status']=='COMPLETED'
    assert meta['seed']==17 and meta['width']==128 and meta['depth']==3 and meta['arm']=='W128' and meta['alpha']==0
    assert meta['experiment_id']==train['experiment_id']==ev['experiment_id']=='W128_LR20_PHASE45_20261031'
    assert meta['train_steps']==list(range(45)) and meta['train_graphs']==14760 and meta['phase_encoding']=='mechanical'
    required={'model_sha256','trainer_sha256','loss_module_sha256','plan_sha256','selection_sha256','bundle_manifest_sha256','mask_manifest_sha256','initial_tensor_state_sha256','data_order_LR_trace_sha256','capacity_contract_sha256'}
    assert set(ev['metadata_hashes'])==required
    for k,v in ev['metadata_hashes'].items():assert meta[k]==train[k]==v
    for k,v in chain['operator_identity'].items():assert ev[k]==v
    assert meta['model_sha256']==digest(BASE/'src/dual_solver_mgn.py')
    assert meta['plan_sha256']==chain['training_plan']['sha256']
    train_groups=sorted({r['geometry_group'] for r in selection['records'] if r['split']=='train'})
    dev_groups=sorted({r['geometry_group'] for r in selection['records'] if r['split']=='development'})
    assert sorted(meta['train_groups'])==train_groups and sorted(meta['development_groups'])==dev_groups
    group_hash=hashlib.sha256(json.dumps(meta['train_groups']).encode()).hexdigest()
    domain=DomainContract(tuple(bundle['node_features']),tuple(bundle['edge_features']),('native_element_Bx_T','native_element_By_T'),group_hash,'native_element_reported','native_explicit_ID')
    work=(BASE/'contour_LR20_R1').resolve();assert not work.exists();work.mkdir()
    result=dict(status='RUNNING',owner=own,release_sha256=args.release_sha256,qualification='STORED_NATIVE_HDF_REPLAY_ONLY',source_vector_basis='UNKNOWN',seed=17,files=[],new_optimizer_updates=0,new_FEA=0,retry=0)
    with (work/'OWNER.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    try:
        check(600);model=restore_capacity(checkpoint,{d:domain for d in ('elmer','motorcad')},len(bundle['design_features']),'W128',128).cuda().eval()
        model.requires_grad_(False);del checkpoint
        operators={}
        for item in selected['conditions']:
            case=item['case_index'];assert development[case]['source_sha256']==records[case]['source_sha256']==item['source_sha256']
            op_row=next(r for r in geometry['records'] if r['case_index']==case);op_root=Path(template['geometry_manifest']['path']).parent.resolve()
            op_path=(op_root/op_row['operator_file']).resolve();assert op_path.parent==op_root and op_row['status']=='ADMITTED'
            assert digest(op_path)==op_row['operator_sha256'];operators[case]=common.load_operator(op_path)
        with torch.inference_mode():
            for item in selected['conditions']:
                case=item['case_index']
                for step in selected['steps']:
                    check();row=next(x for x in records[case]['files'] if x['name']==f'case{case:04d}_step{step:02d}.npz')
                    path=(bundle_path.parent/row['name']).resolve();assert path.parent==bundle_path.parent.resolve()
                    assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
                    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in ('node','edge_index','edge_attr','weights','design_op','target_B_T','native_element_id','native_tri_node_ids','native_reg_code','step','mechanical_angle_deg')}
                    assert digest(path)==row['sha256'];common.validate_graph(arrays,operators[case],step)
                    support=next(x for x in binding['rows'] if (x['case_index'],x['step'])==(case,step));assert support['graph_sha256']==row['sha256']
                    graph={k:torch.as_tensor(arrays[k],device='cuda',dtype=torch.long if k=='edge_index' else torch.float32) for k in ('node','edge_index','edge_attr','weights')}
                    graph['batch']=torch.zeros(len(arrays['node']),device='cuda',dtype=torch.long)
                    design=common.encoded_design(arrays['design_op'],arrays['mechanical_angle_deg'],'mechanical')
                    pred=model.forward_fields('motorcad',graph,torch.as_tensor(design,device='cuda')[None],torch.zeros(1,device='cuda',dtype=torch.long))['fields'].cpu().numpy()
                    assert pred.shape==arrays['target_B_T'].shape and np.isfinite(pred).all() and torch.cuda.max_memory_allocated()<10*1024**3
                    output=work/f'W128_LR20_case{case:04d}_step{step:02d}.npz'
                    with output.open('xb') as f:np.savez_compressed(f,native_element_id=arrays['native_element_id'],pred_B_T=pred,reference_B_T=arrays['target_B_T'],case_index=np.array(case),step=np.array(step))
                    result['files'].append(dict(model='W128_LR20',case_index=case,step=step,path=str(output),sha256=digest(output),bytes=output.stat().st_size,graph_sha256=row['sha256'],checkpoint_sha256=spec['checkpoint']['sha256']))
                    del graph,pred
        assert len(result['files'])==12 and {(x['case_index'],x['step']) for x in result['files']}=={(c,s) for c in (208,408,49) for s in (0,11,22,33)}
        result.update(status='COMPLETED_FIXED12_LR20_CANDIDATE_ONLY',peak_cuda_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
        check();archive=work/'candidate_fixed12.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for row in result['files']:
                p=Path(row['path']);assert digest(p)==row['sha256'];z.write(p,p.name)
            z.writestr('INFERENCE_MANIFEST.json',json.dumps(result,indent=2))
        result['archive']=dict(path=str(archive),sha256=digest(archive),bytes=archive.stat().st_size)
        check()
    except BaseException as exc:
        result.update(status='INCOMPLETE_NO_CLAIM',error=repr(exc));raise
    finally:
        result.update(finished_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        with (work/'INFERENCE_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],files=len(result['files']),elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
