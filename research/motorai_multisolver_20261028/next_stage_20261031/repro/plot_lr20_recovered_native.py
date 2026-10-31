"""Exact retained native polygons, saved LR30 and final LR20; fixed common scales."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'main_training_visual_20261030_01a18cac'
NAS=Path('N:/MotorAI_NextLR20_20261031_01a18cac').resolve()


def sha(p):
    p=Path(p);assert p.stat().st_size<=32*1024**2
    with p.open('rb') as f:b=f.read(32*1024**2+1)
    assert len(b)<=32*1024**2;return hashlib.sha256(b).hexdigest()
def read(p):
    p=Path(p);assert p.stat().st_size<=2*1024**2
    with p.open('rb') as f:b=f.read(2*1024**2+1)
    assert len(b)<=2*1024**2;return json.loads(b.decode('utf-8-sig'))


def main():
    assert not any((ROOT/n).exists() for n in ('CONTOUR_LR20_RECOVERY_FAILURE.json','CONTOUR_LR20_RECOVERY_GUARD_FAILURE.json'))
    execution_path=ROOT/'CONTOUR_LR20_RECOVERY_RENDER_EXECUTION_CONTRACT.json';execution=read(execution_path)
    assert execution['status']=='RELEASED_HASH_BOUND_AUXILIARY_RECOVERED_NATIVE_RENDER'
    assert execution['renderer_sha256']==sha(Path(__file__)) and execution['helpers_sha256']==sha(ROOT/'native_plot_helpers.py')
    captures={}
    for p in execution['input_pins']:
        path=Path(p['path']).resolve()
        assert path.is_relative_to(ROOT.resolve()) or path.is_relative_to(OLD.resolve()) or path.is_relative_to(NAS.resolve())
        assert path.stat().st_size==p['bytes'] and sha(path)==p['sha256'];captures[path]=p['sha256']
    plan=read(ROOT/'CONTOUR_LR20_RENDER_PLAN.json')
    assert plan['status']=='FROZEN_LR20_SHARED_NATIVE_RENDER_PLAN_BEFORE_NEW_FIELD_READS'
    assert plan['cases']==[208,408,49] and plan['steps']==[0,11,22,33] and plan['seed']==17
    pointer=read(ROOT/'CONTOUR_LR20_RECOVERY_POINTER.json');collection_path=Path(pointer['path']).resolve()
    assert collection_path==NAS/'contour_readonly_recovery_R1/READ_ONLY_RECOVERY_RECEIPT.json' and sha(collection_path)==pointer['sha256']
    collection=read(collection_path);assert collection['status']=='PASS_RECOVERED_FIXED12_ARTIFACT_HASH_AND_FIELD_BINDING_ONLY'
    guard_path=Path(pointer['guard_receipt']['path']).resolve();assert guard_path==ROOT/'CONTOUR_LR20_RECOVERY_GUARD_RECEIPT.json'
    assert sha(guard_path)==pointer['guard_receipt']['sha256']==execution['guard_receipt_sha256']
    guard=read(guard_path);assert guard['status']=='PASS_OWNED_JOB_BOUNDED_READ_ONLY_RECOVERY' and guard['elapsed_seconds']<180
    exit_path=ROOT/'CONTOUR_LR20_RECOVERY_GUARD_EXIT_ATTESTATION.json';exit_proof=read(exit_path)
    assert sha(exit_path)==execution['guard_exit_attestation_sha256']
    assert exit_proof['status']=='PASS_GUARD_OS_EXIT0_OBSERVED_WITHIN180' and exit_proof['OS_exit_code']==0 and exit_proof['observed_total_seconds']<180
    assert exit_proof['guard_receipt_sha256']==sha(guard_path) and exit_proof['canonical_pointer_sha256']==sha(ROOT/'CONTOUR_LR20_RECOVERY_POINTER.json')
    assert guard['source_sha256']==exit_proof['guard_source_sha256']==sha(ROOT/'guard_lr20_readonly_recovery.py')
    assert guard['contract_sha256']==exit_proof['execution_contract_sha256']==sha(ROOT/'CONTOUR_LR20_RECOVERY_EXECUTION_CONTRACT.json')
    assert guard['pending_pointer_sha256']==sha(ROOT/'CONTOUR_LR20_RECOVERY_PENDING_POINTER.json')
    assert guard['worker']['returncode']==0 and not guard['worker']['timed_out'] and guard['worker']['all_owned_job_processes_exited']
    assert guard['worker']['peak_job_commit_bytes']<1024**3 and collection['peak_local_RSS_bytes']<1024**3
    assert collection['original_operational_status']=='INCOMPLETE_NO_CLAIM' and collection['process_ancestry']=='UNKNOWN' and collection['auxiliary_visualization_only'] is True
    binding_path=Path(collection['field_binding']['path']).resolve();assert binding_path==collection_path.parent/'EXACT_FIELD_BINDING_RECOVERY_RECEIPT.json'
    assert sha(binding_path)==collection['field_binding']['sha256']
    recovered=read(binding_path);assert recovered['status']=='PASS_ALL12_EXACT_RECOVERED_FIELD_BINDINGS' and len(recovered['rows'])==12
    proof={(p['case_index'],p['step']):p for p in recovered['rows']}
    assert len(proof)==12 and set(proof)=={(c,s) for c in (208,408,49) for s in (0,11,22,33)}
    manifest_path=Path(collection['manifest']['path']).resolve();assert manifest_path==collection_path.parent/'INFERENCE_MANIFEST.json'
    assert sha(manifest_path)==collection['manifest']['sha256'];candidate=read(manifest_path)
    assert candidate['status']=='COMPLETED_FIXED12_LR20_CANDIDATE_ONLY' and len(candidate['files'])==12
    assert candidate['release_sha256']==collection['release_sha256']==sha(ROOT/'CONTOUR_LR20_RELEASE.json')
    release=read(ROOT/'CONTOUR_LR20_RELEASE.json')
    assert sha(ROOT/'CHAIN_LR20.json')=='3fc6975ade9d535728a770210f72f1e413032ea7fbe1943e7540e1fb7558ca32'
    chain=read(ROOT/'CHAIN_LR20.json');control_train_pin=execution['control_train']
    control_train_path=Path(control_train_pin['path']).resolve()
    assert control_train_path.is_relative_to((OLD/'completed_return_20261030_1615').resolve())
    assert control_train_path.stat().st_size==control_train_pin['bytes']==chain['controls']['17']['train']['bytes']
    assert sha(control_train_path)==control_train_pin['sha256']==chain['controls']['17']['train']['sha256']
    control_train=read(control_train_path)
    assert control_train['status']=='COMPLETED' and control_train['seed']==17 and control_train['width']==128 and control_train['training_updates']==88565
    assert control_train['checkpoint_sha256']==execution['control_checkpoint_sha256']
    control=read(OLD/'contour_inference/INFERENCE_MANIFEST.json')
    assert control['status']=='COMPLETED_ALL36_FIXED_CONTOUR_INFERENCES' and len(control['files'])==36
    binding=read(OLD/'CONTOUR_GEOMETRY_BINDING_RESULT.json');assert binding['status']=='PASS_ALL12_STORED_NATIVE_GEOMETRY_BINDINGS'
    assert sha(OLD/'CONTOUR_GEOMETRY_BINDING_RESULT.json')==plan['binding']['sha256']
    selection=read(OLD/'CONTOUR_SELECTION.json');assert sha(OLD/'CONTOUR_SELECTION.json')==plan['selection']['sha256']
    required_metadata={
        (ROOT/'CONTOUR_LR20_RENDER_PLAN.json').resolve(),(ROOT/'CONTOUR_LR20_RECOVERY_POINTER.json').resolve(),collection_path,manifest_path,
        (ROOT/'CONTOUR_LR20_RELEASE.json').resolve(),(ROOT/'CHAIN_LR20.json').resolve(),control_train_path,
        (OLD/'contour_inference/INFERENCE_MANIFEST.json').resolve(),(OLD/'CONTOUR_GEOMETRY_BINDING_RESULT.json').resolve(),(OLD/'CONTOUR_SELECTION.json').resolve()}
    required_metadata.update({binding_path,Path(collection['review']['path']).resolve(),guard_path,exit_path,(ROOT/'guard_lr20_readonly_recovery.py'),(ROOT/'CONTOUR_LR20_RECOVERY_EXECUTION_CONTRACT.json'),(ROOT/'CONTOUR_LR20_RECOVERY_PENDING_POINTER.json')})
    assert sha(Path(collection['review']['path']))==collection['review']['sha256']
    assert required_metadata<=set(captures),'exact required metadata inventory must be pinned, not just an optional list'
    # Library loading is postponed until completed collection and every source/input pin pass.
    sys.path.insert(0,str(OLD/'runtime_lib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    import native_plot_helpers as helpers
    font=Path('C:/Windows/Fonts/malgun.ttf');font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'font.size':9})
    def arrays(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    cache={};array_pins=[]
    for case in plan['cases']:
        for step in plan['steps']:
            support=next(x for x in binding['rows'] if (x['case_index'],x['step'])==(case,step))
            gp=Path(support['output']['path']).resolve()
            assert gp.is_relative_to((OLD/'contour_joined').resolve()) and gp.suffix=='.npz'
            assert sha(gp)==support['output']['sha256'];g=arrays(gp);values={}
            for label,manifest,folder in (('LR30',control,OLD/'contour_inference'),('LR20',candidate,collection_path.parent)):
                model='W128_w1' if label=='LR30' else 'W128_LR20'
                row=next(x for x in manifest['files'] if (x['model'],x['case_index'],x['step'])==(model,case,step))
                filename=f'{model}_case{case:04d}_step{step:02d}.npz';assert Path(row['path']).name==filename
                path=(folder/filename).resolve();assert path.parent==folder.resolve()
                assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
                bound=proof[case,step]['candidate' if label=='LR20' else 'control']
                assert Path(bound['path']).resolve()==path and bound['sha256']==row['sha256'] and bound['bytes']==row['bytes']
                joined_pin=proof[case,step]['joined'];assert Path(joined_pin['path']).resolve()==gp and joined_pin['sha256']==support['output']['sha256']
                assert proof[case,step]['graph_sha256']==support['graph_sha256']
                a=arrays(path);assert np.array_equal(a['native_element_id'],g['native_element_id'])
                assert int(a['case_index'])==case and int(a['step'])==step and row['graph_sha256']==support['graph_sha256']
                if label=='LR20':assert row['checkpoint_sha256']==release['candidate']['checkpoint']['sha256']
                else:assert row['checkpoint_sha256']==execution['control_checkpoint_sha256']
                if 'reference' in values:assert np.array_equal(values['reference'],a['reference_B_T'])
                values['reference']=a['reference_B_T'].astype(np.float64);values[label]=a['pred_B_T'].astype(np.float64)
                assert values[label].shape==values['reference'].shape==(len(g['native_element_id']),2)
                assert np.isfinite(values[label]).all() and np.isfinite(values['reference']).all()
                array_pins.append(dict(path=str(path),sha256=row['sha256'],bytes=row['bytes']))
            assert g['vertex_xy_mm'].shape==(len(g['native_element_id']),3,2) and len(g['weights'])==len(g['native_element_id'])
            cache[case,step]=(g,values)
    output_directory=Path(plan['output_assets_directory']).resolve();assert output_directory==NAS/'report_assets_R1'
    assert not any((ROOT/n).exists() for n in ('CONTOUR_LR20_RECOVERY_FAILURE.json','CONTOUR_LR20_RECOVERY_GUARD_FAILURE.json'))
    output_directory.mkdir(exist_ok=True,parents=True)
    result=dict(status='RUNNING',started_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),renderer_sha256=sha(Path(__file__)),helpers_sha256=execution['helpers_sha256'],
                execution_contract_sha256=sha(execution_path),plan_sha256=sha(ROOT/'CONTOUR_LR20_RENDER_PLAN.json'),qualification='STORED_NATIVE_HDF_REPLAY_ONLY',source_vector_basis='UNKNOWN',native_BC_sign_pair='UNKNOWN',
                smoothing=False,rotation=False,wrap=False,replication=False,new_inference=0,new_control_inference=0,plots=[],clipping=[],array_pins=array_pins)
    for case in plan['cases']:
        item=next(x for x in selection['conditions'] if x['case_index']==case)
        points=np.concatenate([cache[case,s][0]['vertex_xy_mm'].reshape(-1,2) for s in plan['steps']])
        whole=(points[:,0].min()-.5,points[:,0].max()+.5,points[:,1].min()-.5,points[:,1].max()+.5)
        for mode in plan['modes']:
            for crop in ((False,True) if mode=='norm' else (False,)):
                steps=[22] if crop else plan['steps']
                fig,axes=plt.subplots(len(steps),5,figsize=(16,3.3 if crop else 11),squeeze=False,layout='constrained')
                for row_index,step in enumerate(steps):
                    g,v=cache[case,step];reference=v['reference'];arms=[reference,v['LR30'],v['LR20']]
                    if mode=='norm':
                        fields=[np.linalg.norm(x,axis=1) for x in arms];errors=[np.linalg.norm(x-reference,axis=1) for x in arms[1:]]
                        field_limits=plan['limits']['stored_B_component_norm_T'];error_limits=plan['limits']['stored_vector_error_norm_T'];title='저장성분 norm';error_label='||ΔB||';cmap='viridis'
                    else:
                        k={'Bx':0,'By':1}[mode];fields=[x[:,k] for x in arms];errors=[np.abs(x[:,k]-reference[:,k]) for x in arms[1:]]
                        field_limits=plan['limits']['Stored_'+mode+'_T'];error_limits=plan['limits']['absolute_component_error_T'];title='Stored '+mode;error_label='|Δ'+mode+'|';cmap='RdBu_r'
                    bounds=helpers.a1_crop(g['vertex_xy_mm'],g['a1_mask']) if crop else whole
                    panels=['저장 FEM 기준','W128 / LR30','W128 / LR20','LR30 '+error_label,'LR20 '+error_label]
                    for col,(value,label) in enumerate(zip(fields+errors,panels)):
                        limits=field_limits if col<3 else error_limits
                        artist,clipping=helpers.paint(axes[row_index,col],g['vertex_xy_mm'],value,limits,cmap if col<3 else 'inferno',f'step {step} · '+label,bounds,g['weights'])
                        if row_index==0:fig.colorbar(artist,ax=axes[:,col].tolist(),shrink=.8,label='T',extend='both' if col<3 and mode!='norm' else 'max')
                        result['clipping'].append(dict(case_index=case,step=step,mode=mode,a1_crop=crop,panel=label,**clipping))
                name=f'lr20_recovered_native_case{case:04d}_{mode}'+('_a1crop' if crop else '')+'.png';path=output_directory/name;assert not path.exists()
                scalar_note='scalar 진단: '+recovered['scalar_status']+' · 전체24값 표 참조 · CUDA 결정성 미검증'
                fig.suptitle(f"case {case} · PA {item['native_input']['PhaseAdvance']:.3f}° · {title} · "+('a1 기하 기준 확대' if crop else '전체 native support')+'\n운영 실패 후 회수한 저장 산출물 · 프로세스 계보 UNKNOWN · 보조 시각화\nseed17 final88565 · dev32 노출 이력 · basis/BC/sign/pair UNKNOWN · 회색=제외/미지원 공간(a2 polygon 아님)\n'+scalar_note,fontsize=10)
                fig.savefig(path,dpi=145,facecolor='white');plt.close(fig)
                result['plots'].append(dict(case_index=case,mode=mode,a1_crop=crop,path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    result.update(original_operational_status='INCOMPLETE_NO_CLAIM',process_ancestry='UNKNOWN',auxiliary_visualization_only=True,recovery_receipt_sha256=sha(collection_path),guard_exit_attestation_sha256=sha(exit_path),scalar_status=recovered['scalar_status'],scalar_diagnostics=[{k:p[k] for k in ('case_index','step','scalar_status','scalar_comparison')} for p in recovered['rows']])
    assert len(result['plots'])==12
    for path,digest in captures.items():assert sha(path)==digest,'pinned metadata changed while rendering'
    for p in array_pins:assert sha(p['path'])==p['sha256'],'native field array changed while rendering'
    result.update(status='COMPLETED_AUXILIARY_RECOVERED_NATIVE_RENDERING_WITH_OPERATIONAL_FAILURE',finished_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),numpy=np.__version__,matplotlib=matplotlib.__version__,visual_QA='PENDING_STANDALONE_PNG_INSPECTION')
    with (ROOT/'CONTOUR_LR20_RECOVERED_RENDER_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],plots=12)))


if __name__=='__main__':main()
