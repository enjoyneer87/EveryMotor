"""Freeze unseen-speed OP inputs and predictions before any new Full solve."""
from pathlib import Path
import ast,datetime,hashlib,json,math,sys,csv
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PRIMARY=RESEARCH/'mgn_development_20261027';FOLLOW=RESEARCH/'mgn_followup_20261027';SOURCES=RESEARCH/'baseline_audit/sources'
sys.path.insert(0,str(SOURCES));sys.path.insert(0,str(PRIMARY))
from jeet_acloss_rbf.rea_hybrid.e10_sampling import sampling_geometry,evaluate_active_samples
from residual import feature_matrix,predict
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (ROOT/'PREDICTIONS_FROZEN.json').exists(),'Never replace frozen predictions'
    contract=json.loads((SOURCES/'jeet_acloss_rbf/rea_hybrid/examples/e10_ref_input.json').read_text());g=sampling_geometry(contract['geometry_settings_mm'])
    cache={k:np.load(SOURCES/'handoff/20261019_rea_hybrid_remote_results/samples'/f'{k}.npz')['B_magnitude_T'] for k in ['Ref_Ref_P105','Ref_Ref_P117']}
    old=json.loads((PRIMARY/'artifacts/inputs/inference228.json').read_text());sources=json.loads((PRIMARY/'SPLIT_MANIFEST.json').read_text())['MS_sources']
    for r in old:
        if r['model']=='HalfSC' or r['source_key'] not in cache:continue
        scale={'Ref':1.,'SC':2.}[r['model']];c=dict(contract,Ref_current_Arms=r['current']/scale)
        val=evaluate_active_samples(cache[r['source_key']],g,c,r['current'],r['speed'],scale)['hybrid_W'];assert abs(val-r['hybrid'])<1e-6
    cases=[]
    for model,scale in [('Ref',1.),('SC',2.)]:
        for normcurrent,key in [(230.05,'Ref_Ref_P105'),(460.,'Ref_Ref_P117')]:
            for speed in [6000.,12000.]:
                current=normcurrent*scale;case_id=f'{model}_I{current:g}_B36_N{speed:g}'
                assert not any(x['model']==model and x['current']==current and x['beta']==36 and x['speed']==speed for x in old)
                hy=evaluate_active_samples(cache[key],g,dict(contract,Ref_current_Arms=normcurrent),current,speed,scale)
                cases.append(dict(case_id=case_id,model=model,current=current,beta=36.,speed=speed,source_key=key,hybrid=hy['hybrid_W'],
                    normalized_Ref_current_Arms=normcurrent,radial_factor=scale,source_current_mapping='exact_for_new_condition',geometry_group='JEET_Ref_family_same_template',
                    numerical_steps_per_period=256,numerical_periods=3,Full_training=False,Full_evaluation=True))
    assert len(cases)==8
    text=(RESEARCH/'baseline_audit/replay_baseline.py').read_text();defs=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ['poly','predict']];ns={'math':math};exec(compile(ast.Module(body=defs,type_ignores=[]),'pinned_AF','exec'),ns)
    af={m:json.loads((SOURCES/f'handoff/20261019_AF_loaded96/AF_model_{m}_loaded96.json').read_text()) for m in ['Ref','SC']}
    predictions={'latest_AF':{r['case_id']:r['hybrid']*ns['predict'](af[r['model']],r['speed'],r['current'],r['beta']) for r in cases}}
    hashes={f'AF_{m}':sha(SOURCES/f'handoff/20261019_AF_loaded96/AF_model_{m}_loaded96.json') for m in af}
    methods=[('scalar',PRIMARY/'artifacts/stage1',None)]+[(f'MS_MGN_seed{s}',PRIMARY/'artifacts/stage2','MS_pretrained') for s in [17,29,43]]+[(f'A_curl_seed{s}',FOLLOW/'artifacts','A_curl') for s in [17,29,43]]
    for name,folder,kind in methods:
        head=folder/(name+'_head.npz');h=dict(np.load(head));emb=keys=None
        if kind:
            z=np.load(folder/f'embeddings_seed{name.split("seed")[-1]}.npz');emb=z[kind];keys=z['source_keys'];hashes[name+'_embeddings']=sha(folder/f'embeddings_seed{name.split("seed")[-1]}.npz')
        vals=np.array([r['hybrid'] for r in cases])*np.exp(predict(h,feature_matrix(cases,emb,keys)));assert np.isfinite(vals).all() and (vals>0).all();predictions[name]=dict(zip([r['case_id'] for r in cases],vals.tolist()));hashes[name+'_head']=sha(head)
    out=dict(frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(ROOT/'PLAN.json'),code_sha256=sha(__file__),
        Full_calibration_budget=63,new_Full_evaluation_budget=8,new_target_values_seen=False,models_retrained=False,HalfSC_reevaluated=False,
        conditions=cases,predictions_W=predictions,model_hashes=hashes,MS_sources=[x for x in sources if x['source_key'] in cache],
        cache_hashes={k:sha(SOURCES/'handoff/20261019_rea_hybrid_remote_results/samples'/f'{k}.npz') for k in cache},
        canonical_evaluator_hash=sha(SOURCES/'jeet_acloss_rbf/rea_hybrid/e10_sampling.py'),canonical_contract_hash=sha(SOURCES/'jeet_acloss_rbf/rea_hybrid/examples/e10_ref_input.json'),
        limitation='Known geometry and2knownMS source groups at new speeds. Not an independent-template holdout. All8targets unavailable at freeze time.')
    (ROOT/'PREDICTIONS_FROZEN.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    with (ROOT/'case_queue.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0])+['status','max_seconds','max_retries','owner','stop']);w.writeheader()
        for r in cases:w.writerow(dict(r,status='INPUTS_FROZEN_NOT_LAUNCHED',max_seconds=5400,max_retries=0,owner='new_OP_iso_worker_pending',stop='ownedSTOP;UTC2026-10-27T23:00Z'))
    print(json.dumps(dict(cases=8,models=len(predictions),Full_calibration=63,frozen_sha256=sha(ROOT/'PREDICTIONS_FROZEN.json'))))
if __name__=='__main__':main()
