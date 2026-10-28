"""Saved-result arithmetic only; six fixed runs or INCOMPLETE_NO_CLAIM."""
from pathlib import Path
import csv,hashlib,json,statistics,sys,math

R=Path(__file__).resolve().parent
P=R.parent/'phase2_multigeometry_turns_20261027'
sys.path.insert(0,str(P))
from judge_l12_a2 import summarize


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan=json.loads((R/'CONTEXT_CG_PLAN.json').read_text())
    selection=json.loads((P/'SCALE360_SELECTION_FROZEN.json').read_text())
    pa={r['case_index']:r['native_input']['PhaseAdvance'] for r in selection['records']}
    dev={r['case_index'] for r in selection['records'] if r['split']=='development'}
    per,errors,train_metadata={},{},{}
    collection_path=R/'artifacts/COLLECTION_RECEIPT.json'
    collected={r['path']:r for r in json.loads(collection_path.read_text())['files']} if collection_path.exists() else {}
    for arm in ('C','G'):
        for seed in plan['fixed_seeds']:
            rid=f'{arm}_s{seed}';folder=R/'artifacts/runs'/rid
            try:
                paths=[folder/'train/RESULT.json',folder/'eval/RESULT.json',folder/'train_RECEIPT.json',folder/'evaluate_RECEIPT.json']
                assert all(p.exists() for p in paths),'Missing terminal training/evaluation/receipts'
                for p in paths+[folder/'eval/snapshot_errors.csv']:
                    row=collected[p.relative_to(R/'artifacts').as_posix()]
                    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Collected source hash/size mismatch'
                train,ev,tr,er=[json.loads(p.read_text()) for p in paths]
                assert train['status']==ev['status']=='COMPLETED'
                assert all(x['returncode']==0 and x['stop_reason'] is None for x in (tr,er))
                assert train['training_updates']==88565 and train['original_val_test_used']==0 and not train['old_checkpoint_reused']
                assert train['seed']==ev['seed']==seed and train['arm']==ev['arm']==arm
                assert train['context_scope']==ev['context_scope']==plan['arms'][arm]['context_scope']
                assert train['phase_encoding']==ev['phase_encoding']==plan['phase_encoding']=='mechanical'
                assert train['parameter_count']==71206 and ev['graphs_processed']==1440
                for obj in (train,ev):
                    assert obj['model_sha256']==sha(P/'dual_solver_mgn.py')
                    assert obj['context_model_sha256']==sha(R/'context_field_model.py')
                    assert obj['selection_sha256']==sha(P/'SCALE360_SELECTION_FROZEN.json')
                assert train['trainer_sha256']==sha(R/'train_context.py') and ev['evaluator_sha256']==sha(R/'evaluate_context.py')
                assert train['plan_sha256']==ev['context_plan_sha256']==sha(R/'CONTEXT_CG_PLAN.json')
                assert train['bundle_manifest_sha256']==plan['bundle_manifest_sha256']
                rows=list(csv.DictReader((folder/'eval/snapshot_errors.csv').open(encoding='utf-8')))
                keys=[(int(row['case']),int(row['step'])) for row in rows]
                assert len(keys)==1440 and set(keys)=={(case,step) for case in dev for step in range(45)}
                per[rid]=summarize(ev,rows,pa,set(plan['seen_steps']));train_metadata[rid]=train
                assert all(math.isfinite(v) for v in per[rid].values() if isinstance(v,(int,float))),'Nonfinite metric'
            except (AssertionError,KeyError,FileNotFoundError) as e:
                errors[rid]=str(e) or type(e).__name__
    for seed in plan['fixed_seeds']:
        c,g=f'C_s{seed}',f'G_s{seed}'
        if c in train_metadata and g in train_metadata:
            if train_metadata[c]['initial_tensor_state_sha256']!=train_metadata[g]['initial_tensor_state_sha256']:
                errors[f'pair_s{seed}']='Initial tensor states differ'
    decision={'verdict':'INCOMPLETE_NO_CLAIM'}
    if not errors:
        mean=lambda arm,key:statistics.mean(per[f'{arm}_s{s}'][key] for s in plan['fixed_seeds'])
        ranges={a:max(per[f'{a}_s{s}']['all45_vector'] for s in plan['fixed_seeds'])-min(per[f'{a}_s{s}']['all45_vector'] for s in plan['fixed_seeds']) for a in ('C','G')}
        delta=mean('C','all45_vector')-mean('G','all45_vector');spread=max(ranges.values())
        harm={k:{'change_G_minus_C_pp':mean('G',k)-mean('C',k),'limit_pp':v} for k,v in [('seen9_vector',.5),('all45_copper_vector',1.),('PA_ge_70_mean_case_vector',1.)]}
        for h in harm.values():h['pass']=h['change_G_minus_C_pp']<=h['limit_pp']
        improvement=delta>=1. and delta>2*spread
        decision={'arm_means':{a:{k:mean(a,k) for k in ('all45_vector','all45_magnitude','all45_vector_RMSE_T','seen9_vector','unused36_vector','all45_copper_vector','copper_B2_bias')} for a in ('C','G')},
                  'delta_C_minus_G_pp':delta,'paired_deltas':{s:per[f'C_s{s}']['all45_vector']-per[f'G_s{s}']['all45_vector'] for s in plan['fixed_seeds']},
                  'within_arm_ranges_pp':ranges,'spread_pp':spread,'range_scope':'Descriptive max-min, not CI or best-seed selection',
                  'arm_all45_min_max_percent':{a:[min(per[f'{a}_s{s}']['all45_vector'] for s in plan['fixed_seeds']),max(per[f'{a}_s{s}']['all45_vector'] for s in plan['fixed_seeds'])] for a in ('C','G')},
                  'no_harm':harm,'verdict':'G_IMPROVES_DEV' if improvement and all(h['pass'] for h in harm.values()) else 'NO_IMPROVEMENT_CLAIM'}
    result={'plan_sha256':sha(R/'CONTEXT_CG_PLAN.json'),'fixed_seeds':plan['fixed_seeds'],'runs':per,'missing_failed_or_inconsistent':errors,
            'decision':decision,'scope':'innerdev32/E1 oldTRAIN exposed, no independent test, a1 torque or AC qualification',
            'scientific_review':'Final interpretation reserved for Astra Max; saved arithmetic does not authorize another experiment'}
    (R/'CONTEXT_CG_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete_runs':len(per),'verdict':decision['verdict'],'missing_failed_or_inconsistent':errors}))


if __name__=='__main__':main()
