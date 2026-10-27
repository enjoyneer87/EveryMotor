"""One post-freeze evaluation, never called by training or hyperparameter search."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
RETURN=HERE.parent/'remote_return/desktop-6qivkic_20261027'


def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(truth,pred,ids):
    error=pred-truth;relative=100*error/truth;k=int(np.argmax(abs(relative)))
    return dict(n=len(truth),wMAE_percent=float(100*abs(error).sum()/truth.sum()),
        MARE_percent=float(abs(relative).mean()),max_abs_relative_percent=float(abs(relative).max()),
        RMSE_W=float(np.sqrt((error**2).mean())),mean_error_W=float(error.mean()),
        max_abs_error_W=float(abs(error).max()),worst_relative_case=ids[k])


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage1',type=Path,required=True);p.add_argument('--stage2',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False)
    freezes={}
    for name,path in [('stage1',a.stage1),('stage2',a.stage2)]:
        freeze=json.loads((path/'TRAINING_COMPLETE.json').read_text())
        assert freeze['status']=='complete' and not freeze['protected_eval_truth_seen']
        for filename,h in freeze['frozen_files'].items():assert sha(path/filename)==h
        freezes[name]=sha(path/'TRAINING_COMPLETE.json')
    split=json.loads((HERE/'SPLIT_MANIFEST.json').read_text())
    assert sha(RETURN/'comparison_ready.csv')==split['source_comparison_sha256']
    rows=list(csv.DictReader((RETURN/'comparison_ready.csv').open()))
    ids=[r['case_id'] for r in rows];assert len(set(ids))==228
    truth=np.array([float(r['Full_AC_excess_W']) for r in rows]);assert (truth>0).all()
    methods={'Hybrid_unadjusted':np.array([float(r['canonical_Hybrid_W']) for r in rows]),
             'latest_frozen_AF':np.array([float(r['frozen_AF_Hybrid_W']) for r in rows])}
    for name,path in [('stage1',a.stage1),('stage2',a.stage2)]:
        predictions=json.loads((path/'predictions_frozen.json').read_text())
        for method,values in predictions.items():
            key=method if not method.startswith('MS_MGN') else name+'_'+method
            assert set(values)==set(ids)
            methods[key]=np.array([values[k] for k in ids])
    groups=split['groups'].copy();groups['combined_development117']=groups['Ref_excluded60']+groups['SC_strict57']
    groups['calibration63_in_sample']=split['calibration_case_ids']
    results={};summary=[];rng=np.random.default_rng(20261027)
    for group,selected in groups.items():
        mask=np.array([i in selected for i in ids]);assert mask.sum()==len(selected)
        selected_ids=[i for i,m in zip(ids,mask) if m]
        results[group]={m:metrics(truth[mask],pred[mask],selected_ids) for m,pred in methods.items()}
        for family,names in [('MS_MGN_primary',[f'stage2_MS_MGN_seed{s}' for s in [17,29,43]]),
                             ('random_graph_control',[f'random_graph_seed{s}' for s in [17,29,43]]),
                             ('MS_MGN_stage1_diagnostic',[f'stage1_MS_MGN_seed{s}' for s in [17,29,43]])]:
            values=[results[group][m]['wMAE_percent'] for m in names]
            summary.append(dict(group=group,method=family,n=len(selected),wMAE_mean_percent=float(np.mean(values)),wMAE_seed_sd_percent=float(np.std(values)),seed_values=values))
        for method in ['latest_frozen_AF','scalar','Hybrid_unadjusted']:
            summary.append(dict(group=group,method=method,n=len(selected),wMAE_mean_percent=results[group][method]['wMAE_percent'],wMAE_seed_sd_percent=0))
        # Paired source-cluster intervals: duplicated scaled uses of one MS history stay together.
        source=np.array([r['MS_source_key'] for r in rows])[mask];unique=np.unique(source)
        target=truth[mask];aferr=abs(methods['latest_frozen_AF'][mask]-target)
        mgnerr=np.mean([abs(methods[f'stage2_MS_MGN_seed{s}'][mask]-target) for s in [17,29,43]],axis=0)
        interval=[]
        for _ in range(2000):
            sample=np.concatenate([np.where(source==k)[0] for k in rng.choice(unique,len(unique),replace=True)])
            interval.append(100*(mgnerr[sample]-aferr[sample]).sum()/target[sample].sum())
        results[group]['paired_cluster_interval']=dict(delta_MGN_minus_AF_wMAE_percentage_points=float(100*(mgnerr-aferr).sum()/target.sum()),
            CI95_percentile=np.percentile(interval,[2.5,97.5]).tolist(),clusters=len(unique),resamples=2000,
            interpretation='conditional on these known-template MS source clusters; not a new-geometry population confidence interval')
    write(a.out/'metrics.json',dict(frozen_before_evaluation=freezes,Full_budget_unique=63,MS_sources=24,results=results,summary=summary,
        new_final_test=False,HalfSC_scope='previously inspected historical predeclared32 transfer set; excluded from this run training/tuning; one post-freeze evaluation',
        labels_quality='P2 accepted current audit; historical period/mesh qualification UNKNOWN; new calibration-anchor convergence reported separately'))
    with (a.out/'metrics_summary.csv').open('w',newline='') as f:
        fields=['group','method','n','wMAE_mean_percent','wMAE_seed_sd_percent'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summary)
    with (a.out/'predictions228.csv').open('w',newline='') as f:
        fields=['case_id','geometry_id','MS_source_key','Full_AC_excess_W','canonical_Hybrid_W','own_calibration','transferred_coordinate','HalfSC_predeclared32','HalfSC_exploratory4']+list(methods)
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for k,r in enumerate(rows):
            out={key:r[key] for key in fields if key in r};out.update({m:float(v[k]) for m,v in methods.items()});w.writerow(out)
    write(a.out/'EVALUATION_RECEIPT.json',dict(training_freezes=freezes,split_sha256=sha(HERE/'SPLIT_MANIFEST.json'),eval_code_sha256=sha(Path(__file__)),
        source_sha256=sha(RETURN/'comparison_ready.csv'),metrics_sha256=sha(a.out/'metrics.json'),predictions_sha256=sha(a.out/'predictions228.csv'),
        no_model_changes_after_evaluation=True,random_seed_selection=False))
    print(json.dumps([r for r in summary if r['group'] in ['Ref_excluded60','SC_strict57','HalfSC_predeclared32'] and r['method'] in ['latest_frozen_AF','scalar','MS_MGN_primary']],indent=2))


if __name__=='__main__':main()
