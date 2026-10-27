"""Compare frozen predictions with collected P2 labels; no training or selection."""
from pathlib import Path
import json,hashlib,csv,datetime
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    p=ROOT/'PREDICTIONS_FROZEN.json';frozen=read(p)
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='c238ac8b61242d2c804a8e03639df7c1ccb4aa510ba88afa42b96e58afa7f459'
    assert frozen['Full_calibration_budget']==63 and len(frozen['conditions'])==8
    conditions=frozen['conditions'];truth={};quality={};status={};rows=[]
    for c in conditions:
        key=c['case_id'];d=ROOT/'results'/key
        if not (d/'READBACK_RECEIPT.json').exists():status[key]='NOT_COLLECTED';continue
        r=read(d/'READBACK_RECEIPT.json');audit=read(d/'audit.json')
        assert r['case_id']==key and r['source_status']=='PASS' and r['native_steps']==768
        assert hashlib.sha256((d/'waveforms.npz').read_bytes()).hexdigest()==r['waveforms_sha256']
        assert hashlib.sha256((d/'raw_source_manifest.json').read_bytes()).hexdigest()==r['raw_manifest_sha256']
        target=audit['periods'][1]['AC_W'];assert np.isfinite(target) and target>0
        truth[key]=target;quality[key]=dict(current=r['current_status'],nonlinear=r['nonlinear_status'],formal='UNKNOWN');status[key]='COLLECTED'
        for method,predictions in frozen['predictions_W'].items():
            y=predictions[key];rows.append(dict(case_id=key,model=c['model'],method=method,Full_P2_AC_W=target,prediction_W=y,absolute_error_W=abs(y-target),relative_error_percent=100*(y-target)/target,current_status=r['current_status'],nonlinear_status=r['nonlinear_status'],formal_convergence='UNKNOWN',Full_training=False,Full_evaluation=True))
    metrics=[]
    for group in ['Ref','SC','ALL']:
        ids=[c['case_id'] for c in conditions if c['case_id'] in truth and (group=='ALL' or c['model']==group)]
        for method,pred in frozen['predictions_W'].items():
            metrics.append(dict(group=group,method=method,completed_n=len(ids),requested_n=8 if group=='ALL' else 4,
                wMAE_percent=(100*sum(abs(pred[k]-truth[k]) for k in ids)/sum(truth[k] for k in ids)) if ids else None,
                metric_scope='all collected targets, including flagged numerical quality; no failed-quality exclusion',
                current_nonlinear_pass_n=sum(quality[k]['current']=='PASS' and quality[k]['nonlinear']=='PASS' for k in ids)))
    with (ROOT/'per_case_errors.csv').open('w',encoding='utf-8',newline='') as f:
        names=list(rows[0]) if rows else ['case_id','model','method','Full_P2_AC_W','prediction_W','absolute_error_W','relative_error_percent','current_status','nonlinear_status','formal_convergence','Full_training','Full_evaluation']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
    seed_summary=[]
    for group in ['Ref','SC','ALL']:
        for family in ['MS_MGN','A_curl']:
            values=[m['wMAE_percent'] for m in metrics if m['group']==group and m['method'].startswith(family+'_seed') and m['wMAE_percent'] is not None]
            seed_summary.append(dict(group=group,family=family,seeds=[17,29,43],mean_wMAE_percent=float(np.mean(values)) if values else None,population_SD_percent=float(np.std(values)) if values else None,best_seed_selected=False))
    summary=dict(generated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen_predictions_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),Full_calibration_budget=63,new_evaluation_budget=8,completed=len(truth),not_collected=8-len(truth),case_status=status,metrics=metrics,quality=quality,seed_summary=seed_summary,
        formal_qualification='UNKNOWN',missing_does_not_mean_failed=True,models_retrained=False,seed_selection=False,statistical_scope='8 new OP targets,2 existing MS groups,known Ref/SC variants; not new-template generalization',exact_cross_solver_bridge=0)
    (ROOT/'EVALUATION.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(completed=len(truth),not_collected=8-len(truth))))
if __name__=='__main__':main()
