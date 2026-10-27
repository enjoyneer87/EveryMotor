"""Adopt the original 96-candidate/24-anchor AF policy with current Elmer data."""
from pathlib import Path
import contextlib,io,json,sys,hashlib,platform,subprocess
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import lowcurrent_necessity as audit
from jeet_acloss_rbf.ref_only_hybrid import record_result,mask_truth
from jeet_acloss_rbf.RbfModelBuilder import RbfModelBuilder as Builder
from jeet_acloss_rbf.AcLossDataset import AcLossDataset
D=R/'handoff/20261019_rea_hybrid_remote_results'
O=R/'handoff/20261019_AF_loaded96'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def main():
 O.mkdir(parents=True,exist_ok=True)
 inputs=[D/'AF_POINTS_240.json',D/'AF_FIXED_DESIGN.json',D/'AF_model_Ref_REA.json',D/'AF_model_SC_REA.json',
  R/'handoff/20261019_figure_interview/Figure6_sampling_audit/LOWCURRENT_NECESSITY_AUDIT.json',
  R/'handoff/20261019_lowcurrent_completed/LOWCURRENT_RESULTS_48.json',
  R/'handoff/20261019_lowcurrent_completed/SUMMARY.json']
 hashes={p.relative_to(R).as_posix():sha(p) for p in inputs}
 manifest={l.split('  ',1)[1]:l.split('  ',1)[0] for l in (R/'MANIFEST.sha256').read_text().splitlines()}
 for p in inputs:assert sha(p)==manifest[p.relative_to(R).as_posix()],p
 original=load(inputs[0]);past=load(inputs[4]);datasets={};source={};plans={}
 for s in ['Ref','SC']:
  rr=sorted([r for r in original if r['model']==s and r['current_Arms']>=50],key=lambda r:r['index'])
  source[s]=rr;ds=AcLossDataset([audit.point(r) for r in rr]);datasets[s]=ds
  assert len(ds)==96 and len(set(zip(ds.irms_arr,ds.phase_arr)))==24
  ip=Builder.plan_sampling_indices(ds,24,4 if s=='Ref' else 6,base_speed=16,
   placement='structured',seed=9 if s=='Ref' else 6)
  if s=='SC':ip['by_speed'][8.]=Builder.plan_sampling_indices(ds,24,3,base_speed=16,placement='structured',seed=6)['by_speed'][8.]
  plans[s]=ip
 models,trains,logs=audit.build(datasets,plans)
 masked={s:mask_truth(datasets[s],trains[s]) for s in datasets}
 masked_models,_,_=audit.build(masked,plans)
 records={};rows=[];coords={};replay={};regression={}
 for s,ds in datasets.items():
  m=models[s];record,error=record_result(ds,m,trains[s]);assert error is None,error
  records[s]=record
  own=set(map(int,trains[s]));assert len(own)==(36 if s=='Ref' else 27)
  assert len(plans[s]['base'])==24
  prior=past['scenarios']['loaded96_current_kappa_rule']['models'][s]
  pred=audit.predict(m,ds)
  np.testing.assert_allclose(pred,np.array(prior['prediction_W'])[[r['index'] for r in source[s]]],rtol=1e-12,atol=1e-7)
  delta=float(np.max(np.abs(pred-audit.predict(masked_models[s],ds))));assert delta<1e-7
  replay[s]=dict(withheld_truth_nan_max_delta_W=delta,diagnostic_scenario_replayed=True)
  meta=dict(profile='adopted_Elmer_mean_square_field_loaded96_anchor24',scale=s,base_speed_kRPM=16,
   calibration_candidate_rule='I_rms >= 50 A; 4 speeds x 4 loaded currents x 6 phase angles',
   calibration_placement='original deterministic kappa-span rule applied to current anchor AF',
   n_base=24,n_spd=4 if s=='Ref' else 3,own_train_coordinates=record['own_train_coordinates'],
   near_no_load_in_training=False,field_source='Ref Elmer MS only; bounded current matching retained',
   selection_by_error_comparison=False)
  payload=dict(_meta=meta,separable_model=dict(model='Separable_1D_2D_RBF',n_base_centers=24,**audit.export(m)))
  path=O/f'AF_model_{s}_loaded96.json';save(path,payload)
  np.testing.assert_allclose(audit.predict(audit.model(path),ds),pred,rtol=1e-13,atol=1e-7)
  coords[s]=dict(own_count=len(own),own_indices=sorted(own),own_source240_indices=[source[s][i]['index'] for i in sorted(own)],
   own_coordinates=record['own_train_coordinates'],plan=audit.packplan(plans[s],ds),
   original_submission_own_coordinates=load(audit.LEGACY)['cases']['deposited_hybrid'][s]['own_train_coordinates'])
  old={audit.key(*v) for v in coords[s]['original_submission_own_coordinates']};new={audit.key(*v) for v in record['own_train_coordinates']}
  coords[s]['own_removed_vs_submission']=sorted(old-new);coords[s]['own_added_vs_submission']=sorted(new-old)
  for i,r in enumerate(source[s]):
   v=dict(r);v.update(index=i,source240_index=r['index'],previous120_AF_prediction=r['AF_prediction'],
    previous120_prediction_W=r['prediction_W'],previous120_own_calibration=r['own_calibration'],
    AF_prediction=float(pred[i]/r['hybrid_W']),prediction_W=float(pred[i]),own_calibration=i in own)
   rows.append(v)
  samples={}
  reg=[]
  for sp in [2.,4.,8.]:
   for idx in plans[s]['by_speed'][sp]:
    i=int(idx);I=float(ds.irms_arr[i]);beta=float(ds.phase_arr[i]);g=float(m.predict_g(I,beta))
    transfer=s=='SC' and sp in [2.,4.]
    af=float(ds.af_arr[i]);extra={}
    if transfer:
     donor=datasets['Ref'];target_speed=sp*4;requested_I=I/2
     match=(np.abs(donor.irms_arr-requested_I)<.2)&(np.abs(donor.phase_arr-beta)<.1)
     mi=np.flatnonzero(match);assert len(mi)==4
     order=np.argsort(donor.speeds_k[mi]);mi=mi[order]
     hd=float(np.exp(np.interp(np.log(target_speed),np.log(donor.speeds_k[mi]),np.log(donor.h_ac_arr[mi]))))
     donor_af=float(models['Ref'].predict(target_speed*1000,requested_I,beta))
     af=donor_af*hd/float(ds.h_ac_arr[i])
     extra=dict(donor_speed_rpm=int(target_speed*1000),requested_donor_current_Arms=requested_I,
      matched_donor_current_Arms=float(donor.irms_arr[mi[0]]),donor_hybrid_W=hd*1000,donor_AF_prediction=donor_af)
    ratio=af/(g+1e-12);accepted=.3<=ratio<=3
    reg.append(dict(model=s,index=i,point_id=source[s][i]['point_id'],speed_rpm=int(sp*1000),
     current_Arms=I,beta_deg=beta,kind='transferred_predicted_loss' if transfer else 'own_Full_FEA',
     AF_regression_input=af,kappa=g,AF_over_kappa=ratio,accepted=bool(accepted),**extra))
    if accepted:samples.setdefault(sp,[]).append((af,g))
  pc,qc=Builder._fit_speed_scaling(samples,16,True,verbose=False)
  np.testing.assert_allclose(pc,m.p_coeffs,rtol=1e-10,atol=1e-10)
  np.testing.assert_allclose(qc,m.q_coeffs,rtol=1e-10,atol=1e-10)
  regression[s]=reg
  assert all(r['accepted'] for r in reg),'Plot rejected samples separately before adoption'
  assert len(reg)==(12 if s=='Ref' else 15)
 source_same={p.relative_to(R).as_posix():sha(p) for p in inputs};assert source_same==hashes
 low=[r for r in original if r['current_Arms']<50];assert len(low)==48
 exact=load(inputs[5]);assert len(exact)==48
 summaries={}
 for label,rs in [('original240_near_no_load',low),('exact_current_supplement',exact)]:
  grouped=[]
  for s in ['Ref','SC']:
   for speed in [2000,4000,8000,16000]:
    subset=[r for r in rs if r['model']==s and r['speed_rpm']==speed];assert len(subset)==6
    v=dict(model=s,speed_rpm=speed,current_Arms=subset[0]['current_Arms'],cases=[r['case_id'] for r in subset],n=6)
    for name,field in [('Full_AC','P2_AC_excess_W'),('Hybrid_AC','hybrid_W')]:
     a=np.array([r[field] for r in subset]);v.update({name+'_mean_W':float(a.mean()),name+'_min_W':float(a.min()),name+'_max_W':float(a.max()),name+'_beta_spread_percent':float(100*np.ptp(a)/a.mean())})
    grouped.append(v)
  summaries[label]=grouped
 save(O/'AF_POINTS_192.json',rows);save(O/'SAMPLING_PLAN.json',coords);save(O/'REGRESSION_INPUTS.json',regression)
 save(O/'LOWCURRENT_SEPARATE_VALIDATION.json',dict(original_excluded_count=48,exact_current_supplement_count=48,
  same_physical_SC_conditions_not_independent_validation=True,used_for_AF_fit=False,fit_speed_curve=False,
  source_paths=[str(inputs[5].relative_to(R)),str(inputs[6].relative_to(R))],groups=summaries,
  limits=load(inputs[6])['limitations'],note='Means and beta ranges, not uncertainty bounds; lines only join sampled speeds. No extra additive loss term.'))
 report=dict(schema='loaded96-adoption-v1',status='adopted_by_user',user_decision='96/24 유지 + 저전류 별도 검증 (추천)',
  host=platform.node(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
  inputs_sha256=hashes,code_sha256=sha(__file__),new_FEA_runs=0,source240_preserved=True,
  previous120_AF_models_preserved=True,calibration_policy='96/24 with original kappa-span rule; physical points selected from current REA kappa, not frozen by hand or selected by error',
  cases=records,validation=replay,original_near_no_load_count=48,separate_exact_current_supplement_count=48,
  proposed_procedure_budget=dict(Ref_MS_existing_grid=30,Ref_MS_loaded_needed=24,Ref_MS_near_no_load_auxiliary=6,
   Ref_Full_FEA=36,SC_Full_FEA=27,Ref_legacy_acquisition_total=66,Ref_loaded_only_minimum_total=60),
  limitations=['No low-current loss accuracy or P2 periodic convergence inferred from loaded AF results.',
   'All24 anchor candidates calibrate kappa; full-map wMAE includes calibration points.',
   '96 SC points retain bounded Ref-field current matching, up to 0.05 A in Ref / 0.1 A target; skin uses prescribed current.',
   'No HalfSC full-map AF model is fitted.',
   'The existing 30-case Ref MS acquisition is retained; only24 loaded MS conditions feed the main192-point AF calculation.'])
 save(O/'MODEL_REPORT.json',report)
 print(json.dumps(dict(output=str(O),metrics={s:dict(full=records[s]['full'],heldout=records[s]['heldout_own_only']) for s in records},coords_changed={s:coords[s]['own_added_vs_submission'] for s in coords})))
if __name__=='__main__':main()
