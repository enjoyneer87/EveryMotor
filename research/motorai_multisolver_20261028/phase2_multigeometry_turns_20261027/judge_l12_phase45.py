"""Apply the preregistered L12 decision rule (COMMON_EVALUATION_CONTRACT.json) to collected run results; reads saved files only."""
from pathlib import Path
import json,csv,hashlib,statistics as st,collections
R=Path(__file__).parent;L=R/'master_plan_artifacts'/'l12'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(run):
 ev=json.loads((L/run/'eval_RESULT.json').read_text());rows=list(csv.DictReader(open(L/run/'snapshot_errors.csv',encoding='utf-8')))
 return ev,rows
def main():
 plan=json.loads((R/'L12_PHASE45_PLAN.json').read_text());contract=json.loads((R/'master_plan_artifacts'/'COMMON_EVALUATION_CONTRACT.json').read_text())
 sel=json.loads((R/'SCALE360_SELECTION_FROZEN.json').read_text());pa={r['case_index']:r['native_input']['PhaseAdvance'] for r in sel['records']}
 seen=set(plan['seen_steps']);runs={'A0':['A0_s17_ref','A0_s29'],'A1':['A1_s17','A1_s29']};per={}
 for arm,ids in runs.items():
  for rid in ids:
   ev,rows=load(rid);m=ev['metrics'];cases=sorted({int(k.split(':')[1]) for k in m if k.startswith('all45/case:')})
   high=[c for c in cases if pa[c]>=70];low=[c for c in cases if pa[c]<70]
   region=collections.defaultdict(list)
   for r in rows:
    s=int(r['step']);region['seen9' if s in seen else ('after_last_seen' if s>max(seen) else 'interior_unused')].append(float(r['vector_nRMSE_percent']))
   per[rid]=dict(arm=arm,seed=ev['seed'],checkpoint_sha256=ev['checkpoint_sha256'],
    all45_vector=m['all45/all']['B_vector_nRMSE_percent'],all45_magnitude=m['all45/all']['B_magnitude_nRMSE_percent'],all45_vector_RMSE_T=m['all45/all']['B_vector_RMSE_T'],
    seen9_vector=m['seen9/all']['B_vector_nRMSE_percent'],unused36_vector=m['unused36/all']['B_vector_nRMSE_percent'],
    all45_copper_vector=m['all45/copper']['B_vector_nRMSE_percent'],copper_B2_bias=ev['copper_B2']['all45']['signed_bias_percent'],copper_B2_relMAE=ev['copper_B2']['all45']['area_weighted_relative_MAE_percent'],
    PA_ge_70_mean_case_vector=st.mean(m[f'all45/case:{c}']['B_vector_nRMSE_percent'] for c in high),PA_lt_70_mean_case_vector=st.mean(m[f'all45/case:{c}']['B_vector_nRMSE_percent'] for c in low),
    worst_case_vector=max(m[f'all45/case:{c}']['B_vector_nRMSE_percent'] for c in cases),median_case_vector=st.median(m[f'all45/case:{c}']['B_vector_nRMSE_percent'] for c in cases),
    phase_region_mean_snapshot_vector={k:st.mean(v) for k,v in sorted(region.items())},
    step_mean_snapshot_vector={s:st.mean(float(r['vector_nRMSE_percent']) for r in rows if int(r['step'])==s) for s in range(45)})
 mean=lambda arm,k:st.mean(per[r][k] for r in runs[arm])
 spread=max(abs(per[a][k]-per[b][k]) for a,b in [runs['A0'],runs['A1']] for k in ['all45_vector'])
 delta=mean('A0','all45_vector')-mean('A1','all45_vector')
 harm={'seen9_vector':(mean('A1','seen9_vector')-mean('A0','seen9_vector'),0.5),'all45_copper_vector':(mean('A1','all45_copper_vector')-mean('A0','all45_copper_vector'),1.0),
  'PA_ge_70_mean_case_vector':(mean('A1','PA_ge_70_mean_case_vector')-mean('A0','PA_ge_70_mean_case_vector'),1.0)}
 paired=[per[a]['all45_vector']-per[b]['all45_vector'] for a,b in [('A0_s17_ref','A1_s17'),('A0_s29','A1_s29')]]
 improvement=delta>=1.0 and delta>2*spread;no_harm=all(v<=lim for v,lim in harm.values())
 reg=lambda arm,k:st.mean(per[r]['phase_region_mean_snapshot_vector'][k] for r in runs[arm])
 predictions={
  'P1_gain_concentrated_after_last_seen':{'delta_after_last_seen':reg('A0','after_last_seen')-reg('A1','after_last_seen'),'delta_interior_unused':reg('A0','interior_unused')-reg('A1','interior_unused'),'delta_seen9':reg('A0','seen9')-reg('A1','seen9')},
  'P2_seen_and_interior_within_1pp':{'seen9_vector_change':mean('A1','seen9_vector')-mean('A0','seen9_vector'),'interior_unused_mean_snapshot_change':reg('A1','interior_unused')-reg('A0','interior_unused')},
  'P3_copper_B2_bias_A1':[per[r]['copper_B2_bias'] for r in runs['A1']],
  'P4_PA_gap_A1':[per[r]['PA_ge_70_mean_case_vector']-per[r]['PA_lt_70_mean_case_vector'] for r in runs['A1']]}
 predictions['P1_gain_concentrated_after_last_seen']['consistent']=predictions['P1_gain_concentrated_after_last_seen']['delta_after_last_seen']>max(predictions['P1_gain_concentrated_after_last_seen']['delta_interior_unused'],predictions['P1_gain_concentrated_after_last_seen']['delta_seen9'])
 predictions['P2_seen_and_interior_within_1pp']['consistent']=all(abs(v)<=1.0 for k,v in predictions['P2_seen_and_interior_within_1pp'].items() if k!='consistent')
 predictions['P3_copper_B2_bias_A1_consistent']=all(-10<=v<=-9 for v in predictions['P3_copper_B2_bias_A1'])
 predictions['P4_PA_gap_A1_consistent']=all(v>5 for v in predictions['P4_PA_gap_A1'])
 out=dict(experiment='L12 phase9 vs phase45, equal updates',evaluation_contract_sha256=sha(R/'master_plan_artifacts'/'COMMON_EVALUATION_CONTRACT.json'),plan_sha256=sha(R/'L12_PHASE45_PLAN.json'),runs=per,
  decision=dict(rule=contract['decision_rule_L12'],delta_all45_vector_A0_minus_A1=delta,paired_deltas_by_seed=paired,seed_spread_all45_vector=spread,improvement=improvement,
   no_harm={k:{'change_pp':v,'limit_pp':lim,'pass':v<=lim} for k,(v,lim) in harm.items()},no_harm_pass=no_harm,
   verdict='A1_IMPROVES_DEV' if improvement and no_harm else ('NO_IMPROVEMENT_CLAIM' if not improvement else 'IMPROVES_BUT_HARM_LIMIT_EXCEEDED')),
  preregistered_predictions=predictions,
  scope='inner development set dev32 (E1 oldTRAIN exposed); two seeds per arm; not an independent final test; not AC loss accuracy')
 (L/'L12_PHASE45_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'delta':delta,'paired':paired,'spread':spread,'verdict':out['decision']['verdict'],'no_harm':out['decision']['no_harm']},indent=1))
if __name__=='__main__':main()
