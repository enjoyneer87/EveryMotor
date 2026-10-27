"""W03 secondary analysis of the completed frozen diagnostic; reads saved CSV/JSON only, no model or solver run."""
from pathlib import Path
import csv,json,hashlib,statistics as st,collections
R=Path(__file__).parent;OUT=R/'master_plan_artifacts'/'FROZEN_DIAGNOSTIC_PHASE_ANALYSIS.json'
HIGH_PA_DEG=70.0
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 inputs=['FROZEN_DIAGNOSTIC_snapshot_errors.csv','FROZEN_DIAGNOSTIC_conductor_B2.csv','FROZEN_FIELD_DIAGNOSTIC_RESULT.json','SCALE360_SELECTION_FROZEN.json']
 rows=list(csv.DictReader(open(R/inputs[0],encoding='utf-8')));result=json.loads((R/inputs[2]).read_text(encoding='utf-8'));sel=json.loads((R/inputs[3]).read_text(encoding='utf-8'))
 native={r['case_index']:r['native_input'] for r in sel['records']}
 seen=sorted({int(r['step']) for r in rows if r['phase_scope']=='seen9'});last_seen=max(seen)
 dev=sorted({int(r['case']) for r in rows});high={c for c in dev if native[c]['PhaseAdvance']>=HIGH_PA_DEG}
 def summary(v):return dict(n=len(v),mean_snapshot_vector_nRMSE_percent=st.mean(v))
 by_distance=collections.defaultdict(list);by_step=collections.defaultdict(list);interior=collections.defaultdict(list)
 for r in rows:
  s=int(r['step']);c=int(r['case']);d=min(abs(s-x) for x in seen);v=float(r['vector_nRMSE_percent']);grp='PA_ge_70' if c in high else 'PA_lt_70'
  by_distance[(grp,d)].append(v);by_step[(grp,s)].append(v)
  interior[(grp,'seen9' if d==0 else ('after_last_seen' if s>last_seen else 'interior_unused'))].append(v)
 per_case={c:result['metrics'][f'all45/case:{c}'] for c in dev}
 train=[r['native_input'] for r in sel['records'] if r['split']=='train']
 out=dict(
  scope='Secondary analysis of completed frozen SCALE360 diagnostic. Means of per-snapshot nRMSE are descriptive and are not the pooled area-weighted metric. No new inference, training or FEA.',
  inputs={n:sha(R/n) for n in inputs},seen_steps=seen,
  phase_encoding='design_op uses sin/cos of cumulative native mechanical angle (native_element_graph.py); electrical periodicity over the 45-step cycle is not encoded',
  high_phase_advance_threshold_deg=HIGH_PA_DEG,
  dev_cases={'PA_ge_70':sorted(high),'PA_lt_70':sorted(set(dev)-high)},
  dev_all45_pooled_vector_nRMSE_percent_range={g:[min(per_case[c]['B_vector_nRMSE_percent'] for c in cs),max(per_case[c]['B_vector_nRMSE_percent'] for c in cs)] for g,cs in [('PA_ge_70',high),('PA_lt_70',set(dev)-high)]},
  train_condition_PhaseAdvance_ge_70=sum(r['PhaseAdvance']>=HIGH_PA_DEG for r in train),train_conditions=len(train),
  by_seen_distance={f'{g}/d{d}':summary(v) for (g,d),v in sorted(by_distance.items())},
  by_phase_region={f'{g}/{k}':summary(v) for (g,k),v in sorted(interior.items())},
  by_step_PA_lt_70={s:st.mean(v) for (g,s),v in sorted(by_step.items()) if g=='PA_lt_70'},
  copper_B2_signed_bias_percent={k:result['copper_B2'][k]['signed_bias_percent'] for k in ['seen9','unused36','all45']})
 OUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:out[k] for k in ['by_phase_region','dev_all45_pooled_vector_nRMSE_percent_range','copper_B2_signed_bias_percent']},indent=1))
if __name__=='__main__':main()
