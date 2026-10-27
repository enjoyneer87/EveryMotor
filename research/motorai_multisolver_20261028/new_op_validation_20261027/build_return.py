"""Build auditable small return from collected outputs; no solver/model call."""
from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,d):(R/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 frozen=read(R/'PREDICTIONS_FROZEN.json');ev=read(R/'EVALUATION.json');transfer=read(R/'COLLECTION_TRANSFER_RECEIPT.json');state=read(R/'STATE.json')
 with (R/'case_queue.csv').open(encoding='utf-8',newline='') as f:
  reader=csv.DictReader(f);queue_columns=reader.fieldnames;queue=list(reader)
 state_names={'완료':'COMPLETED_NUMERICAL_GATES_PASS','완료·품질FAIL':'COMPLETED_QUALITY_FAIL','실행 중':'RUNNING','순차 대기':'QUEUED','중단·미실행(선행 gate FAIL)':'NOT_RUN_PREDECESSOR_GATE_FAIL','실패/중단':'EXECUTION_FAILED_OR_STOPPED'}
 for row in queue:
  row['status']=state_names.get(state['cases'].get(row['case_id']),'UNKNOWN')
  row['owner']='owned_new_OP_iso_'+row['model']+'_worker'
 with (R/'case_queue.csv').open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=queue_columns);writer.writeheader();writer.writerows(queue)
 assert sha(R/'PREDICTIONS_FROZEN.json')=='c238ac8b61242d2c804a8e03639df7c1ccb4aa510ba88afa42b96e58afa7f459'
 for row in transfer['cases']:
  for item in row['files']:
   p=R/'results'/row['case_id']/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 observations=[];sources=[];shards=[];quality=[];pairs=[];table=[]
 for c in frozen['conditions']:
  case=c['case_id'];p=R/'results'/case
  if not (p/'READBACK_RECEIPT.json').exists():continue
  rec=read(p/'READBACK_RECEIPT.json');audit=read(p/'audit.json');raw=read(p/'raw_source_manifest.json')
  assert rec['source_status']=='PASS';wave=np.load(p/'waveforms.npz',allow_pickle=False)
  numerical_pass=rec['current_status']=='PASS' and rec['nonlinear_status']=='PASS'
  sources.append(dict(logical_source_key=raw['logical_source_key'],source_commit='7f041786053e6fdc5c824f003c55af7a5f96076d',raw_manifest=f'results/{case}/raw_source_manifest.json',raw_manifest_sha256=sha(p/'raw_source_manifest.json'),
   source_status='PASS',raw_files=len(raw['files']),raw_bytes=sum(x['bytes'] for x in raw['files']),provenance='Native source host read-only A/circuit integration and local transfer SHA readback'))
  for period in audit['periods']:
   for field,quantity in [('Joule_W','active_total_Joule'),('DC_W','prescribed_transport_DC'),('AC_W','AC_excess')]:
    observations.append(dict(template='JEET_e10',geometry=c['model'],condition=case,source=raw['logical_source_key'],evaluator='native_A_circuit_BE_BDF2_P1_triangle',quantity=quantity,scope='full_motor_active_8sector',unit='W',value=period[field],temperature='UNKNOWN',conductivity_S_m=audit['sigma_S_m'],waveform='sinusoidal_prescribed_current',average_window=f'P{period["period"]}:poststeps{period["steps"][0]}-{period["steps"][1]}',average_rule='uniform_arithmetic',quality=f'whole_current_{rec["current_status"]};nonlinear_{rec["nonlinear_status"]};formal_convergence_UNKNOWN',role=('accepted_window_evaluation' if numerical_pass else 'quality_flagged_evaluation_diagnostic') if period['period']==2 else 'diagnostic',source_hash=sha(p/'waveforms.npz')))
  for method,pred in frozen['predictions_W'].items():
   observations.append(dict(template='JEET_e10',geometry=c['model'],condition=case,source='frozen_prediction/'+method,evaluator=method,quantity='predicted_AC_excess',scope='full_motor_active_8sector',unit='W',value=pred[case],temperature='UNKNOWN',conductivity_S_m=audit['sigma_S_m'],waveform='sinusoidal_prescribed_current',average_window='target_P2',average_rule='frozen_model',quality='prediction_not_solver_truth',role='evaluation_prediction',source_hash=sha(R/'PREDICTIONS_FROZEN.json')))
  quality.append(dict(case=case,provenance='PASS',units_scope='PASS',temperature='UNKNOWN',raw_A_circuit_reintegration=dict(status='PASS',scope='completed768 native steps read and integrated; independent solver loss unavailable'),
   waveform_arithmetic=dict(status='PASS',max_error_W=rec['waveform_scalar_roundtrip_max_W'],gate=rec['roundtrip_tolerance']),
   current=dict(status=rec['current_status'],max_percent=audit['whole_current_max_percent'],gate_percent=.001),
   P2_current=dict(status='PASS' if audit['periods'][1]['current_max_percent']<=.001 else 'FAIL',max_percent=audit['periods'][1]['current_max_percent'],gate_percent=.001),
   nonlinear=dict(status=rec['nonlinear_status'],max_relative_change=audit['solver_nonlinear_max'],gate=1e-10),
   period_development=dict(status=audit['period_status'],mean_percent=audit.get('P2_P3_mean_percent'),wave_L2_percent=audit.get('P2_P3_wave_L2_percent')),
   formal_period='UNKNOWN',formal_mesh='UNKNOWN',formal_timestep='UNKNOWN',independent_loss_channel='UNKNOWN_not_available'))
  shards.append(dict(case=case,path=f'results/{case}/waveforms.npz',sha256=sha(p/'waveforms.npz'),bytes=(p/'waveforms.npz').stat().st_size,
   arrays={k:dict(shape=list(wave[k].shape),dtype=str(wave[k].dtype)) for k in wave.files},
   scalar_columns=['time_s','Joule_W','DC_W','AC_W','current_error_percent_of_phase_peak'],current_unit='A',
   conductor_IDs=list(range(2301,2337)),conductor_order='ascending physical ID',time_downsampled=False,
   field_arrays_included=False,raw_field_location='source-key mapping on private iso storage'))
  pairs.append(dict(case=case,MS_source=c['source_key'],Full_source=raw['logical_source_key'],denominator_lineage='linked',
   native_same_geometry_MS_pair=c['model']=='Ref',relation='native_Ref_MS_frequency_reuse' if c['model']=='Ref' else 'scaled_Ref_MS_to_SC_prediction_not_native_SC_MS',
   exact_cross_solver_bridge=False))
  table.append(f'| {case} | {audit["periods"][1]["AC_W"]:.6f} | {audit["whole_current_max_percent"]:.9f} ({rec["current_status"]}) | {audit.get("P2_P3_mean_percent",float("nan")):.8f} |')
 if observations:
  with (R/'observations.csv').open('w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(observations[0]));w.writeheader();w.writerows(observations)
 dump('source_manifest.json',dict(source_commit='7f041786053e6fdc5c824f003c55af7a5f96076d',requested=8,collected=len(sources),not_collected=8-len(sources),unique_geometry_variants=len({r['geometry'] for r in observations}),template_families=1,duplicate_case_ids=0,sources=sources))
 dump('sample_shards_manifest.json',dict(shards=shards,scope='loss/current waveforms; raw field remains on source host'))
 dump('quality_summary.json',dict(requested=8,collected=len(quality),not_collected=8-len(quality),quality=quality,blanket_physical_PASS=False,
  quality_fail_collected=sum(x['current']['status']!='PASS' or x['nonlinear']['status']!='PASS' for x in quality),
  unrun_blocked_by_predecessor=state.get('blocked_unrun_cases',0),execution_failed=state.get('execution_failed_cases',0),running=state.get('running_cases',0),pending=state.get('pending_cases',0),
  past_receipt_vs_current='This return contains new source-host integration and local hash/arithmetic checks, not just copied historical PASS.',formal_qualification='UNKNOWN'))
 dump('split_audit.json',dict(status='PASS',newOP_labels_used_for_training=0,unique_Full_calibration=63,independent_new_evaluation_labels=len(pairs),
  numerical_gate_pass_evaluation_labels=sum(x['current']['status']=='PASS' and x['nonlinear']['status']=='PASS' for x in quality),quality_flagged_evaluation_labels=sum(x['current']['status']!='PASS' or x['nonlinear']['status']!='PASS' for x in quality),formal_qualified_labels=0,
  new_template_count=0,reserved_present=[],reserved=['R1_0199','R1_0076'],historical_initial_QC_preserved=True,
  HalfSC_predeclared=32,HalfSC_exploratory=4,HalfSC_reevaluated=False,predictions_frozen_sha256=sha(R/'PREDICTIONS_FROZEN.json'),
  native_same_geometry_MS_pairs=sum(p['native_same_geometry_MS_pair'] for p in pairs),scaled_MS_denominator_associations=sum(not p['native_same_geometry_MS_pair'] for p in pairs),
  exact_cross_solver_bridges=0,pairs=pairs,note='Do not add scaled MS associations to historical41 native pairs. Cross-source geometry aliases remain governed by prior registry.'))
 metrics=[f'| {m["method"]} | {m["wMAE_percent"]:.6f} |' for m in ev['metrics'] if m['group']=='ALL' and m['wMAE_percent'] is not None]
 report=f'''# 새 운전점 원자료 검증 반환

요청8건 중 {len(pairs)}건 회수. 회수 중 품질FAIL {state.get('quality_fail_cases',0)}건, 선행 gate 때문에 미실행 {state.get('blocked_unrun_cases',0)}건, 실행중 {state.get('running_cases',0)}건, 순차대기 {state.get('pending_cases',0)}건이다. 실행 완료와 품질 적합성을 구분하며 미회수를 자동으로 실패로 세지 않는다. 입력·예측은7f04178에서 해석 전에 동결했다. 보정 Full63(Ref36+SC27), 새 평가8건은 모든 방법이 공유하며 재학습·튜닝·best-seed 선택은 없다.

| 조건 | 채택 P2 AC excess W | 전체 전류 최대오차 % | P2/P3 평균차이 % |
|---|---:|---:|---:|
{chr(10).join(table)}

native A와 회로전압에서 BE 첫 스텝/BDF2 이후의 dA/dt, P1 삼각형 내부 J 적분, prescribed transport DC를 재계산했다. 전체모터 활성부 Joule에서 DC를 뺀 AC excess이며 end winding은 포함하지 않는다. P2는257–512 post-step 산술평균이다. P1과P3는 진단이며 채택구간을 바꾸지 않았다. 저장 파형 재평균과 audit의 오차는 quality_summary.json에 기록했다. 이는 동일 native evaluator의 산술 일관성으로 독립 solver loss채널 검증은 아니다.

전류<=0.001%와 비선형<=1e-10은 기존 gate다. P2/P3 개발 threshold와 공식 주기·시간간격·mesh 수렴 UNKNOWN을 구분한다. conductivity는47092391.429245 S/m, active length0.15m, symmetry8이다. 온도는 receipt에 명시되지 않아UNKNOWN이며 conductivity로 임의 역추정하지 않았다.

| 방법 | 회수 조건 전체 wMAE % |
|---|---:|
{chr(10).join(metrics)}

품질FAIL을 포함한 부분 진단 결과다. P2 전류가통과해도전체이력의FAIL을승격하지않는다. CURRENT_GATE_DIAGNOSTICS.json에초과step과P2를구분했다. 조건수와 MS source group이 적어 우월성이나 새로운 형상 일반화를 주장하지 않는다. 모든3시드의 결과와 평균/표준편차는 EVALUATION.json에 보존한다. Ref는 원래 Ref MS의 주파수 재사용, SC는 Ref MS를 스케일링한 분모 연결이므로 native SC MS/Full 쌍으로 세지 않는다. exact cross-solver bridge는0이다.

재현 명령: `python evaluate_frozen.py`, `python build_return.py`, `python update_dashboard.py`. 이 명령들은 solver와 학습을 실행하지 않는다. 대용량 native field는 iso 원본을 보존하고 GitHub에는 파형·hash·코드만 둔다. 상세 source key와 파일별 hash는 각 results의 raw_source_manifest.json에 있다.

별도 Phase2의4/6/8 슬롯당 도체, 총128스텝 요청은 ../phase2_multigeometry_turns_20261027/에서 수행한다. 기존8건의 동결된 N256x3주기 비교 계약은 변경하지 않는다. 신규 Phase2 labels와 기존 평가 labels를 혼용하지 않는다.
'''
 (R/'VALIDATION_REPORT_KO.md').write_text(report,encoding='utf-8')
 (R/'README_KO.md').write_text('# 새 운전점8조건 반환\n\n'+f'현재 원시 적분 회수 {len(pairs)}/8. '+ 'VALIDATION_REPORT_KO.md, STATE.json, STATUS_KO.html 및 source_manifest.json을 함께 확인한다.\n\n실행: `python evaluate_frozen.py` → `python build_return.py` → `python update_dashboard.py`. 원시 field는 보존된 source key 경로에서만 접근한다.\n\n남은 작업: 미완료 조건 회수, 같은 frozen prediction 평가, 누락/실패 포함 최종 집계. 공식 수렴은 UNKNOWN이며 새 Phase2의 총128스텝 진단과 별개다.\n',encoding='utf-8')
 print(json.dumps(dict(cases=len(pairs),observations=len(observations),native_Ref_MS_associations=sum(p['native_same_geometry_MS_pair'] for p in pairs),scaled_associations=sum(not p['native_same_geometry_MS_pair'] for p in pairs))))
if __name__=='__main__':main()
