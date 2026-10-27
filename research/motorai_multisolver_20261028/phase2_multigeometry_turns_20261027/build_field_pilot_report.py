"""Summarize saved receipts; never launches training or a solver."""
from pathlib import Path
import json,csv,datetime
R=Path(__file__).resolve().parent
def main():
 data=json.loads((R/'B_FIELD_TRAINING_STATUS.json').read_text());rows=[]
 for host,item in data.items():
  result=item.get('result/RESULT.json',{});metrics=result.get('metrics',[]);last=metrics[-1] if metrics else {}
  rows.append(dict(host=host,seed=result.get('seed'),status=result.get('status','UNKNOWN'),epochs=len(metrics),updates=result.get('training_updates',0),
   inner_development_vector_nRMSE_percent=last.get('inner_development',{}).get('vector_nRMSE_percent'),train_vector_nRMSE_percent=last.get('train',{}).get('vector_nRMSE_percent'),
   peak_torch_allocated_MiB=result.get('peak_GPU_allocated_bytes',0)/1024**2,elapsed_seconds=result.get('elapsed_seconds'),torch=result.get('torch'),
   checkpoint=item.get('checkpoint'),guard_exitcode=item.get('GUARD_RECEIPT.json',{}).get('returncode'),native_process_live=item.get('owned_process_live'),container_state=item.get('owned_container_state')))
 report=dict(checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Motor-CAD native element Bx/By pretraining diagnostic; not AC loss or equal-Full-budget comparison',
  original_manifest_rows=480,declared_geometry_groups=360,original_copy_HDF5_SHA256_matches=440,initial_source_unlinked=40,native_manifest_geometry_mismatch_quarantined=80,
  eligible_old_train_cases=360,prepared_cases=20,train_geometries=16,inner_development_geometries=4,graphs=180,original_test_cases_used=0,new_FEA_runs=0,
  historical_Full_donor_training_cases=16,historical_Full_donor_development_cases=4,MS_teacher_cases=0,exact_cross_solver_pairs=0,
  field_reader_arithmetic='PASS_all20_cases_all45_steps',formal_mesh_period_convergence='UNKNOWN',AC_loss_accuracy='NOT_EVALUATED',paired_alignment=False,
  metric_definition='100*sqrt(sum(area*||Bpred-B||^2)/sum(area*||B||^2)); pooled geometry snapshots; a2 excluded',
  memory_contract='10GiB checks PyTorch peak allocated between graphs; not a hard whole-device limit',
  seed_comparison='Fixed final epochs, no seed selection. Host/Torch and seed differ together; not pure seed variance.',runs=rows)
 (R/'B_FIELD_PILOT_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
 state=json.loads((R/'STATE.json').read_text());state.update(checked_UTC=report['checked_UTC'],training_status=';'.join(x['host']+':'+str(x['status']) for x in rows),
  new_multigeometry_MGN_models=sum(x['status']=='COMPLETED' for x in rows),real_multigeometry_training_updates=sum(x['updates'] for x in rows),
  MGN_architecture='Two domain encoders conditioned by design/OP/fidelity; Motor-CAD B-only pretraining; paired alignment inactive',
  existing_field_reuse={k:report[k] for k in ['original_manifest_rows','declared_geometry_groups','original_copy_HDF5_SHA256_matches','native_manifest_geometry_mismatch_quarantined','prepared_cases','train_geometries','inner_development_geometries','graphs']})
 (R/'STATE.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n')
 q=list(csv.DictReader((R/'configuration_queue.csv').open(encoding='utf-8')));qa=json.loads((R/'GEOMETRY_QA.json').read_text());by={r['configuration_id']:r for r in qa['results']}
 for r in q:r['geometry_status']='ANALYTIC_GEOMETRY_'+by[r['configuration_id']]['status']
 with (R/'configuration_queue.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(q[0]));w.writeheader();w.writerows(q)
 lines=['# Motor-CAD 다형상 FE 자계 pilot 결과','','실제 native FE 요소 인접 그래프에서 Bx/By 사전학습을 실행했다. 새 FEA0건, 기존 Full donor16형상 학습+4형상 내부 검증의 비용을 별도 계산한다. 기존 Full63/새OP8의 모델·예측은 동결 유지한다.','',
 '| 서버 / seed | 상태 | epoch / updates | train nRMSE | 내부 검증 nRMSE | PyTorch peak MiB |','|---|---|---:|---:|---:|---:|']
 for x in rows:
  fmt=lambda v:'—' if v is None else f'{v:.3f}%'
  lines.append(f"| {x['host']} / {x['seed']} | {x['status']} | {x['epochs']} / {x['updates']} | {fmt(x['train_vector_nRMSE_percent'])} | {fmt(x['inner_development_vector_nRMSE_percent'])} | {x['peak_torch_allocated_MiB']:.1f} |")
 lines+=['','수치는 a2 제외 native 요소 면적 가중 B벡터 nRMSE다. 4형상 내부 검증이며 최종시험이 아니다. 계획한 마지막 epoch를 보고하고 best epoch/seed를 선택하지 않는다. GPU/Torch 환경도 다르므로 차이를 순수seed 효과로 해석하지 않는다.','',
 '이 작은 pilot은 데이터→FE graph→GPU 학습의 실행 증거다. 큰 자계 오차를 물리정확도 PASS나 AC 손실 우위로 표현하지 않는다. J/Je 전류성분 정의, Joule/DC/AC 적분, 공통 AF 분모와 native MS/Full 대응은 별도 검증이 필요하다.','',
 '480행은360개 선언 형상 그룹이다.440행 원본/사본HDF5 hash일치,80행 native/manifest 입력불일치와초기40 source미연결을첫pilot에서제외했다. 남은360은기존TRAIN뿐이므로입력만으로20그룹을선택해16/4동결했다. 기존val/test와과거checkpoint는사용하지않았다.','',
 '45스텝 전체 ID·기하·B 산술 검사 후9스텝씩180graph를저장했다. 시간축소의AC손실보존은검증하지않았다.45step transient field이며MS교사값이아니다.','',
 '10GiB는PyTorch allocated peak의스텝간소프트검사다.전체GPU하드한도보장이아니다.각run은3600초,UTC22:59,STOP과소유PID/시작시각또는containerID로제한한다.','',
 '재현: B_FIELD_PILOT_PLAN.json → audit_native_doe_catalog.py → prepare_existing_field_pilot.py → bundle hash 검증 → train_existing_field_pilot.py. raw/graph/checkpoint경로는private mapping,공유물은logical key/hash로연결한다.','',
 '다음과제: 입력불일치80행계보해결,도체·영역별B및J/Je손실적분검증,공통형상의Motor-CAD/Elmer교차점확보. mesh/time fidelity를분리하며그전paired학습이나AF대비AC우위를주장하지않는다.']
 (R/'B_FIELD_PILOT_REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'runs':rows},ensure_ascii=False))
if __name__=='__main__':main()
