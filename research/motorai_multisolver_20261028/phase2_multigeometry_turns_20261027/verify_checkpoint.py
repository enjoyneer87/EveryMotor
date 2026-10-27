"""Hash/split/step/model-contract checks; no training or solver invocation."""
from pathlib import Path
import argparse,hashlib,json,csv,ast
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write-manifests',action='store_true');a=ap.parse_args()
 folders=[ROOT,ROOT.parent/'new_op_validation_20261027'];counts={}
 for r in folders:
  candidates=[p for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in ['PACKAGE_MANIFEST.json','CHECKPOINT_VERIFICATION.json']]
  for p in candidates:
   if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
  if a.write_manifests:
   manifest={'scope':'small source/return package only','files':[{'path':p.relative_to(r).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(candidates)]}
   (r/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
  m=json.loads((r/'PACKAGE_MANIFEST.json').read_text())
  for row in m['files']:
   p=r/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
  counts[r.name]=len(m['files'])
 rows=list(csv.DictReader((ROOT/'configuration_queue.csv').open(encoding='utf-8')))
 assert len(rows)==816 and len({x['configuration_id'] for x in rows})==816
 assert len({x['geometry_alias_group'] for x in rows})==272
 assert not {'R1_0199','R1_0076'}.intersection(x['geometry_alias_group'] for x in rows)
 assert all(int(x['transient_total_steps_max'])==128 for x in rows)
 assert sum(x['split']=='train' for x in rows)==768 and sum(x['split']=='development' for x in rows)==48
 for g in {x['geometry_alias_group'] for x in rows}:assert len({x['split'] for x in rows if x['geometry_alias_group']==g})==1
 tests=json.loads((ROOT/'DUAL_NETWORK_CONTRACT_TESTS.json').read_text())
 for result in tests.values():assert result['status']=='PASS' and result['training_updates']==0 and result['model_sha256']==sha(ROOT/'dual_solver_mgn.py')
 for p in (ROOT/'step128_results').glob('*/audit.json'):
  d=json.loads(p.read_text());assert d['steps']==128 and d['whole_current_status']=='PASS' and d['nonlinear_status']=='PASS'
 n=folders[1];assert sha(n/'PREDICTIONS_FROZEN.json')=='c238ac8b61242d2c804a8e03639df7c1ccb4aa510ba88afa42b96e58afa7f459'
 ev=json.loads((n/'EVALUATION.json').read_text());assert ev['Full_calibration_budget']==63 and not ev['models_retrained'] and not ev['seed_selection']
 pilot=json.loads((ROOT/'B_FIELD_BUNDLE_MANIFEST.json').read_text());assert pilot['reader_pass_cases']==20 and pilot['failed_cases']==0
 groups={s:{r['geometry_group'] for r in pilot['records'] if r['split']==s} for s in ['train','development']}
 assert len(groups['train'])==16 and len(groups['development'])==4 and not groups['train']&groups['development']
 assert all(r['original_split']=='train' and len(r['files'])==9 and r['status']=='READER_ARITHMETIC_PASS' for r in pilot['records'])
 training=json.loads((ROOT/'B_FIELD_TRAINING_STATUS.json').read_text());updates=0
 for host,item in training.items():
  t=item['result/RESULT.json'];assert t['model_sha256']==sha(ROOT/'dual_solver_mgn.py') and t['trainer_sha256']==sha(ROOT/'train_existing_field_pilot.py')
  assert t['original_test_used']==0 and not t['AC_loss_training'] and not t['paired_alignment'] and not t['seed_selection']
  assert t['train_groups']==sorted(groups['train']) and t['inner_development_groups']==sorted(groups['development'])
  updates+=t['training_updates']
  if t['status']=='COMPLETED':assert len(t['metrics'])==10 and t['training_updates']==1445 and item['GUARD_RECEIPT.json']['returncode']==0
 selection=json.loads((ROOT/'SCALE360_SELECTION_FROZEN.json').read_text());expanded=json.loads((ROOT/'SCALE360_BUNDLE_MANIFEST.json').read_text());plan=json.loads((ROOT/'SCALE360_PLAN.json').read_text())
 assert len(selection['records'])==360 and len({r['geometry_group'] for r in selection['records']})==320
 sg={s:{r['geometry_group'] for r in selection['records'] if r['split']==s} for s in ['train','development']}
 assert len(sg['train'])==288 and len(sg['development'])==32 and not sg['train']&sg['development']
 assert groups['train']<=sg['train'] and groups['development']<=sg['development']
 assert sum(r['split']=='train' for r in selection['records'])==328
 assert expanded['reader_pass_cases']==360 and expanded['failed_cases']==0
 assert expanded['selection_sha256']==sha(ROOT/'SCALE360_SELECTION_FROZEN.json') and expanded['plan_sha256']==sha(ROOT/'SCALE360_PLAN.json')
 assert all(r['QA']['all_45_steps_checked'] and len(r['files'])==9 for r in expanded['records'])
 assert plan['training']['expected_optimizer_updates']==328*9*30+5 and plan['training']['host']=='133'
 expanded_status=json.loads((ROOT/'SCALE360_TRAIN_STATUS.json').read_text());expanded_result=json.loads((ROOT/'SCALE360_TRAIN_RESULT.json').read_text())
 assert expanded_result['model_sha256']==sha(ROOT/'dual_solver_mgn.py') and expanded_result['trainer_sha256']==sha(ROOT/'train_field_scale360.py')
 assert expanded_result['original_val_test_used']==0 and not expanded_result['old_checkpoint_reused'] and not expanded_result['seed_selection']
 if expanded_result['status']=='COMPLETED':
  assert expanded_result['epochs_completed']==30 and expanded_result['training_updates']==88565
  assert expanded_status['GUARD_RECEIPT.json']['returncode']==0 and not expanded_status['owned_process_live']
 register=json.loads((ROOT/'IMPROVEMENT_REGISTER.json').read_text(encoding='utf-8'));assert len(register['items'])==len({r['id'] for r in register['items']})
 report=dict(status='PASS',scope='package integrity/split/caps/frozen-model invariants only',files=counts,configurations=816,
  source_geometries=272,reserved_rows=0,total_step_cap=128,dual_network_contract_hosts=2,real_dual_network_training_updates=updates,
  completed_newOP_native_reintegration=ev['completed'],exact_cross_solver_bridges=0,expanded_reader_cases=360,expanded_geometry_groups=320,expanded_graphs=3240,
  expanded_training_status=expanded_result['status'],expanded_training_updates=expanded_result['training_updates'],improvement_items=len(register['items']),formal_physical_qualification='UNKNOWN')
 (ROOT/'CHECKPOINT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
