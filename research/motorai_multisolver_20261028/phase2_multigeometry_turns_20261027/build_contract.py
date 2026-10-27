"""Solver-free registration of geometry/turn groups and bounded experiments."""
from pathlib import Path
import csv,json,hashlib,datetime
ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 split= json.loads((RESEARCH/'ms_coverage_audit/SPLIT_PREDECLARED.json').read_text(encoding='utf-8'))
 groups={g:'train' for g in split['training_order256']}
 groups.update({g:'development' for g in split['development16']})
 reserved={'R1_0199','R1_0076'}
 assert len(groups)==272 and not reserved.intersection(groups)
 inventory={r['design']:r for r in csv.DictReader((RESEARCH/'inventory/r1_design_metadata.csv').open(encoding='utf-8-sig'))}
 existing={p.stem for p in (RESEARCH/'ms_coverage_audit/cond_numeric').glob('*.npz')}
 assert set(groups)==existing
 rows=[]
 envelope=6*1.686+5*.35
 for g,split_name in groups.items():
  for n in [4,6,8]:
   h=(envelope-(n-1)*.35)/n
   row=dict(configuration_id=f'{g}_slot{n}',geometry_alias_group=g,split=split_name,template='R1_doubleV_48slot_8pole',
      conductors_per_slot=n,sector_conductors=6*n,transient_total_steps_max=128,
      bare_height_mm=h,bare_width_mm=float(inventory[g]['Bs1'])-.919,corner_chamfer_leg_mm=.197,
      inter_bar_gap_mm=.35,stack_envelope_mm=envelope,packing_policy='fixed_original_stack_envelope_and_gap',
      phase_current_Arms_equal_ampere_turns=250*6/n,current_gamma_deg=46.7,
      nominal_total_slots=48,parallel_circuits=1,phase_series_turns_if_two_sides_per_turn=8*n,
      MS_status='EXISTING_CONDUCTOR_SUMMARY_ONLY' if n==6 else 'NOT_SOLVED',
      transient_status='NOT_SOLVED',MGN_FE_graph_status='NOT_BUILT',
      geometry_status='PENDING_GENERATION_QA',automatic_retry=0,
      deadline_UTC='2026-10-27T23:00:00Z')
   for k in ['RotorDia','Airgap','Bs0','Bs1','L1_PoleVAngle','L2_PoleVAngle']:row[k]=float(inventory[g][k])
   rows.append(row)
 with (ROOT/'configuration_queue.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump('PLAN.json',dict(schema='motorai-phase2-turns-v1',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  user_confirmed={'conductors_per_slot':[4,6,8],'transient_total_steps_max':128,'deadline_KST':'2026-10-28T08:00:00+09:00'},
  configurations=816,geometries=272,templates=1,train_configurations=768,development_configurations=48,
  reserved_excluded=sorted(reserved),historical_reserved_initial_QC=True,
  packing={'policy':'Keep original R1 stack envelope, width rule, bottom offset, corner chamfer and 0.35mm bare-to-bare gap; recompute bare radial height.',
   'heights_mm':{str(n):(envelope-(n-1)*.35)/n for n in [4,6,8]},'six_conductor_baseline_must_reproduce':True,
   'copper_area_not_constant':True,'manufacturing_qualification':'UNKNOWN'},
  excitation={'pilot':'equal slot ampere-turns to original 6x250 Arms; sinusoidal gamma46.7 deg',
   'Arms_by_conductors':{'4':375,'6':250,'8':187.5},'phase_series_turns_formula':'48*n/(3*2*parallel); circuit audit required before use'},
  solver_scope={'R1_source':'ipmfea a46e5d8bc0e5621c455915e627279356c1a1269e',
   'available':'GetDP nonlinear MS source + parameterized geometry; native transient R1 implementation not yet qualified',
   'do_not_count':'geometry generation, scalar caches or synthetic graphs are not Full-FEA or MGN training'},
  pilot_selection={'geometry_IDs':split['training_order256'][:2],'uses':'ID order only, no held-out labels','turns':[4,6,8],
   'max_MS_cases':6,'MS_phase_samples_max':30,'max_transient_cases':6,'each_transient_steps_max':128,
   'launch_gate':'mesh/current/region mapping and solver identity must pass; no blind bulk execution'},
  step128_validation={'known_Ref_anchor':{'Irms':460,'beta_deg':36,'rpm':16000},
   'max_new_runs':2,'variants':['128 samples x 1 period','64 samples x 2 periods'],
   'total_steps_each':128,'initialization':'cold A=0; initialization sensitivity explicitly diagnosed',
   'reference':'already completed Ref117 N256P3 and N512P3; compare full P1 and P2 means as separately defined',
   'automatic_extension':False,'automatic_retry':0,'production_accuracy_gate':'UNKNOWN until numerical evidence; no invented PASS'},
  resources={'234':'Existing 2x8-thread frozen new-OP runs preserved. Two additional diagnostic workers at8 threads, disjoint affinity, only after preflight.',
   '133':'L40S CUDA training environment verified; Linux/GetDP/Elmer availability not verified; no assumed FEA readiness.',
   'training_dispatch':'Independent fixed seeds/ablations, one GPU job per host. Actual FE graph+physical targets required before dispatch; no DDP claim.',
   '128step_run_estimate_seconds':700,'128step_case_cap_seconds':1800,'128step_runs_max':2,'aggregate_128step_cap_seconds':3600,
   'RAM_GiB_each':16,'disk_GiB_each':3,'minimum_disk_free_GiB':32,'max_retries':0},
  fair_comparison={'existing_newOP8':'Frozen predictions and63 Full calibration unchanged; never reuse these test labels to tune Phase2.',
   'new_R1_campaign':'Same nested independent Full-label IDs for AF-form recalibration, scalar residual and MS-pretrained MGN; proposed budgets12/24/63 are not yet acquired labels.',
   'AF_denominator':'canonical mean-square-field Hybrid must be reconstructed for the new geometry before fitting/applying AF. Frozen Ref coefficients must not multiply GetDP proximity labels.',
   'reference_baseline':'latest frozen JEET AF remains mandatory in existing Ref/SC evaluation',
   'cross_solver_exact_bridges':0,'split':'all turns, OPs, solvers and derived labels inherit base geometry/alias group'},
  strategy={'Thesis_SKKU_commit':'6a597523d0ebcc9d9cd367565d4d2a76d41418e2',
   'document':'mainmatter_v2/CODESIGN_CH07_SCOPE_20260920.md',
   'application':'Assess geometry/OP-dependent AC-loss and sparse-FEA correction on design-selection decisions, with equal solver budgets and total pipeline time. Energy/mass system study is downstream, not completed here.'},
  stop={'deadline_UTC':'2026-10-27T23:00:00Z','guard_target_UTC':'2026-10-27T22:59:00Z',
   'scope':'owned process group only after PID start-time verification; preserve partial outputs; never restart other campaigns'}))
 dump('split_audit.json',dict(status='PASS',geometries=272,configurations=816,train_geometries=256,development_geometries=16,
  train_configurations=768,development_configurations=48,reserved_field_archives_opened=0,reserved_used_in_model_or_evaluation=0,reserved_in_queue=[],
  reserved_excluded=sorted(reserved),source_split_sha256=sha(RESEARCH/'ms_coverage_audit/SPLIT_PREDECLARED.json'),
  caveat='Existing development16 is not a new untouched final test. Historical source calls reserved_unopened; these two had earlier QC and are not globally never-opened.',
  group_by_configuration={r['configuration_id']:r['geometry_alias_group'] for r in rows}))
 dump('source_manifest.json',dict(source_files=[{'logical_key':str(p.relative_to(RESEARCH)), 'sha256':sha(p),'bytes':p.stat().st_size} for p in [RESEARCH/'ms_coverage_audit/SPLIT_PREDECLARED.json',RESEARCH/'inventory/r1_design_metadata.csv']],
  source_revisions={'ipmfea':'a46e5d8bc0e5621c455915e627279356c1a1269e','Thesis_SKKU':'6a597523d0ebcc9d9cd367565d4d2a76d41418e2'},
  raw_geometry_location='private source-key path map; not GitHub',absolute_paths_published=False))
 print(json.dumps(dict(configurations=len(rows),train=768,development=48,reserved=0,total_transient_steps_cap=128)))
if __name__=='__main__':main()
