"""Metadata/source audit only. Never imports model modules or unpickles weights."""
from pathlib import Path
import argparse,json,hashlib,subprocess,time,datetime,zipfile

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();start=time.monotonic();a.out.mkdir(exist_ok=False)
 def check():
  if (a.out/'STOP').exists() or time.monotonic()-start>1800 or datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-10-27T22:59:00+00:00'):raise InterruptedError('STOP/time')
 def run(cmd):return subprocess.check_output(cmd,cwd=a.repo,text=True,timeout=30).strip()
 head=run(['git','rev-parse','HEAD']);assert head=='65dda206e28770b242fa0dd888fd394f7a155ca0'
 paths=run(['git','ls-files']).splitlines();modules=[p for p in paths if p.endswith('.py') and (p.startswith(('r3_models/','fem_warmstart/')) or p in ['r3_models.py','eval/feature_guard.py','eval/predictors.py','eval/doe_dataset.py','train_doe_curl_mgn.py'])]
 report={'source_repo_commit':head,'new_FEA':0,'training_updates':0,'GPU_jobs':0,'model_imports':0,'weight_deserialization':0,'modules':[],'checkpoints':[],'prior_cache':[]}
 with zipfile.ZipFile(a.out/'admission_code.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for key in modules:
   check();p=a.repo/key
   if p.is_file():report['modules'].append({'source_key':key,'bytes':p.stat().st_size,'sha256':sha(p)});z.write(p,key)
 tag='mgn_nodeB_doe_v3_bw30_prior_e1airgap5.pt'
 for base in [a.repo/'results',a.data/'results']:
  for name in [tag,tag+'.last']:
   check();p=base/name
   if p.is_file():
    item={'source_key':('EveryMotor/results/' if base==a.repo/'results' else 'NvidiaNemo/results/')+name,'bytes':p.stat().st_size,'sha256':sha(p)}
    if zipfile.is_zipfile(p):
     with zipfile.ZipFile(p) as z:item['archive_members']=z.namelist()[:12]
    report['checkpoints'].append(item)
 for key in ['results/checkpoints.json','results/e1_airgap5_verdict_20260914.json','eval/splits/doe_v3_case_split.json']:
  check();p=a.repo/key
  if p.exists():
   q=json.loads(p.read_text(encoding='utf-8-sig'));(a.out/p.name).write_text(json.dumps(q,indent=2)+'\n')
 report['source_split_sizes']={}
 split=a.repo/'eval/splits/doe_v3_case_split.json'
 if split.exists():
  s=json.loads(split.read_text(encoding='utf-8'));report['source_split_sizes']={k:len(s.get(k,[])) for k in ['train','val','test']};report['candidate_cases_old_role']={str(i):[k for k in ['train','val','test'] if i in s.get(k,[])] for i in [74,98]}
 cache=a.data/'results/prior_cache'
 if cache.exists():
  import numpy as np
  for p in cache.glob('*/case_*.npz'):
   if p.stem not in ['case_0074','case_0098']:continue
   check()
   with np.load(p,allow_pickle=False) as z:info={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
   report['prior_cache'].append({'source_key':'NvidiaNemo/results/prior_cache/'+p.relative_to(cache).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size,'arrays':info,'native_hash_binding':'UNKNOWN_not_inferred_from_filename'})
 report['gates']={
  'checkpoint_file_identity':'PASS' if report['checkpoints'] else 'UNKNOWN',
  'architecture_source_present':'PASS' if any(r['source_key'].startswith('r3_models') for r in report['modules']) else 'UNKNOWN',
  'prior_code_present':'PASS' if any(r['source_key'].startswith('fem_warmstart/') for r in report['modules']) else 'UNKNOWN',
  'prior_native_geometry_OP_mesh_binding':'UNKNOWN',
  'all_donor_reserved_exclusion':'UNKNOWN_initial40_native_source_unlinked',
  'checkpoint_normalizer_feature_compatibility':'UNKNOWN_not_imported',
  'independent_geometry_holdout_for_E1_warmstart':'FAIL_current_dev32_oldTRAIN',
  'weights_teacher_distillation_admitted':'UNKNOWN_no_execution_authorized_by_file_existence'}
 report['decision']='HOLD_weights_until_provenance_and_reserved_gates';report['elapsed_seconds']=time.monotonic()-start;report['checked_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (a.out/'TRANSFER_ADMISSION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['modules','prior_cache']}))
if __name__=='__main__':main()
