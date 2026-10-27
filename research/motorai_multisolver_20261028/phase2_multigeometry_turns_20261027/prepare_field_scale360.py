"""Four bounded CPU workers; native source read-only. No solver/trainer."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,datetime,concurrent.futures
import numpy as np
from native_element_graph import read_case,sha,NODE_FEATURES,EDGE_FEATURES,DESIGN_FEATURES

def one(args):
 r,base,out,steps=args;rec={k:r[k] for k in ['case_index','split','original_split','geometry_group','source_key','source_sha256']};rec['files']=[]
 try:
  path=Path(base)/r['source_key'];assert sha(path)==r['source_sha256'];graphs,qa=read_case(path,r['native_input'],steps);rec['QA']=qa
  for g in graphs:
   p=Path(out)/f'case{r["case_index"]:04d}_step{int(g["step"]):02d}.npz';assert not p.exists();np.savez_compressed(p,**g)
   rec['files'].append({'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
  rec['status']='READER_ARITHMETIC_PASS'
 except Exception as e:rec.update(status='FAIL',error=repr(e))
 return rec
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=Path(__file__).parent
 plan=json.loads((root/'SCALE360_PLAN.json').read_text());selection=json.loads((root/'SCALE360_SELECTION_FROZEN.json').read_text());a.out.mkdir(exist_ok=False)
 assert len(selection['records'])==360
 start=time.monotonic();report=dict(created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=sha(root/'SCALE360_SELECTION_FROZEN.json'),plan_sha256=sha(root/'SCALE360_PLAN.json'),
  records=[],node_features=NODE_FEATURES,edge_features=EDGE_FEATURES,design_features=DESIGN_FEATURES,source_steps=45,selected_steps=plan['selected_steps'],new_FEA=0,training_updates=0)
 (a.out/'SELECTION_FROZEN.json').write_bytes((root/'SCALE360_SELECTION_FROZEN.json').read_bytes())
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(one,(r,str(a.base),str(a.out),plan['selected_steps'])) for r in selection['records']]
  for f in concurrent.futures.as_completed(futures):
   if (root/'STOP').exists() or time.monotonic()-start>=1800 or datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(plan['guard_deadline_UTC']):raise InterruptedError('Preparation STOP/wall/deadline')
   report['records'].append(f.result());report.update(elapsed_seconds=time.monotonic()-start,completed_cases=len(report['records']))
   (a.out/'BUNDLE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
 report['records'].sort(key=lambda x:x['case_index']);report.update(reader_pass_cases=sum(r['status']=='READER_ARITHMETIC_PASS' for r in report['records']),failed_cases=sum(r['status']=='FAIL' for r in report['records']))
 (a.out/'BUNDLE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['records','node_features','edge_features','design_features']}))
 if report['failed_cases'] or report['reader_pass_cases']!=360:raise SystemExit(2)
if __name__=='__main__':main()
