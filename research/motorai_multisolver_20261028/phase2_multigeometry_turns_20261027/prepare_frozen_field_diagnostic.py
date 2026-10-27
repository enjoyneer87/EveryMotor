from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,argparse,concurrent.futures,time,datetime
from prepare_field_scale360 import one
from native_element_graph import sha,NODE_FEATURES,EDGE_FEATURES,DESIGN_FEATURES
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();r=Path(__file__).parent
 selection=json.loads((r/'SCALE360_SELECTION_FROZEN.json').read_text());records=[x for x in selection['records'] if x['split']=='development'];assert len(records)==32 and len({x['geometry_group'] for x in records})==32 and all(x['original_split']=='train' for x in records)
 a.out.mkdir(exist_ok=False);start=time.monotonic();report=dict(selection_sha256=sha(r/'SCALE360_SELECTION_FROZEN.json'),plan_sha256=sha(r/'FROZEN_FIELD_DIAGNOSTIC_PLAN.json'),records=[],node_features=NODE_FEATURES,edge_features=EDGE_FEATURES,design_features=DESIGN_FEATURES,new_FEA=0,training_updates=0)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(one,(x,str(a.base),str(a.out),list(range(45)))) for x in records]
  for f in concurrent.futures.as_completed(futures):
   if (r/'STOP').exists() or time.monotonic()-start>1200 or datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-10-27T22:59:00+00:00'):raise InterruptedError('STOP/time')
   report['records'].append(f.result());report.update(completed_cases=len(report['records']),elapsed_seconds=time.monotonic()-start);(a.out/'BUNDLE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
 report['records'].sort(key=lambda x:x['case_index']);report.update(reader_pass_cases=sum(x['status']=='READER_ARITHMETIC_PASS' for x in report['records']),failed_cases=sum(x['status']=='FAIL' for x in report['records']))
 (a.out/'BUNDLE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['records','node_features','edge_features','design_features']}))
 if report['reader_pass_cases']!=32 or report['failed_cases']:raise SystemExit(2)
if __name__=='__main__':main()
