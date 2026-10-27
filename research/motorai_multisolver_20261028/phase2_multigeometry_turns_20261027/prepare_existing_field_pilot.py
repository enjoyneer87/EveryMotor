"""Prepare a bounded native B-field pilot from audited existing source data."""
from pathlib import Path
import argparse,json,hashlib,datetime,time
import numpy as np
from native_element_graph import read_case,sha,NODE_FEATURES,EDGE_FEATURES,DESIGN_FEATURES

def select(records,count):
 # Inputs only; no target/error-based selection. Unique conservative groups.
 bygroup={}
 for r in sorted(records,key=lambda x:x['case_index']):bygroup.setdefault(r['conservative_group'],r)
 rows=list(bygroup.values())
 if not rows or count==0:return []
 x=np.array([[r['native_input'][k] for k in ['Stator_Bore','Slot_Depth']] for r in rows])
 x=(x-x.min(0))/np.maximum(np.ptp(x,axis=0),1e-9);selected=[int(np.argmin(x.sum(1)))];distance=np.full(len(rows),np.inf)
 while len(selected)<min(count,len(rows)):
  distance=np.minimum(distance,np.sum((x-x[selected[-1]])**2,axis=1));distance[selected]=-1;selected.append(int(np.argmax(distance)))
 return [rows[i] for i in selected]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--train',type=int,default=16);ap.add_argument('--val',type=int,default=4);args=ap.parse_args()
 assert args.train<=16 and args.val<=4
 data=json.loads(args.catalog.read_text());candidates=[r for r in data['records'] if r['status']=='NATIVE_SOURCE_HASH_LINKED' and not r['manifest_native_geometry_mismatch'] and all(c['status']=='DISTINCT_NOMINAL_BASE' for c in r['reserved_alias_checks'])]
 train_pool=[r for r in candidates if r['conservative_split']=='train'];val_pool=[r for r in candidates if r['conservative_split']=='val']
 if len(val_pool)>=args.val:
  chosen=[dict(r,pilot_split='train') for r in select(train_pool,args.train)]+[dict(r,pilot_split='development') for r in select(val_pool,args.val)]
  selection='Native-input-only farthest selection, existing train/val split'
 else:
  # All native-linked original val cases belong to the quarantined CurrentAxis
  # set. Freeze an INNER development subset of old TRAIN, without using any old
  # val/test labels or pretrained checkpoint. It is not an unseen final test.
  all_chosen=select(train_pool,args.train+args.val);assert len(all_chosen)==args.train+args.val
  dev_positions=set(np.random.default_rng(17).choice(len(all_chosen),args.val,replace=False).tolist())
  chosen=[dict(r,pilot_split='development' if i in dev_positions else 'train') for i,r in enumerate(all_chosen)]
  selection='Frozen inner train/development split within old TRAIN only; farthest native bore/depth sampling then seed17 development assignment. Old val/test untouched; not final test.'
 args.out.mkdir(parents=True,exist_ok=False);steps=list(range(0,45,5));start=time.monotonic();report=dict(created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),catalog_sha256=sha(args.catalog),selected_case_indices=[r['case_index'] for r in chosen],
  selection=selection,selected_steps=steps,source_steps=45,new_FEA=0,
  field='native element-reported Bx/By in T',quantity='transient_field_not_AC_loss',source_solver='Motor-CAD',source_fidelity='45poststeps_onecycle_formal_convergence_UNKNOWN',
  native_R1_exact_pairs=0,paired_alignment=False,normalization='fixed engineering input units; field target remains tesla; no fitted validation normalization',
  node_features=NODE_FEATURES,edge_features=EDGE_FEATURES,design_features=DESIGN_FEATURES,records=[])
 (args.out/'SELECTION_FROZEN.json').write_text(json.dumps(report,indent=2)+'\n')
 for r in chosen:
  if time.monotonic()-start>1200:raise TimeoutError('CPU preparation wall cap')
  path=args.base/r['training_source_key'];assert sha(path)==r['training_HDF5_sha256']
  rec=dict(case_index=r['case_index'],split=r['pilot_split'],original_split=r['conservative_split'],geometry_group=r['conservative_group'],source_sha256=r['training_HDF5_sha256'],source_key=r['training_source_key'],files=[])
  try:
   graphs,qa=read_case(path,r['native_input'],steps);rec['QA']=qa
   for g in graphs:
    p=args.out/f'case{r["case_index"]:04d}_step{int(g["step"]):02d}.npz';np.savez_compressed(p,**g);rec['files'].append({'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
   rec['status']='READER_ARITHMETIC_PASS'
  except Exception as error:rec.update(status='FAIL',error=repr(error))
  report['records'].append(rec);report['elapsed_seconds']=time.monotonic()-start
  (args.out/'BUNDLE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
 report.update(reader_pass_cases=sum(r['status']=='READER_ARITHMETIC_PASS' for r in report['records']),failed_cases=sum(r['status']=='FAIL' for r in report['records']),training_updates=0)
 (args.out/'BUNDLE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['records','node_features','edge_features','design_features']}))
 if report['failed_cases']:raise SystemExit(2)
if __name__=='__main__':main()
