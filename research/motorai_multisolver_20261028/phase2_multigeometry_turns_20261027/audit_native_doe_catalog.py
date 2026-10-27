"""Join existing DOE HDF5 to native input by file hash; no field payload/training.

Source index ranges propose a join only. A match is accepted only after original
and training-copy HDF5 SHA-256 equality. Source/geometry groups remain conservative.
"""
from pathlib import Path
import argparse,json,hashlib,datetime,time,os,math,collections

KEYS=('Ratio_Bore','Ratio_SlotDepth_ParallelSlot','Stator_Lam_Dia','Stator_Bore','Slot_Depth','Airgap','Slot_Width','Slot_Opening',
 'Copper_Width','Copper_Height','Copper_Corner_Radius','Insulation_Thickness','Liner_Thickness','ConductorSeparation','ConductorsPerSlot','ParallelPaths',
 'PoleVAngle_Array[0]','PoleVAngle_Array[1]','TorquePointsPerCycle','TorqueNumberCycles','TransientCalculationType','ProximityLossModel',
 'ProximityLossModel_FullFEA_CurrentPlacement','RMSCurrent','PeakCurrent','CurrentDefinition','PhaseAdvance','ShaftSpeed','ArmatureConductor_Temperature',
 'Stator_Lam_Length','Rotor_Lam_Length','Magnet_Length')
GEOM=KEYS[:21]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--raw-root',type=Path,required=True);ap.add_argument('--reserved',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--wall-seconds',type=int,default=300);args=ap.parse_args()
 start=time.monotonic();mp=args.base/'backup/doe_data_v3/doe_manifest.json';doc=json.loads(mp.read_text(encoding='utf-8-sig'));reserved=json.loads(args.reserved.read_text())
 split_path=Path.cwd()/'eval/splits/doe_v3_case_split.json';split=json.loads(split_path.read_text());split_of={int(i):s for s in ['train','val','test'] for i in split[s]}
 records=[]
 for row in doc['cases']:
  if time.monotonic()-start>args.wall_seconds:break
  idx=int(row['index']);r={'case_index':idx,'source_geometry_index':row.get('source_geometry_index'),'declared_geometry':row['geometry'],'original_split':split_of[idx],'status':'UNKNOWN'}
  if idx<40:r['reason']='Original initial-40 native source missing; do not substitute remeshed DOE4TrainingData';records.append(r);continue
  if idx<120:family,local='DOE_Ext',idx-40
  elif idx<240:family,local='DOE_Ext2',idx-120
  elif idx<320:family,local='DOE_CurrentAxis',idx-240
  else:family,local='DOE_Ext3',idx-320
  case=args.raw_root/family/f'case_{local:04d}';mot=case/'TestCAD1.mot';orig=case/'postproc/Mag_OnLoadTorque_result_1.h5'
  name=next(n for n in row['h5_paths'] if 'OnLoadTorque' in n);hp=Path(name);hp=hp if hp.is_absolute() else args.base/hp
  r.update(logical_source_key=f'{family}/case_{local:04d}',training_source_key=hp.relative_to(args.base).as_posix(),candidate_join='range mapping, requires hash match')
  if not all(p.exists() for p in [mot,orig,hp]):r['reason']='Missing native input or original HDF5';records.append(r);continue
  original_sha,copy_sha=sha(orig),sha(hp);r.update(original_HDF5_sha256=original_sha,training_HDF5_sha256=copy_sha,HDF5_bytes=hp.stat().st_size,native_mot_sha256=sha(mot))
  if original_sha!=copy_sha:r['status']='FAIL';r['reason']='HDF5 hash mismatch; no canonical replacement';records.append(r);continue
  vals={}
  for line in mot.read_text(encoding='utf-8',errors='replace').splitlines():
   if '=' in line:
    k,v=line.split('=',1)
    if k in KEYS:
     try:vals[k]=float(v)
     except ValueError:vals[k]=v
  r['native_input']=vals;r['native_geometry_hash']=hashlib.sha256(json.dumps({k:vals.get(k) for k in GEOM},sort_keys=True).encode()).hexdigest()
  r['manifest_native_geometry_mismatch']={k:dict(manifest=v,native=vals.get(k)) for k,v in row['geometry'].items() if k not in vals or abs(float(v)-float(vals[k]))>1e-6}
  r['reserved_alias_checks']=[]
  for forbid in reserved['geometries']:
   differences={k:abs(vals[key]-forbid[k]) for k,key in [('L1_PoleVAngle','PoleVAngle_Array[0]'),('L2_PoleVAngle','PoleVAngle_Array[1]')] if key in vals}
   # Any known magnet angle differs: distinct nominal base geometry. This is
   # exclusion evidence only, never positive proof of exact physical matching.
   different=any(v>.01 for v in differences.values())
   r['reserved_alias_checks'].append({'reserved_ID':forbid['design'],'nominal_angle_differences_deg':differences,'status':'DISTINCT_NOMINAL_BASE' if different else 'POSSIBLE_ALIAS_QUARANTINE','tolerance_deg':.01})
  r['status']='NATIVE_SOURCE_HASH_LINKED';r['formal_qualification']='UNKNOWN';r['J_Je_semantic_qualification']='UNKNOWN'
  r['source_format_counts']=dict(collections.Counter(p.suffix.lower() for p in case.rglob('*') if p.is_file() and p.suffix.lower() in ['.mes','.npz','.txt','.h5','.mot']))
  records.append(r)
  if len(records)%40==0:
   args.out.parent.mkdir(parents=True,exist_ok=True);args.out.with_suffix('.progress.json').write_text(json.dumps({'pid':os.getpid(),'completed_rows':len(records),'elapsed_seconds':time.monotonic()-start}))
 # Transitive grouping: source identity OR declared parameters OR native input.
 parent={r['case_index']:r['case_index'] for r in records}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 aliases={}
 for r in records:
  keys=[('source_index',r['source_geometry_index']),('declared',json.dumps(r['declared_geometry'],sort_keys=True))]
  if 'native_geometry_hash' in r:keys.append(('native',r['native_geometry_hash']))
  for key in keys:
   if key in aliases:parent[find(r['case_index'])]=find(aliases[key])
   else:aliases[key]=r['case_index']
 groups=collections.defaultdict(list)
 for r in records:groups[find(r['case_index'])].append(r)
 for members in groups.values():
  splits={r['original_split'] for r in members};assigned='test' if 'test' in splits else ('val' if 'val' in splits else 'train')
  for r in members:r['conservative_group']='EveryMotor_DOE_group_'+str(min(x['case_index'] for x in members));r['conservative_split']=assigned;r['original_split_conflict']=len(splits)>1
 report={'checked_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_rows':480,'checked_rows':len(records),'wall_seconds':time.monotonic()-start,
  'manifest_sha256':sha(mp),'old_split_sha256':sha(split_path),'reserved_input_sha256':sha(args.reserved),'native_HDF5_hash_linked':sum(r['status']=='NATIVE_SOURCE_HASH_LINKED' for r in records),
  'manifest_native_geometry_mismatch_rows':sum(bool(r.get('manifest_native_geometry_mismatch')) for r in records),
  'conservative_group_count':len(groups),'group_split_conflicts':sum(len({r['original_split'] for r in members})>1 for members in groups.values()),
  'original_raw_preserved':True,'field_numeric_payload_read':False,'new_FEA':0,'training_updates':0,'records':records}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='records'}))
if __name__=='__main__':main()
