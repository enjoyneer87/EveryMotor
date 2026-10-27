"""Read-only EveryMotor manifest/HDF5 inventory on the source host.

Never loads field numeric values or trains. The newest manifest is canonical;
older nested datasets are not summed. Output paths are a private caller option.
"""
from pathlib import Path
import argparse,json,hashlib,collections,datetime

def main():
 import h5py
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 base=args.base;root=base/'backup';manifest=root/'doe_data_v3/doe_manifest.json'
 doc=json.loads(manifest.read_text(encoding='utf-8-sig'));rows=[];schemas={};formats=collections.Counter();seen_files=set();geo_groups={}
 for row in doc['cases']:
  index=int(row['index']);geom=row['geometry'];key=hashlib.sha256(json.dumps(geom,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  geo_groups.setdefault(key,[]).append(index)
  rec={'case_index':index,'source_geometry_index':row.get('source_geometry_index'),'manifest_geometry':geom,'geometry_parameter_hash':key,
       'electrical':row.get('electrical',{}),'files':[],'split_status':'NOT_YET_BRIDGED_TO_R1_RESERVED_ALIAS_REGISTRY','field_payload_read':False}
  for name in row['h5_paths']:
   hp=Path(name);choices=[hp if hp.is_absolute() else base/hp,root/'doe_data_v3'/f'case_{index:04d}'/'postproc'/hp.name]
   found=next((p for p in choices if p.exists()),None)
   if found is None:rec['files'].append({'kind':hp.name,'found':False});continue
   try:logical=found.relative_to(base).as_posix()
   except ValueError:logical='original_campaign/'+found.parent.parent.name+'/'+found.name
   info={'logical_source_key':logical,'kind':hp.name,'found':True,'bytes':found.stat().st_size,'full_file_sha256':'NOT_COMPUTED_metadata_only'}
   with h5py.File(found,'r') as h:
    info['field_keys']=list(h['fields']) if 'fields' in h else []
    info['has_je']='fields/je' in h
    if 'OnLoadTorque' in hp.name and index in [0,40,240,320,479]:
     tree={}
     def visit(n,o):
      if isinstance(o,h5py.Dataset):tree[n]={'shape':list(o.shape),'dtype':str(o.dtype),'attributes':{str(k):str(v) for k,v in o.attrs.items()}}
     h.visititems(visit);schemas[str(index)]={'root_attributes':{str(k):str(v) for k,v in h.attrs.items()},'datasets':tree,'logical_source_key':logical}
   rec['files'].append(info)
   # Count only small, known case directories and their immediate export children.
   case=found.parent.parent
   for folder in [case,found.parent,case/'FEResultsData',case/'FEA']:
    if folder.is_dir():
     for p in folder.iterdir():
      if p.is_file() and p.suffix.lower() in ['.mes','.npz','.txt','.h5','.hdf5','.mot'] and str(p) not in seen_files:
       seen_files.add(str(p));formats[p.suffix.lower()]+=1
  rows.append(rec)
 trans=[f for r in rows for f in r['files'] if 'OnLoadTorque' in f['kind']]
 report={'checked_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'EveryMotor newest v3 manifest','manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
  'manifest_rows':len(rows),'unique_declared_geometry_parameter_sets':len(geo_groups),'unique_source_geometry_indices':len({r['source_geometry_index'] for r in rows}),
  'found_transient_HDF5':sum(f['found'] for f in trans),'transient_with_je':sum(f.get('has_je',False) for f in trans),
  'scoped_file_counts':dict(formats),'inventory_scope':'resolved manifest case directories and immediate export children only; not a whole-drive census',
  'older_nested_manifest_rows_added':False,'actual_unique_physical_geometries':'UNKNOWN_until_native_geometry_alias_audit','training_launched':False,
  'field_numeric_payload_read':False,'rows':rows,'representative_HDF5_schema':schemas,
  'geometry_parameter_groups':geo_groups,'qualification':'HDF5 field presence and mesh schema only; J/Je definition/units and reserved aliases need separate verification'}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k not in ['rows','representative_HDF5_schema','geometry_parameter_groups']}))
if __name__=='__main__':main()
