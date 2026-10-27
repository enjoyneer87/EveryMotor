"""Freeze expanded inner split from input metadata only, preserving pilot roles."""
from pathlib import Path
import json,hashlib,random
R=Path(__file__).resolve().parent
def main():
 dst=R/'SCALE360_SELECTION_FROZEN.json';assert not dst.exists(), 'Frozen split already exists'
 source=R/'NATIVE_DOE_CATALOG.json';cat=json.loads(source.read_text());pilot=json.loads((R/'B_FIELD_BUNDLE_MANIFEST.json').read_text())
 eligible=[r for r in cat['records'] if r['status']=='NATIVE_SOURCE_HASH_LINKED' and not r['manifest_native_geometry_mismatch'] and r['conservative_split']=='train' and all(x['status']=='DISTINCT_NOMINAL_BASE' for x in r['reserved_alias_checks'])]
 assert len(eligible)==360 and len({r['conservative_group'] for r in eligible})==320
 fixed={r['geometry_group']:r['split'] for r in pilot['records']};assert len(fixed)==20
 pool=sorted({r['conservative_group'] for r in eligible if r['conservative_group'] not in fixed})
 dev=set(random.Random(20261027).sample(pool,28))|{g for g,s in fixed.items() if s=='development'}
 records=[dict(case_index=r['case_index'],geometry_group=r['conservative_group'],split='development' if r['conservative_group'] in dev else 'train',original_split='train',source_key=r['training_source_key'],source_sha256=r['training_HDF5_sha256'],native_input=r['native_input']) for r in eligible]
 assert len({r['geometry_group'] for r in records if r['split']=='train'})==288 and len(dev)==32
 assert all(next(r['split'] for r in records if r['geometry_group']==g)==s for g,s in fixed.items())
 result=dict(catalog_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selection='All eligible oldTRAIN, pilot16/4 roles fixed; remaining28 dev groups metadata-only random seed20261027',
  train_groups=288,inner_development_groups=32,train_cases=sum(r['split']=='train' for r in records),development_cases=sum(r['split']=='development' for r in records),old_val_test_used=0,reserved_field_payload_used=0,records=records)
 dst.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
