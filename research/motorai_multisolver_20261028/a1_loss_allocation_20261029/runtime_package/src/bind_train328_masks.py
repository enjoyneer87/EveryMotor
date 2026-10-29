"""CPU-only native a1 mask admission on fixed train328x9 and development32x9.

No model imports and no target_B or other solution-array deserialization. Reuses
the admitted dev32 operator sidecars; opens no development HDF5. Result arrays are
derived from IDs/regions/geometry only. Output is new and preserves partial failure.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import re
import time
import numpy as np

SELECTION_SHA = '21e78f734250ace514f2d2e646e9a382ca271cc6cf703a54068b7c3e0a9acb7d'
BUNDLE_SHA = '0ab3743b64274f3a46222de4ece9ab8138ca07ab87b0603a0f9fa642d892c4d4'
DEV_BUNDLE_SHA = '5bf699ac5b89937f0ac7acb194c1ce8a075bef6b81c9e9e72b2834be4fc7c186'
DEV_GEOMETRY_SHA = '060222bb23cefaa9b33490c3e7f880a5ce7ba2459394356a10f25e649e6fb4dc'
STEPS = list(range(0, 45, 5))
BASE_OPERATOR_KEYS = ('admitted_element_ids','a1_element_ids','a1_triangle_node_ids','region_codes','area_m2','radius_m','cos_theta','sin_theta')
TRAIN_OPERATOR_KEYS = ('admitted_triangle_node_ids','admitted_region_codes','admitted_area_m2_by_train_step','native_area_steps')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_file(path, check=lambda:None):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4*1024**2):
            check(); result.update(block)
    return result.hexdigest()


def hash_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def pinned_json(path, digest, check):
    require(hash_file(path,check)==digest, 'pinned JSON SHA differs: '+Path(path).name)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def child_file(root, name):
    require(isinstance(name,str) and '\\' not in name and ':' not in name, 'unsafe child name')
    path = (root/name).resolve(strict=True)
    require(path.is_relative_to(root.resolve()) and path.is_file(), 'child escapes bound root')
    return path


def positions(have, need):
    require(have.ndim==1 and have.dtype.kind in 'iu' and len(np.unique(have))==len(have), 'invalid native IDs')
    order=np.argsort(have); offsets=np.searchsorted(have[order],need)
    require(np.all(offsets<len(have)), 'native ID absent')
    require(np.array_equal(have[order][offsets],need), 'native ID mismatch')
    return order[offsets]


def phase_files(record):
    found={}
    for row in record['files']:
        match=re.fullmatch(r'case(\d+)_step(\d+)\.npz',Path(row['name']).name)
        require(match is not None and int(match.group(1))==record['case_index'], 'graph case filename differs')
        step=int(match.group(2))
        require(step not in found, 'duplicate graph phase')
        found[step]=row
    expected=set(range(45)) if record['split']=='train' else set(STEPS)
    require(set(found)==expected, 'frozen training bundle phase coverage differs')
    return [dict(found[step],step=step) for step in STEPS]


def validate_graph(arrays, operator, step, is_training=True):
    """Validate one graph without targets and return trainer-compatible mask pins."""
    ids,tri,codes,weights,node,native_step=(arrays[key] for key in
        ('native_element_id','native_tri_node_ids','native_reg_code','weights','node','step'))
    require(native_step.shape==() and native_step.dtype.kind in 'iu' and int(native_step)==step, 'graph step mismatch')
    require(ids.ndim==1 and ids.dtype.kind in 'iu' and len(np.unique(ids))==len(ids), 'invalid/duplicate graph native IDs')
    require(np.array_equal(np.sort(ids),operator['admitted_element_ids']), 'admitted native ID set differs')
    require(tri.shape==(len(ids),3) and tri.dtype.kind in 'iu', 'invalid native incidence')
    require(codes.shape==ids.shape and codes.dtype.kind in 'iu', 'invalid native region codes')
    require(weights.shape==ids.shape and weights.dtype.kind=='f' and np.isfinite(weights).all() and np.all(weights>0), 'invalid physical weights')
    require(node.ndim==2 and len(node)==len(ids) and node.shape[1]>=2 and np.isfinite(node[:,:2]).all(), 'invalid graph coordinates')
    a1_ids=np.asarray(operator['a1_element_ids'],dtype=np.int64)
    require(a1_ids.ndim==1 and len(a1_ids)>0 and np.array_equal(np.unique(a1_ids),a1_ids), 'invalid sorted a1 IDs')
    a1_positions=positions(ids,a1_ids)
    mask=np.isin(ids,a1_ids)
    require(np.array_equal(mask,np.isin(codes,operator['region_codes'])), 'a1 ID/region mask differs')
    require(np.array_equal(tri[a1_positions],operator['a1_triangle_node_ids']), 'a1 native incidence differs')
    tolerance=4*np.finfo(np.float32).eps
    require(np.isfinite(operator['area_m2']).all() and np.all(operator['area_m2']>0), 'invalid pinned a1 area')
    require(np.allclose(weights[a1_positions].astype(float),operator['area_m2'],rtol=tolerance,atol=0), 'a1 native area differs')
    centroid=np.column_stack([operator['radius_m']*operator['cos_theta'],operator['radius_m']*operator['sin_theta']])
    require(np.allclose(node[a1_positions,:2].astype(float)*.1,centroid,rtol=tolerance,atol=tolerance*.1), 'a1 native centroid differs')
    max_area_rel=float(np.max(np.abs(weights[a1_positions].astype(float)-operator['area_m2'])/operator['area_m2']))
    if is_training:
        require(np.array_equal(operator['native_area_steps'],STEPS), 'native area phase list differs')
        all_positions=positions(ids,operator['admitted_element_ids'])
        require(np.array_equal(tri[all_positions],operator['admitted_triangle_node_ids']), 'all admitted native incidence differs')
        require(np.array_equal(codes[all_positions],operator['admitted_region_codes']), 'all admitted native regions differ')
        areas=operator['admitted_area_m2_by_train_step']
        require(areas.shape==(9,len(ids)) and np.isfinite(areas).all() and np.all(areas>0), 'invalid pinned native areas')
        reference=areas[STEPS.index(step)]
        require(np.allclose(weights[all_positions].astype(float),reference,rtol=tolerance,atol=0), 'all admitted native area differs')
        max_area_rel=float(np.max(np.abs(weights[all_positions].astype(float)-reference)/reference))
    values=weights.astype(np.float64)
    area_all=float(values.sum()); area_a1=float(values[mask].sum())
    require(np.isfinite([area_all,area_a1]).all() and 0<area_a1<=area_all, 'invalid q areas')
    q=area_a1/area_all
    return {'step':step,'native_element_id_sha256':hash_array(ids),'native_element_id_dtype':ids.dtype.str,
            'a1_mask_sha256':hash_array(mask.astype(np.bool_)),'a1_mask_dtype':'bool',
            'a1_count':int(mask.sum()),'all_count':len(ids),'area_a1_m2':area_a1,'area_all_m2':area_all,
            'q':q,'effective_a1_allocation_alpha05':.5+.5*q,
            'native_area_relative_difference_max':max_area_rel,'area_relative_tolerance':tolerance,
            'native_area_validation_scope':'ALL_ADMITTED' if is_training else 'A1_REUSED_DEV32_ADMISSION'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('selection','bundle','train-audit','dev-geometry','dev-bundle','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--train-audit-sha256',required=True)
    parser.add_argument('--deadline-UTC',required=True)
    args=parser.parse_args()
    deadline=dt.datetime.fromisoformat(args.deadline_UTC)
    require(deadline.tzinfo is not None and dt.datetime.now(dt.timezone.utc)<deadline,'invalid/expired binding deadline')
    require(not args.out.exists(),'no repeated binding output; retry0')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    def check():
        if time.monotonic()-start>=1800 or dt.datetime.now(dt.timezone.utc)>=deadline or (args.out.parent/'STOP').exists():
            raise InterruptedError('CPU binding STOP/time/deadline')
    selection=pinned_json(args.selection,SELECTION_SHA,check)
    selected={row['case_index']:row for row in selection['records']}
    require(len(selected)==360 and sum(row['split']=='train' for row in selected.values())==328,'frozen selection count differs')
    geometry=pinned_json(args.train_audit,args.train_audit_sha256,check)
    require(geometry['status']=='GEOMETRY_CHECKED_BINDING_PENDING' and geometry['selection_sha256']==SELECTION_SHA
            and len(geometry['records'])==328 and geometry['train_steps']==STEPS,'training geometry incomplete')
    dev_geometry=pinned_json(args.dev_geometry,DEV_GEOMETRY_SHA,check)
    require(dev_geometry['status']=='ADMITTED' and len(dev_geometry['records'])==32 and dev_geometry['selection_sha256']==SELECTION_SHA,'dev32 admission differs')
    manifest=pinned_json(args.bundle/'BUNDLE_MANIFEST.json',BUNDLE_SHA,check)
    require(manifest['selection_sha256']==SELECTION_SHA and manifest['reader_pass_cases']==360 and manifest['failed_cases']==0,'training bundle admission differs')
    records={row['case_index']:row for row in manifest['records']}
    require(len(records)==len(manifest['records'])==360 and set(records)==set(selected),'training bundle case set differs')
    dev_manifest=pinned_json(args.dev_bundle/'BUNDLE_MANIFEST.json',DEV_BUNDLE_SHA,check)
    dev_records={row['case_index']:row for row in dev_manifest['records']}
    require(len(dev_records)==len(dev_manifest['records'])==32,'common dev bundle count differs')
    train_geometry={row['case_index']:row for row in geometry['records']}
    dev_operators={row['case_index']:row for row in dev_geometry['records']}
    require(set(train_geometry)=={case for case,row in selected.items() if row['split']=='train'},'training geometry cases differ')
    require(set(dev_operators)==set(dev_records)=={case for case,row in selected.items() if row['split']=='development'},'development geometry cases differ')
    result={'status':'RUNNING','selection_sha256':SELECTION_SHA,'bundle_manifest_sha256':BUNDLE_SHA,
            'geometry_audit_sha256':args.train_audit_sha256,'dev_geometry_manifest_sha256':DEV_GEOMETRY_SHA,
            'common_dev_bundle_sha256':DEV_BUNDLE_SHA,'train_steps':STEPS,'records':[],'development_records':[],
            'target_field_arrays_read':0,'new_training_updates':0,'new_inference':0,'new_FEA':0,'automatic_retries':0,
            'started_UTC':dt.datetime.now(dt.timezone.utc).isoformat()}
    save_json(args.out,result)
    try:
        for case in sorted(selected):
            check(); declaration=selected[case]; rec=records[case]
            for key in ('split','geometry_group','source_key','source_sha256','original_split'):
                require(rec[key]==declaration[key],'case source binding differs: '+key)
            require(rec['original_split']=='train' and rec['status']=='READER_ARITHMETIC_PASS','unadmitted source record')
            is_training=rec['split']=='train'
            row=train_geometry[case] if is_training else dev_operators[case]
            require(row['source_sha256']==rec['source_sha256'] and row['steps_checked']==45,'native geometry source/phase differs')
            require(row['a1_candidate_moving_nodes']==row['a1_verified_stationary_candidates'],'a1 motion admission incomplete')
            geometry_root=args.train_audit.parent if is_training else args.dev_geometry.parent
            operator_path=child_file(geometry_root,row['operator_file'])
            require(hash_file(operator_path,check)==row['operator_sha256'],'operator SHA differs')
            keys=BASE_OPERATOR_KEYS+(TRAIN_OPERATOR_KEYS if is_training else ())
            with np.load(operator_path,allow_pickle=False) as archive:
                operator={key:archive[key] for key in keys}
            require(hash_array(np.asarray(operator['a1_element_ids'],dtype=np.int64))==row['a1_id_sha256'],'a1 identity hash differs')
            graph_rows=[]
            common_by_name={Path(item['name']).name:item for item in dev_records[case]['files']} if not is_training else {}
            for item in phase_files(rec):
                check(); path=child_file(args.bundle,item['name'])
                require(path.stat().st_size==item['bytes'] and hash_file(path,check)==item['sha256'],'graph bytes differ')
                if not is_training:
                    counterpart=common_by_name[Path(item['name']).name]
                    require(counterpart['sha256']==item['sha256'] and counterpart['bytes']==item['bytes'],'development graph differs from common dev32 bundle')
                with np.load(path,allow_pickle=False) as archive:
                    arrays={key:archive[key] for key in ('native_element_id','native_tri_node_ids','native_reg_code','weights','node','step')}
                graph_rows.append(dict(item,**validate_graph(arrays,operator,item['step'],is_training)))
            entry={key:rec[key] for key in ('case_index','split','geometry_group','source_key','source_sha256')}
            entry.update(status='ADMITTED',a1_native_element_ids=operator['a1_element_ids'].astype(np.int64).tolist(),
                         a1_id_sha256=row['a1_id_sha256'],a1_id_dtype='int64',operator_file=row['operator_file'],
                         operator_sha256=row['operator_sha256'],graphs=graph_rows)
            result['records' if is_training else 'development_records'].append(entry)
            result['elapsed_seconds']=time.monotonic()-start
            save_json(args.out,result)
        require(len(result['records'])==328 and len(result['development_records'])==32,'case coverage incomplete')
        q=np.array([item['q'] for row in result['records'] for item in row['graphs']])
        require(len(q)==2952 and np.isfinite(q).all(),'training q coverage incomplete')
        result.update(status='ADMITTED',train_graphs=2952,development_graphs=288,
                      training_q_distribution={'count':len(q),'min':float(q.min()),'max':float(q.max()),'mean':float(q.mean()),
                                               'median':float(np.median(q)),'weighting':'one value per fixed training graph; no target_B read'},
                      effective_a1_allocation_alpha05={'min':float((.5+.5*q).min()),'max':float((.5+.5*q).max()),'mean':float((.5+.5*q).mean())})
    except BaseException as exc:
        result.update(status='INCOMPLETE_NO_CLAIM',error=repr(exc),failed_case=case if 'case' in locals() else None)
        raise
    finally:
        result.update(completed_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        save_json(args.out,result)
        print(json.dumps({'status':result['status'],'train_cases':len(result['records']),'development_cases':len(result['development_records']),
                          'elapsed_seconds':result['elapsed_seconds'],'sha256':hash_file(args.out)}))


if __name__=='__main__':
    main()
