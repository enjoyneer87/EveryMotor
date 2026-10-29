"""Bounded CPU native-geometry admission for the frozen 328 training cases.

Reuses the already reviewed R3 dev32 reader, including exact all45 a1 stationarity.
Reads mesh/region arrays only. Full-file SHA hashing does not deserialize fields.
The caller must pin this file, audit_dev32_geometry.py, strict_airgap.py and eval
dependencies in its release package. No source-file or previous-output mutation.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import time
import numpy as np
from audit_dev32_geometry import inspect_case, hash_array
from eval.mesh_regions import element_areas_m2

SELECTION_SHA = '21e78f734250ace514f2d2e646e9a382ca271cc6cf703a54068b7c3e0a9acb7d'
TRAIN_STEPS = list(range(0, 45, 5))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_file(path, check=lambda: None):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            check(); result.update(block)
    return result.hexdigest()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def exact_positions(have, need):
    have, need = np.asarray(have), np.asarray(need)
    require(have.ndim == 1 and have.dtype.kind in 'iu' and len(np.unique(have)) == len(have), 'invalid native ID vector')
    order = np.argsort(have)
    offsets = np.searchsorted(have[order], need)
    require(np.all(offsets < len(have)), 'missing native ID')
    require(np.array_equal(have[order][offsets], need), 'native ID mismatch')
    return order[offsets]


def admitted_geometry_arrays(h, operator, check):
    """Exact incidence/region/area proof for all admitted elements at train phases."""
    node_ids = h['mesh/node_id'][:]
    element_ids = h['mesh/tri_index'][:]
    tri_ids = np.column_stack([h[f'mesh/node_{k}'][:] for k in (1, 2, 3)])
    region_codes = h['mesh/reg_code'][:]
    # Match build_strict_airgap's float64 native-coordinate arithmetic exactly.
    # Canonical element_areas_m2 preserves input dtype; float32 here would produce
    # a different area before conversion, even from the identical native values.
    xy_ref = np.column_stack([h['mesh/node_x_mm'][:], h['mesh/node_y_mm'][:]]).astype(np.float64)
    moving_positions = h['mesh/moving_node_indices'][:]
    moving_x, moving_y = h['mesh/node_x_mm_by_step_moving'], h['mesh/node_y_mm_by_step_moving']
    rows = exact_positions(element_ids, operator.admitted_element_ids)
    admitted_tri = tri_ids[rows]
    tri_positions = exact_positions(node_ids, admitted_tri.ravel()).reshape(admitted_tri.shape)
    areas = []
    for step in TRAIN_STEPS:
        check()
        xy = xy_ref.copy()
        xy[moving_positions, 0], xy[moving_positions, 1] = moving_x[step], moving_y[step]
        require(np.isfinite(xy[tri_positions]).all(), 'nonfinite admitted native coordinates')
        area = element_areas_m2(xy[:, 0], xy[:, 1], tuple(tri_positions.T))
        require(area.shape == (len(rows),) and np.isfinite(area).all() and np.all(area > 0), 'invalid admitted native area')
        areas.append(area)
    areas = np.asarray(areas, dtype=np.float64)
    a1_rows = exact_positions(operator.admitted_element_ids, operator.a1_element_ids)
    require(np.array_equal(areas[:, a1_rows], np.broadcast_to(operator.band.area_m2, (len(TRAIN_STEPS), len(a1_rows)))),
            'all admitted/a1 area calculations disagree')
    return {
        'admitted_triangle_node_ids': np.asarray(admitted_tri, dtype=np.int64),
        'admitted_region_codes': np.asarray(region_codes[rows], dtype=np.int64),
        'admitted_area_m2_by_train_step': areas,
        'native_area_steps': np.asarray(TRAIN_STEPS, dtype=np.int64),
    }


def inspect_training_case(h, record, check, operator_sink=None):
    captured = []
    row = inspect_case(h, record, check, captured.append)
    require(len(captured) == 1, 'native operator missing')
    operator = captured[0]
    extra = admitted_geometry_arrays(h, operator, check)
    if operator_sink is not None:
        operator_sink(operator, extra)
    row.update(split='train', geometry_group=record['geometry_group'],
               train_steps_checked=TRAIN_STEPS, all_admitted_area_steps_checked=9,
               a1_stationarity_steps_checked=45, field_array_reads=0,
               admitted_incidence_sha256=hash_array(extra['admitted_triangle_node_ids']),
               admitted_region_sha256=hash_array(extra['admitted_region_codes']),
               native_area_sums_m2=extra['admitted_area_m2_by_train_step'].sum(axis=1).tolist(),
               train_bundle_exact_ID_join='REQUIRED_BEFORE_RELEASE')
    return row


def validate_request(request, selection):
    require(request['selection_sha256'] == SELECTION_SHA, 'wrong frozen selection')
    require(request['steps'] == TRAIN_STEPS, 'training request is not phase9')
    require(request['cpu_wall_seconds_max'] == 1800, 'CPU wall bound differs')
    rows = request['records']
    selected = {row['case_index']: row for row in selection['records'] if row['split'] == 'train'}
    require(len(selected) == 328 and len(rows) == 328 and len({row['case_index'] for row in rows}) == 328,
            'training request must cover328 unique cases')
    require({row['case_index'] for row in rows} == set(selected), 'request/selection case set differs')
    for row in rows:
        reference = selected[row['case_index']]
        for key in ('geometry_group', 'source_key', 'source_sha256', 'split', 'original_split'):
            require(row[key] == reference[key], 'request/selection binding differs: ' + key)
        require(row['split'] == 'train' and row['original_split'] == 'train', 'only frozen oldTRAIN training cases admitted')
        require(row['stack_length_m'] == reference['native_input']['Stator_Lam_Length'] * .001,
                'native stack length differs')
    return sorted(rows, key=lambda row: row['case_index'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(hash_file(args.selection) == SELECTION_SHA, 'selection SHA mismatch')
    require(hash_file(args.request) == args.request_sha256, 'request SHA mismatch')
    request = json.loads(args.request.read_text(encoding='utf-8'))
    records = validate_request(request, json.loads(args.selection.read_text(encoding='utf-8')))
    deadline = dt.datetime.fromisoformat(request['deadline_UTC'])
    require(deadline.tzinfo is not None and dt.datetime.now(dt.timezone.utc) < deadline, 'expired or unzoned CPU deadline')
    base = args.source_root.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def check():
        if time.monotonic() - start >= 1800 or dt.datetime.now(dt.timezone.utc) >= deadline or (args.out/'STOP').exists() or (args.out.parent/'STOP').exists():
            raise InterruptedError('CPU STOP/time/deadline')
    result = {'status':'RUNNING', 'selection_sha256':SELECTION_SHA, 'request_sha256':args.request_sha256,
              'started_UTC':dt.datetime.now(dt.timezone.utc).isoformat(), 'records':[], 'train_steps':TRAIN_STEPS,
              'new_inference':0, 'new_training_updates':0, 'new_FEA':0, 'target_field_arrays_read':0,
              'automatic_retries':0}
    out = args.out/'GEOMETRY_AUDIT.json'
    save_json(out, result)
    try:
        import h5py
        for record in records:
            check()
            path = (base/record['source_key']).resolve(strict=True)
            require(path.is_relative_to(base), 'source escapes declared root')
            before = path.stat()
            require(hash_file(path, check) == record['source_sha256'], 'source SHA differs')
            operator_path = args.out/f"operator_case_{record['case_index']:04d}.npz"
            def sink(operator, extra):
                arrays = {key:getattr(operator, key) for key in ('admitted_element_ids','a1_element_ids','a1_node_ids',
                          'a1_node_xy_mm','a1_triangle_node_ids','a1_orientation','axial_length_m')}
                arrays.update({key:getattr(operator.band, key) for key in ('radius_m','area_m2','cos_theta','sin_theta',
                               'r_inner_m','r_outer_m','sector_span_rad','symmetry_multiplier','region_codes')})
                arrays.update(extra)
                np.savez_compressed(operator_path, **arrays)
            with h5py.File(path, 'r') as h:
                row = inspect_training_case(h, record, check, sink)
            after = path.stat()
            require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
                    and hash_file(path, check) == record['source_sha256'], 'source changed during audit')
            row.update(operator_file=operator_path.name, operator_sha256=hash_file(operator_path,check),
                       operator_bytes=operator_path.stat().st_size)
            result['records'].append(row)
            result['elapsed_seconds'] = time.monotonic()-start
            save_json(out,result)
        require(len(result['records']) == 328, 'incomplete training geometry')
        result['status'] = 'GEOMETRY_CHECKED_BINDING_PENDING'
    except BaseException as exc:
        result.update(status='INCOMPLETE_NO_CLAIM', error=repr(exc), failed_case_index=record['case_index'] if 'record' in locals() else None)
        raise
    finally:
        result.update(completed_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        save_json(out,result)
        print(json.dumps({'status':result['status'],'cases':len(result['records']),'elapsed_seconds':result['elapsed_seconds'],
                          'sha256':hash_file(out)}))


if __name__ == '__main__':
    main()
