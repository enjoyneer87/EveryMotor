"""Prepared metadata-only dev32 admission reader; NOT RUN on real data in this stage.

Requires explicit source root; opens HDF5 in r mode and reads only mesh/regions.
No model/field/prior/solver imports. Run only under a separately released CPU plan.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from strict_airgap import build_strict_airgap, validate_all45_stationarity


def hash_file(path, check):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4 * 1024**2):
            check()
            h.update(chunk)
    return h.hexdigest()


def hash_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def inspect_case(h, rec, check, operator_sink=None):
    """Explicit HDF5 dataset allowlist: no solution/field dataset access."""
    node_ids = h['mesh/node_id'][:]
    element_ids = h['mesh/tri_index'][:]
    triangles = np.column_stack([h[f'mesh/node_{k}'][:] for k in (1, 2, 3)])
    codes = h['mesh/reg_code'][:]
    name_codes, name_values = h['regions/reg_code'][:], h['regions/name'].asstr()[:]
    if len(name_codes) != len(set(name_codes.tolist())) or len(name_codes) != len(name_values):
        raise ValueError('duplicate or mismatched region-name mapping')
    names = dict(zip(name_codes.tolist(), name_values.tolist()))
    moving_codes = h['mesh/moving_reg_codes'][:]
    moving_ids = h['mesh/moving_node_id'][:]
    moving_positions = h['mesh/moving_node_indices'][:]
    if (moving_positions.dtype.kind not in 'iu' or len(np.unique(moving_positions)) != len(moving_positions)
            or np.any(moving_positions < 0) or np.any(moving_positions >= len(node_ids))
            or not np.array_equal(node_ids[moving_positions], moving_ids)):
        raise ValueError('moving native IDs do not bind to positions')
    xy_ref = np.column_stack([h['mesh/node_x_mm'][:], h['mesh/node_y_mm'][:]])
    moving_x = h['mesh/node_x_mm_by_step_moving']
    moving_y = h['mesh/node_y_mm_by_step_moving']
    if moving_x.shape != (45, len(moving_ids)) or moving_y.shape != moving_x.shape:
        raise ValueError('expected 45 moving-coordinate steps')
    # Native export's candidate table includes stationary interface nodes.
    # Admission requires exact equality to reference in every stored step.
    moving_x_values, moving_y_values = moving_x[:], moving_y[:]
    stable_candidates = np.all(moving_x_values == xy_ref[moving_positions, 0], axis=0) & np.all(
        moving_y_values == xy_ref[moving_positions, 1], axis=0)
    verified_stationary_ids = moving_ids[stable_candidates]
    a1 = np.array([names[int(c)].lower() == 'a1' for c in codes])
    admitted = np.array([names[int(c)].lower() != 'a2' for c in codes])
    if not a1.any():
        raise ValueError('missing native a1; no fallback')
    # Native a1 node angular coverage, including boundary nodes, supports n_sym.
    # Retain the measured span; do not assume the same sector for every design.
    order = np.argsort(node_ids)
    a1_node_ids = np.unique(triangles[a1])
    indices = np.searchsorted(node_ids[order], a1_node_ids)
    if np.any(indices >= len(node_ids)) or not np.array_equal(node_ids[order][indices], a1_node_ids):
        raise ValueError('unmapped a1 node ID')
    a1_xy = xy_ref[order[indices]]
    angles = np.unique(np.mod(np.arctan2(a1_xy[:, 1], a1_xy[:, 0]), 2*np.pi))
    if len(angles) < 2 or not np.isfinite(angles).all():
        raise ValueError('insufficient angular coverage evidence')
    gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
    span = float(2*np.pi - gaps.max())
    ratio = 2*np.pi/span if span > 0 else float('inf')
    if not np.isfinite(ratio):
        raise ValueError('invalid sector span')
    multiplier = round(ratio)
    if multiplier < 1 or abs(ratio-multiplier) > .02*multiplier:
        raise ValueError('a1 angular span does not bind a sector multiplier')
    source_xy = xy_ref[(np.hypot(xy_ref[:, 0], xy_ref[:, 1]) > 1e-9)
                       & ~np.isin(node_ids, moving_ids)]
    source_angles = np.unique(np.mod(np.arctan2(source_xy[:, 1], source_xy[:, 0]), 2*np.pi))
    source_span = float(2*np.pi - np.diff(np.r_[source_angles, source_angles[0]+2*np.pi]).max())
    source_ratio = 2*np.pi/source_span
    if not np.isfinite(source_ratio) or abs(source_ratio-multiplier) > .02*multiplier:
        raise ValueError('stationary reference mesh and a1 sector support disagree')
    operators = {}
    for step in range(45):
        check()
        xy = xy_ref.copy()
        xy[moving_positions, 0] = moving_x[step]
        xy[moving_positions, 1] = moving_y[step]
        operators[step] = build_strict_airgap(
            node_ids, xy, element_ids, triangles, codes, names, moving_codes, moving_ids,
            admitted_element_ids=element_ids[admitted], axial_length_m=rec['stack_length_m'],
            symmetry_multiplier=multiplier, verified_stationary_node_ids=verified_stationary_ids)
    validate_all45_stationarity(operators)
    op = operators[0]
    if operator_sink is not None:
        operator_sink(op)
    return {
        'case_index': rec['case_index'], 'source_sha256': rec['source_sha256'],
        'status': 'GEOMETRY_CHECKED_BINDING_PENDING', 'steps_checked': 45,
        'a1_native_element_ids': op.a1_element_ids.tolist(),
        'a1_id_sha256': hash_array(op.a1_element_ids),
        'admitted_id_sha256': hash_array(op.admitted_element_ids),
        'a1_incidence_sha256': hash_array(op.a1_triangle_node_ids),
        'a1_coordinates_sha256': hash_array(op.a1_node_xy_mm),
        'operator': op.band.describe(), 'axial_length_m': op.axial_length_m,
        'a1_node_angular_span_rad': span,
        'source_symmetry_confirmation': 'STATIONARY_REFERENCE_MESH_AND_A1_GEOMETRIC_SUPPORT_AGREE',
        'source_reference_angular_span_rad': source_span,
        'candidate_moving_node_id_sha256': hash_array(moving_ids),
        'verified_stationary_candidate_id_sha256': hash_array(verified_stationary_ids),
        'a1_candidate_moving_nodes': int(np.intersect1d(op.a1_node_ids, moving_ids).size),
        'a1_verified_stationary_candidates': int(np.intersect1d(op.a1_node_ids, verified_stationary_ids).size),
        'stationarity_rule': 'all45 native candidate coordinates exactly equal native reference; no tolerance relaxation',
        'symmetry_scope': 'geometric sector multiplier; not a new anti-periodic boundary-condition proof',
        'dev_bundle_exact_ID_join': 'REQUIRED_BEFORE_RELEASE',
        'field_array_reads': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True,
                        help='Expected digest frozen in the future release contract')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.request.read_bytes()).hexdigest() != args.request_sha256:
        raise ValueError('request SHA differs from frozen release')
    request = json.loads(args.request.read_text(encoding='utf-8'))
    records = request['records']
    if len(records) != 32 or len({r['case_index'] for r in records}) != 32 or request['steps'] != list(range(45)):
        raise ValueError('request must be exact dev32 by45')
    if any(r['split'] != 'development' or r['original_split'] != 'train' for r in records):
        raise ValueError('only frozen oldTRAIN innerdev is admitted')
    base = args.source_root.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def check():
        if time.monotonic()-start > 600 or (args.out/'STOP').exists():
            raise InterruptedError('STOP or 600s CPU bound')
    result = {'status': 'RUNNING', 'records': [], 'new_inference': 0, 'new_FEA': 0,
              'request_sha256': hash_file(args.request, check)}
    try:
        import h5py
        for rec in records:
            check()
            path = (base/rec['source_key']).resolve(strict=True)
            if not path.is_relative_to(base):
                raise ValueError('source key escapes declared root')
            before = path.stat()
            if hash_file(path, check) != rec['source_sha256']:
                raise ValueError('source SHA mismatch')
            with h5py.File(path, 'r') as h:
                operator_path = args.out / ('operator_case_%04d.npz' % rec['case_index'])
                def sink(op):
                    arrays = {k: getattr(op, k) for k in ('admitted_element_ids', 'a1_element_ids',
                        'a1_node_ids', 'a1_node_xy_mm', 'a1_triangle_node_ids', 'a1_orientation', 'axial_length_m')}
                    arrays.update({k: getattr(op.band, k) for k in ('radius_m', 'area_m2', 'cos_theta', 'sin_theta',
                        'r_inner_m', 'r_outer_m', 'sector_span_rad', 'symmetry_multiplier', 'region_codes')})
                    np.savez_compressed(operator_path, **arrays)
                row = inspect_case(h, rec, check, sink)
                row.update(operator_file=operator_path.name, operator_sha256=hash_file(operator_path, check))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or hash_file(path, check) != rec['source_sha256']:
                raise ValueError('source changed during audit')
            result['records'].append(row)
        result['status'] = 'GEOMETRY_CHECKED_BINDING_PENDING'
    except Exception as exc:
        result.update(status='INCOMPLETE_NO_CLAIM', error=repr(exc),
                      failed_case_index=rec['case_index'] if 'rec' in locals() else None)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-start
        (args.out/'GEOMETRY_AUDIT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
