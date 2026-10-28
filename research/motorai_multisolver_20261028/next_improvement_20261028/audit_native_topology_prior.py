"""Train-only HDF5 metadata and prior-cache header audit; no fields, solve or GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import h5py
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def components(count, pairs):
    parent = np.arange(count)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    for a, b in pairs:
        left, right = root(int(a)), root(int(b))
        if left != right:
            parent[left] = right
    return len({root(i) for i in range(count)})


def triangle_pairs(tri):
    pairs = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    owners = np.tile(np.arange(len(tri)), 3)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, owners = pairs[order], owners[order]
    same = np.all(pairs[1:] == pairs[:-1], axis=1)
    assert not np.any(same[1:] & same[:-1]), 'Non-manifold edge in export topology'
    where = np.flatnonzero(same)
    return np.column_stack([owners[where], owners[where + 1]])


def inspect_cache(path, n_nodes):
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue
            with archive.open(name) as entry:
                version = np.lib.format.read_magic(entry)
                fn = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, order, dtype = fn(entry)
                headers[name[:-4]] = {'shape': list(shape), 'dtype': str(dtype), 'fortran': order}
    identity_names = [name for name in headers if any(token in name.lower() for token in ('hash', 'node_id', 'mesh_id', 'source', 'geometry', 'material'))]
    steps = sorted(name for name in headers if name.startswith('step_'))
    return {'sha256': sha(path), 'bytes': path.stat().st_size, 'keys': sorted(headers),
            'step_count': len(steps), 'all_nodal3_shapes_match': bool(steps) and all(headers[key]['shape'] == [n_nodes, 3] for key in steps),
            'step0_header': headers.get('step_0'), 'embedded_identity_keys': identity_names,
            'native_source_binding': 'UNKNOWN_not_proved_by_name_shape_or_current_scale',
            'cache_payload_arrays_read': 0}


def audit(base, request):
    rows = []
    for item in request['records']:
        assert item['split'] == item['original_split'] == 'train'
        path = base / item['source_key']
        before = path.stat()
        assert sha(path) == item['source_sha256'], item['source_key']
        with h5py.File(path, 'r') as h:
            ids = h['mesh/node_id'][:].astype(np.int64)
            elem = h['mesh/tri_index'][:].astype(np.int64)
            tri_native = np.column_stack([h['mesh/node_' + str(k)][:] for k in (1, 2, 3)]).astype(np.int64)
            reg = h['mesh/reg_code'][:]
            names = dict(zip(h['regions/reg_code'][:].tolist(), h['regions/name'].asstr()[:].tolist()))
            assert len(set(ids.tolist())) == len(ids) and len(set(elem.tolist())) == len(elem)
            order = np.argsort(ids)
            positions = np.searchsorted(ids[order], tri_native)
            assert positions.max() < len(ids) and np.array_equal(ids[order][positions], tri_native)
            tri = order[positions]
            keep = np.array([names[int(code)].lower() != 'a2' for code in reg])
            a1 = np.array([names[int(code)].lower() == 'a1' for code in reg])
            full = components(len(tri), triangle_pairs(tri_native))
            removed = components(int(keep.sum()), triangle_pairs(tri_native[keep]))
            used = np.unique(tri)
            node_edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
            lookup = np.full(len(ids), -1, dtype=np.int64)
            lookup[used] = np.arange(len(used))
            node_components = components(len(used), lookup[node_edges])
            base_xy = np.column_stack([h['mesh/node_x_mm'][:], h['mesh/node_y_mm'][:]])
            moving = h['mesh/moving_node_indices'][:].astype(np.int64)
            assert np.array_equal(ids[moving], h['mesh/moving_node_id'][:])
            a1_areas, a1_xy_hashes, invalid_elsewhere = [], [], []
            for step in range(45):
                xy = base_xy.copy()
                xy[moving, 0] = h['mesh/node_x_mm_by_step_moving'][step]
                xy[moving, 1] = h['mesh/node_y_mm_by_step_moving'][step]
                v = xy[tri]
                ab, ac = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
                signed = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
                if step == 0:
                    reference_sign = np.sign(signed)
                invalid = ~np.isfinite(signed) | (abs(signed) < 1e-10) | (np.sign(signed) != reference_sign)
                invalid_elsewhere.append(int(invalid[keep].sum()))
                a1_areas.append(float(np.abs(signed[a1]).sum() * .5e-6))
                a1_xy_hashes.append(hashlib.sha256(v[a1].tobytes()).hexdigest())
            row = {key: item[key] for key in ('case_index', 'geometry_group', 'source_key', 'source_sha256')}
            row.update(native_nodes=len(ids), native_elements=len(elem), a2_elements=int((~keep).sum()),
                       full_export_dual_components=full, without_a2_dual_components=removed,
                       full_export_used_node_components=node_components,
                       a1_elements=int(a1.sum()), a1_area_m2_min=min(a1_areas), a1_area_m2_max=max(a1_areas),
                       a1_coordinates_identical_all45=len(set(a1_xy_hashes)) == 1,
                       invalid_non_a2_elements_max=max(invalid_elsewhere), full_native_node_ID_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                       full_native_element_ID_sha256=hashlib.sha256(elem.tobytes()).hexdigest())
        candidates = []
        for cache_root in (base / 'results/prior_cache', Path('C:/work/integrate/EveryMotor/results/prior_cache')):
            if not cache_root.exists():
                continue
            for cache in sorted(cache_root.glob('*/case_' + f"{item['case_index']:04d}" + '.npz')):
                candidates.append({'logical_cache_key': 'prior_cache/' + cache.parent.name + '/' + cache.name,
                                   **inspect_cache(cache, len(ids))})
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        row['prior_cache_candidates'] = candidates
        rows.append(row)
    return {'scope': 'six admitted train HDF5 meshes and existing prior archive headers, no field-array access',
            'records': rows, 'source_hashes_matched': len(rows), 'field_arrays_read': 0, 'old_val_test_payloads_opened': 0,
            'new_FEA': 0, 'new_prior_solves': 0, 'training_or_inference_updates': 0,
            'prior_admission': 'HOLD_shape_or_filename_matches_are_not_native_source_binding',
            'export_topology_note': 'Full export connectivity is a reference topology; connectivity alone does not validate rotated a2 elements or current-step remeshing'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.base, json.loads(args.request.read_text(encoding='utf-8')))))
