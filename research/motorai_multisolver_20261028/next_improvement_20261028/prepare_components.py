"""Create component IDs from admitted graph native adjacency, without field reads."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def component_ids(count, edges):
    parent = np.arange(count)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    for left, right in edges.T:
        a, b = root(int(left)), root(int(right))
        if a != b:
            parent[a] = b
    labels = np.array([root(i) for i in range(count)])
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int64)


def prepare(bundle, selection_path, out):
    start = time.monotonic()
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    manifest = json.loads((bundle/'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    assert sha(selection_path) == manifest['selection_sha256'] == '21e78f734250ace514f2d2e646e9a382ca271cc6cf703a54068b7c3e0a9acb7d'
    assert sha(bundle/'BUNDLE_MANIFEST.json') == '0ab3743b64274f3a46222de4ece9ab8138ca07ab87b0603a0f9fa642d892c4d4'
    admitted = {r['case_index']:r for r in selection['records']}
    out.mkdir(exist_ok=False)
    rows = []
    for record in manifest['records']:
        ref = admitted[record['case_index']]
        assert record['split'] == ref['split'] and record['geometry_group'] == ref['geometry_group'] and record['source_sha256'] == ref['source_sha256']
        assert record['original_split'] == ref['original_split'] == 'train'
        entry = next(f for f in record['files'] if f['name'].endswith('_step00.npz'))
        path = bundle/entry['name']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            native_ids = arrays['native_element_id']
            edges = arrays['edge_index']
            labels = component_ids(len(native_ids), edges)
        target = out/f"case{record['case_index']:04d}_components.npz"
        np.savez_compressed(target, component_id=labels, native_element_id=native_ids)
        rows.append({'case_index':record['case_index'], 'split':record['split'], 'geometry_group':record['geometry_group'],
                     'source_sha256':record['source_sha256'], 'reference_graph_name':entry['name'], 'reference_graph_sha256':entry['sha256'],
                     'edge_index_sha256':hashlib.sha256(edges.tobytes()).hexdigest(),
                     'native_element_id_sha256':hashlib.sha256(native_ids.tobytes()).hexdigest(),
                     'component_count':int(labels.max())+1, 'triangles':len(labels),
                     'file':target.name, 'bytes':target.stat().st_size, 'sha256':sha(target)})
    assert len(rows) == 360 and sum(r['split']=='train' for r in rows)==328
    result={'status':'COMPLETED','records':rows,'cases':len(rows),'elapsed_seconds':time.monotonic()-start,
            'selection_sha256':sha(selection_path),'source_bundle_manifest_sha256':sha(bundle/'BUNDLE_MANIFEST.json'),
            'field_arrays_read':0,'training_updates':0,'new_FEA':0,
            'scope':'native-edge connected-component metadata, not new physical topology or field truth'}
    (out/'COMPONENT_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(rows),'elapsed_seconds':result['elapsed_seconds'],
                      'component_counts':sorted(set(r['component_count'] for r in rows))}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.bundle,args.selection,args.out)
