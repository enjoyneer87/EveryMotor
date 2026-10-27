"""Load actual copper FE topology for development; no coordinate graph inference.

The graph is a copper-only submesh, not a full-machine MeshGraphNet checkpoint.
J remains element-node data; shared nodes are not averaged across regions.
"""
from pathlib import Path
import numpy as np


def load_copper_graph(path):
    with np.load(Path(path),allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    tri=d["connectivity_local"]
    assert np.array_equal(d["node_ids"][tri],d["connectivity_node_ids"])
    edges=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]],axis=0)
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    d["edge_index"]=np.concatenate([edges,edges[:,::-1]],axis=0).T
    assert d["edge_index"].min()>=0 and d["edge_index"].max()<len(d["node_ids"])
    # Never create edges between conductor centers or across missing non-copper domains.
    return d
