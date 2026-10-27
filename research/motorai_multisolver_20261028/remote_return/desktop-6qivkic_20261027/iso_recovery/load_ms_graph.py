"""Actual stationary homogenized-slot FE graph from a recovered MS case.

The moving rotor/antiperiodic boundary is deliberately outside this reader's
verified coordinate scope. MS has no individual copper interfaces or J target.
"""
from pathlib import Path
import json
import numpy as np


def load_ms_slots(case_directory):
    p=Path(case_directory)
    m=json.loads((p/'MS_MANIFEST.json').read_text(encoding='utf-8-sig'))
    with np.load(p/'ALL_A_HISTORY.npz',allow_pickle=False) as z:
        ids=z['node_ids'];xy=z['node_xy_m'];tri=z['triangles'];ph=z['physical'];A=z['A_node_Wb_per_m']
    e=np.loadtxt(p/'raw/mesh/mesh.elements',dtype=np.int64)
    nodes=np.loadtxt(p/'raw/mesh/mesh.nodes');lookup={int(n):i for i,n in enumerate(nodes[:,0])}
    selection=(ph>=2401)&(ph<=2406);tri=tri[selection];ph=ph[selection]
    used=np.unique(tri);old_to_new=np.full(len(ids),-1,dtype=np.int64);old_to_new[used]=np.arange(len(used));conn=old_to_new[tri]
    id_to_element={(int(row[1]),tuple(sorted(map(int,row[3:])))):int(row[0]) for row in e if row[2]==303}
    element_ids=np.array([id_to_element[(int(tag),tuple(sorted(map(int,ids[t]))))] for tag,t in zip(ph,tri)])
    assert np.max(np.abs(xy[used]-nodes[[lookup[int(n)] for n in ids[used]],2:4]))<1e-12
    edges=np.concatenate([conn[:,[0,1]],conn[:,[1,2]],conn[:,[2,0]]]);edges=np.unique(np.sort(edges,axis=1),axis=0)
    edge_index=np.concatenate([edges,edges[:,::-1]],axis=0).T
    v=xy[tri];u=v[:,1]-v[:,0];w=v[:,2]-v[:,0];det=u[:,0]*w[:,1]-u[:,1]*w[:,0]
    area=np.abs(det)/2;av=A[:,tri];dv=av[:,:,1:]-av[:,:,:1]
    gx=(dv[:,:,0]*w[:,1]-dv[:,:,1]*u[:,1])/det;gy=(u[:,0]*dv[:,:,1]-w[:,0]*dv[:,:,0])/det
    assert np.all(area>0) and A.shape[0]==m['samples_per_period']==128
    return dict(node_ids=ids[used],node_xy_m=xy[used],element_ids=element_ids,connectivity_local=conn,
        connectivity_node_ids=ids[tri],physical=ph,slot_ids=ph-2400,edge_index=edge_index,element_area_m2=area,
        A_node_Wb_per_m=A[:,used],B_element_T=np.stack([gy,-gx],axis=-1),
        time_s=np.arange(1,129)*m['dt_s'],electrical_phase_deg=np.arange(1,129)*360/128,
        source_current_Arms=np.array(m['current_rms_A']),source_beta_deg=np.array(m['beta_deg']),
        source_radial_factor=np.array(m['radial_factor']),source_speed_rpm=np.array(m['rpm']))
