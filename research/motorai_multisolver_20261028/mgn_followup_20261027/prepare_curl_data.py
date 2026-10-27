"""Build A/curl supervision from the same immutable Ref MS24, no Full eval."""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'remote_return/desktop-6qivkic_20261027/iso_recovery'))
from load_ms_graph import load_ms_slots

def coefficients(xy,tri):
    v=xy[tri];u=v[:,1]-v[:,0];w=v[:,2]-v[:,0]
    det=u[:,0]*w[:,1]-u[:,1]*w[:,0]
    gx=np.stack([u[:,1]-w[:,1],w[:,1],-u[:,1]],axis=1)/det[:,None]
    gy=np.stack([w[:,0]-u[:,0],-w[:,0],u[:,0]],axis=1)/det[:,None]
    return np.stack([gy,-gx],axis=-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--ms',type=Path,required=True);p.add_argument('--primary-data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False);old=np.load(a.primary_data/'MS_graph_data.npz');fieldsA=[];fieldsB=[];rows=[]
    tri=old['connectivity_local'];xy=old['node_xy_m'];area=old['element_area_m2'];ew=area/area.sum();nw=old['node_area_weights'];coef=coefficients(xy,tri)
    for i,key in enumerate(old['source_keys']):
        d=load_ms_slots(a.ms/str(key));assert np.array_equal(d['connectivity_local'],tri)
        slot=np.full(len(xy),-1)
        for t,s in zip(tri,d['slot_ids']):
            assert np.all((slot[t]==-1)|(slot[t]==s));slot[t]=s
        A=d['A_node_Wb_per_m'].copy();B=d['B_element_T']
        for s in np.unique(slot):
            take=slot==s;A[:,take]-=(A[:,take]*nw[take]).sum(-1)[:,None]/nw[take].sum()
        direct=np.einsum('ftj,tjk->ftk',A[:,tri],coef)
        roundtrip=old['B_node_area_projected_T'][i][:,tri].mean(2)
        def rel(x):return float(100*np.sqrt(np.sum((x-B)**2*ew[None,:,None])/np.sum(B**2*ew[None,:,None])))
        rows.append(dict(source_key=str(key),A_curl_relative_L2_percent=rel(direct),A_curl_max_abs_T=float(abs(direct-B).max()),
            projected_node_B_roundtrip_relative_L2_percent=rel(roundtrip),
            squared_B_area_integral_bias_percent=float(100*((roundtrip**2*ew[None,:,None]).sum()/(B**2*ew[None,:,None]).sum()-1)),
            source_sha256=hashlib.sha256((a.ms/str(key)/'ALL_A_HISTORY.npz').read_bytes()).hexdigest()))
        fieldsA.append(A.astype(np.float32));fieldsB.append(B.astype(np.float32))
    assert len(rows)==24 and max(x['A_curl_max_abs_T'] for x in rows)<1e-10
    base={k:old[k] for k in old.files if k!='B_node_area_projected_T'}
    np.savez_compressed(a.out/'MS_curl_data.npz',**base,A_gauged_Wb_per_m=np.array(fieldsA),B_element_T=np.array(fieldsB),curl_coefficients_per_m=coef,node_slot=slot)
    for f in ['calibration63.json','SPLIT_MANIFEST.json']:(a.out/f).write_bytes((a.primary_data/f).read_bytes())
    report=dict(MS_sources=24,geometry='same6stationary slots',rows=rows,operator_status='PASS',
        interpretation='Measured on Elmer slots only. Area-integrated B squared is a representation diagnostic, not canonical sampled Hybrid or Full AC loss. No Motor-CAD17percent/torque59.5percent value is transferred.',
        data_sha256=hashlib.sha256((a.out/'MS_curl_data.npz').read_bytes()).hexdigest())
    (ROOT/'REPRESENTATION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(sources=24,curl_max_abs_T=max(x['A_curl_max_abs_T'] for x in rows),roundtrip_L2_percent_range=[min(x['projected_node_B_roundtrip_relative_L2_percent'] for x in rows),max(x['projected_node_B_roundtrip_relative_L2_percent'] for x in rows)])))

if __name__=='__main__':main()
