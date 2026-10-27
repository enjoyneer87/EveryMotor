"""Freeze equal-budget selectors and build MS FE tensors; no training or solver."""
from pathlib import Path
import argparse,hashlib,json,csv,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RETURN=ROOT/'remote_return/desktop-6qivkic_20261027'
sys.path.insert(0,str(RETURN/'iso_recovery'))
from load_ms_graph import load_ms_slots

def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--ms',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    with (RETURN/'comparison_ready.csv').open() as f:rows=list(csv.DictReader(f))
    selected=[r for r in rows if r['own_calibration']=='True'];assert len(selected)==63
    assert sum(r['geometry_id']=='jeet_elmer:Ref' for r in selected)==36
    assert sum(r['geometry_id']=='jeet_elmer:SC' for r in selected)==27
    assert not any('HalfSC' in r['geometry_id'] for r in selected)
    features=[];fields=[];keys=[];contracts=[]
    for case in sorted(a.ms.iterdir()):
        d=load_ms_slots(case);keys.append(case.name);xy=d['node_xy_m'];tri=d['connectivity_local'];area=d['element_area_m2']
        w=np.bincount(tri.ravel(),weights=np.repeat(area/3,3),minlength=len(xy));node_slot=np.zeros(len(xy))
        for t,ph in zip(tri,d['physical']):node_slot[t]=ph-2401
        nodal=np.zeros((128,len(xy),2))
        for t in range(3):
            for k in range(2):
                for frame in range(128):np.add.at(nodal[frame,:,k],tri[:,t],d['B_element_T'][frame,:,k]*area/3)
        nodal/=w[None,:,None]
        if len(keys)==1:
            geometry=dict(node_xy_m=xy,node_ids=d['node_ids'],element_ids=d['element_ids'],connectivity_local=tri,edge_index=d['edge_index'],
                element_area_m2=area,node_area_weights=w/w.sum(),node_static=np.column_stack([xy/.1,np.sin(node_slot*np.pi/3),np.cos(node_slot*np.pi/3)]))
        else:
            for k in ['node_ids','element_ids','connectivity_local','edge_index']:assert np.array_equal(geometry[k],d[k])
        fields.append(nodal.astype(np.float32));features.append([float(d['source_current_Arms'])/460,float(d['source_beta_deg'])/90])
        contracts.append(dict(source_key=case.name,field_sha256=hashlib.sha256((case/'ALL_A_HISTORY.npz').read_bytes()).hexdigest()))
    np.savez_compressed(a.out/'MS_graph_data.npz',**geometry,B_node_area_projected_T=np.array(fields),source_features=np.array(features),source_keys=np.array(keys),
        phases_rad=np.arange(1,129)*2*np.pi/128)
    # Calibration truth is separate from evaluation truth; learner loads only calibration.
    train=[dict(case_id=r['case_id'],model=r['geometry_id'].split(':')[-1],current=float(r['current_Arms']),beta=float(r['beta_deg']),speed=float(r['speed_rpm']),
        source_key=r['MS_source_key'],hybrid=float(r['canonical_Hybrid_W']),truth=float(r['Full_AC_excess_W'])) for r in selected]
    infer=[dict(case_id=r['case_id'],model=r['geometry_id'].split(':')[-1],current=float(r['current_Arms']),beta=float(r['beta_deg']),speed=float(r['speed_rpm']),
        source_key=r['MS_source_key'],hybrid=float(r['canonical_Hybrid_W'])) for r in rows]
    write(a.out/'calibration63.json',train);write(a.out/'inference228.json',infer)
    groups={name:[r['case_id'] for r in rows if rule(r)] for name,rule in {
        'Ref_excluded60':lambda r:r['geometry_id']=='jeet_elmer:Ref' and r['own_excluded_development']=='True',
        'SC_strict57':lambda r:r['geometry_id']=='jeet_elmer:SC' and r['own_transfer_excluded_development']=='True',
        'SC_excluded69':lambda r:r['geometry_id']=='jeet_elmer:SC' and r['own_excluded_development']=='True',
        'HalfSC_predeclared32':lambda r:r['HalfSC_predeclared32']=='True','HalfSC_exploratory4':lambda r:r['HalfSC_exploratory4']=='True'}.items()}
    split=dict(calibration_case_ids=[r['case_id'] for r in train],calibration_count=63,groups=groups,MS_sources=contracts,
        reserved_count=0,HalfSC_training_count=0,task='known-template LF-assisted OP development and protected transfer; not inductive geometry test',
        MS_source_reuse='All24 are Ref pretraining/inference sources. Same-source history shared across target scales. This is explicitly known-template assistance.',
        group_CV='inner holdout MS source key among calibration63; no Full eval labels loaded during training',
        protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.json').read_bytes()).hexdigest(),source_comparison_sha256=hashlib.sha256((RETURN/'comparison_ready.csv').read_bytes()).hexdigest())
    write(HERE/'SPLIT_MANIFEST.json',split);write(a.out/'SPLIT_MANIFEST.json',split)
    print(json.dumps(dict(calibration=len(train),inference=len(infer),MS_sources=len(keys),nodes=len(xy),groups={k:len(v) for k,v in groups.items()})))

if __name__=='__main__':main()
