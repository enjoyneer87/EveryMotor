"""Lossless Cu FE snapshots plus full scalar history; sparse snapshots are not loss labels."""
from pathlib import Path
import argparse,json,re
import numpy as np
from field_reader import native_frames,square_integral,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--derived',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    read=lambda p:json.loads(p.read_text());raw=a.raw;params=read(raw/'PARAMETERS.json');model=read(raw/'MODEL.json');contract=read(raw/'INPUT_CONTRACT.json')
    spp=params['NbSteps'];assert params['NbPeriods']==3
    nodes=np.loadtxt(raw/'mesh/mesh.nodes');elements=np.loadtxt(raw/'mesh/mesh.elements',dtype=np.int64)
    cu=elements[(elements[:,1]>=2301)&(elements[:,1]<=2336)];ids=np.unique(cu[:,3:]);mapping={int(n):i for i,n in enumerate(ids)}
    lookup={int(n):i for i,n in enumerate(nodes[:,0])};tri=np.array([[mapping[int(n)] for n in t] for t in cu[:,3:]])
    xy=nodes[[lookup[int(n)] for n in ids],2:4];v=xy[tri];u=v[:,1]-v[:,0];w=v[:,2]-v[:,0]
    det=u[:,0]*w[:,1]-u[:,1]*w[:,0];area=abs(det)/2;bar=cu[:,1]-2301;areas=np.bincount(bar,weights=area,minlength=36)
    e=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]);e=np.unique(np.sort(e,axis=1),axis=0);edges=np.concatenate([e,e[:,::-1]]).T
    names={m[2].strip():int(m[1])-1 for m in re.finditer(r'^\s*(\d+):\s*(.*)$',(raw/'circuits.dat.names').read_text(),re.M)}
    circuit=np.loadtxt(raw/'circuits.dat');components=sorted([c for c in contract['components'] if c['kind']=='copper'],key=lambda c:c['physical'])
    sigma=params['sigma_cu'];length=model['axial_length_m'];dt=1/(params['rpm']*4/60)/spp
    steps=np.arange(spp//8,spp+1,spp//8)+2*spp;selected=set(steps.tolist());last=np.zeros(len(ids));older=last.copy()
    snapshots_A=[];snapshots_B=[];snapshots_J=[];snapshot_losses=[];snapshot_currents=[]
    for step,t,perm,fullA in native_frames(raw/'mesh/native.result'):
        if step==1:
            perm_map={int(n):i for i,n in enumerate(perm[:,0])};which=np.array([perm_map[int(n)] for n in ids])
        field=fullA[which];dadt=(field-last)/dt if step==1 else (3*field-4*last+older)/(2*dt)
        if step in selected:
            data=circuit[step-1];voltage=np.array([data[names[f"res: v_component({c['component']})"]] for c in components])
            J=sigma*(voltage[bar,None]/length-dadt[tri]);cur=np.bincount(bar,weights=area*J.mean(1),minlength=36)
            dv=field[tri][:,1:]-field[tri][:,:1];gx=(dv[:,0]*w[:,1]-dv[:,1]*u[:,1])/det;gy=(u[:,0]*dv[:,1]-w[:,0]*dv[:,0])/det
            snapshots_A.append(field);snapshots_B.append(np.stack([gy,-gx],-1));snapshots_J.append(J);snapshot_currents.append(cur)
            snapshot_losses.append(float(8*length/sigma*square_integral(area,J).sum()))
        older,last=last,field
    full=np.load(a.derived/'waveforms.npz');scalar=full['scalar'];assert len(scalar)==3*spp and len(snapshots_A)==8
    selected_scalar=scalar[steps-1];jerror=np.array(snapshot_losses)-selected_scalar[:,1]
    cerr=np.array(snapshot_currents)-full['current_A'][steps-1]
    np.testing.assert_allclose(jerror,0,rtol=0,atol=1e-6)
    np.testing.assert_allclose(cerr,0,rtol=0,atol=1e-7)
    a.out.mkdir(parents=True,exist_ok=False)
    shard=a.out/'copper_graph_8phases.npz'
    np.savez_compressed(shard,node_ids=ids,node_xy_m=xy,element_ids=cu[:,0],connectivity_local=tri,connectivity_node_ids=cu[:,3:],edge_index=edges,
        physical_ids=cu[:,1],slot_ids=bar//6+1,conductor_ids=bar+1,circuit_component_ids=np.array([c['component'] for c in components]),
        element_area_m2=area,full_motor_element_volume_m3=8*length*area,source_steps=steps,time_s=selected_scalar[:,0],phase_deg=(steps-2*spp)*360/spp,
        A_node_Wb_m=np.array(snapshots_A),B_element_T=np.array(snapshots_B),J_total_element_node_A_m2=np.array(snapshots_J),current_bar_A=np.array(snapshot_currents),
        full_history_scalar=scalar,full_history_current_A=full['current_A'],full_history_prescribed_A=full['prescribed_A'])
    with np.load(shard) as z:arrays={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
    p3=scalar[2*spp:3*spp];mean8=selected_scalar[:,3].mean();fullmean=p3[:,3].mean()
    report=dict(source_native_sha256=sha(raw/'mesh/native.result'),source_mesh_sha256=sha(raw/'mesh/mesh.elements'),shard_sha256=sha(shard),shard_bytes=shard.stat().st_size,
        arrays=arrays,snapshot_steps=steps.tolist(),period=3,source_steps_per_period=spp,sparse_phase_count=8,
        Joule_snapshot_reintegration_max_error_W=float(abs(jerror).max()),current_snapshot_reintegration_max_error_A=float(abs(cerr).max()),
        sparse8_AC_mean_W=float(mean8),full_P3_AC_mean_W=float(fullmean),sparse8_vs_full_AC_mean_difference_percent=float(100*(mean8-fullmean)/fullmean),
        full_scalar_history_preserved_losslessly=True,float_precision='float64, lossless NPZ; no quantization',
        sparse_field_warning='8 snapshots do not supply a validated period average; use included full-time scalar curve for loss labels. Quantified sparse mean error is diagnostic, no new acceptance gate.',
        units=dict(A='Wb/m nodal Az',B='T element P1curl',J='A/m2 total conduction at element nodes',current='A,bar signed instantaneous',scalar_columns=['time_s','Joule_active_fullmotor_W','prescribed_DC_fullmotor_W','AC_excess_fullmotor_W','current_max_percent_phase_peak']),
        integration='P1 exact square integral, sigma47092391.429245 S/m,8sectors,active length0.15m; no end winding',
        J_definition='sigma*(circuit_component_voltage/active_length-BE_or_BDF2_dA_dt); total conduction, not Motor-CAD Je; no artificial skin/proximity decomposition',
        mapping='original IDs from native permutation and mesh; no coordinate matching; B and J keep element/material-side values',
        geometry_group='same Ref_P117 original geometry; numerical mesh/time variant, not independent training condition',
        training_use='No fields from this new numerical study entered the frozen models')
    (a.out/'shard_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['arrays','units','snapshot_steps']}))


if __name__=='__main__':main()
