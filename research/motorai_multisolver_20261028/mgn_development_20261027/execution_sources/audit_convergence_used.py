"""General-length BE/BDF2 native P1 audit; original node IDs, no spatial lookup."""
from pathlib import Path
import json,re
import numpy as np
from field_reader import native_frames,square_integral,sha,write_json,read_json

def audit(raw,out):
    raw=Path(raw);out=Path(out);out.mkdir(exist_ok=True)
    params=read_json(raw/'PARAMETERS.json');model=read_json(raw/'MODEL.json');contract=read_json(raw/'INPUT_CONTRACT.json')
    nodes=np.loadtxt(raw/'mesh/mesh.nodes');lookup={int(n):i for i,n in enumerate(nodes[:,0])}
    elements=np.loadtxt(raw/'mesh/mesh.elements',dtype=np.int64);cu=elements[(elements[:,1]>=2301)&(elements[:,1]<=2336)]
    assert np.all(cu[:,2]==303);ids=np.unique(cu[:,3:]);mapping={int(n):i for i,n in enumerate(ids)}
    tri=np.array([[mapping[int(n)] for n in r] for r in cu[:,3:]]);xy=nodes[[lookup[int(n)] for n in ids],2:4];v=xy[tri]
    u=v[:,1]-v[:,0];w=v[:,2]-v[:,0];area=abs(u[:,0]*w[:,1]-u[:,1]*w[:,0])/2;bar=cu[:,1]-2301
    areas=np.bincount(bar,weights=area,minlength=36);assert (areas>0).all()
    names={m[2].strip():int(m[1])-1 for m in re.finditer(r'^\s*(\d+):\s*(.*)$',(raw/'circuits.dat.names').read_text(),re.M)}
    circuit=np.loadtxt(raw/'circuits.dat').reshape(-1,len(names));components=sorted([c for c in contract['components'] if c['kind']=='copper'],key=lambda c:c['physical'])
    sigma=params['sigma_cu'];length=model['axial_length_m'];frequency=params['rpm']*4/60;dt=1/frequency/params['NbSteps']
    assert 'BDF Order = 2' in (raw/'case.sif').read_text() and contract['initial_A']==0
    last=np.zeros(len(ids));older=last.copy();rows=[];currents=[];prescribed=[]
    for step,t,perm,allA in native_frames(raw/'mesh/native.result'):
        assert step==len(rows)+1
        if step==1:
            permap={int(n):i for i,n in enumerate(perm[:,0])};which=np.array([permap[int(n)] for n in ids])
        a=allA[which];data=circuit[step-1];time=data[names['res: time']]
        assert abs(time-step*dt)<1e-10
        dadt=(a-last)/dt if step==1 else (3*a-4*last+older)/(2*dt)
        voltage=np.array([data[names[f"res: v_component({c['component']})"]] for c in components])
        imp=np.array([c['amplitude_A']*np.sin(2*np.pi*frequency*time+np.deg2rad(c['phase_deg'])) for c in components])
        j=sigma*(voltage[bar,None]/length-dadt[tri]);current=np.bincount(bar,weights=area*j.mean(axis=1),minlength=36)
        total=8*length/sigma*square_integral(area,j).sum();dc=8*length/sigma*np.sum(imp**2/areas)
        error=100*np.max(abs(current-imp))/(np.sqrt(2)*params['Irms'])
        rows.append([time,total,dc,total-dc,error]);currents.append(current);prescribed.append(imp);older,last=last,a
    rows=np.array(rows);assert len(rows)==len(circuit)
    nl=[]
    for block in re.split(r'MAIN: Time:',(raw/'solver.log').read_text(errors='replace'))[1:]:
        values=re.findall(r'ComputeChange: NS \(ITER=(\d+)\).*?\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s*\) :: mag',block)
        if values:nl.append(float(values[-1][2]))
    spp=params['NbSteps'];periods=[]
    for k in range(len(rows)//spp):
        r=rows[k*spp:(k+1)*spp];periods.append(dict(period=k+1,steps=[k*spp+1,(k+1)*spp],Joule_W=float(r[:,1].mean()),DC_W=float(r[:,2].mean()),AC_W=float(r[:,3].mean()),current_max_percent=float(r[:,4].max())))
    result=dict(steps=len(rows),steps_per_period=spp,periods=periods,solver_nonlinear_max=max(nl),nonlinear_status='PASS' if len(nl)==len(rows) and max(nl)<=1e-10 else 'FAIL',
        whole_current_max_percent=float(rows[:,4].max()),whole_current_status='PASS' if rows[:,4].max()<=.001 else 'FAIL',
        current_gate_percent=.001,area_m2_by_bar=areas.tolist(),nodes=len(nodes),elements=len(elements),copper_elements=len(cu),
        sigma_S_m=sigma,active_length_m=length,average='uniform post-step arithmetic mean',initial_A=0,BDF='BE step1, BDF2 thereafter',
        period_status='UNKNOWN',mesh_status='UNKNOWN',time_step_status='UNKNOWN',input_sha256={name:sha(raw/name) for name in ['case.sif','PARAMETERS.json','MODEL.json','INPUT_CONTRACT.json','circuits.definitions','CircuitsBDF2Consistent.so','mesh/mesh.nodes','mesh/mesh.elements']})
    if len(periods)>=3:
        p2=rows[spp:2*spp,3];p3=rows[2*spp:3*spp,3]
        dm=100*abs(p3.mean()-p2.mean())/abs(p3.mean());dl=100*np.linalg.norm(p3-p2)/np.linalg.norm(p3)
        result.update(P2_P3_mean_percent=float(dm),P2_P3_wave_L2_percent=float(dl),P3_current_status='PASS' if periods[2]['current_max_percent']<=.001 else 'FAIL',
            period_status='PASS' if dm<=.1 and dl<=.5 else 'FAIL',period_gate=dict(mean_percent=.1,wave_L2_percent=.5))
    np.savez_compressed(out/'waveforms.npz',scalar=rows,current_A=np.array(currents),prescribed_A=np.array(prescribed))
    write_json(out/'audit.json',result);return result
