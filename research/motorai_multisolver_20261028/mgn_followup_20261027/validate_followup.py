"""Independent curl invariants and same-support primary B checkpoint readback."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mgn_development_20261027'))
from model import MeshGraphNet,edge_features
from train_curl import CurlMGN,curl,gauge
def coefficients(xy,tri):
    v=xy[tri];x=v[:,:,0];y=v[:,:,1]
    det=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0])
    dx=np.stack([y[:,1]-y[:,2],y[:,2]-y[:,0],y[:,0]-y[:,1]],1)/det[:,None]
    dy=np.stack([x[:,2]-x[:,1],x[:,0]-x[:,2],x[:,1]-x[:,0]],1)/det[:,None]
    return np.stack([dy,-dx],-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--primary',type=Path,required=True);p.add_argument('--curl-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);d=np.load(a.data/'MS_curl_data.npz');tri=d['connectivity_local'];xy=d['node_xy_m'];coef=coefficients(xy,tri)
    reordered=tri[:,[2,1,0]];reverse=coefficients(xy,reordered);A=d['A_gauged_Wb_per_m'][0].astype(float)
    get=lambda x,t,c:np.einsum('btj,tjk->btk',x[:,t],c)
    reverse_error=float(abs(get(A,tri,coef)-get(A,reordered,reverse)).max());gauge_error=float(abs(get(A+0.125,tri,coef)-get(A,tri,coef)).max())
    assert reverse_error<1e-10 and gauge_error<1e-10
    q=torch.randn(1,3,dtype=torch.float64,requires_grad=True);ct=torch.as_tensor(coef[:1],dtype=torch.float64);tt=torch.arange(3).reshape(1,3)
    gradient_check=torch.autograd.gradcheck(lambda x:curl(x,tt,ct),(q,),eps=1e-6,atol=1e-5,rtol=1e-4);assert gradient_check
    dev='cuda';tensor=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device=dev)
    static=tensor(d['node_static']);edges=tensor(d['edge_index'],torch.long);geom=edge_features(static,edges);ttri=tensor(tri,torch.long);tcoef=tensor(coef)
    conditions=tensor(d['source_features']);phases=tensor(d['phases_rad']);B=tensor(d['B_element_T']);weights=tensor(d['element_area_m2']/d['element_area_m2'].sum())
    primary_scale=float(json.loads((a.primary/'INPUT_FREEZE.json').read_text())['target_scale_T'])
    curl_scale=float(json.loads((a.curl_run/'INPUT_FREEZE.json').read_text())['A_scale_Wb_per_m'])
    slots=d['node_slot'];indicator=tensor(np.array([slots==s for s in np.unique(slots)]));sw=indicator.T*tensor(d['node_area_weights'])[:,None];sw/=sw.sum(0)
    results=[]
    for method in ['primary_node_B','followup_A_curl']:
        for seed in [17,29,43]:
            path=a.primary/f'MS_MGN_seed{seed}.pt' if method=='primary_node_B' else a.curl_run/f'A_curl_seed{seed}.pt'
            net=(MeshGraphNet() if method=='primary_node_B' else CurlMGN()).to(dev);net.load_state_dict(torch.load(path,map_location=dev,weights_only=True));net.eval();error=norm=0.;count=0
            with torch.no_grad():
                for s in range(24):
                    for k in [np.arange(7,128,8)[:8],np.arange(7,128,8)[8:]]:
                        pred,_=net(static,edges,geom,conditions[s].expand(len(k),-1),phases[k])
                        if method=='primary_node_B':pred=(pred*primary_scale)[:,ttri].mean(2)
                        else:pred=curl(gauge(pred[...,0],sw,indicator)*curl_scale,ttri,tcoef)
                        truth=B[s,k];error+=float(((pred-truth).square().sum(-1)*weights).sum());norm+=float((truth.square().sum(-1)*weights).sum());count+=len(k)
            results.append(dict(method=method,seed=seed,element_B_held_phase_relative_L2_percent=100*(error/norm)**.5,element_B_held_phase_RMSE_T=(error/count)**.5))
    report=dict(status='PASS',operator_tests=dict(orientation_reversal_max_abs_T=reverse_error,constant_gauge_max_abs_T=gauge_error,float64_gradient_check=gradient_check),
        same_support_comparison=results,scope='MS24 same-source held phases on original elements; no Full AC evaluation. Seeds not selected by this result.')
    a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
