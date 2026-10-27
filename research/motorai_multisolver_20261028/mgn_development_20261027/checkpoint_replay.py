"""Recompute all24 MS embeddings from each final saved MGN checkpoint."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from model import MeshGraphNet,edge_features
from train import embed,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stage2',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);d=np.load(a.data/'MS_graph_data.npz');device='cuda'
    static=torch.as_tensor(d['node_static'],dtype=torch.float32,device=device);edges=torch.as_tensor(d['edge_index'],dtype=torch.long,device=device)
    geom=edge_features(static,edges);cond=torch.as_tensor(d['source_features'],dtype=torch.float32,device=device)
    phase=torch.as_tensor(d['phases_rad'],dtype=torch.float32,device=device);weights=torch.as_tensor(d['node_area_weights'],dtype=torch.float32,device=device)
    reports=[]
    for seed in [17,29,43]:
        net=MeshGraphNet().to(device);net.load_state_dict(torch.load(a.stage2/f'MS_MGN_seed{seed}.pt',map_location=device,weights_only=True))
        observed=embed(net,static,edges,geom,cond,phase,weights);saved=np.load(a.stage2/f'embeddings_seed{seed}.npz')['MS_pretrained']
        np.testing.assert_allclose(observed,saved,rtol=1e-5,atol=1e-5)
        reports.append(dict(seed=seed,max_abs_embedding_error=float(abs(observed-saved).max()),status='PASS'))
    write(a.out,dict(status='PASS',models=3,sources_per_model=24,tolerance=dict(rtol=1e-5,atol=1e-5),checks=reports,
        scope='CUDA saved checkpoint to latent embedding replay; nondeterministic scatter rounding allowed'))


if __name__=='__main__':main()
