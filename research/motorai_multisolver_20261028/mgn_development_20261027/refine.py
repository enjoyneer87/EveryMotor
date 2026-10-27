"""Declared MS-only continuation; never loads Full evaluation labels."""
from pathlib import Path
import argparse,json,time,os,math
import numpy as np
import torch,psutil
from model import MeshGraphNet,edge_features
from train import write,sha,check_bounds,embed,field_monitor,head


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False);torch.set_num_threads(4);device='cuda'
    initial_freeze=json.loads((a.initial/'TRAINING_COMPLETE.json').read_text())
    for name,h in initial_freeze['frozen_files'].items():assert sha(a.initial/name)==h
    write(a.out/'OWNER.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),stop_file='STOP',automatic_retry=False))
    train=json.loads((a.data/'calibration63.json').read_text());infer=json.loads((a.data/'inference228.json').read_text())
    assert len(train)==63 and all(r['model'] in ['Ref','SC'] for r in train) and all('truth' not in r for r in infer)
    d=np.load(a.data/'MS_graph_data.npz');keys=d['source_keys']
    static=torch.as_tensor(d['node_static'],dtype=torch.float32,device=device);edges=torch.as_tensor(d['edge_index'],dtype=torch.long,device=device)
    edge_geom=edge_features(static,edges);conditions=torch.as_tensor(d['source_features'],dtype=torch.float32,device=device)
    phases=torch.as_tensor(d['phases_rad'],dtype=torch.float32,device=device);target=torch.as_tensor(d['B_node_area_projected_T'],device=device)
    weights=torch.as_tensor(d['node_area_weights'],dtype=torch.float32,device=device);train_phases=np.array([p for p in range(128) if p%8!=7])
    scale=target[:,train_phases].square().mean().sqrt()
    write(a.out/'INPUT_FREEZE.json',dict(stage1_freeze_sha256=sha(a.initial/'TRAINING_COMPLETE.json'),amendment_sha256=sha(Path(__file__).parent/'AMENDMENT_02.json'),
        code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},eval_truth_loaded=False,Full_budget=63,MS_sources=24,seeds=[17,29,43],updates_additional=16200))
    predictions={};receipts=[]
    for seed in [17,29,43]:
        start=time.monotonic();torch.manual_seed(seed+1800);torch.cuda.manual_seed_all(seed+1800);rng=np.random.default_rng(seed+1800)
        net=MeshGraphNet().to(device);net.load_state_dict(torch.load(a.initial/f'MS_MGN_seed{seed}.pt',map_location=device,weights_only=True))
        optimizer=torch.optim.Adam(net.parameters(),lr=.0003);curve=[];torch.cuda.reset_peak_memory_stats()
        for update in range(1,16201):
            lr=.00001+(.0003-.00001)*.5*(1+math.cos(math.pi*(update-1)/16200))
            for group in optimizer.param_groups:group['lr']=lr
            net.train();s=rng.integers(0,24,size=8);p=rng.choice(train_phases,size=8)
            pred,_=net(static,edges,edge_geom,conditions[s],phases[p]);loss=((pred-target[s,p]/scale).square().sum(-1)*weights).sum(-1).mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();check_bounds(a.out,start)
            if update==1 or update%500==0 or update==16200:
                r=dict(update_total=1800+update,normalized_B_training_MSE=float(loss),lr=lr,elapsed_seconds=time.monotonic()-start)
                curve.append(r);write(a.out/'progress.json',dict(seed=seed,**r));print(json.dumps(dict(seed=seed,**r)),flush=True)
        pretrain=time.monotonic()-start;monitor=field_monitor(net,static,edges,edge_geom,conditions,phases,target,weights,scale)
        before=time.monotonic();learned=embed(net,static,edges,edge_geom,conditions,phases,weights);elapsed=time.monotonic()-before
        name=f'MS_MGN_seed{seed}';torch.save(net.state_dict(),a.out/(name+'.pt'))
        np.savez_compressed(a.out/f'embeddings_seed{seed}.npz',source_keys=keys,MS_pretrained=learned)
        predictions[name]=head(name,train,infer,keys,learned,a.out)
        r=dict(seed=seed,total_updates=18000,additional_updates=16200,pretrain_seconds=pretrain,wall_seconds=time.monotonic()-start,monitor=monitor,
            extract24_embeddings_seconds=elapsed,peak_VRAM_bytes=torch.cuda.max_memory_allocated(),RAM_bytes=psutil.Process().memory_info().rss,training_curve=curve)
        receipts.append({k:v for k,v in r.items() if k!='training_curve'});write(a.out/f'training_seed{seed}.json',r)
    write(a.out/'predictions_frozen.json',predictions)
    write(a.out/'TRAINING_COMPLETE.json',dict(status='complete',primary_MGN=True,Full_budget=63,MS_sources=24,protected_eval_truth_seen=False,
        seeds=[17,29,43],total_updates_per_seed=18000,training_receipts=receipts,automatic_resume=False,
        frozen_files={p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file() and p.name not in ['OWNER.json','progress.json']}))
    print('REFINEMENT_COMPLETE',flush=True)


if __name__=='__main__':main()
