"""Bounded MS pretraining and equal-budget residual fitting. No eval truth reader."""
from pathlib import Path
import argparse, hashlib, json, os, time
import numpy as np
import psutil
import torch
from model import MeshGraphNet, edge_features
from residual import feature_matrix, select_and_fit, predict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def check_bounds(out, start):
    if (out/'STOP').exists():
        raise RuntimeError('owned STOP requested')
    if time.monotonic()-start > 3600:
        raise RuntimeError('seed wall limit')
    if psutil.Process().memory_info().rss > 16*1024**3:
        raise RuntimeError('RAM limit')
    if torch.cuda.max_memory_allocated() > 12*1024**3:
        raise RuntimeError('VRAM limit')


@torch.no_grad()
def embed(net, static, edges, edge_geom, conditions, phases, weights):
    net.eval()
    vectors=[]
    for condition in conditions:
        chunks=[]
        for k in range(0, len(phases), 8):
            phase=phases[k:k+8]
            _, hidden=net(static, edges, edge_geom, condition.expand(len(phase), -1), phase)
            chunks.append((hidden*weights[None,:,None]).sum(1).cpu().numpy())
        latent=np.concatenate(chunks)
        vectors.append(np.concatenate([latent.mean(0), latent.std(0)]))
    return np.array(vectors)


@torch.no_grad()
def field_monitor(net, static, edges, edge_geom, conditions, phases, target, weights, scale):
    net.eval()
    squared_error=0.; squared_target=0.;count=0
    for s in range(len(conditions)):
        for k in [np.arange(7,128,8)[:8], np.arange(7,128,8)[8:]]:
            pred,_=net(static, edges, edge_geom, conditions[s].expand(len(k),-1), phases[k])
            truth=target[s,k]
            squared_error+=float(((pred*scale-truth).square().sum(-1)*weights).sum())
            squared_target+=float((truth.square().sum(-1)*weights).sum())
            count+=len(k)
    return dict(held_phase_vector_B_area_RMSE_T=(squared_error/count)**.5,
                held_phase_vector_B_relative_L2_percent=100*(squared_error/squared_target)**.5,
                scope='auxiliary projected MS nodal B, not Full AC error',frames=count)


def head(name, train, infer, keys, embeddings, output):
    start=time.perf_counter()
    x=feature_matrix(train, embeddings, keys)
    model,scores=select_and_fit(x, train)
    fitted=time.perf_counter()
    values=np.array([r['hybrid'] for r in infer])*np.exp(predict(model, feature_matrix(infer,embeddings,keys)))
    assert np.isfinite(values).all() and (values>0).all()
    np.savez_compressed(output/(name+'_head.npz'), **model)
    write(output/(name+'_CV.json'),dict(scores=scores,selected=dict(multiplier=model['multiplier'],regularization=model['regularization']),
         calibration_count=len(train),unique_MS_groups=len(set(r['source_key'] for r in train)),
         group_CV_fit_seconds=fitted-start,predict228_seconds=time.perf_counter()-fitted))
    return dict(zip([r['case_id'] for r in infer], values.tolist()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    device=torch.device('cuda')
    write(a.out/'OWNER.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),stop_file='STOP',automatic_retry=False))
    # The learner receives no observations/comparison/evaluation truth file.
    train=json.loads((a.data/'calibration63.json').read_text())
    infer=json.loads((a.data/'inference228.json').read_text())
    assert len(train)==63 and len(infer)==228
    assert set(r['model'] for r in train)=={'Ref','SC'}
    assert all('truth' not in r for r in infer)
    assert not any(any(t in r['case_id'] for t in ['R1_0199','R1_0076']) for r in train+infer)
    data=np.load(a.data/'MS_graph_data.npz',allow_pickle=False)
    keys=data['source_keys'];assert len(keys)==24
    static=torch.as_tensor(data['node_static'],dtype=torch.float32,device=device)
    edges=torch.as_tensor(data['edge_index'],dtype=torch.long,device=device)
    edge_geom=edge_features(static,edges)
    conditions=torch.as_tensor(data['source_features'],dtype=torch.float32,device=device)
    phases=torch.as_tensor(data['phases_rad'],dtype=torch.float32,device=device)
    target=torch.as_tensor(data['B_node_area_projected_T'],device=device)
    weights=torch.as_tensor(data['node_area_weights'],dtype=torch.float32,device=device)
    train_phases=np.array([p for p in range(128) if p%8!=7])
    scale=target[:,train_phases].square().mean().sqrt()
    inputs={p.name:sha(p) for p in a.data.iterdir() if p.is_file()}
    code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    write(a.out/'INPUT_FREEZE.json',dict(inputs=inputs,code=code,torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),nodes=len(static),directed_edges=edges.shape[1],MS_sources=24,
        target_scale_T=float(scale),Full_budget=63,eval_truth_loaded=False,seeds=[17,29,43]))
    # Sanity run is discarded; independent final networks use specified seeds.
    torch.manual_seed(1001);torch.cuda.manual_seed_all(1001)
    sanity=MeshGraphNet().to(device);optimizer=torch.optim.Adam(sanity.parameters(),lr=.001)
    start=time.monotonic(); losses=[]
    for step in range(20):
        pred,_=sanity(static,edges,edge_geom,conditions[:8],phases[:8])
        truth=target[torch.arange(8,device=device),torch.arange(8,device=device)]/scale
        loss=((pred-truth).square().sum(-1)*weights).sum(-1).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(sanity.parameters(),1.);optimizer.step()
        losses.append(float(loss));check_bounds(a.out,start)
    write(a.out/'SANITY.json',dict(updates=20,first_loss=losses[0],final_loss=losses[-1],finite=True,wall_seconds=time.monotonic()-start,
        peak_VRAM_bytes=torch.cuda.max_memory_allocated(),passed=losses[-1]<losses[0]))
    assert losses[-1]<losses[0]
    del sanity,optimizer
    torch.cuda.empty_cache()
    predictions={'scalar':head('scalar',train,infer,keys,None,a.out)}
    all_receipts=[]
    for seed in [17,29,43]:
        start=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
        net=MeshGraphNet().to(device);torch.cuda.reset_peak_memory_stats()
        initial=embed(net,static,edges,edge_geom,conditions,phases,weights)
        random_name=f'random_graph_seed{seed}'
        predictions[random_name]=head(random_name,train,infer,keys,initial,a.out)
        torch.save(net.state_dict(),a.out/(random_name+'.pt'))
        optimizer=torch.optim.Adam(net.parameters(),lr=.001)
        curve=[];pretrain_start=time.monotonic()
        for update in range(1,1801):
            net.train();s=rng.integers(0,24,size=8);p=rng.choice(train_phases,size=8)
            pred,_=net(static,edges,edge_geom,conditions[s],phases[p])
            loss=((pred-target[s,p]/scale).square().sum(-1)*weights).sum(-1).mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            check_bounds(a.out,start)
            if update==1 or update%50==0:
                rec=dict(update=update,normalized_B_training_MSE=float(loss),elapsed_seconds=time.monotonic()-pretrain_start)
                curve.append(rec);write(a.out/'progress.json',dict(seed=seed,**rec));print(json.dumps(dict(seed=seed,**rec)),flush=True)
        pretrain_seconds=time.monotonic()-pretrain_start
        monitor=field_monitor(net,static,edges,edge_geom,conditions,phases,target,weights,scale)
        before=time.monotonic();learned=embed(net,static,edges,edge_geom,conditions,phases,weights);embed_seconds=time.monotonic()-before
        name=f'MS_MGN_seed{seed}'
        torch.save(net.state_dict(),a.out/(name+'.pt'))
        np.savez_compressed(a.out/f'embeddings_seed{seed}.npz',source_keys=keys,random=initial,MS_pretrained=learned)
        predictions[name]=head(name,train,infer,keys,learned,a.out)
        receipt=dict(seed=seed,updates=1800,checkpoint_selection='fixed final',pretrain_seconds=pretrain_seconds,
            extract24_embeddings_seconds=embed_seconds,wall_seconds=time.monotonic()-start,monitor=monitor,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated(),RAM_bytes=psutil.Process().memory_info().rss,
            parameters=sum(p.numel() for p in net.parameters()),training_curve=curve)
        write(a.out/f'training_seed{seed}.json',receipt);all_receipts.append(receipt)
    write(a.out/'predictions_frozen.json',predictions)
    write(a.out/'TRAINING_COMPLETE.json',dict(status='complete',seeds=[17,29,43],Full_budget=63,MS_sources=24,
        fixed_final_updates=1800,training_receipts=[{k:v for k,v in r.items() if k!='training_curve'} for r in all_receipts],
        frozen_files={p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file() and p.name not in ['OWNER.json','progress.json']},
        protected_eval_truth_seen=False,automatic_resume=False))
    print('TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    main()
