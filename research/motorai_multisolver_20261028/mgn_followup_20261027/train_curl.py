"""Fixed A/curl followup, isolated from all Full evaluation truths."""
from pathlib import Path
import argparse,datetime,json,sys,time,os,math,hashlib
import numpy as np
import torch,psutil
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mgn_development_20261027'))
from model import MeshGraphNet,edge_features,mlp
from train import embed
from residual import feature_matrix,select_and_fit
DEADLINE=datetime.datetime.fromisoformat('2026-10-27T23:00:00+00:00').timestamp()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def check(out,start):
    if time.time()>=DEADLINE:raise RuntimeError('08:00 KST deadline reached')
    if (out/'STOP').exists():raise RuntimeError('owned STOP requested')
    if time.monotonic()-start>3600:raise RuntimeError('seed wall limit3600s')
    if psutil.Process().memory_info().rss>16*1024**3:raise RuntimeError('RAM limit16GiB')
    if torch.cuda.max_memory_allocated()>12*1024**3:raise RuntimeError('GPU tensor limit12GiB')

class CurlMGN(MeshGraphNet):
    def __init__(self):
        super().__init__();self.decoder=mlp(48,1,48,normalize=False)

def gauge(A,slot_weights,slot_indicator):
    return A-(A@slot_weights)@slot_indicator

def curl(A,tri,coef):
    return torch.einsum('btj,tjk->btk',A[:,tri],coef)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--sanity-only',action='store_true');a=ap.parse_args()
    assert time.time()+300<DEADLINE,'Insufficient cutoff margin'
    a.out.mkdir(exist_ok=False);torch.set_num_threads(4);assert torch.cuda.is_available();dev='cuda'
    write(a.out/'OWNER.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),deadline_UTC='2026-10-27T23:00:00Z',stop_file='STOP',automatic_retry=False))
    d=np.load(a.data/'MS_curl_data.npz');train=json.loads((a.data/'calibration63.json').read_text());assert len(train)==63 and set(r['model'] for r in train)=={'Ref','SC'}
    split=json.loads((a.data/'SPLIT_MANIFEST.json').read_text());assert set(split['calibration_case_ids'])==set(r['case_id'] for r in train)
    tensor=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device=dev)
    static=tensor(d['node_static']);edges=tensor(d['edge_index'],torch.long);geom=edge_features(static,edges);tri=tensor(d['connectivity_local'],torch.long);coef=tensor(d['curl_coefficients_per_m'])
    cond=tensor(d['source_features']);phase=tensor(d['phases_rad']);A=tensor(d['A_gauged_Wb_per_m']);B=tensor(d['B_element_T'])
    nw=tensor(d['node_area_weights']);ew=tensor(d['element_area_m2']/d['element_area_m2'].sum())
    slots=d['node_slot'];indicator=tensor(np.array([slots==s for s in np.unique(slots)]));sw=indicator.T*nw[:,None];sw=sw/sw.sum(0)
    trainphase=np.array([p for p in range(128) if p%8!=7]);monitorphase=np.arange(7,128,8)
    ascale=((A[:,trainphase].square()*nw).sum(-1).mean()).sqrt();bscale=((B[:,trainphase].square().sum(-1)*ew).sum(-1).mean()/2).sqrt()
    def forward(net,s,p):
        raw,latent=net(static,edges,geom,cond[s],phase[p]);an=gauge(raw[...,0],sw,indicator);bp=curl(an*ascale,tri,coef)
        return an,bp,latent
    def objective(net,s,p):
        an,bp,_=forward(net,s,p)
        lb=(((bp-B[s,p])/bscale).square().sum(-1)*ew).sum(-1).mean()
        la=((an-A[s,p]/ascale).square()*nw).sum(-1).mean()
        return lb+.1*la,lb,la
    write(a.out/'INPUT_FREEZE.json',dict(inputs={p.name:sha(p) for p in a.data.iterdir() if p.is_file()},code={p.name:sha(p) for p in ROOT.glob('*.py')},
        inherited_code={p.name:sha(p) for p in (ROOT.parent/'mgn_development_20261027').glob('*.py')},protocol_sha256=sha(ROOT/'PROTOCOL.json'),
        torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),A_scale_Wb_per_m=float(ascale),B_scale_T=float(bscale),Full_budget=63,Full_eval_truth_loaded=False))
    # Analytic linear potential and orientation reversal check the discrete operator.
    analytic=tensor(2*d['node_xy_m'][:,0]+3*d['node_xy_m'][:,1]+.01)[None]
    got=curl(analytic,tri,coef);analytic_error=float((got-tensor([3.,-2.])).abs().max())
    assert analytic_error<.001
    raw_replay=curl(A[0],tri,coef);reader_error=float((raw_replay-B[0]).abs().max());assert reader_error<1e-4
    torch.manual_seed(1001);torch.cuda.manual_seed_all(1001);net=CurlMGN().to(dev);opt=torch.optim.Adam(net.parameters(),lr=.001);losses=[];start=time.monotonic()
    for update in range(20):
        loss,_,_=objective(net,np.arange(8),trainphase[:8]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(float(loss));check(a.out,start)
    sanity=dict(status='PASS' if losses[-1]<losses[0] else 'FAIL',first_loss=losses[0],last_loss=losses[-1],updates=20,discarded=True,analytic_curl_max_abs_T=analytic_error,float32_raw_replay_max_abs_T=reader_error,seconds=time.monotonic()-start)
    write(a.out/'SANITY.json',sanity);assert sanity['status']=='PASS';del net,opt;torch.cuda.empty_cache()
    if a.sanity_only:print(json.dumps(sanity));return
    receipts=[];total_start=time.monotonic()
    for seed in [17,29,43]:
        assert time.time()+3600<DEADLINE,'Do not start a seed without its full deadline allowance'
        start=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);net=CurlMGN().to(dev);opt=torch.optim.Adam(net.parameters(),lr=.001);torch.cuda.reset_peak_memory_stats();curve=[]
        try:
            for update in range(1,18001):
                if update==1801:
                    opt=torch.optim.Adam(net.parameters(),lr=.0003);rng=np.random.default_rng(seed+1800)
                lr=.001 if update<=1800 else .00001+(.0003-.00001)*.5*(1+math.cos(math.pi*(update-1801)/16200))
                for group in opt.param_groups:group['lr']=lr
                s=rng.integers(0,24,8);p=rng.choice(trainphase,8);loss,lb,la=objective(net,s,p)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();check(a.out,start)
                if time.monotonic()-total_start>10800:raise RuntimeError('aggregate10800s limit')
                if update==1 or update%500==0 or update==18000:
                    row=dict(seed=seed,update=update,loss=float(loss),B_loss=float(lb),A_loss=float(la),lr=lr,seconds=time.monotonic()-start);curve.append(row);write(a.out/'progress.json',row);print(json.dumps(row),flush=True)
            training_seconds=time.monotonic()-start;net.eval();err=norm=0.;nframes=0
            with torch.no_grad():
                for s in range(24):
                    for p in [monitorphase[:8],monitorphase[8:]]:
                        _,bp,_=forward(net,np.repeat(s,len(p)),p);err+=float(((bp-B[s,p]).square().sum(-1)*ew).sum());norm+=float((B[s,p].square().sum(-1)*ew).sum());nframes+=len(p)
            latent=embed(net,static,edges,geom,cond,phase,nw);head,scores=select_and_fit(feature_matrix(train,latent,d['source_keys']),train)
            torch.save(net.state_dict(),a.out/f'A_curl_seed{seed}.pt');np.savez_compressed(a.out/f'embeddings_seed{seed}.npz',source_keys=d['source_keys'],A_curl=latent)
            np.savez_compressed(a.out/f'A_curl_seed{seed}_head.npz',**head)
            receipt=dict(seed=seed,updates=18000,training_seconds=training_seconds,wall_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in net.parameters()),
                element_B_held_phase_relative_L2_percent=100*(err/norm)**.5,element_B_held_phase_RMSE_T=(err/nframes)**.5,monitor_frames=nframes,
                calibration_group_CV=scores,selected_head=dict(multiplier=head['multiplier'],regularization=head['regularization']),
                Full_eval_performed=False,peak_VRAM_bytes=torch.cuda.max_memory_allocated(),RAM_bytes=psutil.Process().memory_info().rss,training_curve=curve)
            write(a.out/f'training_seed{seed}.json',receipt);receipts.append({k:v for k,v in receipt.items() if k!='training_curve'})
        except Exception as exc:
            torch.save(net.state_dict(),a.out/f'A_curl_seed{seed}_PARTIAL.pt');write(a.out/'INTERRUPTED.json',dict(seed=seed,update=update,reason=str(exc),not_completed=True));raise
    write(a.out/'TRAINING_COMPLETE.json',dict(status='COMPLETE',receipts=receipts,Full_budget=63,Full_eval_performed=False,protected_HalfSC_training=False,
        frozen_files={p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file() and p.name not in ['OWNER.json','progress.json']},automatic_resume=False))
    print('CURL_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
