"""NumPy-only replay of all released frozen heads, no fitting or evaluation labels."""
from pathlib import Path
import argparse,json
import numpy as np
from residual import feature_matrix,predict


def verify(root,rows):
    reports=[]
    for stage in ['stage1','stage2']:
        folder=root/stage;predictions=json.loads((folder/'predictions_frozen.json').read_text())
        for name,expected in predictions.items():
            if name=='scalar':emb=None;keys=None
            else:
                seed=int(name.split('seed')[-1]);z=np.load(folder/f'embeddings_seed{seed}.npz',allow_pickle=False)
                keys=z['source_keys'];emb=z['random' if name.startswith('random') else 'MS_pretrained']
            with np.load(folder/(name+'_head.npz'),allow_pickle=False) as z:model=dict(z)
            values=np.array([r['hybrid'] for r in rows])*np.exp(predict(model,feature_matrix(rows,emb,keys)))
            saved=np.array([expected[r['case_id']] for r in rows]);error=abs(values-saved)
            np.testing.assert_allclose(values,saved,rtol=1e-10,atol=1e-8)
            reports.append(dict(stage=stage,method=name,n=len(rows),max_abs_error_W=float(error.max()),mean_abs_error_W=float(error.mean()),
                tolerance=dict(rtol=1e-10,atol_W=1e-8),status='PASS'))
    return dict(status='PASS',models=len(reports),checks=reports,scope='frozen head arithmetic replay, not accuracy or convergence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--inference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=verify(a.models,json.loads(a.inference.read_text()));a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
