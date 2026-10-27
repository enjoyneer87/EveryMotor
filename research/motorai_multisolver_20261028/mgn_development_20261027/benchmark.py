"""Frozen predictor timing only; excludes cached MS/Hybrid construction costs."""
from pathlib import Path
import argparse,ast,json,math,time,csv,hashlib,platform
import numpy as np
from residual import feature_matrix,predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root
    baseline=root.parent/'baseline_audit';text=(baseline/'replay_baseline.py').read_text()
    defs=[node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name in ['poly','predict']]
    assert len(defs)==2
    namespace={'math':math};exec(compile(ast.Module(body=defs,type_ignores=[]),'frozen_pure_AF_functions','exec'),namespace)
    modeldir=baseline/'sources/handoff/20261019_AF_loaded96'
    af={name:json.loads((modeldir/f'AF_model_{name}_loaded96.json').read_text()) for name in ['Ref','SC']}
    rows=json.loads((root/'artifacts/inputs/inference228.json').read_text())
    def eval_af():
        out=[]
        for r in rows:
            half=r['model']=='HalfSC'
            factor=namespace['predict'](af['SC' if half else r['model']],r['speed']*(.5625 if half else 1),r['current']*(4/3 if half else 1),r['beta'])
            out.append(r['hybrid']*factor)
        return np.array(out)
    sources=list(csv.DictReader((root.parent/'remote_return/desktop-6qivkic_20261027/comparison_ready.csv').open()))
    np.testing.assert_allclose(eval_af(),[float(r['frozen_AF_Hybrid_W']) for r in sources],rtol=1e-12,atol=1e-6)
    af_inputs=[]
    for name in ['Ref','SC','HalfSC']:
        index=np.array([i for i,r in enumerate(rows) if r['model']==name]);half=name=='HalfSC'
        m=af['SC' if half else name]['separable_model']
        n=np.array([rows[i]['speed'] for i in index])*(.5625 if half else 1)
        current=np.array([rows[i]['current'] for i in index])*(4/3 if half else 1)
        beta=np.array([rows[i]['beta'] for i in index]);hybrid=np.array([rows[i]['hybrid'] for i in index])
        af_inputs.append((index,m,n,current,beta,hybrid))
    def eval_af_batched():
        out=np.empty(len(rows))
        for index,m,n,current,beta,hybrid in af_inputs:
            d2=((current[:,None]-np.array(m['base_centers_i']))/m['ls_i'])**2+((beta[:,None]-np.array(m['base_centers_p']))/m['ls_p'])**2
            k=(d2*np.log(np.sqrt(d2)+1e-12))@np.array(m['base_weights'])
            out[index]=hybrid*np.polyval(m['speed_poly_coeffs'],n/1000)*np.maximum(k,1e-3)**np.polyval(m['spread_poly_coeffs'],n/1000)
        return out
    np.testing.assert_allclose(eval_af_batched(),eval_af(),rtol=1e-12,atol=1e-6)
    functions={'latest_AF_pure_frozen_evaluator':eval_af,'latest_AF_batched_NumPy':eval_af_batched}
    for name,stage in [('scalar','stage1')]+[(f'MS_MGN_seed{s}','stage2') for s in [17,29,43]]:
        path=root/'artifacts'/stage
        model=dict(np.load(path/(name+'_head.npz')))
        if name=='scalar':emb=None;keys=None
        else:
            z=np.load(path/f'embeddings_seed{name.split("seed")[-1]}.npz');emb=z['MS_pretrained'];keys=z['source_keys']
        x=feature_matrix(rows,emb,keys);hybrid=np.array([r['hybrid'] for r in rows])
        functions[name+'_cached_features']=lambda m=model,x=x:hybrid*np.exp(predict(m,x))
    times={}
    for name,fn in functions.items():
        for _ in range(5):fn()
        values=[]
        for _ in range(100):
            start=time.perf_counter();fn();values.append(time.perf_counter()-start)
        times[name]=dict(batch_rows=228,median_ms=float(np.median(values)*1000),p95_ms=float(np.percentile(values,95)*1000),repetitions=100)
    out=dict(scope='implementation-dependent CPU evaluator timing on precomputed inputs, not end-to-end solver replacement or equal total compute',
        python=platform.python_version(),numpy=np.__version__,AF_pure_function_source_sha256=hashlib.sha256(text.encode()).hexdigest(),timings=times,
        exclusions=['MS FEA','canonical Hybrid sample evaluation','GPU graph encoding','disk reads','feature assembly'],
        graph_encoding_cost='See stage2 training receipts: approximately1.9s GPU for24 sources x128 phases; cacheable only for those sources')
    a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
