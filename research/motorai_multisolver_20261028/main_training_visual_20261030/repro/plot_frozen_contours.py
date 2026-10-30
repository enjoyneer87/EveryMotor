"""Render only hash-collected real inference arrays on exact native triangles."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime_lib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def clip_record(value,area,limits):
    lo,hi=value<limits[0],value>limits[1]
    return dict(below_elements=int(lo.sum()),above_elements=int(hi.sum()),
                count_scope='full retained native support; crop changes viewport only',
                below_area_fraction=float(area[lo].sum()/area.sum()),
                above_area_fraction=float(area[hi].sum()/area.sum()),
                minimum_T=float(value.min()),maximum_T=float(value.max()))


def paint(ax,vertices,value,limits,cmap,title,bounds,area):
    c=clip_record(value,area,limits)
    artist=PolyCollection(vertices,array=value,cmap=cmap,norm=Normalize(*limits),
                          edgecolors='none',antialiaseds=False)
    ax.set_facecolor('#d9dde2');ax.add_collection(artist)
    ax.set(xlim=bounds[:2],ylim=bounds[2:],aspect='equal',xlabel='x [mm]',ylabel='y [mm]')
    ax.set_title(title+f"\n전체 요소 범위 밖 ↓{c['below_elements']} / ↑{c['above_elements']}",fontsize=8)
    ax.tick_params(labelsize=7)
    return artist,c


def a1_crop(vertices,mask):
    points=vertices[mask].reshape(-1,2)
    theta=np.arctan2(points[:,1],points[:,0]);radius=np.linalg.norm(points,axis=1)
    assert theta.max()-theta.min()<np.pi, 'no inferred angle wrapping allowed'
    middle=(theta.min()+theta.max())/2
    low=max(theta.min(),middle-np.deg2rad(5));high=min(theta.max(),middle+np.deg2rad(5))
    corners=np.asarray([(r*np.cos(t),r*np.sin(t)) for r in (radius.min(),radius.max()) for t in (low,high)])
    return (float(corners[:,0].min()-.5),float(corners[:,0].max()+.5),
            float(corners[:,1].min()-.5),float(corners[:,1].max()+.5))


def main():
    contract=json.loads((ROOT/'CONTOUR_RENDER_EXECUTION_CONTRACT.json').read_text())
    assert sha(Path(__file__).resolve())==contract['renderer_sha256']
    for pin in contract['input_pins']:
        assert sha(Path(pin['path']))==pin['sha256']
    plan=json.loads((ROOT/'CONTOUR_RENDER_PLAN.json').read_text())
    assert plan['status']=='FROZEN_SHARED_RENDER_LIMITS_BEFORE_FIELD_READS'
    binding=json.loads((ROOT/'CONTOUR_GEOMETRY_BINDING_RESULT.json').read_text())
    assert binding['status']=='PASS_ALL12_STORED_NATIVE_GEOMETRY_BINDINGS'
    manifest=json.loads((ROOT/'contour_inference/INFERENCE_MANIFEST.json').read_text())
    assert manifest['status']=='COMPLETED_ALL36_FIXED_CONTOUR_INFERENCES' and len(manifest['files'])==36
    font=Path('C:/Windows/Fonts/malgun.ttf');font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus':False,'font.size':9})
    assets=ROOT.parents[1]/'outputs/MOTORAI_MAIN_TRAINING_VISUAL_20261030_assets'
    output=dict(status='RUNNING',created_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                render_plan_sha256=sha(ROOT/'CONTOUR_RENDER_PLAN.json'),
                qualification='STORED_NATIVE_HDF_REPLAY_ONLY',source_vector_basis='UNKNOWN',
                interpolation=False,symmetry_replication=False,plots=[],clipping=[])
    cache={}
    for case in plan['cases']:
        for step in plan['steps']:
            support=next(x for x in binding['rows'] if (x['case_index'],x['step'])==(case,step))
            gp=Path(support['output']['path']);assert sha(gp)==support['output']['sha256']
            g=read_npz(gp);values={}
            for label in ('W64_w1','W64_w3','W128_w1'):
                row=next(x for x in manifest['files'] if (x['model'],x['case_index'],x['step'])==(label,case,step))
                path=ROOT/'contour_inference'/Path(row['path']).name;assert sha(path)==row['sha256']
                a=read_npz(path)
                assert np.array_equal(a['native_element_id'],g['native_element_id'])
                assert int(a['case_index'])==case and int(a['step'])==step
                assert row['graph_sha256']==support['graph_sha256']
                if 'reference' in values:assert np.array_equal(values['reference'],a['reference_B_T'])
                values['reference']=a['reference_B_T'].astype(np.float64)
                values[label]=a['pred_B_T'].astype(np.float64)
                assert values[label].shape==values['reference'].shape==(len(g['native_element_id']),2)
                assert np.isfinite(values[label]).all() and np.isfinite(values['reference']).all()
            cache[case,step]=(g,values)
    selection=json.loads((ROOT/'CONTOUR_SELECTION.json').read_text())
    for case in plan['cases']:
        item=next(x for x in selection['conditions'] if x['case_index']==case)
        allpoints=np.concatenate([cache[case,s][0]['vertex_xy_mm'].reshape(-1,2) for s in plan['steps']])
        whole=(allpoints[:,0].min()-.5,allpoints[:,0].max()+.5,allpoints[:,1].min()-.5,allpoints[:,1].max()+.5)
        for group,second,modes in (('width','W128_w1',('Bx','By','norm')),('historical_w3','W64_w3',('norm',))):
            for mode in modes:
                for zoom in (False,True) if mode=='norm' else (False,):
                    steps=[22] if zoom else plan['steps']
                    fig,axes=plt.subplots(len(steps),5,figsize=(16,3.3 if zoom else 11),squeeze=False,layout='constrained')
                    for r,step in enumerate(steps):
                        g,values=cache[case,step];ref=values['reference'];base=values['W64_w1'];candidate=values[second]
                        if mode=='norm':
                            fields=[np.linalg.norm(x,axis=1) for x in (ref,base,candidate)]
                            errors=[np.linalg.norm(x-ref,axis=1) for x in (base,candidate)]
                            fieldlimits=plan['stored_B_component_norm_T'];errorlimits=plan['stored_vector_error_norm_T']
                            title='저장 B성분의 norm';errtitle='||ΔB||';cmap='viridis'
                        else:
                            k={'Bx':0,'By':1}[mode];fields=[x[:,k] for x in (ref,base,candidate)]
                            errors=[np.abs(x[:,k]-ref[:,k]) for x in (base,candidate)]
                            fieldlimits=plan['Stored_'+mode+'_T'];errorlimits=plan['absolute_component_error_T']
                            title='Stored '+mode;errtitle='|Δ'+mode+'|';cmap='RdBu_r'
                        bounds=a1_crop(g['vertex_xy_mm'],g['a1_mask']) if zoom else whole
                        names=['저장 FEM 기준','W64 / w1',second.replace('_',' / '),'W64 '+errtitle,second+' '+errtitle]
                        for col,(value,name) in enumerate(zip(fields+errors,names)):
                            limits=fieldlimits if col<3 else errorlimits
                            art,clipping=paint(axes[r,col],g['vertex_xy_mm'],value,limits,cmap if col<3 else 'inferno',
                                               f'step {step} · '+name,bounds,g['weights'])
                            if r==0:
                                fig.colorbar(art,ax=axes[:,col].tolist(),shrink=.8,label='T',extend='both' if col<3 and mode!='norm' else 'max')
                            output['clipping'].append(dict(case_index=case,step=step,group=group,mode=mode,zoom=zoom,panel=name,**clipping))
                    suffix='_a1crop' if zoom else ''
                    filename=f'contour_{group}_case{case:04d}_{mode}{suffix}.png'
                    p=assets/filename;assert not p.exists()
                    fig.suptitle(f"case {case} · PA {item['native_input']['PhaseAdvance']:.3f}° · {title} · "+('a1 geometry crop' if zoom else '전체 native support')+
                                 '\nseed17 final88565 · stored-native HDF replay · basis UNKNOWN · 회색=제외/미지원 공간 (a2 polygon 아님)',fontsize=11)
                    fig.savefig(p,dpi=145,facecolor='white');plt.close(fig)
                    output['plots'].append(dict(case_index=case,group=group,mode=mode,a1_crop=zoom,path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    assert len(output['plots'])==18
    output['status']='COMPLETED_REAL_STORED_NATIVE_CONTOUR_RENDERING'
    with (ROOT/'CONTOUR_RENDER_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(output,f,indent=2)
    print(json.dumps({'status':output['status'],'plots':len(output['plots'])}))


if __name__=='__main__':
    main()
