"""Audit saved HalfSC outputs and plot direct frozen SC-AF validation. Never runs FEA or fits AF."""
from pathlib import Path
import argparse,json,hashlib,math,csv
import numpy as np
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def metric(rows,key):
    truth=np.array([r['P2_AC_excess_W'] for r in rows])
    pred=np.array([r['predictions'][key]['prediction_W'] for r in rows])
    e=100*(pred/truth-1)
    return dict(count=len(rows),wMAE_percent=float(100*np.abs(pred-truth).sum()/truth.sum()),MARE_percent=float(np.abs(e).mean()),max_absolute_error_percent=float(np.abs(e).max()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    a=p.parse_args();r=a.repo;o=a.output;o.mkdir(parents=True,exist_ok=True)
    assert not (o/'FINAL_AUDIT.json').exists()
    h=r/'handoff/20261020_halfsc_direct_af';comp=read(h/'COMPARISON.json');plan=read(h/'PLAN.json')
    assert comp['complete'] and comp['observed']==36 and comp['new_observed']==32
    for rel,d in plan['pinned_sources'].items():assert sha(r/rel)==d,rel
    byid={p['case_id']:p for p in plan['points']};rows=comp['rows']
    assert len({x['case_id'] for x in rows})==36
    for row in rows:assert row['predictions']==byid[row['case_id']]['predictions']
    new=[x for x in rows if x['role']=='new_validation'];old=[x for x in rows if x['role']=='existing_exploratory']
    assert len(new)==32 and len(old)==4
    B=a.metadata
    D=Path('D:/KDH/JEET_HALFSC_DIRECT_AF_20261020')
    for item in read(B/'SNAPSHOT_MANIFEST.json'):assert sha(B/item['path'])==item['sha256']
    receipts=[];relax={}
    for row in new:
        cid=row['case_id'];w=B/cid;dest=D/cid
        done=read(w/'DONE.json');receipt=read(dest/'PRESERVATION_RECEIPT.json')
        assert done['all_files_SHA_readback_verified'] and receipt['all_files_SHA_readback_verified']
        assert done['manifest_sha256']==sha(dest/'PRESERVATION_MANIFEST.json')==receipt['manifest_sha256']
        assert done['result_sha256']==sha(w/'RESULT_SUMMARY.json')==sha(dest/'RESULT_SUMMARY.json')
        assert read(w/'RESULT_SUMMARY.json')['P2_AC_excess_W']==row['P2_AC_excess_W']
        result=read(dest/'RESULT.json');assert result['nonlinear_passed'] and len(result['nonlinear_steps'])==512
        assert all(x['passed'] for x in result['nonlinear_steps'])
        relax[cid]=result['effective']['relaxation']
        manifest=read(dest/'PRESERVATION_MANIFEST.json')
        receipts.append(dict(case_id=cid,receipt_sha256=sha(dest/'PRESERVATION_RECEIPT.json'),manifest_sha256=done['manifest_sha256'],summary_sha256=done['result_sha256'],files=len(manifest['files']),bytes=sum(x['bytes'] for x in manifest['files']),all_files_SHA_readback_verified_at_preservation=True,metadata_rechecked_now=True))
    assert list(relax.values()).count(.7)==31 and list(relax.values()).count(.5)==1
    groups={'new32':new,'new_in_domain24':[x for x in new if not x['primary_AF_speed_extrapolated']],'new_extrapolation8':[x for x in new if x['primary_AF_speed_extrapolated']],'existing4':old}
    for speed in [2000,4000,8000,16000]:groups['speed_'+str(speed)]=[x for x in new if x['speed_rpm']==speed]
    methods=['uncorrected','normalized_current_same_speed','normalized_current_similarity_speed']
    metrics={g:{key:metric(vals,key) for key in methods} for g,vals in groups.items()}
    for key in methods:
        assert abs(metrics['new32'][key]['wMAE_percent']-comp['metrics']['new32_all_observed'][key]['wMAE_percent'])<1e-10
    worst=max(new,key=lambda x:abs(x['errors_percent'][methods[2]]))
    audit=dict(status='PASS',new_points=32,existing_exploratory_points=4,total_points=36,AF_refitted=False,new_target_labels_in_fit=0,
        predictions_predeclared_and_unchanged=True,metrics=metrics,primary=methods[2],mapping='AF_SC(0.5625*n_H,4/3*I_H,beta)',
        max_error_case={k:worst[k] for k in ['case_id','speed_rpm','current_Arms','beta_deg','errors_percent']},
        P2_current_failures=sum(not x['P2_current_pass'] for x in rows),whole_history_flags_retained=sum(not x['original_whole_history_current_pass'] for x in rows),
        period_convergence_qualified=False,mesh_convergence_qualified=False,scope='Second-period saved-field comparison at predeclared coordinates, not full-map or continuous-speed validation.',
        relaxation_per_case=relax,receipts=receipts,pinned_source_sha256={n:sha(h/n) for n in ['PLAN.json','COMPARISON.json','REF_ROUTE_COMPARISON.json','RELAXATION05_EXCEPTION.json']})
    write(o/'FINAL_AUDIT.json',audit);write(o/'HALFSC_POINTS_36.json',rows)
    with (o/'HALFSC_POINTS_36.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['case_id','role','speed_rpm','current_Arms','beta_deg','P2_AC_excess_W','uncorrected_W','direct_SC_AF_W','signed_error_percent','AF_speed_extrapolated','relaxation','P2_current_pass','whole_history_current_pass']
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for x in rows:wr.writerow(dict(case_id=x['case_id'],role=x['role'],speed_rpm=x['speed_rpm'],current_Arms=x['current_Arms'],beta_deg=x['beta_deg'],P2_AC_excess_W=x['P2_AC_excess_W'],uncorrected_W=x['predictions']['uncorrected']['prediction_W'],direct_SC_AF_W=x['predictions'][methods[2]]['prediction_W'],signed_error_percent=x['errors_percent'][methods[2]],AF_speed_extrapolated=x['primary_AF_speed_extrapolated'],relaxation=relax.get(x['case_id'],.7),P2_current_pass=x['P2_current_pass'],whole_history_current_pass=x['original_whole_history_current_pass']))
    plot_rows(r,o,rows,metrics)
    print(json.dumps({k:audit[k] for k in ['status','new_points','total_points','P2_current_failures','whole_history_flags_retained']}))
def plot_rows(r,o,rows,metrics):
    new=[x for x in rows if x['role']=='new_validation'];old=[x for x in rows if x['role']=='existing_exploratory']
    methods=['uncorrected','normalized_current_same_speed','normalized_current_similarity_speed']
    import sys
    sys.path.insert(0,str(r/'scripts'))
    from jeet_review.style import apply_style
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    apply_style('dq_sc')
    fig,ax=plt.subplots(1,3,figsize=(7.25,3.25),gridspec_kw={'width_ratios':[1.0,1.35,1.35]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.27,top=.82,wspace=.45)
    coords=sorted({(x['current_Arms']/690,x['beta_deg']) for x in new})
    ax[0].scatter([x[1] for x in coords],[x[0] for x in coords],s=22,c='#2166ac',zorder=3)
    ax[0].scatter([36],[1],s=60,marker='*',facecolors='none',edgecolors='#333333',zorder=4)
    ax[0].set(xlim=(-7,97),ylim=(.15,1.09),xticks=[0,36,54,90],yticks=[.25,.5,.75,1],xlabel=r'$\beta$ [deg]',ylabel=r'$I_{\mathrm{rms}}/690$')
    ax[0].set_title('Same coordinates at all four speeds',fontsize=7.4,pad=10)
    for i,key in enumerate([methods[0],methods[2]],1):
        z=ax[i]
        z.axvspan(.65,1.35,color='#fff0d9',zorder=0)
        for j,sp in enumerate([2000,4000,8000,16000],1):
            group=sorted([x for x in new if x['speed_rpm']==sp],key=lambda x:(x['current_Arms'],x['beta_deg']))
            z.scatter(j+np.linspace(-.18,.18,8),[x['errors_percent'][key] for x in group],s=16,c='#2166ac',zorder=3)
            oo=[x for x in old if x['speed_rpm']==sp][0]
            z.scatter([j],[oo['errors_percent'][key]],s=48,marker='*',facecolors='none',edgecolors='#333333',linewidths=.8,zorder=4)
            value=metrics['speed_'+str(sp)][key]['wMAE_percent']
            z.text(j,.98,f'{value:.2f}%',transform=z.get_xaxis_transform(),ha='center',va='top',fontsize=6.7)
        z.axhline(0,color='#555555',lw=.65)
        z.set(xlim=(.6,4.4),xticks=[1,2,3,4],xticklabels=['2','4','8','16'],xlabel='HalfSC speed [kRPM]',ylabel='Signed loss error [%]')
        z.set_ylim((-90,12) if i==1 else (-16,19))
        z.set_title('Uncorrected Hybrid' if i==1 else 'Direct frozen SC AF',fontsize=8.0,pad=10)
    for i,z in enumerate(ax):
        z.grid(alpha=.2,lw=.4);z.tick_params(labelsize=7,direction='in',length=2)
        z.text(.5,-.30,'('+chr(97+i)+')',transform=z.transAxes,ha='center',fontsize=8)
    fig.text(.51,.96,'HalfSC: 32 predeclared points + 4 prior checks; no target AF fitting',ha='center',fontsize=8.2)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2166ac',ms=4,label='Predeclared grid (8 per speed)'),
                        Line2D([],[],marker='*',ls='',mfc='none',mec='#333333',ms=7,label='Prior rated-current check (1 per speed)')],
               loc='lower center',bbox_to_anchor=(.52,.035),ncol=2,frameon=False,fontsize=7)
    fig.text(.52,.005,'Badges: wMAE of evaluation grid only. Shaded 2 kRPM maps to SC 1.125 kRPM (extrapolation).',ha='center',fontsize=6.8)
    for ext in ['pdf','png','svg']:fig.savefig(o/('FigureB2_HalfSC_Direct_AF.'+ext),bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

if __name__=='__main__':main()
