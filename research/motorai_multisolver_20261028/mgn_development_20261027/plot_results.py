"""Static scientific figures, from frozen evaluation and training receipts only."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    summary=json.loads((r/'evaluation/metrics.json').read_text())['summary']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#234b79','#ea9c34','#8d98a7','#248776']
    groups=['Ref_excluded60','SC_strict57','HalfSC_predeclared32']
    names=['latest_frozen_AF','scalar','random_graph_control','MS_MGN_primary']
    labels=['Latest AF','Scalar residual','Random graph','MS-pretrained MGN']
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    x=np.arange(3);width=.2
    for j,(method,label,color) in enumerate(zip(names,labels,colors)):
        values=[next(s for s in summary if s['group']==g and s['method']==method) for g in groups]
        means=[v['wMAE_mean_percent'] for v in values];sd=[v['wMAE_seed_sd_percent'] for v in values]
        bars=ax.bar(x+(j-1.5)*width,means,width,label=label,color=color,yerr=sd,capsize=3)
        ax.bar_label(bars,fmt='%.2f',padding=5,fontsize=9)
    ax.set_xticks(x,['Ref: excluded 60','SC: strict excluded 57','HalfSC: predeclared 32'])
    ax.set_ylabel('AC-excess loss wMAE (%)');ax.set_ylim(0,10.2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.legend(ncol=2,loc='upper right',frameon=False)
    ax.set_title('Same 63 unique Full-FEA calibration labels; known-template LF-assisted comparison',pad=14)
    (r/'figures').mkdir(exist_ok=True);fig.savefig(r/'figures/equal_budget.png',dpi=200);fig.savefig(r/'figures/equal_budget.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for seed,color in zip([17,29,43],['#234b79','#ea9c34','#248776']):
        s1=json.loads((r/f'artifacts/stage1/training_seed{seed}.json').read_text());s2=json.loads((r/f'artifacts/stage2/training_seed{seed}.json').read_text())
        x=[z['update'] for z in s1['training_curve']]+[z['update_total'] for z in s2['training_curve']]
        y=[z['normalized_B_training_MSE'] for z in s1['training_curve']+s2['training_curve']]
        axes[0].plot(x,y,label=f'seed {seed}',color=color,lw=1.1)
        axes[1].plot([1800,18000],[s1['monitor']['held_phase_vector_B_relative_L2_percent'],s2['monitor']['held_phase_vector_B_relative_L2_percent']],marker='o',color=color,label=f'seed {seed}')
    axes[0].set(xlabel='MS optimizer updates',ylabel='Area-weighted normalized vector-B MSE',yscale='log')
    axes[1].set(xlabel='Fixed final checkpoint updates',ylabel='Held-phase MS vector-B relative L2 (%)',xticks=[1800,18000],ylim=(0,48))
    for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle('Auxiliary MS field learning; these are not Full-FEA AC loss errors')
    fig.savefig(r/'figures/MS_pretraining.png',dpi=200);fig.savefig(r/'figures/MS_pretraining.svg');plt.close(fig)
    for svg in (r/'figures').glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    print('4 plot files written')


if __name__=='__main__':main()
