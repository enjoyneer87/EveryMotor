"""Numerical comparison plots from completed full-history readback only."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def save(fig,root,name):
    fig.savefig(root/'figures'/(name+'.png'),dpi=200);fig.savefig(root/'figures'/(name+'.svg'));plt.close(fig)
    p=root/'figures'/(name+'.svg');p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--scope-only',action='store_true');a=ap.parse_args();r=a.root
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    graphs=[np.load(r/'artifacts/graph_topology.npz'),np.load(r/'convergence/N256_P3/copper_graph_8phases.npz')]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,d,title,color in zip(axes,graphs,['MS graph: pretraining input topology','Full Cu graph: validation only'],['#248776','#234b79']):
        xy=d['node_xy_m']*1000;edges=d['edge_index'].T
        ax.add_collection(LineCollection(xy[edges],colors=color,linewidths=.25,alpha=.65));ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='x (mm)',ylabel='y (mm)',title=title)
    fig.suptitle('Original FE adjacency in stationary domains; no kNN or coordinate matching')
    save(fig,r,'graph_scope')
    if a.scope_only:return
    result=json.loads((r/'CONVERGENCE_RESULTS.json').read_text());assert result['status']=='COMPLETE'
    cases=['N256_P3','N512_P3','fine_N256_P3'];waves={};audits={}
    for case in cases:
        folder=r/'convergence'/case;waves[case]=np.load(folder/'waveforms.npz')['scalar'];audits[case]=json.loads((folder/'audit.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained');coarse=waves['N256_P3'];means=[p['AC_W']/1000 for p in audits['N256_P3']['periods']]
    bars=axes[0,0].bar(['P1 startup','P2 adopted','P3 check'],means,color=['#9ca6b0','#234b79','#248776']);axes[0,0].bar_label(bars,fmt='%.3f',padding=3)
    axes[0,0].set(ylabel='Mean active AC excess (kW)',ylim=(0,max(means)*1.14),title='Averaging window matters')
    colors=['#234b79','#ea9c34','#248776']
    for case,color in zip(cases,colors):
        spp=audits[case]['steps_per_period'];y=waves[case][2*spp:3*spp,3];phase=np.arange(1,spp+1)*360/spp
        axes[0,1].plot(phase,y/1000,label=case,color=color,lw=1.2)
        if case!='N256_P3':
            aligned=y[1::2] if spp==512 else y
            axes[1,0].plot(np.arange(1,257)*360/256,100*(aligned-coarse[512:768,3])/y.mean(),label=case,color=color,lw=1)
    axes[0,1].set(xlabel='Electrical phase in P3 (deg)',ylabel='Active AC excess (kW)',title='Full-time P3 waveforms');axes[0,1].legend(frameon=False,fontsize=8)
    axes[1,0].set(xlabel='Aligned electrical phase (deg)',ylabel='Difference / refined P3 mean (%)',title='Refined minus coarse waveform');axes[1,0].legend(frameon=False,fontsize=8)
    comp=result['comparisons'];values=[audits['N256_P3']['P2_P3_mean_percent'],comp['time_step']['P3_mean_difference_percent'],comp['mesh']['P3_mean_difference_percent']]
    bars=axes[1,1].bar(['P2 vs P3','N256 vs N512','Coarse vs fine'],np.maximum(values,1e-12),color=colors)
    axes[1,1].set(yscale='log',ylabel='Absolute mean difference (%)',title='Measured sensitivity, not formal qualification')
    for bar,value in zip(bars,values):axes[1,1].annotate(f'{value:.4g}%',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    for ax in axes.ravel():ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Ref_P117: 16,000 rpm, 460 Arms, beta 36 deg; unchanged material/circuit inputs')
    save(fig,r,'convergence')
    print('scope and convergence figures written')


if __name__=='__main__':main()
