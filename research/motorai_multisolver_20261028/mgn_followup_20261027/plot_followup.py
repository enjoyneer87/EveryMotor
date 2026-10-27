"""MS representation and same-support field diagnostics; not Full AC results."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);r=p.parse_args().root
    d=json.loads((r/'REPRESENTATION_AUDIT.json').read_text());v=json.loads((r/'FOLLOWUP_VALIDATION.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    vals=sorted(x['projected_node_B_roundtrip_relative_L2_percent'] for x in d['rows']);axes[0].plot(np.arange(1,25),vals,'o-',color='#cc7f23',markersize=4,label='Element B -> node -> element')
    axes[0].axhline(0,color='#248776',label='Original nodal A -> P1 curl')
    axes[0].set(xlabel='MS source rank (24 sources)',ylabel='Area-weighted vector-B relative L2 (%)',title='Representation before learned predictions',ylim=(-.7,18.5));axes[0].legend(frameon=False,fontsize=8)
    methods=['primary_node_B','followup_A_curl'];colors=['#234b79','#248776']
    for i,method in enumerate(methods):
        a=np.array([x['element_B_held_phase_relative_L2_percent'] for x in v['same_support_comparison'] if x['method']==method])
        axes[1].bar(i,a.mean(),yerr=a.std(),color=colors[i],alpha=.8,capsize=5);axes[1].scatter(i+np.array([-.08,0,.08]),a,color='black',s=15,zorder=4);axes[1].text(i,a.mean()+a.std()+.5,f'{a.mean():.2f}%',ha='center')
    axes[1].set(xticks=[0,1],xticklabels=['Primary node B','Followup A / curl'],ylabel='Original-element B relative L2 (%)',title='Same source / held phase, three fixed seeds',ylim=(0,19))
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Elmer stationary MS slots: field diagnostics, not Full-FEA AC-loss accuracy')
    (r/'figures').mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(r/f'figures/MS_curl_followup.{ext}',dpi=180)
    svg=r/'figures/MS_curl_followup.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')

if __name__=='__main__':main()
