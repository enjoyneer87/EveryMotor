"""Plot saved final three-seed width comparisons; no model or source-array reads."""
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
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract=json.loads((ROOT/'FINAL_DIAGNOSTIC_PLOT_CONTRACT.json').read_text(encoding='utf-8-sig'))
    assert contract['plotter_sha256']==sha(Path(__file__).resolve())
    source=ROOT/'FINAL_WIDTH_DIAGNOSTICS.json';assert sha(source)==contract['summary_sha256']
    d=json.loads(source.read_text(encoding='utf-8-sig'));assert d['status']=='COMPLETE_SAVED_FINAL_WIDTH_DIAGNOSTICS'
    font=Path('C:/Windows/Fonts/malgun.ttf');font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    assets=ROOT.parents[1]/'outputs/MOTORAI_MAIN_TRAINING_VISUAL_20261030_assets'
    seeds=('17','29','43');arms=(('control','W64 / w1','#246a9a'),('candidate','W128 / w1','#d57c24'))
    labels=(('전체 Bx/By','primary_all45_Bvector'),('공극 a1','a1_Bvector'),('구리','copper_Bvector'),
            ('과거9위상 subset','seen9_Bvector'),('PA≥70 case 평균','PA_ge70_case_mean_Bvector'),('같은a1 Arkkio 토크','pooled_torque'))
    files=[]
    provenance='\n내부 dev32 (oldTRAIN 노출 이력) · 독립 test 아님'

    def save(fig,name):
        p=assets/name;assert not p.exists();fig.savefig(p,dpi=145,facecolor='white');plt.close(fig);files.append(p)

    fig,axes=plt.subplots(2,3,figsize=(13,7.8))
    for ax,(title,key) in zip(axes.flat,labels):
        m=d['registered_judgment_replay']['metrics'][key]
        for j,(role,label,color) in enumerate(arms):
            v=m[role];ys=[v['by_seed'][s] for s in seeds];xs=np.asarray([j-.07,j,j+.07])
            ax.scatter(xs,ys,color=color,s=42,zorder=3)
            for x,y,s in zip(xs,ys,seeds):ax.annotate(s,(x,y),xytext=(4,3),textcoords='offset points',fontsize=8,color=color)
            ax.errorbar(j,v['mean'],yerr=[[v['mean']-v['min']],[v['max']-v['mean']]],fmt='D',color=color,capsize=8,markersize=7,zorder=4)
        for k,s in enumerate(seeds):ax.plot([-.07+k*.07,.93+k*.07],[m['control']['by_seed'][s],m['candidate']['by_seed'][s]],color='#95a6b9',lw=.8,alpha=.65)
        ax.set(title=title,xticks=[0,1],xticklabels=['W64 / w1','W128 / w1'],ylabel='nRMSE [%]',xlim=(-.45,1.45));ax.grid(axis='y',alpha=.2)
    fig.suptitle('고정3시드 모두 표시 · ◆=평균 · 선=최소–최대 (신뢰구간 아님)'+provenance,fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'width_seed_distributions.png')

    fig,axes=plt.subplots(1,2,figsize=(13,4.8));styles=('-','--',':')
    for role,label,color in arms:
        for k,s in enumerate(seeds):
            points=d['learning'][role][s]
            x=[r['training_updates'] for r in points]
            for ax,key in zip(axes,('train_phase9_fixed','development')):
                y=[r[key]['all']['vector_nRMSE_percent'] for r in points]
                assert all(v>0 for v in y)
                ax.plot(x,y,styles[k],color=color,marker='o',ms=3,label=f'{label} · seed{s}')
    for ax,title in zip(axes,('Train 고정9위상 진단 (2952 graphs)','Dev 고정9위상 진단 (288 graphs)')):
        ax.set(title=title,xlabel='실제 optimizer update',ylabel='B벡터 nRMSE [%] · log scale');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('저장된6평가점만 연결 · 전체45위상 train-loss 곡선이 아님'+provenance,fontsize=12)
    fig.tight_layout(rect=(0,0,1,.88));save(fig,'width_learning_curves.png')

    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,metric,title in zip(axes,('vector_nrmse_percent','magnitude_nrmse_percent'),('Bx/By 벡터','저장성분 norm')):
        for role,label,color in arms:
            rows=[d['metrics'][f'snapshot_case_arithmetic/step:{i}/{metric}'][role] for i in range(45)]
            ax.plot(range(45),[r['mean'] for r in rows],color=color,label=label)
            ax.fill_between(range(45),[r['min'] for r in rows],[r['max'] for r in rows],color=color,alpha=.12)
        ax.set(title=title,xlabel='저장 위상 index (0…44)',ylabel='32case snapshot 산술평균 nRMSE [%]');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('선=3시드 평균, 음영=최소–최대 · 면적 pooled 지표와 구분 · 45위상 모두 학습'+provenance,fontsize=12)
    fig.tight_layout(rect=(0,0,1,.88));save(fig,'width_phase_diagnostics.png')

    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for row,case in enumerate((208,408,49)):
        reference=None
        for role,label,color in arms:
            values=[]
            for s in seeds:
                points=sorted((x for x in d['fixed_contour_torque_samples'][role][s] if x['case']==case),key=lambda x:x['step'])
                assert [x['step'] for x in points]==list(range(45))
                truth=np.asarray([x['truth_Nm'] for x in points]);pred=np.asarray([x['pred_Nm'] for x in points])
                if reference is None:reference=truth
                else:assert np.array_equal(truth,reference)
                values.append(pred)
            values=np.asarray(values);error=values-reference[None]
            for ax,array in ((axes[row,0],values),(axes[row,1],error)):
                ax.plot(range(45),array.mean(axis=0),color=color,label=label)
                ax.fill_between(range(45),array.min(axis=0),array.max(axis=0),color=color,alpha=.13)
        axes[row,0].plot(range(45),reference,'k--',lw=1.2,label='저장 FEM B의 공통 연산자')
        axes[row,1].axhline(0,color='black',lw=.8)
        for col in (0,1):
            axes[row,col].set(title=f'case {case} · '+('공통 a1 토크' if col==0 else '예측 - 저장기준'),xlabel='저장 위상 index',ylabel='Nm');axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
    fig.suptitle('오차로 고르지 않은3조건 · 선=3시드 평균, 띠=최소–최대 · 독립 실험축토크 검증 아님'+provenance,fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'width_torque_waveforms.png')
    receipt=dict(status='COMPLETE_FINAL_WIDTH_DIAGNOSTIC_PLOTS',created_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                 summary_sha256=sha(source),plotter_sha256=sha(Path(__file__).resolve()),
                 matplotlib=matplotlib.__version__,numpy=np.__version__,new_inference=0,new_training=0,
                 files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],visual_QA='PENDING_STANDALONE_PNG_INSPECTION')
    with (ROOT/'FINAL_DIAGNOSTIC_PLOT_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'status':receipt['status'],'plots':4}))


if __name__=='__main__':main()
