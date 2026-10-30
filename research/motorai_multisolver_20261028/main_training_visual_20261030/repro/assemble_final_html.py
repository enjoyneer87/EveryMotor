"""Assemble completed, reviewed numbers and actual QA-bound plots into the HTML."""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import datetime as dt
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
ASSETS=WORKSPACE/'outputs/MOTORAI_MAIN_TRAINING_VISUAL_20261030_assets'
METRICS=(('전체 Bx/By','primary_all45_Bvector'),('공극 a1','a1_Bvector'),('구리','copper_Bvector'),
         ('과거9위상 subset','seen9_Bvector'),('PA≥70 case 평균','PA_ge70_case_mean_Bvector'),('같은a1 Arkkio 토크','pooled_torque'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def key(path):return str(Path(path).resolve()).casefold()


class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
            if name in ('src','href') and value:self.refs.append(value)


def main():
    assert not (ROOT/'FINAL_HTML_COMPLETION_RECEIPT.json').exists()
    assembler_path=Path(__file__).resolve();assembler_bytes=assembler_path.read_bytes()
    inputs={}
    def capture(path):
        path=Path(path).resolve();data=path.read_bytes();inputs[path]=data
        return json.loads(data)
    def captured_sha(path):return hashlib.sha256(inputs[Path(path).resolve()]).hexdigest()
    summary_path=ROOT/'FINAL_WIDTH_DIAGNOSTICS.json';d=capture(summary_path)
    assert d['status']=='COMPLETE_SAVED_FINAL_WIDTH_DIAGNOSTICS' and d['metric_count']==703
    verdict=d['registered_judgment_replay'];assert verdict['status']=='COMPLETE'
    science_path=ROOT/'science/FINAL_WIDTH_SCIENTIFIC_REVIEW.json';science=capture(science_path)
    assert science['status']=='COMPLETE_FIXED_WIDTH_REVIEW' and science['summary_sha256']==captured_sha(summary_path)
    assert science['verdict']==verdict['verdict']
    contours=capture(ROOT/'CONTOUR_RENDER_RECEIPT.json');diagnostics=capture(ROOT/'FINAL_DIAGNOSTIC_PLOT_RECEIPT.json')
    assert contours['status']=='COMPLETED_REAL_STORED_NATIVE_CONTOUR_RENDERING' and len(contours['plots'])==18
    assert diagnostics['status']=='COMPLETE_FINAL_WIDTH_DIAGNOSTIC_PLOTS' and len(diagnostics['files'])==4
    assert diagnostics['summary_sha256']==captured_sha(summary_path)
    render_plan_path=ROOT/'CONTOUR_RENDER_PLAN.json';capture(render_plan_path)
    assert contours['render_plan_sha256']==captured_sha(render_plan_path)
    png_qa_path=ROOT/'PNG_VISUAL_QA_RECEIPT.json';png_qa=capture(png_qa_path)
    assert png_qa['status']=='PASS_STANDALONE_PNG_IMAGES'
    qa_files={key(x['path']):x for x in png_qa['files']}
    required=contours['plots']+diagnostics['files']
    assert len({key(x['path']) for x in required})==22
    for row in required:
        p=Path(row['path']).resolve();assert p.is_relative_to(ASSETS.resolve()) and sha(p)==row['sha256']
        assert key(p) in qa_files and qa_files[key(p)]['sha256']==row['sha256']
    source=WORKSPACE/'outputs/MOTORAI_MAIN_TRAINING_VISUAL_20261030.html'
    old=source.read_bytes();text=old.decode('utf-8-sig')
    assert 'id="width-final-results"' not in text
    passed=verdict['scientific_gates_passed']
    conclusion='W128이 사전등록 내부검증 개선 기준을 통과했습니다.' if passed else 'W128은 사전등록 개선 기준을 통과하지 못했습니다. 비교 기준은 W64/w1을 유지합니다.'
    body='<section id="width-final-results"><h2>폭128 본학습 최종 결과</h2><p><b>'+conclusion+'</b></p>'
    body+='<p>고정시드17·29·43을 모두 완료했습니다. 각88,565회 갱신과32조건×45위상=1,440개 최종평가, 총265,695회 갱신·4,320개 평가입니다. 폭 외의 그래프·정답·손실·깊이·순서·학습률은 고정했습니다.</p>'
    body+='<p class="note">범위는 oldTRAIN 노출 이력이 있는 내부dev32입니다. 독립test·AC손실·물리 축토크·cross-solver 개선의 검증이 아닙니다. 원천 BC·벡터basis·회전자–고정자 nativepair는UNKNOWN으로 남습니다.</p>'
    body+='<table><tr><th>오차 nRMSE</th><th>W64/w1 평균 [최소–최대]</th><th>W128/w1 평균 [최소–최대]</th><th>후보−기준 [pp]</th></tr>'
    for title,name in METRICS:
        row=verdict['metrics'][name];a,b=row['control'],row['candidate']
        body+=f'<tr><td>{title}</td><td>{a["mean"]:.3f}% [{a["min"]:.3f}–{a["max"]:.3f}]</td><td>{b["mean"]:.3f}% [{b["min"]:.3f}–{b["max"]:.3f}]</td><td>{row["worsening_mean_pp"]:+.3f}</td></tr>'
    body+='</table><p>표는 반올림 표시이며 판정은 저장 원수치로 수행했습니다. 범위는3시드 최소–최대이며 신뢰구간이나 유의성 검정이 아닙니다.</p>'
    primary=verdict['metrics']['primary_all45_Bvector'];delta=primary['improvement_mean_pp']
    body+=f'<p>전체 B벡터 평균 개선은 {delta:+.6f}pp입니다. 채택은 다음 사전등록 기준의 동시 통과로 판정합니다.</p><table><tr><th>전체 자계 기준</th><th>결과</th></tr>'
    gate_names={'mean_improvement_at_least_1pp':'평균 개선≥1pp','mean_improvement_strictly_gt_2max_range':'평균 개선&gt;2×두 모델 중 큰 시드범위','mean_improvement_strictly_gt_frozen_W64_phase45_bound':'평균 개선&gt;2×고정 W64 시드범위'}
    for name,value in verdict['primary_gates'].items():body+=f'<tr><td>{gate_names[name]}</td><td>{"PASS" if value else "FAIL"}</td></tr>'
    body+='</table><table><tr><th>악화 제한</th><th>허용 [pp]</th><th>실제 후보−기준 [pp]</th><th>결과</th></tr>'
    label_map=dict((name,title) for title,name in METRICS)
    for name,row in verdict['no_harm_gates'].items():body+=f'<tr><td>{label_map[name]}</td><td>≤{row["maximum_worsening_pp"]:g}</td><td>{row["actual_worsening_pp"]:+.6f}</td><td>{"PASS" if row["pass"] else "FAIL"}</td></tr>'
    body+='</table><details><summary>시드별 결과 전부 보기</summary><div style="overflow:auto"><table><tr><th>지표</th><th>모델</th><th>seed17</th><th>seed29</th><th>seed43</th><th>평균</th><th>범위</th></tr>'
    for title,name in METRICS:
        for role,label in (('control','W64/w1'),('candidate','W128/w1')):
            row=verdict['metrics'][name][role]
            body+=f'<tr><td>{title} [%]</td><td>{label}</td>'+''.join(f'<td>{row["by_seed"][s]:.6f}</td>' for s in ('17','29','43'))+f'<td>{row["mean"]:.6f}</td><td>{row["range"]:.6f}</td></tr>'
    body+='</table></div></details><p><a href="../work/main_training_visual_20261030_01a18cac/FINAL_WIDTH_DIAGNOSTICS.json">703개 저장 진단·원수치·고정 판정</a> · <a href="../work/main_training_visual_20261030_01a18cac/science/FINAL_WIDTH_SCIENTIFIC_REVIEW.json">Astra Max 최종 과학 검토</a></p></section>'

    def image(path,caption):
        p=Path(path).resolve();assert key(p) in qa_files and sha(p)==qa_files[key(p)]['sha256']
        href=ASSETS.name+'/'+p.name
        return f'<figure style="margin:22px 0"><a href="{href}"><img src="{href}" style="width:100%;height:auto" alt="{escape(p.stem)}"></a><figcaption>{caption}</figcaption></figure>'

    body+='<section id="width-final-diagnostics"><h2>학습·시드·위상·토크 비교</h2>'
    captions={'width_seed_distributions.png':'모든3시드와 평균·최소–최대. 좋은 시드만 고르지 않았습니다.',
              'width_learning_curves.png':'저장된6개 평가점입니다. train2952/dev288의 고정9위상 진단이며 전체45위상 train-loss 곡선이 아닙니다.',
              'width_phase_diagnostics.png':'각 저장위상의32case snapshot 산술평균이며 면적 pooled 지표와 다릅니다. 두 모델 모두45위상을 학습했습니다.',
              'width_torque_waveforms.png':'점수 전에 고정한3조건의45개 표본. 같은 a1 연산자 결과이며 독립 측정 축토크의 검증이 아닙니다.'}
    for row in diagnostics['files']:body+=image(row['path'],captions[Path(row['path']).name])
    body+='</section>'
    gallery='<section id="native-contours"><h2>실제 저장 FEM 기준·예측·오차 컨투어</h2><p>아래는 실제 native 삼각형의 평탄 색상입니다. 각 그림을 누르면 원본 해상도로 볼 수 있습니다. 조건·시드17·위상0/11/22/33은 결과 확인 전에 고정했습니다.</p><p>Stored Bx/By [T]는±3T, 저장성분 norm은0–3T, 절대성분/벡터오차는0–1.5T의 공통 범위입니다. 범위 밖 요소 수는 각 패널에 표시하며 확대도 전체 retained 요소를 기준으로 집계합니다. 회색은제외/미지원 공간으로 실제 a2 polygon을 뜻하지 않습니다. smoothing·대칭복제·추정회전은 사용하지 않았습니다.</p><p><b>STORED_NATIVE_HDF_REPLAY_ONLY / source vector basis UNKNOWN.</b> 원 TXT와 직접 메시를 대조한 사례는case40뿐입니다. 이 그림이 native BC·주기 부호·회전자–고정자 대응 또는 전체corpus의 물리 검증을 뜻하지 않습니다.</p>'
    for case in (208,408,49):
        rows=[x for x in contours['plots'] if x['case_index']==case]
        overview=next(x for x in rows if x['group']=='width' and x['mode']=='norm' and not x['a1_crop'])
        gallery+=f'<h3>사전 지정 case {case}</h3>'+image(overview['path'],'고정 seed17/final88565. FEM 기준·W64/w1·W128/w1과각 벡터오차를같은범위로 비교합니다.')
        gallery+='<details><summary>저장 Bx·By / a1 확대 / 과거 공극w3 비교 펼치기</summary>'
        for row in rows:
            if row is overview:continue
            caption=('완료된 W64/w3 음성 실험의 별도 역사 비교입니다. 새 W128 후보와 혼동하지 않습니다.' if row['group']=='historical_w3' else '같은 native support에서의 저장성분 비교입니다.')
            if row['a1_crop']:caption+=' 확대는사전고정 step22의 geometry 기준 Cartesian crop이며 메시나색상범위를바꾸지 않았습니다.'
            gallery+=image(row['path'],caption)
        gallery+='</details>'
    gallery+='<p><a href="../work/main_training_visual_20261030_01a18cac/CONTOUR_RENDER_RECEIPT.json">그림·클리핑 수·실제 수치 범위 기록</a></p></section>'
    cost='<section><h2>계산 비용과 확인 범위</h2><table><tr><th>모델 / 시드</th><th>학습worker [분]</th><th>최종평가worker [초]</th></tr>'
    for role,label in (('control','W64/w1'),('candidate','W128/w1')):
        for seed in ('17','29','43'):
            row=d['run_costs'][role][seed];cost+=f'<tr><td>{label} / {seed}</td><td>{row["training_worker_elapsed_seconds"]/60:.2f}</td><td>{row["evaluation_worker_elapsed_seconds"]:.2f}</td></tr>'
    cost+='</table><p>학습worker 시간에는저장된6회 진단 평가가포함됩니다. 프로세스 부팅·전송·최종보고 시간과구분하며 동일갱신 수가동일GPU비용을뜻하지 않습니다. FEM 대비 배속은 측정하지 않았습니다.</p><p>브라우저 보안정책이 로컬HTML 렌더링을차단하여 브라우저 화면은미검증입니다. 실제 standalonePNG 검토와해시·정적링크 검사결과를별도로 남겼습니다.</p></section>'
    text,replacements=re.subn(r'<section><h2>컨투어와 학습 진단</h2>.*?</section>',gallery,text,count=1,flags=re.DOTALL)
    assert replacements==1
    text=re.sub(r'<p class="note">.*?</p>','<p class="note">폭128의고정3시드 학습·전체평가·최종체크포인트 컨투어를완료했습니다. 판정과원수치는아래 최종결과에서 확인할수있습니다. 주기·공극 물리연결은보류합니다.</p>',text,count=1,flags=re.DOTALL)
    text=text.replace('</header>','</header>'+body,1)
    text=text.replace('<footer>',cost+'<footer>',1)
    text=re.sub(r'<span class="badge">.*?</span>','<span class="badge">3시드 본학습·전체평가 완료 · 내부검증 범위</span>',text,count=1,flags=re.DOTALL)
    text=text.replace('폭만 바꾸는 본학습을 진행하며','폭만 바꾸는 본학습을 완료했으며')
    text=text.replace('실제 자계 컨투어는 최종 체크포인트 추론 후 추가합니다.','실제 최종 체크포인트 컨투어를 아래에 추가했습니다.')
    parser=Links();parser.feed(text);count=0
    for target in parser.refs:
        if target.startswith(('http:','https:','data:','#','mailto:')):continue
        path=(source.parent/target.split('#')[0]).resolve();assert path.is_file(),target
        if path.suffix.lower()=='.png':assert key(path) in qa_files and qa_files[key(path)]['sha256']==sha(path)
        count+=1
    assert source.read_bytes()==old,'input HTML changed during assembly'
    assert assembler_path.read_bytes()==assembler_bytes,'assembler source changed during execution'
    for path,data in inputs.items():assert path.read_bytes()==data,'validated report input changed before write'
    with (ROOT/'HTML_BEFORE_FINAL_ASSEMBLY.html').open('xb') as f:f.write(old)
    source.write_text(text,encoding='utf-8')
    finalqa=dict(status='PASS_STANDALONE_PNG_QA_AND_STATIC_LINKS',created_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                 browser_rendering='UNVERIFIED_SECURITY_POLICY_BLOCK',png_visual_qa_sha256=captured_sha(png_qa_path),
                 files=png_qa['files'],required_new_PNGs=22,local_HTML_links_verified=count,
                 html_sha256=sha(source),scientific_review_sha256=captured_sha(science_path),
                 qualification='stored native replay; original private source/corpus physical qualification remains unknown')
    with (ROOT/'FINAL_VISUAL_QA_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(finalqa,f,indent=2)
    completion=dict(status='COMPLETE_LOCAL_HTML_WITH_VALIDATED_PNGS',created_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                    html_sha256=sha(source),summary_sha256=captured_sha(summary_path),visual_qa_sha256=sha(ROOT/'FINAL_VISUAL_QA_RECEIPT.json'),
                    assembler_sha256=hashlib.sha256(assembler_bytes).hexdigest(),previous_html_sha256=hashlib.sha256(old).hexdigest(),
                    scientific_verdict=verdict['verdict'],new_contours=18,new_diagnostic_plots=4,local_links_verified=count,
                    browser_QA='UNVERIFIED_SECURITY_POLICY_BLOCK')
    with (ROOT/'FINAL_HTML_COMPLETION_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(completion,f,indent=2)
    print(json.dumps(completion))


if __name__=='__main__':main()
