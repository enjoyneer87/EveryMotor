"""Assemble only completed LR20 evidence and eighteen actually reviewed PNGs."""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import datetime as dt
import hashlib
import json

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
NAS=Path('N:/MotorAI_NextLR20_20261031_01a18cac').resolve()
CAPTURED={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,expected_sha256=None):
    p=Path(p).resolve();assert p.suffix=='.json' and p.stat().st_size<=32*1024**2
    with p.open('rb') as f:raw=f.read(32*1024**2+1)
    assert len(raw)<=32*1024**2
    digest=hashlib.sha256(raw).hexdigest();assert p not in CAPTURED or CAPTURED[p]==digest;CAPTURED[p]=digest
    assert expected_sha256 is None or digest==expected_sha256,'captured bytes differ from approved input pin'
    return json.loads(raw.decode('utf-8-sig'))
def pin(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def number(value):
    import math
    assert type(value) in (int,float) and math.isfinite(value)
    return f'{value:.6f}'
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


class Links(HTMLParser):
    def __init__(self):super().__init__();self.files=[];self.images=[];self.external=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key not in ('src','href') or not value:continue
            url=urlparse(value)
            if url.scheme in ('http','https'):self.external.append(value);continue
            assert url.scheme=='file','only explicitly resolved local evidence URIs are allowed'
            path=unquote(url.path)
            if url.netloc:path='//'+url.netloc+path
            elif path.startswith('/') and len(path)>2 and path[2]==':':path=path[1:]
            p=Path(path).resolve();assert p.is_relative_to(NAS.resolve()) and p.is_file()
            self.files.append(p)
            if tag=='img':self.images.append(p)


def main():
    assert not any((ROOT/n).exists() for n in ('CONTOUR_LR20_RECOVERY_FAILURE.json','CONTOUR_LR20_RECOVERY_GUARD_FAILURE.json'))
    initial_source_sha=sha(Path(__file__))
    contract_path=ROOT/'HTML_LR20_RECOVERY_EXECUTION_CONTRACT.json';contract=read(contract_path)
    assert contract['status']=='RELEASED_FINAL_LR20_HTML_WITH_AUXILIARY_RECOVERED_IMAGES' and contract['assembler_sha256']==initial_source_sha
    pointer=read(ROOT/'FINAL_DIAGNOSTICS_POINTER.json');summary_path=Path(pointer['path']).resolve()
    assert summary_path==NAS/'final_return_R1/FINAL_LR20_DIAGNOSTICS.json'
    assert sha(summary_path)==pointer['sha256']==contract['summary_sha256']
    d=read(summary_path,contract['summary_sha256']);assert d['status']=='COMPLETE_LR20_SAVED_FINAL_DIAGNOSTICS' and d['metric_count']==703
    science_path=ROOT/'science/FINAL_LR20_SCIENTIFIC_REVIEW.json'
    assert science_path.stat().st_size==contract['science_bytes']
    science=read(science_path,contract['science_sha256'])
    assert science['status']=='COMPLETE_FIXED_LR20_REVIEW' and science['summary_sha256']==pointer['sha256']
    verdict=d['registered_judgment_replay'];assert verdict['status']=='COMPLETE' and verdict['fixed_seeds']==[17,29,43]
    assert verdict['control_arm']=='saved_W128_LR30' and verdict['candidate_arm']=='W128_LR20_equal88565'
    assert science['verdict']==verdict['verdict'] and verdict['verdict'] in ('IMPROVEMENT_CLAIM_WITHIN_INNERDEV','NO_IMPROVEMENT_CLAIM')
    native_path=ROOT/'CONTOUR_LR20_RECOVERED_RENDER_RECEIPT.json';native=read(native_path,contract['native_receipt_sha256'])
    diagnostics_path=ROOT/'LR20_DIAGNOSTIC_PLOT_RECEIPT.json';diagnostics=read(diagnostics_path,contract['diagnostic_receipt_sha256'])
    assert native['status']=='COMPLETED_AUXILIARY_RECOVERED_NATIVE_RENDERING_WITH_OPERATIONAL_FAILURE' and len(native['plots'])==12
    assert native['source_vector_basis']==native['native_BC_sign_pair']=='UNKNOWN'
    assert native['original_operational_status']=='INCOMPLETE_NO_CLAIM' and native['process_ancestry']=='UNKNOWN' and native['auxiliary_visualization_only'] is True
    exit_path=ROOT/'CONTOUR_LR20_RECOVERY_GUARD_EXIT_ATTESTATION.json';exit_proof=read(exit_path,contract['guard_exit_attestation_sha256'])
    assert native['guard_exit_attestation_sha256']==contract['guard_exit_attestation_sha256']
    assert exit_proof['status']=='PASS_GUARD_OS_EXIT0_OBSERVED_WITHIN180' and exit_proof['OS_exit_code']==0 and exit_proof['observed_total_seconds']<180
    assert diagnostics['status']=='COMPLETE_FINAL_LR20_DIAGNOSTIC_PLOTS' and diagnostics['summary_sha256']==pointer['sha256'] and len(diagnostics['files'])==6
    qa_path=ROOT/'PNG_LR20_VISUAL_QA_RECEIPT.json';qa=read(qa_path,contract['png_qa_sha256'])
    assert qa['status']=='PASS_STANDALONE_PNG_IMAGES' and qa['review_method']
    assert contract['png_qa_sha256']==sha(qa_path) and contract['native_receipt_sha256']==sha(native_path) and contract['diagnostic_receipt_sha256']==sha(diagnostics_path)
    expected=native['plots']+diagnostics['files'];qa_by_path={Path(p['path']).resolve():p for p in qa['files']}
    assert len(expected)==len(qa_by_path)==18
    for p in expected:
        path=Path(p['path']).resolve();assert path.parent==NAS/'report_assets_R1' and path.suffix=='.png'
        assert path in qa_by_path and path.stat().st_size==p['bytes']
        assert sha(path)==p['sha256']==qa_by_path[path]['sha256']
    primary=verdict['metrics']['primary_all45_Bvector'];passed=verdict['scientific_gates_passed']
    outcome='노출된 내부검증 범위에서 개선 기준 충족' if passed else '사전등록 개선 기준을 충족하지 못함'
    sections=[]
    sections.append('<section class="status"><h2>컨투어 운영 실패와 별도 산출물 회수</h2><p>원래 컨투어 실행은 INCOMPLETE_NO_CLAIM입니다. Popen child와 산출물 owner의 PID가 달라 소유 신원 검사가 실패했습니다. 실제 프로세스 계보는 UNKNOWN이며, 이 실패를 성공으로 바꾸지 않았습니다.</p><p>추가 추론·재시도 없이 이미 저장된12개 artifact를 별도 읽기 전용 감사로 회수했습니다. 필수 파일 해시·native ID 순서·참조 자계·고정 geometry/support 결속 검증을 통과한 자료만 보조 시각화로 표시하며, scalar 차이는 별도 진단으로 전부 공개합니다. 이는 원래 운영 계약의 성공, 독립 정확도 평가 또는 물리 경계 인증을 뜻하지 않습니다.</p></section>')
    sections.append('<header><p class="eyebrow">MotorAI · 고정 3시드 · LR 정책 단일축</p><h1>W128 / LR30 → LR20</h1><p class="badge">'+escape(outcome)+'</p><p>전체 Bx/By 평균 nRMSE '+number(primary['control']['mean'])+'% → '+number(primary['candidate']['mean'])+'%. 감소량은 '+number(primary['improvement_mean_pp'])+' pp입니다. 이 결과는 노출 이력이 있는 내부 dev32 비교이며 독립 시험 결과가 아닙니다.</p></header>')
    sections.append('<section><h2>무엇을 바꿨나</h2><p>폭128·깊이3·저장45위상·공극 가중1·alpha0, 초기 가중치와 표본 순서, Adam·clip·손실·분할·평가를 유지했습니다. cosine 학습률의 감소 구간만 30에서 20 등가 구간으로 줄였습니다. 같은 세 초기화에서 각각 88,565회 새로 학습했고, 완료된 LR30 대조군과 그 저장된 추론값은 재실행하지 않았습니다.</p><p>학습률을 더 일찍 낮추면 같은 업데이트 예산에서 오차가 더 줄어들 수 있다는 가설을 시험했습니다. 중간 점수나 좋은 시드로 조건을 바꾸지 않고 final88565만 판정했습니다. 전체45는 저장된 표본45개이며 실제360도 주기성의 증명이 아닙니다.</p><h3>고정 시드의 뜻</h3><p>17·29·43은 초기값과 학습 순서를 반복할 수 있게 정하는 번호입니다. 세 결과를 모두 제시합니다. 최소–최대와 범위는 초기화·순서에 따른 흔들림이며 신뢰구간이나 독립 표본 검정이 아닙니다.</p></section>')
    scalar_rows=[]
    assert len(native['scalar_diagnostics'])==12 and len({(p['case_index'],p['step']) for p in native['scalar_diagnostics']})==12
    for p in sorted(native['scalar_diagnostics'],key=lambda x:(x['case_index'],x['step'])):
        for metric,v in p['scalar_comparison'].items():
            scalar_rows.append([p['case_index'],p['step'],metric,format(v['saved'],'.12g'),format(v['recovered'],'.12g'),format(v['signed_delta_pp'],'.6g'),format(v['absolute_delta_pp'],'.6g'),'기존 산술 오차 내 일치' if v['within_saved_arithmetic_tolerance'] else 'SCALAR_DIFFERENCE_UNEXPLAINED'])
    assert len(scalar_rows)==24
    sections.append('<section><h2>회수 자계의 scalar 일관성 진단</h2><p>'+escape(native['scalar_status'])+'</p><p>고정12snapshot의 vector/norm nRMSE 24값을 모두 표시합니다. rel/abs 1e-10은 기존 저장 산술 비교값이며 별도 CUDA 추론의 검증된 허용오차가 아닙니다. 차이는 숨기거나 새 문턱으로 통과시키지 않습니다. 컨투어는 보조 저장 산출물이며 저장 평가를 재현한 자계 증거로 주장하지 않습니다.</p>'+table(['case','step','지표','저장 [%]','회수 [%]','회수−저장 [pp]','절댓값 [pp]','진단'],scalar_rows)+'</section>')
    names=[('전체 Bx/By','primary_all45_Bvector'),('과거9위상 부분집합','seen9_Bvector'),('공극 a1','a1_Bvector'),('구리','copper_Bvector'),('PA≥70 case 산술평균','PA_ge70_case_mean_Bvector'),('공통 a1 Arkkio 토크','pooled_torque')]
    rows=[]
    for label,key in names:
        m=verdict['metrics'][key]
        for role,arm in (('control','W128 LR30'),('candidate','W128 LR20')):
            v=m[role];rows.append([label,arm,*[number(v['by_seed'][str(s)]) for s in (17,29,43)],number(v['mean']),number(v['min']),number(v['max']),number(v['range'])])
    sections.append('<section><h2>모든 시드와 고정 판정</h2><p>아래 nRMSE는 %입니다. PA는 Phase Advance 조건이며 고 회전자 위치각을 뜻하지 않습니다. 토크는 같은 저장 자계와 공통 a1 연산자로 계산한 진단입니다.</p>'+table(['지표','모델','seed17','seed29','seed43','평균','최소','최대','범위'],rows))
    gate_labels={'mean_improvement_at_least_1pp':'전체 평균 감소 ≥1 pp','mean_improvement_strictly_gt_2max_range':'감소량 >2×더 큰 시드 범위','mean_improvement_strictly_gt_frozen_W128_bound':'감소량 >동결 LR30 범위 하한'}
    gates=[[gate_labels[key],'PASS' if value else 'FAIL'] for key,value in verdict['primary_gates'].items()]
    harm=[[names[[k for _,k in names].index(key)][0],number(row['actual_worsening_pp']),number(row['maximum_worsening_pp']),'PASS' if row['pass'] else 'FAIL'] for key,row in verdict['no_harm_gates'].items()]
    sections.append('<p>전체 자계 감소량 ≥1 pp이면서 두 모델 중 더 큰 시드 범위의 2배를 엄격히 초과해야 합니다. 동결된 LR30 범위에 따른 하한은 1.421114316269 pp이며 모든 비악화 조건도 통과해야 합니다.</p>'+table(['사전등록 primary 조건','판정'],gates)+table(['비악화 지표','후보−대조 평균 [pp]','허용 상한 [pp]','판정'],harm))
    paired=[]
    for label,key in names:
        m=verdict['metrics'][key];paired.append([label,*[number(m['paired_control_minus_candidate_pp'][str(s)]) for s in (17,29,43)],number(m['improvement_mean_pp'])])
    sections.append(table(['대조−후보 감소 [pp]','seed17','seed29','seed43','평균'],paired)+'<p>별도 토크 개선 조건: '+('PASS' if verdict['separate_torque_improvement'] else 'FAIL')+'. 독립 실험축 토크나 native 경계조건의 검증으로 해석하지 않습니다.</p></section>')
    def figures(rows):
        return ''.join('<figure><img loading="lazy" src="'+escape(Path(p['path']).resolve().as_uri(),quote=True)+'" alt="'+escape(Path(p['path']).stem,quote=True)+'"><figcaption>'+escape(Path(p['path']).name)+' · SHA256 '+p['sha256'][:16]+'…</figcaption></figure>' for p in rows)
    sections.append('<section><h2>학습·학습률·오차 분포</h2><p>학습 곡선은 실제 저장된 6개 평가점입니다. 전체45위상의 연속 train-loss 곡선이 아닙니다. LR 그림의 선은 동결 정책 계산이고 점은 기록된 평가시점 LR입니다. CDF는 각 시드의 1440개 snapshot 전체이며 면적 pooled 지표와 분모가 다릅니다.</p>'+figures(diagnostics['files'])+'</section>')
    detailed=[]
    for phase in ('all45','seen9','unused36'):
        for region in ('all','a1','copper'):
            for metric in ('vector_nrmse_percent','magnitude_nrmse_percent','vector_rmse_T','magnitude_rmse_T'):
                key=f'field/area_pooled/{phase}/{region}/{metric}';m=d['metrics'][key]
                detailed.append([phase,region,metric,number(m['control']['mean']),number(m['candidate']['mean'])])
    sections.append('<section><h2>영역·위상·단위별 진단</h2><p>seen9·unused36은 과거의 명칭입니다. 이번 양쪽 학습은 저장45위상을 모두 사용했습니다. 각 영역·위상의 상대오차 분모가 다르므로 nRMSE만으로 전체 SSE 기여율을 추정하지 않습니다.</p><details><summary>전체·공극·구리의 벡터/크기 오차와 절대 T</summary>'+table(['저장 위상 부분집합','영역','지표 / 단위','LR30 평균','LR20 평균'],detailed)+'</details>')
    extra=[]
    prefixes=('copper_B2/all45/','torque/pooled/','torque/case_arithmetic/')
    for key,m in d['metrics'].items():
        if key.startswith(prefixes):extra.append([key,number(m['control']['mean']),number(m['candidate']['mean'])])
    sections.append('<details><summary>구리 B²·토크 평균/리플 진단</summary><p>signed B² bias는 부호가 있는 편향이며 감소 방향 자체가 정확도 개선을 뜻하지 않습니다. B²·토크는 저장 자계의 파생 진단이며 AC 손실이나 실측 토크의 검증이 아닙니다.</p>'+table(['저장 진단 / 명시된 단위','LR30 평균','LR20 평균'],extra)+'</details></section>')
    sections.append('<section><h2>실제 native 자계 비교</h2><p>점수를 보기 전에 고정한 seed17, case208·408·49, step0·11·22·33을 사용했습니다. 세 조건은 모두460 Arms 메타데이터로 전류 범위 실험이 아닙니다. 저장 FEM 기준·LR30·LR20와 오차는 같은 색 범위와 T 단위입니다. 보간·평활·회전·주기 복제를 하지 않았습니다.</p><p>회색은 a2 제외 또는 미지원 공간이며 stale a2 polygon을 그리지 않았습니다. 확대 위치는 오차가 아닌 기존 a1 기하로 고정했습니다. clipping 개수와 면적비는 확대 화면이 아닌 전체 retained support 기준입니다. source basis 및 BC/sign/pair는 UNKNOWN입니다.</p>'+figures(native['plots'])+'</section>')
    costs=[]
    for role,arm in (('control','저장 LR30'),('candidate','신규 LR20')):
        for seed in ('17','29','43'):
            v=d['run_costs'][role][seed];costs.append([arm,seed,number(v['training_worker_elapsed_seconds']),number(v['training_assessment_wall_seconds']),number(v['evaluation_worker_elapsed_seconds']),v['training_updates'],v['evaluation_graphs']])
    sections.append('<section><h2>실행 비용과 적용 범위</h2>'+table(['모델','seed','학습 worker [s]','평가점 포함 진단 [s]','최종 평가 worker [s]','업데이트','평가 graph'],costs)+'<p>같은 업데이트 수가 같은 GPU 비용을 뜻하지는 않습니다. 학습 assessment 시간은 학습 worker 시간에 포함됩니다. FEM 대비 속도 배수는 측정하지 않았습니다.</p><h3>대리모델이 하는 일</h3><p>형상·운전조건·회전자 위치와 native 요소의 특징을 입력받아 메시 이웃의 정보를 모으고 요소별 Bx/By를 예측합니다. 상세한 자계 분포를 빠르게 근사하는 데 활용할 수 있지만, 학습 범위 밖 형상·전류·주기 경계의 신뢰도는 별도 검증이 필요합니다. 이번 변경은 학습률뿐이며 회전자–고정자 새 연결을 추가하지 않았습니다.</p><h3>남겨 둔 FAIL / UNKNOWN</h3><p>native BC·부호·성분 기저·boundary pair는 UNKNOWN입니다. case40의 no-wrap P0 공극 연결은 전체45위상 coverage FAIL로 유지합니다. a2 감독·면적 제외, 보호 R1_0199/R1_0076와 alias·old val/test 제외, HalfSC32/4·exact bridge0·canonical AF 범위를 유지했습니다. case40 원천 pilot의 일치를 전체 corpus 물리 검증으로 확대하지 않습니다.</p><p>새 FEA·prior·teacher는 사용하지 않았고 과거 A2/C/G/alpha/w3 결과도 재실행하거나 덮어쓰지 않았습니다. 이번 한 실험 이후 추가 학습을 자동으로 시작하지 않습니다.</p></section>')
    sections.append('<footer><p><a href="'+escape(summary_path.as_uri(),quote=True)+'">전체703개 저장 진단 JSON</a> · <a href="https://github.com/enjoyneer87/EveryMotor/tree/25f0834ed0fa197b3671289703b8c3122c505d75/research/motorai_multisolver_20261028/main_training_visual_20261030">이전 W128 완료 보고</a></p><p>PNG 개별 검수와 정적 링크·해시 검사를 수행했습니다. HTML 브라우저 렌더링은 보안정책 차단 이력으로 미검증이며 다른 surface/protocol로 우회하지 않았습니다.</p></footer>')
    style='body{font:16px/1.7 "Malgun Gothic",sans-serif;max-width:1180px;margin:32px auto;padding:0 18px;background:#eef3f8;color:#173047}header,section,footer{background:white;border:1px solid #d8e2eb;border-radius:14px;margin:20px 0;padding:26px}header{background:#12324c;color:white}.eyebrow{color:#a4d8f4}.badge{font-weight:bold;font-size:20px}h1{font-size:32px}h2{font-size:24px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:9px;border-bottom:1px solid #d8e2eb;text-align:left;white-space:nowrap}.table{overflow:auto}img{width:100%;height:auto;border:1px solid #d8e2eb}figure{margin:24px 0}figcaption{font-size:12px;color:#506579}details{margin:16px 0}a{color:#1572a1}'
    html='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MotorAI LR20 최종 비교</title><style>'+style+'</style></head><body>'+''.join(sections)+'</body></html>'
    links=Links();links.feed(html)
    assert len(links.images)==18 and set(links.images)==set(qa_by_path)
    for path,digest in CAPTURED.items():assert sha(path)==digest,'captured evidence changed before HTML write'
    for path,p in qa_by_path.items():assert sha(path)==p['sha256']
    assert not any((ROOT/n).exists() for n in ('CONTOUR_LR20_RECOVERY_FAILURE.json','CONTOUR_LR20_RECOVERY_GUARD_FAILURE.json'))
    assert sha(Path(__file__))==initial_source_sha
    target=WORKSPACE/'outputs/MOTORAI_NEXT_STAGE_20261031.html';backup=ROOT/'HTML_PRE_FINAL_BACKUP.html'
    with backup.open('xb') as f:f.write(target.read_bytes())
    target.write_text(html,encoding='utf-8')
    combined=dict(status='PASS_STANDALONE_PNG_QA_AND_STATIC_LINKS',browser_rendering='UNVERIFIED_SECURITY_POLICY_BLOCK',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),html_sha256=sha(target),png_qa_sha256=sha(qa_path),files=qa['files'],local_reference_count=len(links.files),external_links=links.external)
    combined_path=ROOT/'FINAL_LR20_VISUAL_QA_RECEIPT.json'
    with combined_path.open('x',encoding='utf-8') as f:json.dump(combined,f,indent=2)
    combined_scope=dict(original_operational_status='INCOMPLETE_NO_CLAIM',process_ancestry='UNKNOWN',auxiliary_visualization_only=True,scalar_status=native['scalar_status'])
    completion=dict(**combined_scope,status='COMPLETE_LOCAL_LR20_HTML_WITH_AUXILIARY_RECOVERED_CONTOURS',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),html=pin(target),summary_sha256=pointer['sha256'],visual_qa_sha256=sha(combined_path),assembler_sha256=initial_source_sha,execution_contract_sha256=sha(contract_path),verdict=verdict['verdict'],PNGs=18,browser_rendering='UNVERIFIED_SECURITY_POLICY_BLOCK')
    with (ROOT/'FINAL_LR20_HTML_COMPLETION_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(completion,f,indent=2)
    print(json.dumps(dict(status=completion['status'],path=str(target),PNGs=18,verdict=completion['verdict'])))


if __name__=='__main__':main()
