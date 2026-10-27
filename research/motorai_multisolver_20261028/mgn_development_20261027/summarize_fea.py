"""Summarize numerical diagnostics without inventing formal qualification."""
from pathlib import Path
import json,csv
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    cases=['N256_P3','N512_P3','fine_N256_P3'];data={};observations=[]
    for case in cases:
        folder=ROOT/'convergence'/case
        if not (folder/'audit.json').exists():continue
        a=json.loads((folder/'audit.json').read_text());e=json.loads((folder/'execution.json').read_text());s=json.loads((folder/'shard_manifest.json').read_text())
        data[case]=dict(audit=a,execution=e,shard=s)
        for period in a['periods']:
            for quantity in ['Joule_W','DC_W','AC_W','current_max_percent']:
                observations.append(dict(template='JEET_e10_Ref_family',geometry='jeet_elmer:Ref',condition='Ref_P117;16000rpm;460Arms;beta36',
                    source='jeet_mgn_20261027:'+case,evaluator='native_P1_BE_BDF2_circuit_integration',quantity=quantity,scope='full_motor_active_only' if quantity!='current_max_percent' else 'max36bar_error_over_prescribed_phase_peak',
                    unit='W' if quantity.endswith('_W') else 'percent',value=period[quantity],period=period['period'],window_steps=str(period['steps']),
                    average='uniform_post_step_mean' if quantity.endswith('_W') else 'maximum_over_period',sigma_S_m=a['sigma_S_m'],active_length_m=a['active_length_m'],
                    source_hash=s['source_native_sha256'],new_independent_training_label=False,formal_convergence_qualification='UNKNOWN'))
    comparisons={}
    if 'N256_P3' in data:
        coarse=data['N256_P3']['audit'];old=json.loads((ROOT.parent/'remote_return/desktop-6qivkic_20261027/shards/Ref_P117_audit.json').read_text())
        error=np.array([coarse['periods'][1][k] for k in ['Joule_W','DC_W','AC_W']])-old['P2_saved_W']
        comparisons['repeat_original_P2']=dict(errors_W=error.tolist(),max_abs_error_W=float(abs(error).max()),gate_W=1e-6,
            status='PASS' if max(abs(error))<=1e-6 else 'FAIL',scope='same physical/time/mesh inputs, extended total history only; old target unchanged')
        base=coarse['periods'][2]['AC_W'];base_wave=np.load(ROOT/'convergence/N256_P3/waveforms.npz')['scalar'][512:768,3]
        for key,case in [('time_step','N512_P3'),('mesh','fine_N256_P3')]:
            if case not in data:continue
            other=data[case]['audit'];p3=other['periods'][2]['AC_W'];spp=other['steps_per_period'];wave=np.load(ROOT/'convergence'/case/'waveforms.npz')['scalar'][2*spp:3*spp,3]
            aligned=wave[1::2] if spp==512 else wave
            diff=100*abs(p3-base)/abs(p3);wave_error=100*np.linalg.norm(aligned-base_wave)/np.linalg.norm(aligned)
            period_ok=coarse['period_development_threshold_status']=='PASS' and other['period_development_threshold_status']=='PASS'
            comparisons[key]=dict(reference_case=case,coarse_case='N256_P3',P3_mean_difference_percent=float(diff),P3_phase_aligned_wave_L2_percent=float(wave_error),
                proposed_mean_threshold_percent=1.,development_threshold_status=('PASS' if diff<=1 else 'FAIL') if period_ok else 'UNKNOWN',
                formal_qualification='UNKNOWN',reason='No separately approved time/mesh qualification contract; measured sensitivity and predeclared engineering threshold reported only',
                period_development_checks_met=period_ok)
    result=dict(status='COMPLETE' if len(data)==3 else 'IN_PROGRESS',requested_numerical_variants=3,completed_numerical_variants=len(data),
        unique_physical_conditions=1,new_independent_training_labels=0,Full_calibration_budget_unchanged=63,comparisons=comparisons,
        period_diagnostics={k:{n:v['audit'][n] for n in ['P2_P3_mean_percent','P2_P3_wave_L2_percent','period_development_threshold_status','P3_current_status','whole_current_status','nonlinear_status']} for k,v in data.items()},
        formal_period_mesh_time_qualification='UNKNOWN',qualification_policy='QUALIFICATION_POLICY.json',historical_HalfSC_receipts_modified=False,
        original_results_modified=False,actual_solver_seconds=sum(v['execution']['wall_seconds'] for v in data.values()),
        whole_dataset_qualification='UNKNOWN; one Ref anchor does not qualify192loaded or36HalfSC conditions')
    (ROOT/'CONVERGENCE_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    if observations:
        with (ROOT/'convergence_observations.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(observations[0]));w.writeheader();w.writerows(observations)
    text=['# Ref_P117 수치 민감도와 원시 적분', '', f"상태: **{result['status']}**, 계획한3개 수치 조건 중{len(data)}개 완료. 실제 물리 운전점은1개이며 새 독립 학습 정답은0개다.", '',
        '기존 보정점 Ref_P117, 16,000rpm·460Arms·beta36°에서 원래 mesh/N256, 원래 mesh/N512, RefFine/N256을 각각3주기로 계산한다. 초기 A=0, 첫 스텝 BE 이후 BDF2, sigma47092391.429245S/m, 활성길이0.15m,8sector를 유지한다. P2는 원래의 균일 post-step 평균이며 전체 이력 평균으로 바꾸지 않는다.', '',
        '**판정 범위:** current0.001%와 nonlinear1e-10은 기존 채택 규칙이다. 주기 평균0.1%·파형L2 0.5%, 시간/메시 평균1%는 계산 전에 기록한 개발용 기준이다. 별도의 승인된 수렴 계약이 없으므로 공식 주기·시간·메시 qualification은UNKNOWN으로 남긴다. 아래 개발 기준 충족을 전체 물리 검증PASS로 확대하지 않는다.', '',
        '| 수치 조건 | 스텝 | P2 AC(W) | P3 AC(W) | P2/P3 평균 차(%) | P3 전류 최대오차(%) | 계산시간(s) |','|---|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        if case not in data:text.append(f'| {case} | 진행 중 | — | — | — | — | — |');continue
        d=data[case];a=d['audit'];text.append(f"| {case} | {a['steps']} | {a['periods'][1]['AC_W']:.9f} | {a['periods'][2]['AC_W']:.9f} | {a['P2_P3_mean_percent']:.8g} | {a['periods'][2]['current_max_percent']:.8g} | {d['execution']['wall_seconds']:.2f} |")
    text+=['','전류 오차는 순간 전류가 아닌 prescribed phase peak(sqrt(2)×Irms)에 대한 최대 비율이다. P2/P3 waveform L2와 전체 이력 current 상태는 CONVERGENCE_RESULTS.json에 별도 보존한다.']
    if 'repeat_original_P2' in comparisons:
        d=comparisons['repeat_original_P2'];text+=['',f"원래 P2 저장 scalar의 새 계산 재현 최대 차이는 **{d['max_abs_error_W']:.3g}W**, 기존1e-6W gate에 대해{d['status']}다. 이는 고정 label 재현 검사이며 다른 수치 해상도의 차이를 오류로 취급하지 않는다."]
        c=data['N256_P3']['audit']['periods'];first=100*(c[0]['AC_W']-c[1]['AC_W'])/c[1]['AC_W'];text+=['',f'첫 주기 AC 평균은 P2보다{first:.3f}% 높다. startup이 포함된 whole-history 평균을 채택 P2 대신 쓰지 않는다.']
    for kind in ['time_step','mesh']:
        if kind in comparisons:
            d=comparisons[kind];text+=['',f"{kind}: P3 평균 차이 **{d['P3_mean_difference_percent']:.6f}%**, 위상을 맞춘 파형L2 차이{d['P3_phase_aligned_wave_L2_percent']:.6f}%. 개발 기준{d['development_threshold_status']}; 공식 qualification UNKNOWN."]
    text+=['','새 scalar/current 시간 이력은 native A와 회로 전압에서 BE/BDF2를 적용해 재구성했다. 저장8위상의 J 재적분과 이 이력의 일치는 shard·reader 일관성 검사이며 별도의 독립 solver 손실 채널 검증이라고 주장하지 않는다. 독립적인 과거 채택 scalar와의 연결은 위 원래 P2 재현 검사이며, 각 시각의 전류는 prescribed current와 직접 비교했다.','',
        '## 메시·필드·보존','',
        'RefFine는 기존 검증된 input의 물성·회로·solver를 복사하고 메시만 교체했다. 구리36영역의 면적 차이 최대1.65e-13%, 중심차1.25e-16m, 외접범위 차0m다. 전체 physical domain 면적 차이는 최대0.00794%로 보고하며 연속 곡면의 완전 동일성을 증명했다고 하지 않는다. 좌표 최근접으로 ID를 복구하지 않았다.', '',
        '대용량 native/VTU/solver log는 원 계산 위치에 보존하고 모든 파일의 SHA256·크기를 raw_source_manifest.json으로 연결했다. 공개 shard는 원래 Cu node/element/회로 ID, 연결 관계, 좌표, 면적/체적, P3의8위상 A/B/total J와 전체 시간 scalar/current 파형을 포함한다. float64 NPZ 압축은 무손실이다. sparse8의 평균은 Full period loss label로 사용하지 않는다.']
    for case,d in data.items():
        s=d['shard'];text+=['',f"{case}:8위상 J 재적분과 원 scalar의 최대 차이{s['Joule_snapshot_reintegration_max_error_W']:.3g}W, current 차이{s['current_snapshot_reintegration_max_error_A']:.3g}A. 8위상만 평균하면 AC 평균이 전체 P3와{s['sparse8_vs_full_AC_mean_difference_percent']:.4f}% 다르므로 전체 파형을 함께 제공한다."]
    text+=['','## 실행 이력과 한계','',
        '첫8스텝 pilot은 MPI 초기화 단계에서 멈췄고 물리 step을 계산하지 않았다. 원본 실패 기록을 보존하고 single-rank MPI 환경을 단독 검증한 뒤 새 root에서 pilot8을 통과했다. 이후 생산 case는3개로 한정했고 재시도하지 않았다. 실제 속도·메모리 측정 후 CPU영역을 분리해 실행했으며 누적 case 경과시간의 최종 상한은17,000초다. 원래 주행 프로세스는 중단·변경하지 않았고 이 작업의 coordinator만 소유 PID 확인 후 교체했다.', '',
        '후속 독립 템플릿, SC/HalfSC 고부하와 저전류, 모든 운전점의 수렴은 남아 있다. 과거 HalfSC의 period_convergence_qualified=false와 mesh_convergence_qualified=false는 그대로다. 현재 원시 검증이 완료된 canonical MS/Full 대응41개 고유 조건에 이 동일 앵커의 수치 변형을 더해 표본 수를 늘리지 않는다.']
    (ROOT/'CONVERGENCE_REPORT_KO.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
