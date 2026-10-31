"""Standard-library, JSON-only final W128 phase45 versus saved W64 phase45 innerdev judgment.

No import from frozen src, checkpoint loading, native arrays, inference, or retry.
Contract pins saved controls, CPU admission, evaluation template, chain-plan SHA,
chain receipt path, and each candidate result path. Candidate hashes come solely
from the completed scientific guard receipt. All three fixed seeds are required.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

SEEDS = (17, 29, 43)
PHASES = ('all45', 'seen9', 'unused36', 'interior_unused', 'after_last_seen')
IDENTITY_KEYS = ('selection_sha256', 'bundle_manifest_sha256', 'geometry_manifest_sha256', 'training_mask_manifest_sha256')
REQUIRED_FIELDS = {'primary_all45_Bvector': 'all45/all/area', 'seen9_Bvector': 'seen9/all/area',
                   'a1_Bvector': 'all45/a1/area', 'copper_Bvector': 'all45/copper/area'}
NO_HARM = {'seen9_Bvector': .5, 'a1_Bvector': 1., 'copper_Bvector': 1., 'PA_ge70_case_mean_Bvector': 1., 'pooled_torque': 1.}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, nonnegative=False, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), 'missing/nonfinite numeric '+name)
    require(not nonnegative or value >= 0, 'negative '+name)
    require(not positive or value > 0, 'nonpositive '+name)
    return float(value)


def close(left, right, label):
    require(math.isclose(number(left, label), number(right, label), rel_tol=1e-10, abs_tol=1e-10), 'arithmetic/reference mismatch '+label)


def all_finite(value, path='JSON'):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite '+path)
    elif isinstance(value, dict):
        for key, child in value.items():
            all_finite(child, path+'.'+key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            all_finite(child, path+f'[{index}]')


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_json(path, digest=None, byte_count=None):
    path = Path(path)
    require(path.is_absolute() and path.suffix.lower() == '.json', 'only absolute JSON paths are admitted')
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if digest is not None:
        require(valid_sha(digest) and actual == digest, 'JSON SHA mismatch '+path.name)
    if byte_count is not None:
        require(type(byte_count) is int and len(raw) == byte_count, 'JSON byte count mismatch '+path.name)
    def reject_constant(token):
        raise ValueError('nonfinite JSON token '+token)
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key '+key)
            result[key] = value
        return result
    payload = json.loads(raw.decode('utf-8-sig'), parse_constant=reject_constant, object_pairs_hook=unique_object)
    require(isinstance(payload, dict), 'JSON object required '+path.name)
    all_finite(payload)
    return payload, {'path': str(path.resolve()), 'sha256': actual, 'bytes': len(raw)}


def pinned(spec):
    return load_json(spec['path'], spec['sha256'], spec.get('bytes'))


def validate_field_statistics(result):
    metrics, stats = result['field_metrics'], result['field_sufficient_statistics']
    require(set(metrics) == set(stats) and metrics, 'field metric/statistic key set differs')
    for key, row in stats.items():
        count = row['count']
        require(type(count) is int and count > 0, 'invalid element-observation count '+key)
        weight = number(row['weight_sum'], key+' weight', positive=True)
        energy = number(row['truth_energy'], key+' truth energy', positive=True)
        vector = number(row['vector_sse'], key+' vector SSE', nonnegative=True)
        magnitude = number(row['magnitude_sse'], key+' magnitude SSE', nonnegative=True)
        summary = metrics[key]
        require(summary['count'] == count and summary['normalization_status'] == 'DEFINED', 'field metric count/normalization differs '+key)
        for name, value in (('vector_nrmse_percent', 100*math.sqrt(vector/energy)),
                            ('magnitude_nrmse_percent', 100*math.sqrt(magnitude/energy)),
                            ('vector_rmse_T', math.sqrt(vector/weight)), ('magnitude_rmse_T', math.sqrt(magnitude/weight))):
            close(summary[name], value, key+'/'+name)
    return stats


def validate_torque(result):
    samples, cases = result['torque_samples'], result['case_torque']
    require(len(cases) == 32 and len(samples) == 1440, 'torque32x45 coverage differs')
    by_case, seen = {}, set()
    for row in samples:
        case, step = row['case'], row['step']
        require(type(case) is int and type(step) is int and 0 <= step < 45, 'invalid native case/step')
        require((case, step) not in seen, 'duplicate torque case/step')
        seen.add((case, step))
        pred = number(row['pred_Nm'], 'pred torque')
        truth = number(row['truth_Nm'], 'truth torque')
        by_case.setdefault(case, []).append((step, pred, truth))
    require({str(case) for case in by_case} == set(cases), 'torque case identities differ')
    require(all({row[0] for row in rows} == set(range(45)) for rows in by_case.values()), 'torque step coverage differs')
    squared_error = math.fsum((row['pred_Nm']-row['truth_Nm'])**2 for row in samples)
    squared_truth = math.fsum(row['truth_Nm']**2 for row in samples)
    truth_rms, rmse = math.sqrt(squared_truth/1440), math.sqrt(squared_error/1440)
    require(truth_rms > 0, 'undefined pooled torque reference')
    pooled = result['pooled_torque']
    require(pooled['count'] == 1440 and pooled['normalization_status'] == 'DEFINED'
            and pooled['ripple'] == 'NOT_POOLED_ACROSS_CASES', 'pooled torque contract differs')
    for key, expected in (('rmse_Nm', rmse), ('truth_rms_Nm', truth_rms), ('nrmse_percent', 100*rmse/truth_rms)):
        close(pooled[key], expected, 'pooled_torque/'+key)
    for case, rows in by_case.items():
        require(cases[str(case)]['count'] == 45 and cases[str(case)]['normalization_status'] == 'DEFINED', 'case torque normalization differs')
        p, t = [r[1] for r in rows], [r[2] for r in rows]
        trms = math.sqrt(math.fsum(x*x for x in t)/45)
        require(trms > 0, 'undefined case torque reference')
        err = math.sqrt(math.fsum((a-b)**2 for a,b in zip(p,t))/45)
        expected = {'rmse_Nm': err, 'truth_rms_Nm': trms, 'nrmse_percent': 100*err/trms,
                    'pred_mean_Nm': statistics.fmean(p), 'truth_mean_Nm': statistics.fmean(t),
                    'mean_error_rms_normalized_percent': 100*abs(statistics.fmean(p)-statistics.fmean(t))/trms,
                    'pred_ripple_Nm': max(p)-min(p), 'truth_ripple_Nm': max(t)-min(t)}
        for key, value in expected.items():
            close(cases[str(case)][key], value, f'case{case}/'+key)
    return by_case


def extract_metrics(result):
    values = {label: number(result['field_metrics'][key]['vector_nrmse_percent'], key, nonnegative=True)
              for label, key in REQUIRED_FIELDS.items()}
    pa = result['PA_mean_case_metrics']['all45/PA_ge_70/all']
    require(pa['normalization_status'] == 'DEFINED' and type(pa['case_count']) is int and pa['case_count'] > 0
            and pa['case_count'] == len(pa['case_ids']) == len(set(pa['case_ids'])), 'PA>=70 case group undefined/duplicate')
    require(set(map(str, pa['case_ids'])) <= set(result['case_torque']), 'PA>=70 cases outside admitted32')
    expected = statistics.fmean(result['field_metrics'][f'all45/case:{case}/all/area']['vector_nrmse_percent'] for case in pa['case_ids'])
    close(pa['mean_case_Bvector_nrmse_percent'], expected, 'PA>=70 mean-of-case metric')
    values['PA_ge70_case_mean_Bvector'] = number(pa['mean_case_Bvector_nrmse_percent'], 'PA>=70', nonnegative=True)
    values['pooled_torque'] = number(result['pooled_torque']['nrmse_percent'], 'pooled torque', nonnegative=True)
    return values


def identity(result, arm, seed, expected, cpu):
    require(result['status'] == 'COMPLETED' and result['arm'] == arm and result['seed'] == seed
            and result['id'] == f'{arm}_s{seed}', 'result identity incomplete/mismatched')
    require(result['graphs_processed'] == 1440 and result['training_updates'] == 0
            and result['new_FEA'] == 0 and result['new_prior_solves'] == 0, 'result work scope/coverage differs')
    require(result['phase_encoding'] == 'mechanical' and type(result['alpha']) in (int,float) and result['alpha'] == 0., 'encoding/loss differs')
    require(result['reproduction_gate']['status'] == 'PASS_ARITHMETIC_ONLY', 'checkpoint reproduction not admitted')
    require(valid_sha(result['checkpoint_sha256']), 'checkpoint SHA missing')
    for key, digest in expected.items():
        require(valid_sha(digest) and result[key] == digest, 'data/operator identity mismatch '+key)
    width_arm = arm
    if arm in ('W64','W128'):
        require(result['width'] == (128 if arm=='W128' else 64) and result['depth'] == 3, 'candidate width/depth differs')
    hashes = result['metadata_hashes']
    require(hashes['selection_sha256'] == expected['selection_sha256']
            and hashes['mask_manifest_sha256'] == expected['training_mask_manifest_sha256']
            and hashes['bundle_manifest_sha256'] == cpu['bundle_manifest_sha256'], 'training data/selection/mask binding differs')
    for key, expected_value in (('initial_tensor_state_sha256', cpu['widths'][width_arm]['initial_state_sha256_by_seed'][str(seed)]),
        ('data_order_LR_trace_sha256', cpu['widths'][width_arm]['expected_data_order_LR_trace_sha256']['88565'][str(seed)])):
        require(hashes[key] == expected_value and valid_sha(expected_value), 'CPU initialization/order binding differs '+key)
    if arm in ('W64','W128'):
        sources = {Path(row['path']).name: row['sha256'] for row in cpu['source_files']}
        for key, filename in (('model_sha256', 'dual_solver_mgn.py'), ('trainer_sha256', 'train_width128.py' if arm=='W128' else 'train_phase45.py'),
                              ('loss_module_sha256', 'a1_allocation_loss.py'), ('capacity_contract_sha256', 'capacity_contract.py')):
            require(hashes[key] == sources[filename], 'candidate frozen source differs '+key)


def compare_reference(left, right):
    a, b = left['field_sufficient_statistics'], right['field_sufficient_statistics']
    require(a.keys() == b.keys(), 'field comparison support key set differs')
    for key in a:
        require(a[key]['count'] == b[key]['count'], 'element support count differs '+key)
        close(a[key]['weight_sum'], b[key]['weight_sum'], key+'/same area')
        close(a[key]['truth_energy'], b[key]['truth_energy'], key+'/same truth energy')
    require(left['PA_mean_case_metrics']['all45/PA_ge_70/all']['case_ids'] == right['PA_mean_case_metrics']['all45/PA_ge_70/all']['case_ids'], 'PA case group changes across models/seeds')
    truths = lambda r: {(row['case'], row['step']): row['truth_Nm'] for row in r['torque_samples']}
    at, bt = truths(left), truths(right)
    require(at.keys() == bt.keys(), 'torque reference case/step support differs')
    for key in at:
        close(at[key], bt[key], 'same reference torque '+str(key))


def distribution(values):
    require(len(values) == 3, 'exactly three fixed seeds required')
    return {'mean': statistics.fmean(values), 'min': min(values), 'max': max(values), 'range': max(values)-min(values),
            'by_seed': {str(seed): value for seed, value in zip(SEEDS, values)}}


def diagnostics(result):
    case_torque = result['case_torque']
    return {'field_metrics_by_phase_region': {key: result['field_metrics'][key] for phase in PHASES for region in ('all','a1','copper')
                for key in (f'{phase}/{region}/area',) if key in result['field_metrics']},
            'copper_B2': result.get('copper_B2', {'status': 'NOT_PROVIDED'}),
            'case_torque': case_torque,
            'mean_case_abs_mean_torque_error_Nm': statistics.fmean(abs(r['pred_mean_Nm']-r['truth_mean_Nm']) for r in case_torque.values()),
            'mean_case_abs_ripple_error_Nm': statistics.fmean(abs(r['pred_ripple_Nm']-r['truth_ripple_Nm']) for r in case_torque.values()),
            'mean_case_torque_nrmse_percent': statistics.fmean(r['nrmse_percent'] for r in case_torque.values())}


def validate_training_identity(train,result,seed,cpu,role):
    arm,width,params,field,experiment=('W128',128,1079430,466434,'W128_PHASE45_EQUAL_UPDATE_20261030') if role=='candidate' else ('W64',64,273478,118530,'W64_PHASE45_EQUAL_UPDATE_20261030')
    require(train['status']=='COMPLETED' and train['arm']==arm and train['seed']==seed and train['width']==width and train['depth']==3
            and train['training_updates']==88565 and train['parameters']==params and train['field_path_parameters']==field
            and train['alpha']==0 and train['checkpoint_sha256']==result['checkpoint_sha256'],'training result identity/checkpoint differs')
    require(train['train_steps']==list(range(45)) and train['data_arm']==result['data_arm']=='A45','training phase support differs')
    require(train['old_checkpoint_reused'] is False and train['new_FEA']==train['new_prior_solves']==train['original_val_test_used']==0,'training source scope differs')
    for key in ('initial_tensor_state_sha256','data_order_LR_trace_sha256','plan_sha256','selection_sha256','bundle_manifest_sha256','mask_manifest_sha256','model_sha256','trainer_sha256','loss_module_sha256','capacity_contract_sha256'):
        require(valid_sha(train[key]) and train[key]==result['metadata_hashes'][key],'training/eval metadata differs '+key)
    require(train['experiment_id']==result['experiment_id']==experiment and train['train_graphs']==14760
            and train['train_phase9_diagnostic_graphs']==2952,'all45 experiment/graph identity differs')
    require(train['metrics'][-1]['epoch']==30 and train['metrics'][-1]['training_updates']==88565
            and train['metrics'][-1]['development']['all']['snapshots']==288,'training final assessment incomplete')


def judge(contract, chain, loaded_results, cpu, template, phase45_cpu, loaded_trains):
    require(contract['fixed_seeds'] == list(SEEDS), 'fixed seeds differ')
    require(valid_sha(contract['chain_plan_sha256']), 'invalid frozen chain-plan SHA')
    require(chain['status'] == 'COMPLETED' and chain['phase'] == 'scientific'
            and chain['chain_plan_sha256'] == contract['chain_plan_sha256'], 'scientific chain incomplete/mismatched')
    require(chain['full_training_updates'] == 265695 and chain['evaluation_graphs'] == 4320, 'scientific chain totals incomplete')
    require(cpu['status'] == phase45_cpu['status'] == 'PASS', 'CPU admission incomplete')
    require(cpu['widths']['W128']['expected_data_order_LR_trace_sha256']==phase45_cpu['widths']['W64']['expected_data_order_LR_trace_sha256'],'W64/W128 data order/LR differs')
    require(cpu['reference_W64']['initial_state_sha256_by_seed']==phase45_cpu['widths']['W64']['initial_state_sha256_by_seed']
            and cpu['reference_W64']['bridge']['status']=='PASS_BASELINE_BRIDGE_CPU_ONLY','W64 baseline CPU bridge differs')
    expected = {'selection_sha256': template['selection']['sha256'], 'bundle_manifest_sha256': template['bundle_manifest_sha256'],
                'geometry_manifest_sha256': template['geometry_manifest']['sha256'],
                'training_mask_manifest_sha256': template['training_mask_manifest']['sha256']}
    eval_rows = [row for row in chain['jobs'] if row['kind'] == 'eval']
    train_rows = [row for row in chain['jobs'] if row['kind'] == 'train']
    require(len(eval_rows) == len(train_rows) == 3 and len(chain['jobs']) == 6, 'scientific job count differs')
    for kind_rows in (eval_rows, train_rows):
        require({row['seed'] for row in kind_rows} == set(SEEDS), 'scientific seed coverage differs')
        for row in kind_rows:
            require(row['status'] == row['worker_status'] == 'COMPLETED' and row['arm'] == 'W128' and row['width'] == 128,
                    'scientific worker identity/final status differs')
    metrics, details = {}, {}
    reference = None
    for role, arm in (('control','W64'), ('candidate','W128')):
        metrics[role], details[role] = {}, {}
        for seed in SEEDS:
            result = loaded_results[role][str(seed)]
            identity(result, arm, seed, expected, cpu if role=='candidate' else phase45_cpu)
            validate_field_statistics(result); validate_torque(result)
            training=loaded_trains[role][str(seed)]
            validate_training_identity(training,result,seed,cpu if role=='candidate' else phase45_cpu,role)
            if role == 'candidate':
                row = next(r for r in eval_rows if r['seed'] == seed)
                require(result['job_sha256'] == row['job_sha256'] and result['checkpoint_sha256'] == row['checkpoint_sha256'],
                        'candidate checkpoint/job differs from completed guard')
                train_row = next(r for r in train_rows if r['seed'] == seed)
                require(train_row['checkpoint_sha256'] == row['checkpoint_sha256'] and train_row['training_updates'] == 88565,
                        'candidate training/eval checkpoint differs')
                require(train_row['plan_sha256']==training['plan_sha256'],'candidate plan differs from guard receipt')
            if reference is not None:
                compare_reference(reference, result)
            else:
                reference = result
            metrics[role][str(seed)] = extract_metrics(result)
            details[role][str(seed)] = diagnostics(result)
    summary = {}
    for metric in next(iter(metrics['control'].values())):
        control = distribution([metrics['control'][str(seed)][metric] for seed in SEEDS])
        candidate = distribution([metrics['candidate'][str(seed)][metric] for seed in SEEDS])
        summary[metric] = {'control': control, 'candidate': candidate, 'improvement_mean_pp': control['mean']-candidate['mean'],
            'worsening_mean_pp': candidate['mean']-control['mean'], 'max_arm_range_pp': max(control['range'],candidate['range']),
            'paired_control_minus_candidate_pp': {str(seed): metrics['control'][str(seed)][metric]-metrics['candidate'][str(seed)][metric] for seed in SEEDS}}
    primary = summary['primary_all45_Bvector']
    primary_gate = {'mean_improvement_at_least_1pp': primary['improvement_mean_pp'] >= 1.,
                    'mean_improvement_strictly_gt_2max_range': primary['improvement_mean_pp'] > 2*primary['max_arm_range_pp']}
    if 'primary_strict_min_improvement_pp' in contract:
        bound=number(contract['primary_strict_min_improvement_pp'],'frozen W64 phase45 range bound',nonnegative=True)
        close(bound,2*primary['control']['range'],'frozen control-only bound')
        primary_gate['mean_improvement_strictly_gt_frozen_W64_phase45_bound']=primary['improvement_mean_pp']>bound
    noharm = {metric: {'maximum_worsening_pp': threshold, 'actual_worsening_pp': summary[metric]['worsening_mean_pp'],
                      'pass': summary[metric]['worsening_mean_pp'] <= threshold} for metric, threshold in NO_HARM.items()}
    passed = all(primary_gate.values()) and all(row['pass'] for row in noharm.values())
    torque = summary['pooled_torque']
    return {'status': 'COMPLETE', 'verdict': 'IMPROVEMENT_CLAIM_WITHIN_INNERDEV' if passed else 'NO_IMPROVEMENT_CLAIM',
            'fixed_seeds': list(SEEDS), 'control_arm': 'saved_W64_phase45', 'candidate_arm': 'W128_phase45_equal88565',
            'primary_gates': primary_gate, 'no_harm_gates': noharm, 'metrics': summary,
            'numerical_primary_direction': 'LOWER_MEAN_ERROR' if primary['improvement_mean_pp']>0 else ('HIGHER_MEAN_ERROR' if primary['improvement_mean_pp']<0 else 'EQUAL_MEAN_ERROR'),
            'scientific_gates_passed': passed,
            'separate_torque_diagnostic_gate': {'mean_improvement_at_least_1pp': torque['improvement_mean_pp'] >= 1.,
                'mean_improvement_strictly_gt_2max_range': torque['improvement_mean_pp'] > 2*torque['max_arm_range_pp'],
                'scope': 'model innerdev torque diagnostic; no physical torque qualification'},
            'diagnostics_by_role_seed': details, 'data_operator_identity_sha256': expected,
            'range_interpretation': 'three-seed range is neither a confidence interval nor a significance test',
            'training_comparison': 'equal88565 optimizer updates and identical all45 support/order/LR; width64 versus width128; both six passes',
            'phase_scope_interpretation': 'seen9/unused36/interior_unused/after_last_seen label historical phase9 scopes; every45 phase is trained in candidate',
            'scope': 'same E1-oldTRAIN-exposed innerdev32; no independent-test, AC-loss, cross-solver, or physical-validation claim',
            'control_replacement': 'NONE; fixed saved W64 phase45 controls retained; all three candidates included',
            'new_training_updates': 0, 'GPU_inference': 0, 'new_FEA': 0, 'binary_payloads_read': 0}


def run(contract_path, contract_sha, chain_path):
    contract, contract_pin = load_json(contract_path, contract_sha)
    require(Path(contract['chain_receipt_path']).resolve() == Path(chain_path).resolve(), 'scientific receipt path differs from contract')
    require(set(contract['control_results_by_seed']) == set(contract['candidate_results_by_seed']) == {str(seed) for seed in SEEDS}, 'contract seed mapping differs')
    chain, chain_pin = load_json(chain_path)
    cpu, cpu_pin = pinned(contract['cpu_admission'])
    phase45_cpu,phase45_cpu_pin=pinned(contract['phase45_cpu_admission'])
    w32_context,w32_context_pin=pinned(contract['original_w32_context'])
    template, template_pin = pinned(contract['evaluation_template'])
    loaded, result_pins = {'control': {}, 'candidate': {}}, {'control': {}, 'candidate': {}}
    loaded_trains,train_pins={'control':{},'candidate':{}},{'control':{},'candidate':{}}
    for seed in SEEDS:
        key = str(seed)
        loaded['control'][key], result_pins['control'][key] = pinned(contract['control_results_by_seed'][key])
        loaded_trains['control'][key],train_pins['control'][key]=pinned(contract['control_train_results_by_seed'][key])
        train_rows=[r for r in chain.get('jobs',[]) if r.get('kind')=='train' and r.get('seed')==seed]
        require(len(train_rows)==1,'candidate training receipt missing/duplicate')
        loaded_trains['candidate'][key],train_pins['candidate'][key]=pinned(train_rows[0]['result'])
        rows = [r for r in chain.get('jobs', []) if r.get('kind') == 'eval' and r.get('seed') == seed]
        require(len(rows) == 1, 'completed candidate receipt missing/duplicate seed'+key)
        result_spec = rows[0]['result']
        require(Path(result_spec['path']).resolve() == Path(contract['candidate_results_by_seed'][key]['path']).resolve(), 'candidate result path differs from contract')
        loaded['candidate'][key], result_pins['candidate'][key] = pinned(result_spec)
        if 'sha256' in contract['candidate_results_by_seed'][key]:
            require(result_pins['candidate'][key]['sha256'] == contract['candidate_results_by_seed'][key]['sha256'], 'contract candidate SHA differs')
    result = judge(contract, chain, loaded, cpu, template,phase45_cpu,loaded_trains)
    require(w32_context['status']=='COMPLETE','original W32 context incomplete')
    result['original_w32_report_context']={'status':'REPORT_ONLY_NO_ACCEPTANCE_GATE','source_verdict':w32_context['verdict'],'metrics':w32_context['metrics']}
    if contract.get('phase45_context'):
        context,context_pin=pinned(contract['phase45_context'])
        require(context['status']=='COMPLETE','prior phase45 context incomplete')
        result['prior_phase45_report_context']={'status':'REPORT_ONLY_NO_ACCEPTANCE_GATE','source_verdict':context['verdict'],'metrics':context['metrics'],'pin':context_pin}
    result['evidence_pins'] = dict(contract=contract_pin, scientific_receipt=chain_pin, cpu_admission=cpu_pin,
                                 evaluation_template=template_pin, results=result_pins,training_results=train_pins,phase45_cpu_admission=phase45_cpu_pin,original_w32_context=w32_context_pin)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--chain-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.out.is_absolute() and args.out.suffix.lower() == '.json' and not args.out.exists(), 'new absolute output JSON required')
    try:
        result = run(args.contract, args.contract_sha256, args.chain_receipt)
        exit_code = 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        result = {'status': 'INCOMPLETE', 'verdict': 'INCOMPLETE_NO_CLAIM', 'error': str(exc),
                  'control_replacement': 'NONE; no implicit baseline replacement', 'new_training_updates': 0,
                  'GPU_inference': 0, 'new_FEA': 0, 'binary_payloads_read': 0}
        exit_code = 1
    result.update(judged_UTC=dt.datetime.now(dt.timezone.utc).isoformat(), judge_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  requested_contract_sha256=args.contract_sha256, python=sys.executable)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'verdict': result['verdict'], 'output': str(args.out)}))
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
