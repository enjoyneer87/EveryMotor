"""Saved-final width diagnostics only; candidate inputs require completed collection/replay.

The five numeric transforms are byte-preserving copies of the completed w3
reporting transforms; the current width judge remains the sole adoption rule.
"""
from pathlib import Path, PureWindowsPath
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import statistics
import platform
import sys

ROOT=Path(__file__).resolve().parent
FIELD_NAMES=('vector_nrmse_percent','magnitude_nrmse_percent','vector_rmse_T','magnitude_rmse_T')
PHASES=('all45','seen9','unused36','interior_unused','after_last_seen')
SEEDS=(17,29,43)
J=None

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def remote_key(path):return PureWindowsPath(str(path)).as_posix().casefold()

def describe_numeric(values):
    return J.distribution([J.number(values[str(seed)], 'distribution') for seed in SEEDS])

def compare_numeric_maps(control, candidate):
    need(set(control) == set(candidate) == {str(s) for s in SEEDS}, 'all three fixed seeds required')
    keys = set(control['17'])
    need(all(set(arm[str(s)]) == keys for arm in (control, candidate) for s in SEEDS), 'numeric summary key coverage differs')
    result = {}
    for metric in sorted(keys):
        a = {str(s): control[str(s)][metric] for s in SEEDS}
        b = {str(s): candidate[str(s)][metric] for s in SEEDS}
        result[metric] = dict(control=describe_numeric(a), candidate=describe_numeric(b),
            paired_control_minus_candidate=describe_numeric({str(s): a[str(s)]-b[str(s)] for s in SEEDS}),
            delta_units='percentage points' if metric.endswith('_percent') else 'same units as metric',
            delta_direction='positive means a decrease; signed bias is diagnostic and is not monotone accuracy')
    return result

def validate_snapshots(result):
    rows = result.get('snapshot_errors')
    if rows is None:
        return None
    keys = {(r['case'], r['step']) for r in rows}
    expected = {(int(c), step) for c in result['case_torque'] for step in range(45)}
    need(len(rows) == 1440 and len(keys) == len(rows) and keys == expected, 'snapshot case/step coverage differs')
    for row in rows:
        need(row['phase_scope'] == ('seen9' if row['step'] % 5 == 0 else 'unused36'), 'historical snapshot phase label differs')
        for metric in ('vector_nrmse_percent', 'magnitude_nrmse_percent'):
            J.number(row[metric], 'snapshot/'+metric, nonnegative=True)
    return rows

def phase_steps(phase):
    return {'all45': set(range(45)), 'seen9': set(range(0,45,5)),
            'unused36': {s for s in range(45) if s % 5},
            'interior_unused': {s for s in range(1,40) if s % 5},
            'after_last_seen': set(range(41,45))}[phase]

def numeric_metrics(result):
    """Saved pooled metrics plus explicitly named arithmetic diagnostics."""
    values, optional = {}, {}
    for phase in PHASES:
        for region in ('all', 'a1', 'copper', 'PA_ge_70/all', 'PA_ge_70/a1', 'PA_ge_70/copper'):
            key = f'{phase}/{region}/area'
            if key in result['field_metrics']:
                for name in FIELD_NAMES:
                    values[f'field/area_pooled/{phase}/{region}/{name}'] = result['field_metrics'][key][name]
        for region in ('all', 'a1', 'copper'):
            pa = result.get('PA_mean_case_metrics', {}).get(f'{phase}/PA_ge_70/{region}')
            if pa:
                ids = pa['case_ids']
                need(ids and len(set(ids)) == len(ids) == pa['case_count'], 'PA case membership invalid')
                for metric in FIELD_NAMES:
                    mean = statistics.fmean(result['field_metrics'][f'{phase}/case:{c}/{region}/area'][metric] for c in ids)
                    values[f'field/PA_ge70_case_arithmetic/{phase}/{region}/{metric}'] = mean
                    if metric == 'vector_nrmse_percent':
                        J.close(mean, pa['mean_case_Bvector_nrmse_percent'], 'stored PA case arithmetic')
    b2 = result.get('copper_B2')
    optional['copper_B2'] = 'PROVIDED' if b2 else 'NOT_PROVIDED'
    for phase, row in (b2 or {}).items():
        den = J.number(row['weighted_reference_T2_m2'], 'CuB2 reference', positive=True)
        mae = 100*J.number(row['weighted_absolute_error_T2_m2'], 'CuB2 MAE numerator', nonnegative=True)/den
        bias = 100*J.number(row['weighted_signed_error_T2_m2'], 'CuB2 signed numerator')/den
        J.close(mae, row['area_weighted_relative_MAE_percent'], 'CuB2 MAE')
        J.close(bias, row['signed_bias_percent'], 'CuB2 bias')
        values[f'copper_B2/{phase}/area_weighted_relative_MAE_percent'] = mae
        values[f'copper_B2/{phase}/signed_bias_percent'] = bias
        values[f'copper_B2/{phase}/absolute_signed_bias_percent'] = abs(bias)
    conductors = result.get('conductor_B2')
    optional['native_conductor_max'] = {}
    if conductors:
        seen = set()
        for row in conductors:
            key = row['case'], row['native_reg_code'], row['phase_scope']
            need(key not in seen, 'duplicate native conductor/phase')
            seen.add(key)
            ref = J.number(row['true_spatial_mean_B2_T2'], 'conductor B2 reference', positive=True)
            bias = 100*(row['pred_spatial_mean_B2_T2']-ref)/ref
            J.close(bias, row['B2_signed_relative_percent'], 'native conductor B2 bias')
        for phase in PHASES:
            selected = [r for r in conductors if r['phase_scope'] == phase]
            if selected:
                worst = max(selected, key=lambda r: abs(r['B2_signed_relative_percent']))
                values[f'native_conductor/{phase}/max_absolute_B2_bias_percent'] = abs(worst['B2_signed_relative_percent'])
                optional['native_conductor_max'][phase] = dict(case=worst['case'], native_reg_code=worst['native_reg_code'],
                    signed_B2_bias_percent=worst['B2_signed_relative_percent'])
    else:
        optional['native_conductor_max'] = {'status': 'NOT_PROVIDED'}
    pooled = result['pooled_torque']
    for metric in ('rmse_Nm', 'truth_rms_Nm', 'nrmse_percent'):
        values['torque/pooled/'+metric] = pooled[metric]
    for case, row in result['case_torque'].items():
        mean_signed = row['pred_mean_Nm']-row['truth_mean_Nm']
        ripple_signed = row['pred_ripple_Nm']-row['truth_ripple_Nm']
        metrics = dict(row)
        metrics.update(mean_error_signed_Nm=mean_signed, mean_error_abs_Nm=abs(mean_signed),
                       ripple_error_signed_Nm=ripple_signed, ripple_error_abs_Nm=abs(ripple_signed),
                       ripple_error_abs_truth_rms_normalized_percent=100*abs(ripple_signed)/row['truth_rms_Nm'])
        for metric in ('rmse_Nm', 'truth_rms_Nm', 'nrmse_percent', 'pred_mean_Nm', 'truth_mean_Nm',
                       'mean_error_signed_Nm', 'mean_error_abs_Nm', 'mean_error_rms_normalized_percent',
                       'pred_ripple_Nm', 'truth_ripple_Nm', 'ripple_error_signed_Nm', 'ripple_error_abs_Nm',
                       'ripple_error_abs_truth_rms_normalized_percent'):
            values[f'torque/case:{case}/'+metric] = metrics[metric]
    for metric, field in (('mean_case_abs_mean_error_Nm', 'mean_error_abs_Nm'),
                          ('mean_case_abs_ripple_error_Nm', 'ripple_error_abs_Nm'),
                          ('mean_case_nrmse_percent', 'nrmse_percent'),
                          ('mean_case_ripple_error_abs_truth_rms_normalized_percent', 'ripple_error_abs_truth_rms_normalized_percent')):
        values['torque/case_arithmetic/'+metric] = statistics.fmean(values[f'torque/case:{case}/'+field] for case in result['case_torque'])
    snapshots = validate_snapshots(result)
    optional['snapshot_diagnostics'] = 'PROVIDED' if snapshots else 'NOT_PROVIDED'
    if snapshots:
        groups = {f'phase_bin/{phase}': [r for r in snapshots if r['step'] in phase_steps(phase)] for phase in PHASES}
        groups.update({f'step:{step}': [r for r in snapshots if r['step'] == step] for step in range(45)})
        optional['snapshot_group_counts'] = {key: len(rows) for key, rows in groups.items()}
        for group, rows in groups.items():
            need(rows, 'empty snapshot group')
            for metric in ('vector_nrmse_percent', 'magnitude_nrmse_percent'):
                values[f'snapshot_case_arithmetic/{group}/'+metric] = statistics.fmean(r[metric] for r in rows)
    return values, optional


def main():
    global J
    parser=argparse.ArgumentParser();parser.add_argument('--collection',type=Path,required=True)
    args=parser.parse_args();directory=args.collection.resolve()
    execution_path=ROOT/'FINAL_SUMMARY_EXECUTION_CONTRACT.json'
    execution=read(execution_path)
    need(execution['status']=='RELEASED_SAVED_FINAL_DIAGNOSTIC_REPLAY','explicit final-summary source/input contract required')
    need(execution['summarizer_sha256']==sha(Path(__file__).resolve()),'summarizer source differs from execution contract')
    need(execution['collection_receipt_sha256']==sha(directory/'COLLECTION_RECEIPT.json'),'summary collection differs from execution contract')
    collection=read(directory/'COLLECTION_RECEIPT.json')
    need(collection['status']=='COMPLETE_COLLECTED_AND_REPLAYED','all final runs and verified collection required before any candidate reads')
    replay_pin=collection['replay_receipt_pin'];need(sha(replay_pin['path'])==replay_pin['sha256'],'replay receipt hash differs')
    replay=read(replay_pin['path']);need(replay['status']=='PASS_INDEPENDENT_JSON_REPLAY','frozen judgment replay required')
    rows={remote_key(x['remote']['path']):x for x in collection['files']}

    def saved(pin):
        row=rows[remote_key(pin['path'])];path=Path(row['local']['path']).resolve()
        need(path.is_relative_to(directory),'collection path outside owned return')
        need(sha(path)==row['local']['sha256']==row['remote']['sha256']==pin['sha256'],'source/local/frozen pin differs')
        need(path.stat().st_size==row['local']['bytes']==row['remote']['bytes']==pin['bytes'],'source/local/frozen size differs')
        return read(path),dict(row['local'])

    verdict_row=next(x for x in collection['files'] if remote_key(x['remote']['path']).endswith('/width128_verdict.json'))
    verdict,verdict_pin=saved(verdict_row['remote'])
    need(verdict['status']=='COMPLETE' and verdict['control_arm']=='saved_W64_phase45'
         and verdict['candidate_arm']=='W128_phase45_equal88565' and verdict['fixed_seeds']==list(SEEDS),
         'wrong or incomplete final verdict identity')
    replay_verdict=replay['server_verdict_pin']
    need(Path(replay_verdict['path']).resolve()==Path(verdict_pin['path']).resolve()
         and replay_verdict['sha256']==verdict_pin['sha256'] and replay_verdict['bytes']==verdict_pin['bytes'],
         'independent replay is not bound to this exact collected verdict')
    judge_path=ROOT/'judge_width128.py';need(sha(judge_path)==verdict['judge_source_sha256'],'current frozen judge source differs')
    spec=importlib.util.spec_from_file_location('final_width_frozen_judge',judge_path)
    J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
    need(replay['judge_source']['sha256']==sha(judge_path),'independent replay used different judge source')
    numeric={role:{} for role in ('control','candidate')};optional={role:{} for role in numeric}
    learning={role:{} for role in numeric};torque={role:{} for role in numeric};costs={role:{} for role in numeric};pins={}
    selection=read(ROOT/'CONTOUR_SELECTION.json');need(sha(ROOT/'CONTOUR_SELECTION.json')=='264e59c849668cfec86766c324ee6ab3e56ab526aeb7733da22579732b1fc116','fixed contour selection differs')
    cases={x['case_index'] for x in selection['conditions']};need(cases=={208,408,49},'fixed operating conditions changed')
    for role,arm in (('control','W64'),('candidate','W128')):
        for seed in SEEDS:
            key=str(seed)
            ev,ev_pin=saved(verdict['evidence_pins']['results'][role][key])
            train,tr_pin=saved(verdict['evidence_pins']['training_results'][role][key])
            need(ev['status']==train['status']=='COMPLETED' and train['training_updates']==88565 and ev['graphs_processed']==1440,'incomplete seed')
            need(ev['seed']==train['seed']==seed and ev['arm']==train['arm']==arm,'wrong seed/arm')
            expected_experiment='W64_PHASE45_EQUAL_UPDATE_20261030' if role=='control' else 'W128_PHASE45_EQUAL_UPDATE_20261030'
            need(ev['experiment_id']==train['experiment_id']==expected_experiment
                 and ev['data_arm']==train['data_arm']=='A45','wrong frozen data or experiment identity')
            need(ev['reproduction_gate']['status']=='PASS_ARITHMETIC_ONLY' and ev['original_train_sha256']==tr_pin['sha256'],'train/evaluation lineage differs')
            numeric[role][key],optional[role][key]=numeric_metrics(ev)
            for name,value in J.extract_metrics(ev).items():
                J.close(value,verdict['metrics'][name][role]['by_seed'][key],'registered result binding '+name)
            learning[role][key]=[{k:m[k] for k in ('epoch','training_updates','learning_rate','train_phase9_fixed','development')}
                                 for m in train['metrics']]
            need(len(learning[role][key])==6 and learning[role][key][-1]['training_updates']==88565,'stored learning anchors incomplete')
            selected=[x for x in ev['torque_samples'] if x['case'] in cases]
            need(len(selected)==135 and {(x['case'],x['step']) for x in selected}=={(c,s) for c in cases for s in range(45)},'fixed torque coverage differs')
            torque[role][key]=selected
            costs[role][key]=dict(training_worker_elapsed_seconds=train['elapsed_seconds'],
                                   training_assessment_wall_seconds=train['assessment_wall_seconds_total'],
                                   evaluation_worker_elapsed_seconds=ev['elapsed_seconds'],
                                   training_updates=88565,evaluation_graphs=1440,
                                   note='worker wall time; equal updates are not equal GPU cost; no FEM speedup claim')
            pins[f'{role}/{key}/train']=tr_pin;pins[f'{role}/{key}/eval']=ev_pin
    metrics=compare_numeric_maps(numeric['control'],numeric['candidate'])
    result=dict(status='COMPLETE_SAVED_FINAL_WIDTH_DIAGNOSTICS',created_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                summarizer_sha256=sha(Path(__file__).resolve()),summary_execution_contract_sha256=sha(execution_path),
                scope={'reference':'W64/full45/w1/alpha0','candidate':'W128/full45/w1/alpha0','single_changed_axis':'width',
                       'fixed_seeds':list(SEEDS),'final_updates_each':88565,'development_conditions':32,'eval_steps':45,
                       'eval_graphs_per_seed':1440,'validation':'exposed inner development, not independent test',
                       'phase_names':'seen9/unused36 are historical labels; all45 were used in training',
                       'uncertainty':'3-seed min/max/range, not a confidence interval','physical_BC_frame_pair':'UNKNOWN'},
                registered_judgment_replay=verdict,metrics=metrics,metric_count=len(metrics),optional=optional,
                learning=learning,fixed_contour_torque_samples=torque,run_costs=costs,
                evidence_pins={'collection':dict(path=str(directory/'COLLECTION_RECEIPT.json'),sha256=sha(directory/'COLLECTION_RECEIPT.json')),
                               'replay':replay_pin,'verdict':verdict_pin,'results':pins,'judge':dict(path=str(judge_path),sha256=sha(judge_path)),
                               'diagnostic_transform_reuse_receipt':dict(path=str(ROOT/'SUMMARY_TRANSFORM_REUSE_RECEIPT.json'),sha256=sha(ROOT/'SUMMARY_TRANSFORM_REUSE_RECEIPT.json'))},
                analysis_origin={'hostname':platform.node(),'interpreter':sys.executable,'python':sys.version},
                new_training=0,new_inference=0,new_FEA=0,adoption_rule_changed=False)
    path=ROOT/'FINAL_WIDTH_DIAGNOSTICS.json'
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'metrics':len(metrics),'verdict':verdict['verdict'],'path':str(path)}))


if __name__=='__main__':main()
