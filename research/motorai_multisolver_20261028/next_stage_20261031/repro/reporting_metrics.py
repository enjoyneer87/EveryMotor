"""Exact reused saved-final scalar transforms; no dataset/model/checkpoint reads."""
import statistics
import saved_metrics as J
FIELD_NAMES=('vector_nrmse_percent','magnitude_nrmse_percent','vector_rmse_T','magnitude_rmse_T')
PHASES=('all45','seen9','unused36','interior_unused','after_last_seen')
SEEDS=(17,29,43)

def need(ok,message):
    if not ok:raise ValueError(message)

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
