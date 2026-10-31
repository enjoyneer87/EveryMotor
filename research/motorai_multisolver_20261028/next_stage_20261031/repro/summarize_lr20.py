"""Saved-final LR30/LR20 diagnostics, gated by completed collection and exact inputs."""
from pathlib import Path, PureWindowsPath
import argparse
import datetime as dt
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'main_training_visual_20261030_01a18cac'
NAS = Path('N:/MotorAI_NextLR20_20261031_01a18cac')
sys.path.insert(0, str(ROOT / 'src'))
CHAIN_SHA = '3fc6975ade9d535728a770210f72f1e413032ea7fbe1943e7540e1fb7558ca32'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(path):
    return PureWindowsPath(str(path)).as_posix().casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=Path, required=True)
    args = parser.parse_args()
    directory = args.collection.resolve()
    assert directory.is_relative_to(NAS.resolve())
    execution_path = ROOT / 'FINAL_SUMMARY_EXECUTION_CONTRACT.json'
    assert execution_path.stat().st_size <= 1024**2
    captured_execution = execution_path.read_bytes()
    execution = json.loads(captured_execution.decode('utf-8-sig'))
    assert execution['status'] == 'RELEASED_COMPLETE_LR20_DIAGNOSTIC_REPLAY'
    assert execution['summarizer_sha256'] == sha(Path(__file__))
    assert execution['transforms_sha256'] == sha(ROOT / 'reporting_metrics.py')
    assert execution['arithmetic_sha256'] == sha(ROOT / 'src/saved_metrics.py')
    assert execution['judge_sha256'] == sha(ROOT / 'src/judge_lr20.py')
    assert sha(ROOT / 'CHAIN_LR20.json') == execution['chain_sha256'] == CHAIN_SHA
    frozen_chain = json.loads((ROOT / 'CHAIN_LR20.json').read_text(encoding='utf-8'))
    assert frozen_chain['judge_arithmetic_sha256'] == execution['arithmetic_sha256']
    assert frozen_chain['judge_sha256'] == execution['judge_sha256']
    # Numeric modules may execute only after their frozen source hashes are admitted.
    import saved_metrics as arithmetic
    import reporting_metrics as transforms
    execution, execution_pin = arithmetic.load_json(execution_path, hashlib.sha256(captured_execution).hexdigest())
    collection, collection_pin = arithmetic.load_json(directory / 'COLLECTION_RECEIPT.json', execution['collection_receipt_sha256'])
    assert collection['status'] == 'COMPLETE_LR20_COLLECTED_AND_REPLAYED'
    assert collection['collector_sha256'] == execution['collector_sha256'] == sha(ROOT / 'collect_lr20_r2.py')
    assert collection['collection_execution_contract_sha256'] == sha(ROOT / 'FINAL_COLLECTION_EXECUTION_CONTRACT.json')
    replay_pin = collection['replay_receipt_pin']
    assert Path(replay_pin['path']).resolve() == directory / 'INDEPENDENT_REPLAY_RECEIPT.json'
    replay, actual_replay_pin = arithmetic.pinned(replay_pin)
    assert replay['status'] == 'PASS_EXACT_LR20_REPLAY_WITH_FULL_EVIDENCE_BINDINGS'
    assert replay['collector_sha256'] == execution['collector_sha256']
    assert execution['chain_sha256'] == sha(ROOT / 'CHAIN_LR20.json')
    chain, chain_pin = arithmetic.load_json(ROOT / 'CHAIN_LR20.json', execution['chain_sha256'])
    assert chain['proposal_sha256'] == sha(ROOT / 'NEXT_STAGE_PLAN.json') == execution['plan_sha256']
    source_by_name = {Path(p['path']).name: p for p in chain['source_files']}
    for name in ('judge_lr20.py', 'saved_metrics.py'):
        assert source_by_name[name]['sha256'] == sha(ROOT / 'src' / name)
    reuse, reuse_pin = arithmetic.load_json(ROOT / 'REPORT_TRANSFORM_REUSE_RECEIPT.json', execution['reuse_receipt_sha256'])
    assert reuse['status'] == 'PASS_EXACT_FIVE_TRANSFORM_SOURCE_REUSE'
    assert reuse['destination']['sha256'] == execution['transforms_sha256']

    old_receipt_path = OLD / 'completed_return_20261030_1615/COLLECTION_RECEIPT.json'
    old_collection, old_pin = arithmetic.load_json(old_receipt_path, execution['control_collection_sha256'])
    assert old_collection['status'] == 'COMPLETE_COLLECTED_AND_REPLAYED'
    rows = {}
    for owner, entries in ((directory, collection['files']), (old_receipt_path.parent.resolve(), old_collection['files'])):
        for row in entries:
            identity = key(row['remote']['path'])
            assert identity not in rows, 'duplicate remote evidence identity'
            local = Path(row['local']['path']).resolve()
            assert local.is_relative_to(owner)
            assert row['local']['sha256'] == row['remote']['sha256']
            assert row['local']['bytes'] == row['remote']['bytes']
            rows[identity] = row

    def saved(pin):
        row = rows[key(pin['path'])]
        assert pin['sha256'] == row['local']['sha256']
        if 'bytes' in pin:
            assert pin['bytes'] == row['local']['bytes']
        return arithmetic.load_json(Path(row['local']['path']).resolve(), pin['sha256'], row['local']['bytes'])

    verdict_path = directory / 'LR20_VERDICT.json'
    verdict, verdict_pin = arithmetic.load_json(verdict_path, collection['server_verdict_sha256'])
    assert replay['server_verdict_pin'] == verdict_pin
    assert replay['judge_sha256'] == execution['judge_sha256'] and replay['arithmetic_sha256'] == execution['arithmetic_sha256']
    assert verdict['status'] == 'COMPLETE' and verdict['fixed_seeds'] == [17, 29, 43]
    assert verdict['control_arm'] == 'saved_W128_LR30' and verdict['candidate_arm'] == 'W128_LR20_equal88565'
    assert verdict['chain_sha256'] == execution['chain_sha256'] and verdict['judge_sha256'] == execution['judge_sha256']
    assert verdict['scientific_receipt_sha256'] == sha(directory / 'runs/SCIENTIFIC_RECEIPT.json')
    selection_path = OLD / 'CONTOUR_SELECTION.json'
    selection, selection_pin = arithmetic.load_json(selection_path, execution['selection_sha256'])
    cases = {row['case_index'] for row in selection['conditions']}
    assert cases == {208, 408, 49}
    numeric = {role: {} for role in ('control', 'candidate')}
    optional = {role: {} for role in numeric}
    learning = {role: {} for role in numeric}
    torque = {role: {} for role in numeric}
    costs = {role: {} for role in numeric}
    pins = {}
    for role in numeric:
        for seed in (17, 29, 43):
            s = str(seed)
            bindings = verdict['evidence_pins'][role][s]
            train, train_pin = saved(bindings['train'])
            ev, eval_pin = saved(bindings['eval'])
            assert train['status'] == ev['status'] == 'COMPLETED'
            assert train['seed'] == ev['seed'] == seed and train['training_updates'] == 88565 and ev['graphs_processed'] == 1440
            expected = 'W128_PHASE45_EQUAL_UPDATE_20261030' if role == 'control' else 'W128_LR20_PHASE45_20261031'
            assert train['experiment_id'] == ev['experiment_id'] == expected
            assert train['width'] == ev['width'] == 128 and train['depth'] == ev['depth'] == 3
            assert train['data_arm'] == ev['data_arm'] == 'A45' and train['alpha'] == ev['alpha'] == 0
            assert train_pin['sha256'] == ev['original_train_sha256']
            assert train['checkpoint_sha256'] == ev['checkpoint_sha256'] and ev['reproduction_gate']['status'] == 'PASS_ARITHMETIC_ONLY'
            arithmetic.validate_field_statistics(ev)
            arithmetic.validate_torque(ev)
            numeric[role][s], optional[role][s] = transforms.numeric_metrics(ev)
            for name, value in arithmetic.extract_metrics(ev).items():
                arithmetic.close(value, verdict['metrics'][name][role]['by_seed'][s], 'registered binding ' + name)
            learning[role][s] = [{k: row[k] for k in ('epoch', 'training_updates', 'learning_rate', 'train_phase9_fixed', 'development')} for row in train['metrics']]
            assert [row['training_updates'] for row in learning[role][s]] == [5, 2957, 14765, 29525, 59045, 88565]
            torque[role][s] = [row for row in ev['torque_samples'] if row['case'] in cases]
            assert len(torque[role][s]) == 135 and {(row['case'], row['step']) for row in torque[role][s]} == {(c, step) for c in cases for step in range(45)}
            costs[role][s] = dict(training_worker_elapsed_seconds=train['elapsed_seconds'], training_assessment_wall_seconds=train['assessment_wall_seconds_total'], evaluation_worker_elapsed_seconds=ev['elapsed_seconds'], training_updates=88565, evaluation_graphs=1440)
            pins[role + '/' + s] = dict(train=train_pin, eval=eval_pin)
    metrics = transforms.compare_numeric_maps(numeric['control'], numeric['candidate'])
    assert len(metrics) == 703, 'diagnostic coverage changed; preserve failure rather than omitting metrics'
    result = dict(status='COMPLETE_LR20_SAVED_FINAL_DIAGNOSTICS', created_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                  summarizer_sha256=sha(Path(__file__)), transforms_sha256=execution['transforms_sha256'], execution_contract=execution_pin,
                  scope=dict(reference='W128/LR30/full45/w1/alpha0', candidate='W128/LR20/full45/w1/alpha0', single_changed_axis='cosine LR horizon30 to20', fixed_seeds=[17,29,43], final_updates_each=88565, eval_graphs_per_seed=1440, validation='exposed innerdev32, not independent test', uncertainty='min/max/range, not CI', native_BC_sign_frame_pair='UNKNOWN'),
                  registered_judgment_replay=verdict, metrics=metrics, metric_count=len(metrics), optional=optional, learning=learning,
                  fixed_contour_torque_samples=torque, run_costs=costs,
                  evidence_pins=dict(collection=collection_pin, replay=actual_replay_pin, control_collection=old_pin, verdict=verdict_pin, chain=chain_pin, results=pins, selection=selection_pin, transform_reuse=reuse_pin),
                  new_training=0, new_inference=0, new_FEA=0, adoption_rule_changed=False)
    for pin in (execution_pin, collection_pin, actual_replay_pin, old_pin, verdict_pin, chain_pin, reuse_pin, selection_pin):
        assert sha(pin['path']) == pin['sha256'], 'input changed during summary'
    output = directory / 'FINAL_LR20_DIAGNOSTICS.json'
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (ROOT / 'FINAL_DIAGNOSTICS_POINTER.json').open('x', encoding='utf-8') as f:
        json.dump(dict(path=str(output), sha256=sha(output), bytes=output.stat().st_size, status=result['status']), f, indent=2)
    print(json.dumps(dict(status=result['status'], metrics=len(metrics), verdict=verdict['verdict'], path=str(output))))


if __name__ == '__main__':
    main()
