"""CPU W128 admission with a pinned full-source/synthetic bridge to saved W64."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import socket
import sys
import time

import numpy as np
from a1_allocation_loss import allocation_loss, learning_rate, require, tensor_state_sha256, trace_update
from capacity_contract import FIELD_PREFIXES, SEEDS, WIDTHS, parameter_counts, validate_model_parameters
from baseline_bridge import source_equivalence, synthetic_source_parity, fingerprint, functions

WIDTHS = {'W128': 128}
EXPERIMENT = 'W128_PHASE45_EQUAL_UPDATE_20261030'

STEPS = (0, 5, 10, 15, 20, 25, 30, 35, 40)
LR = {'start': .001, 'end': .00001}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_metadata(path, digest):
    path = Path(path)
    require(path.suffix.lower() == '.json' and sha(path) == digest, 'only hash-pinned JSON metadata is admitted')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def graph_names(manifest):
    require(manifest['reader_pass_cases'] == 360 and manifest['failed_cases'] == 0
            and len(manifest['records']) == 360, 'training bundle JSON contract differs')
    names, cases, groups = [], set(), set()
    for record in manifest['records']:
        require(record['case_index'] not in cases and record['original_split'] == 'train', 'duplicate/non-oldTRAIN metadata')
        cases.add(record['case_index'])
        if record['split'] != 'train':
            continue
        groups.add(record['geometry_group'])
        require(record['geometry_group'] not in ('R1_0199', 'R1_0076'), 'reserved metadata')
        selected = []
        for row in sorted(record['files'], key=lambda item: item['name']):
            name = row['name']
            require(Path(name).name == name and name.endswith('.npz'), 'unsafe graph name in JSON')
            step = int(Path(name).stem.split('_step')[1])
            require(int(Path(name).stem.split('_step')[0][4:]) == record['case_index'], 'graph-name native case differs')
            require(step in range(45), 'graph step outside all45')
            selected.append((step, name))
        require(len(selected) == 45 and {step for step, _ in selected} == set(range(45)), 'all45 names differ')
        names.extend(name for _, name in selected)
    require(len(groups) == 288 and len(names) == len(set(names)) == 14760, 'train-only all45 JSON name coverage differs')
    return names


def order_traces(names, seed, limits=(2957, 88565)):
    """Mirror smoke5 and the frozen Python shuffle/LR schedule without loading graphs."""
    require(len(names) == 14760 and len(set(names)) == 14760, 'frozen all45 name list differs')
    require(seed in SEEDS and all(type(n) is int and n >= 5 for n in limits), 'unregistered trace request')
    trace, result, count = hashlib.sha256(), {}, 0
    smoke_names = [name for name in names if int(Path(name).stem.split('_step')[1]) in STEPS][:5]
    require(len(smoke_names) == 5, 'historical smoke5 phase9 names missing')
    for name in smoke_names:
        count += 1
        trace_update(trace, count, name, LR['start'])
    if 5 in limits:
        result['5'] = trace.hexdigest()
    rng = random.Random(seed)
    while count < max(limits):
        order = list(names); rng.shuffle(order)
        for name in order:
            count += 1
            trace_update(trace, count, name, learning_rate(count, LR, 2952))
            if count in limits:
                result[str(count)] = trace.hexdigest()
            if count >= max(limits):
                break
    return result


def historical_initials(historical):
    require(historical['status'] == 'PASS' and historical['data_arm']=='A45'
            and historical['experiment_id']=='W64_PHASE45_EQUAL_UPDATE_20261030', 'prior W64 all45 CPU admission incomplete')
    return historical['widths']['W64']['initial_state_sha256_by_seed']


def pinned_source(historical,name):
    rows=[row for row in historical['source_files'] if Path(row['path']).name==name]
    require(len(rows)==1,'missing/duplicate historical source pin '+name)
    row=rows[0];path=Path(row['path'])
    require(path.suffix.lower()=='.py' and path.is_absolute() and sha(path)==row['sha256'], 'historical Python source SHA differs '+name)
    require('bytes' not in row or path.stat().st_size==row['bytes'],'historical source bytes differ '+name)
    return path,row


def make_model(width, seed):
    import torch
    from dual_solver_mgn import DomainContract, DualSolverMGN
    torch.manual_seed(seed)
    contract = DomainContract(tuple(f'n{i}' for i in range(14)), tuple(f'e{i}' for i in range(4)),
        ('native_element_Bx_T', 'native_element_By_T'), 'synthetic-train-only', 'native_element_reported', 'native_explicit_ID')
    model = DualSolverMGN({domain: contract for domain in ('elmer', 'motorcad')}, 12,
                         ('MotorCAD_N45P1_formal_UNKNOWN',), width=width, depth=3)
    validate_model_parameters(model, width)
    return model


def same_tree(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same_tree(a, b) for a, b in zip(left, right))
    return left == right


def synthetic_step_parity(seed=17,old_model_path=None,old_trainer_source=None):
    root=Path(__file__).resolve().parent
    require(old_model_path is not None and old_trainer_source is not None,'pinned original model/trainer required')
    return synthetic_source_parity(old_model_path,root/'dual_solver_mgn.py',old_trainer_source,
                                   (root/'train_width128.py').read_text(encoding='utf-8-sig'),seed)


def run_admission(bundle_metadata, historical, check=lambda: None):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    names = graph_names(bundle_metadata)
    historical_hashes = historical_initials(historical)
    require(historical['torch']==torch.__version__ and historical['numpy']==np.__version__
            and historical['python_version']==sys.version and historical['hostname']==socket.gethostname(),
            'saved W64 runtime/host differs from current CPU admission')
    require(set(historical_hashes) == {str(seed) for seed in SEEDS}, 'historical initial-state seed set differs')
    root=Path(__file__).resolve().parent
    old_model,model_pin=pinned_source(historical,'dual_solver_mgn.py')
    old_trainer,trainer_pin=pinned_source(historical,'train_phase45.py')
    old_evaluator,evaluator_pin=pinned_source(historical,'evaluate_phase45.py')
    old_cpu,cpu_source_pin=pinned_source(historical,'cpu_admission.py')
    require(sha(root/'dual_solver_mgn.py')==model_pin['sha256'],'native model source bytes changed')
    old_source=old_trainer.read_text(encoding='utf-8-sig')
    trainer_proof=source_equivalence(old_source,(root/'train_width128.py').read_text(encoding='utf-8-sig'),'trainer')
    evaluator_proof=source_equivalence(old_evaluator.read_text(encoding='utf-8-sig'),(root/'evaluate_width128.py').read_text(encoding='utf-8-sig'),'evaluator')
    import ast
    old_cpu_functions=functions(ast.parse(old_cpu.read_text(encoding='utf-8-sig')))
    new_cpu_functions=functions(ast.parse(Path(__file__).read_text(encoding='utf-8-sig')))
    require(all(fingerprint(old_cpu_functions[name])==fingerprint(new_cpu_functions[name]) for name in ('graph_names','order_traces')),
            'all45 graph-name/sampling trace source differs')
    unchanged=[]
    for name in ('a1_allocation_loss.py','strict_airgap.py','evaluate_common_airgap.py','mesh_regions.py','torque.py','feature_guard.py','case_split.py'):
        path,pin=pinned_source(historical,name)
        relative=Path('eval')/name if name in ('mesh_regions.py','torque.py','feature_guard.py','case_split.py') else Path(name)
        require(sha(root/relative)==pin['sha256'],'physical/evaluator helper source changed '+name)
        unchanged.append(pin)
    traces = {str(limit): {} for limit in (2957, 88565)}
    for seed in SEEDS:
        check()
        for limit, digest in order_traces(names, seed).items():
            traces[limit][str(seed)] = digest
    require(traces==historical['widths']['W64']['expected_data_order_LR_trace_sha256'], 'all45 order/LR differs from saved W64')
    reference_hashes={}
    for seed in SEEDS:
        check();reference_hashes[str(seed)]=tensor_state_sha256(make_model(64,seed).state_dict())
        require(reference_hashes[str(seed)]==historical_hashes[str(seed)],'W64 reference initial state differs')
    widths = {}
    for arm, width in WIDTHS.items():
        hashes = {}
        shapes = None
        for seed in SEEDS:
            check(); model = make_model(width, seed)
            hashes[str(seed)] = tensor_state_sha256(model.state_dict())
            current_shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
            require(shapes is None or shapes == current_shapes, 'architecture tensor shapes differ by seed')
            shapes = current_shapes
        require(len(set(hashes.values())) == 3, 'seed initialization hashes are not distinct')
        widths[arm] = dict(width=width, depth=3, **parameter_counts(width), initial_state_sha256_by_seed=hashes,
                           expected_data_order_LR_trace_sha256=copy.deepcopy(traces), tensor_shapes=shapes,
                           input_channels={'node': 14, 'edge': 4, 'design_op': 12}, output_channels=2)
    check(); parity = synthetic_step_parity(old_model_path=old_model,old_trainer_source=old_source)
    bridge=dict(status='PASS_BASELINE_BRIDGE_CPU_ONLY',trainer_full_module_AST_equivalent=True,model_source_identical=True,
        numerical_function_AST_equivalent=True,outer_main_AST_equivalent=True,initial_states_match_historical=True,
        all45_order_LR_traces_match_historical=True,synthetic_W64=parity,trainer_source_proof=trainer_proof,
        same_runtime_and_host_as_historical=True,
        evaluator_source_proof=evaluator_proof,unchanged_physical_evaluator_source_pins=unchanged,
        historical_source_pins=[model_pin,trainer_pin,evaluator_pin,cpu_source_pin],
        new_dataset_or_GPU_updates=0,scope='CPU/source equivalence only; no measured hardware resource or accuracy qualification')
    reference=dict(width=64,depth=3,**parameter_counts(64),initial_state_sha256_by_seed=reference_hashes,
                   expected_data_order_LR_trace_sha256=copy.deepcopy(traces),bridge=bridge)
    smoke_names = [name for name in names if int(Path(name).stem.split('_step')[1]) in STEPS][:5]
    return dict(status='PASS', experiment_id=EXPERIMENT, data_arm='A45', widths=widths, reference_W64=reference,
                train_graph_name_count=len(names), train_steps=list(range(45)),
                train_phase9_diagnostic_graphs=2952, smoke_graph_names=smoke_names,
                saved_W64_all45_order_LR_trace_sha256=historical['widths']['W64']['expected_data_order_LR_trace_sha256'],
                train_graph_names_sha256=hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest(),
                native_or_graph_binary_payloads_read=0, checkpoint_payloads_read=0, new_FEA=0,
                new_prior_solves=0, actual_dataset_optimizer_updates=0, GPU_calls=0,
                sampling='first5 historical phase9 smoke, then full all45 shuffle;88565 updates=5+6*14760',
                scope='CPU synthetic arithmetic and metadata-only all45 order admission; no accuracy/resource qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-manifest', type=Path, required=True)
    parser.add_argument('--bundle-manifest-sha256', required=True)
    parser.add_argument('--historical-admission', type=Path, required=True)
    parser.add_argument('--historical-admission-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wall-seconds-max', type=int, default=300)
    args = parser.parse_args()
    require(args.out.suffix.lower() == '.json' and not args.out.exists(), 'new JSON output required')
    require(1 <= args.wall_seconds_max <= 900, 'CPU admission wall cap exceeds900s')
    start = time.monotonic()
    def check():
        require(time.monotonic()-start < args.wall_seconds_max, 'CPU admission wall bound')
    manifest = pinned_metadata(args.bundle_manifest, args.bundle_manifest_sha256)
    historical = pinned_metadata(args.historical_admission, args.historical_admission_sha256)
    payload = run_admission(manifest, historical, check)
    import torch
    root = Path(__file__).resolve().parent
    payload.update(checked_UTC=dt.datetime.now(dt.timezone.utc).isoformat(), hostname=socket.gethostname(),
        python=sys.executable, python_version=sys.version, torch=torch.__version__, numpy=np.__version__,
        elapsed_seconds=time.monotonic()-start, bundle_manifest_sha256=args.bundle_manifest_sha256,
        historical_admission_sha256=args.historical_admission_sha256,
        source_files=[dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size) for path in sorted(root.rglob('*.py'))])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': payload['status'], 'output': str(args.out), 'widths': list(payload['widths'])}))


if __name__ == '__main__':
    main()
