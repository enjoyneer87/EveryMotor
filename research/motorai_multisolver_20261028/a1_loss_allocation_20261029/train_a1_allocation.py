"""Frozen A0 training loop with the single alpha0/alpha0.5 a1-loss axis.

Based on train_field_l12.py: same model creation, sample order, first5 updates,
Adam, clipping, update-indexed LR, assessment checkpoints, and native B targets.
Execution requires a hash-pinned release plan and admitted train328 masks.
"""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import datetime as dt
import hashlib
import json
import math
import queue
import random
import threading
import time
import numpy as np
from a1_allocation_loss import ALPHAS, allocation_loss, learning_rate, native_mask_and_q, tensor_state_sha256, trace_update, require

SEEDS = (17, 29, 43)
STEPS = (0, 5, 10, 15, 20, 25, 30, 35, 40)
ASSESSMENTS = {'0': 5, '1': 2957, '5': 14765, '10': 29525, '20': 59045, '30': 88565}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc)


def pinned_json(path, digest):
    require(sha(path) == digest, f'JSON SHA mismatch: {Path(path).name}')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin(spec):
    path = Path(spec['path'])
    require(path.is_file() and sha(path) == spec['sha256'], f'input SHA mismatch: {path.name}')
    if 'bytes' in spec:
        require(path.stat().st_size == spec['bytes'], f'input size mismatch: {path.name}')
    return path


def replace_retry(tmp, dst):
    """Only retry an atomic local status-file replacement, never a model run."""
    for _ in range(40):
        try:
            return tmp.replace(dst)
        except PermissionError:
            time.sleep(.25)
    return tmp.replace(dst)


def validate_plan(plan, arm, seed, max_updates):
    require(arm in ALPHAS and seed in SEEDS, 'unregistered arm/seed')
    require(plan['status'] in ('RELEASED_FIXED_SIX_RUNS', 'RELEASED_ENGINEERING_ONLY'), 'plan not released')
    require(plan['arms'][arm]['alpha'] == ALPHAS[arm] and plan['fixed_seeds'] == list(SEEDS), 'arm/seed contract differs')
    require(plan['seen_steps'] == list(STEPS), 'phase9 contract differs')
    spec = plan['training']
    require(spec['width'] == 32 and spec['depth'] == 3 and spec['expected_optimizer_updates'] == 88565
            and spec['updates_per_A0_epoch'] == 2952 and spec['smoke_updates'] == 5
            and spec['evaluation_updates'] == ASSESSMENTS, 'A0 training schedule differs')
    require(spec['learning_rate']['start'] == .001 and spec['learning_rate']['end'] == .00001, 'A0 LR differs')
    require(spec['wall_seconds_max'] <= 7200 and spec['torch_allocated_soft_limit_GiB'] <= 10 and spec['retry'] == 0,
            'resource bound exceeds declared proposal')
    if max_updates is None:
        require(plan['status'] == 'RELEASED_FIXED_SIX_RUNS', 'full training not released')
    else:
        require(type(max_updates) is int and 5 <= max_updates <= plan['engineering']['max_updates'], 'undeclared engineering updates')
    deadline = dt.datetime.fromisoformat(plan['guard_deadline_UTC'])
    require(deadline.tzinfo is not None and utc() < deadline, 'UTC deadline missing or expired')
    return spec, deadline


def admit_inputs(plan, bundle, masks_path, masks_sha):
    selection_path = pin(plan['selection'])
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    manifest_path = bundle / 'BUNDLE_MANIFEST.json'
    manifest = pinned_json(manifest_path, plan['bundle_manifest_sha256'])
    require(manifest['selection_sha256'] == plan['selection']['sha256'] and manifest['reader_pass_cases'] == 360
            and manifest['failed_cases'] == 0 and len(manifest['records']) == 360, 'original bundle admission differs')
    require(masks_sha == plan['mask_manifest_sha256'], 'mask manifest differs from release plan')
    masks = pinned_json(masks_path, masks_sha)
    require(masks['status'] == 'ADMITTED' and masks['selection_sha256'] == plan['selection']['sha256']
            and masks['bundle_manifest_sha256'] == plan['bundle_manifest_sha256']
            and masks['train_steps'] == list(STEPS), 'mask admission differs')
    declared = {r['case_index']: r for r in selection['records']}
    require(len(declared) == len(selection['records']) == 360, 'selection duplicate/missing cases')
    mask_cases = {r['case_index']: r for r in masks['records']}
    require(len(mask_cases) == len(masks['records']) == 328
            and set(mask_cases) == {c for c, r in declared.items() if r['split'] == 'train'}, 'train mask case coverage differs')
    files = {'train': [], 'development': []}
    groups = {split: set() for split in files}
    metadata, bindings, graph_pins, qs = {}, {}, {}, []
    seen_cases = set()
    for record in manifest['records']:
        case = record['case_index']
        require(case not in seen_cases, 'duplicate bundle case')
        seen_cases.add(case)
        declared_case = declared[case]
        split = record['split']
        require(record['geometry_group'] == declared_case['geometry_group'] and split == declared_case['split']
                and record['original_split'] == 'train' and record['source_sha256'] == declared_case['source_sha256'], 'bundle source/split mismatch')
        require(record['status'] == 'READER_ARITHMETIC_PASS' and len(record['files']) == (45 if split == 'train' else 9), 'bundle phases differ')
        groups[split].add(record['geometry_group'])
        mr = mask_cases[case] if split == 'train' else None
        if mr is not None:
            require(mr['source_sha256'] == record['source_sha256'], 'mask native source differs')
            a1_ids = np.asarray(mr['a1_native_element_ids'], dtype=np.int64)
            require(len(a1_ids) > 0 and len(np.unique(a1_ids)) == len(a1_ids), 'duplicate/empty admitted a1 IDs')
            require(hashlib.sha256(a1_ids.tobytes()).hexdigest() == mr['a1_id_sha256'], 'admitted a1 ID hash differs')
            mask_rows = {r['name']: r for r in mr['graphs']}
            require(len(mask_rows) == len(mr['graphs']) == 9 and {r['step'] for r in mr['graphs']} == set(STEPS), 'mask phase9 coverage differs')
        selected_names = set()
        for row in sorted(record['files'], key=lambda f: f['name']):
            name = row['name']
            require(Path(name).name == name and name.endswith('.npz'), 'unsafe graph filename')
            path = bundle / name
            step = int(path.stem.split('_step')[1])
            require(int(path.stem.split('_step')[0][4:]) == case, 'graph native case differs')
            require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'frozen graph bytes differ')
            metadata[path.stem] = {'case': case, 'group': record['geometry_group'], 'current_Arms': declared_case['native_input']['RMSCurrent']}
            graph_pins[name] = row['sha256']
            if split == 'development':
                require(step in STEPS, 'development assessment must remain seen9')
                files[split].append(path)
            elif step in STEPS:
                binding = mask_rows[name]
                require(binding['sha256'] == row['sha256'] and binding['bytes'] == row['bytes'] and binding['step'] == step, 'mask graph pin differs')
                bindings[name] = (a1_ids, binding)
                qs.append(float(binding['q']))
                files['train'].append(path)
                selected_names.add(name)
        if mr is not None:
            require(selected_names == set(mask_rows), 'mask graph name set differs')
    require(len(files['train']) == 2952 and len(files['development']) == 288, 'A0 graph counts differ')
    require(len(groups['train']) == 288 and len(groups['development']) == 32 and not groups['train'] & groups['development'], 'group split differs')
    pilot_path = pin(plan['pilot_manifest'])
    pilot = json.loads(pilot_path.read_text(encoding='utf-8'))
    pilotdev = {r['geometry_group'] for r in pilot['records'] if r['split'] == 'development'}
    require(len(pilotdev) == 4 and all(r['geometry_group'] in groups[r['split']] for r in pilot['records']), 'fixed pilot groups differ')
    require(all(math.isfinite(q) and 0 < q <= 1 for q in qs), 'invalid admitted area fractions')
    return manifest, files, groups, metadata, bindings, graph_pins, pilotdev, qs


def load_arrays(path, bindings, graph_pins):
    require(sha(path) == graph_pins[path.name], 'graph changed immediately before use')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ('node', 'edge_index', 'edge_attr', 'weights', 'design_op', 'target_B_T')}
        if path.name in bindings:
            ids, binding = bindings[path.name]
            arrays['a1_mask'], arrays['q'] = native_mask_and_q(archive['native_element_id'], ids, arrays['weights'], binding)
    return arrays


def prefetched(paths, bindings, graph_pins, depth):
    if depth <= 0:
        for path in paths:
            yield path, load_arrays(path, bindings, graph_pins)
        return
    channel = queue.Queue(maxsize=depth)
    done = object()
    def work():
        try:
            for path in paths:
                channel.put((path, load_arrays(path, bindings, graph_pins)))
            channel.put(done)
        except BaseException as exc:
            channel.put(exc)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = channel.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--masks-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(ALPHAS), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--prefetch', type=int, default=8)
    parser.add_argument('--max-updates', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = pinned_json(args.plan, args.plan_sha256)
    spec, deadline = validate_plan(plan, args.arm, args.seed, args.max_updates)
    require(args.prefetch == plan['training']['prefetch_depth'], 'prefetch differs from frozen plan')
    sources = {Path(row['path']).name: pin(row) for row in plan['source_files']}
    require(Path(__file__).resolve() == sources['train_a1_allocation.py'].resolve()
            and (root / 'a1_allocation_loss.py').resolve() == sources['a1_allocation_loss.py'].resolve(), 'trainer/loss module path not pinned')
    manifest, files, groups, metadata, bindings, graph_pins, pilotdev, qs = admit_inputs(plan, args.bundle, args.masks, args.masks_sha256)
    import torch
    import dual_solver_mgn
    from dual_solver_mgn import DomainContract, DualSolverMGN
    require(Path(dual_solver_mgn.__file__).resolve() == sources['dual_solver_mgn.py'].resolve(), 'loaded model source not pinned')
    args.out.mkdir(exist_ok=False)
    start = time.monotonic()
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    require(torch.cuda.is_available(), 'declared CUDA environment unavailable')
    device = torch.device('cuda:0')
    group_hash = hashlib.sha256(json.dumps(sorted(groups['train'])).encode()).hexdigest()
    contract = DomainContract(tuple(manifest['node_features']), tuple(manifest['edge_features']), ('native_element_Bx_T', 'native_element_By_T'), group_hash, 'native_element_reported', 'native_explicit_ID')
    model = DualSolverMGN({domain: contract for domain in ('elmer', 'motorcad')}, len(manifest['design_features']), ('MotorCAD_N45P1_formal_UNKNOWN',), width=32, depth=3).to(device)
    require(sum(p.numel() for p in model.parameters()) == 70182, 'model capacity differs')
    initial_sha = tensor_state_sha256(model.state_dict())
    require(initial_sha == plan['initial_state_sha256_by_seed'][str(seed)], 'initial tensor state differs from paired CPU admission')
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['learning_rate']['start'])
    limit = args.max_updates if args.max_updates is not None else 88565
    assessments = {update: int(epoch) for epoch, update in ASSESSMENTS.items()}
    trace = hashlib.sha256()
    alpha = ALPHAS[args.arm]
    result = dict(status='RUNNING', started_UTC=utc().isoformat(), host='133', arm=args.arm, alpha=alpha, data_arm='A0', phase_encoding='mechanical', seed=seed,
        prefetch_depth=args.prefetch, max_updates=args.max_updates, train_steps=list(STEPS), device=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__,
        model_sha256=sha(sources['dual_solver_mgn.py']), trainer_sha256=sha(__file__), loss_module_sha256=sha(sources['a1_allocation_loss.py']),
        plan_sha256=args.plan_sha256, selection_sha256=plan['selection']['sha256'], bundle_manifest_sha256=plan['bundle_manifest_sha256'], mask_manifest_sha256=args.masks_sha256,
        initial_tensor_state_sha256=initial_sha, data_order_LR_trace_sha256=trace.hexdigest(),
        train_groups=sorted(groups['train']), development_groups=sorted(groups['development']), fixed_pilot_development_groups=sorted(pilotdev),
        train_conditions=328, development_conditions=32, train_graphs=2952, original_val_test_used=0, new_FEA=0, new_prior_solves=0,
        historical_Full_train_donor_conditions=328, AC_loss_training=False, MS_teacher_used=False, paired_alignment=False, old_checkpoint_reused=False, seed_selection=False,
        parameters=70182, field_path_parameters=30594, scope='native Bx/By T; stationary native a1 loss allocation; a2 excluded',
        epochs_requested=30, training_updates=0, epochs_completed=0, passes_completed=0, metrics=[], assessment_wall_seconds_total=0.0,
        q_distribution={'count': len(qs), 'min': min(qs), 'max': max(qs), 'mean': float(np.mean(qs)), 'source': 'target-free admitted native-ID graph areas'},
        loss_formula='(1-alpha)*sum(area*vector_error2)/(2*sum(area))+alpha*sum(area_a1*vector_error2_a1)/(2*sum(area_a1))',
        memory_scope='PyTorch allocated peak, soft between-graph check; guard owns whole process limits')
    def save():
        result.update(checked_UTC=utc().isoformat(), elapsed_seconds=time.monotonic()-start,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), data_order_LR_trace_sha256=trace.hexdigest())
        temp = args.out / 'RESULT.tmp'
        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        replace_retry(temp, args.out / 'RESULT.json')
    def check():
        stop_files = [root / 'STOP', args.out / 'STOP'] + [Path(p) for p in plan['stop_files']]
        if any(path.exists() for path in stop_files):
            raise InterruptedError('STOP')
        if utc() >= deadline or time.monotonic()-start >= spec['wall_seconds_max']:
            raise InterruptedError('time cap')
        if torch.cuda.max_memory_allocated() > spec['torch_allocated_soft_limit_GiB']*1024**3:
            raise InterruptedError('torch allocated soft cap')
    def to_device(arrays):
        graph = {key: torch.as_tensor(arrays[key], device=device, dtype=torch.long if key == 'edge_index' else torch.float32) for key in ('node', 'edge_index', 'edge_attr', 'weights')}
        graph['batch'] = torch.zeros(len(graph['node']), dtype=torch.long, device=device)
        return graph, torch.as_tensor(arrays['design_op'], device=device)[None], torch.as_tensor(arrays['target_B_T'], device=device)
    def forward(graph, design):
        return model.forward_fields('motorcad', graph, design, torch.zeros(1, dtype=torch.long, device=device))['fields']
    def update(path, arrays):
        check()
        graph, design, target = to_device(arrays)
        optimizer.zero_grad(set_to_none=True)
        pred = forward(graph, design)
        mask = torch.as_tensor(arrays['a1_mask'], dtype=torch.bool, device=device)
        loss = allocation_loss(pred, target, graph['weights'], mask, alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        result['training_updates'] += 1
        trace_update(trace, result['training_updates'], path.name, optimizer.param_groups[0]['lr'])
    def evaluate(paths):
        model.eval()
        stats = {}
        def add(key, sse, energy, area):
            row = stats.setdefault(key, [0., 0., 0., 0])
            row[0] += float(sse); row[1] += float(energy); row[2] += float(area); row[3] += 1
        with torch.no_grad():
            for path, arrays in prefetched(paths, bindings, graph_pins, args.prefetch):
                check()
                graph, design, target = to_device(arrays)
                pred = forward(graph, design)
                require(torch.isfinite(pred).all().item(), 'nonfinite evaluation field')
                weights = graph['weights']; error = ((pred-target)**2).sum(1); energy = (target**2).sum(1)
                meta = metadata[path.stem]
                values = ((weights*error).sum(), (weights*energy).sum(), weights.sum())
                for key in ('all', 'case:'+str(meta['case']), 'geometry:'+meta['group'], 'current_Arms:'+str(meta['current_Arms'])):
                    add(key, *values)
                if meta['group'] in pilotdev:
                    add('fixed_pilot_dev4', *values)
                for i, name in enumerate(('stator','rotor','magnet','copper','air','wedge','shaft','impreg','other')):
                    mask = graph['node'][:, 3+i] > .5
                    if mask.any():
                        add('material:'+name, (weights[mask]*error[mask]).sum(), (weights[mask]*energy[mask]).sum(), weights[mask].sum())
        return {key: {'vector_nRMSE_percent': 100*(v[0]/v[1])**.5 if v[1] > 0 else None,
                      'vector_RMSE_T': (v[0]/v[2])**.5, 'snapshots': v[3], 'weighted_SSE_T2_m2': v[0],
                      'target_energy_T2_m2': v[1], 'area_m2': v[2]} for key, v in stats.items()}
    def checkpoint(epoch):
        temp = args.out / 'last.tmp'
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'epoch_equivalent': epoch,
                    'passes': result['passes_completed'], 'metadata': {k:v for k,v in result.items() if k != 'checkpoint_sha256'}}, temp)
        replace_retry(temp, args.out / 'last.pt')
    def assessment(epoch):
        assessment_started = time.monotonic()
        result['metrics'].append({'epoch': epoch, 'training_updates': result['training_updates'],
            'learning_rate': optimizer.param_groups[0]['lr'], 'train_phase9': evaluate(files['train']), 'development': evaluate(files['development'])})
        save(); checkpoint(epoch)
        # Resource projection metadata only: includes both evaluation passes and
        # their first status/checkpoint writes; changes no forward/update order.
        assessment_elapsed = time.monotonic() - assessment_started
        result['metrics'][-1]['assessment_wall_seconds'] = assessment_elapsed
        result['assessment_wall_seconds_total'] += assessment_elapsed
        save()
    save()
    try:
        model.train()
        for path, arrays in prefetched(files['train'][:5], bindings, graph_pins, args.prefetch):
            update(path, arrays)
        torch.cuda.synchronize(); check(); result['GPU_smoke'] = 'PASS'; save(); assessment(0)
        rng = random.Random(seed)
        while result['training_updates'] < limit:
            model.train()
            order = files['train'].copy(); rng.shuffle(order)
            for path, arrays in prefetched(order, bindings, graph_pins, args.prefetch):
                if result['training_updates'] >= limit:
                    break
                lr = learning_rate(result['training_updates']+1, spec['learning_rate'], 2952)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                update(path, arrays)
                count = result['training_updates']
                result['epochs_completed'] = min(max((count-5)//2952, 0), 30)
                if count in assessments and assessments[count] > 0:
                    assessment(assessments[count]); model.train()
                elif count % 100 == 0:
                    save()
            result['passes_completed'] += 1
        require(result['training_updates'] == limit, 'optimizer update count differs')
        if args.max_updates is None:
            require(result['metrics'][-1]['epoch'] == 30 and result['metrics'][-1]['development']['all']['snapshots'] == 288,
                    'final A0 assessment missing')
            result.update(status='COMPLETED', completed_UTC=utc().isoformat())
        else:
            result.update(status='SMOKE_COMPLETED', completed_UTC=utc().isoformat())
        save()
        checkpoint(30 if args.max_updates is None else result['epochs_completed'])
        result['checkpoint_sha256'] = sha(args.out/'last.pt')
    except InterruptedError as exc:
        result.update(status='STOPPED', stop_reason=str(exc))
        raise
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc))
        raise
    finally:
        save()
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','train_groups','development_groups')}, allow_nan=False))


if __name__ == '__main__':
    main()
