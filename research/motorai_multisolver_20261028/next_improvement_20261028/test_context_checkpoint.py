"""CPU-only synthetic roundtrip and scope rejection for the actual evaluator loader."""
import io
import json
import torch
from context_checkpoint import restore_context
from context_field_model import ContextFieldModel
from dual_solver_mgn import DomainContract


def main():
    torch.set_num_threads(1)
    contract = DomainContract(tuple(map(str, range(14))), tuple(map(str, range(4))),
                              ('Bx_T', 'By_T'), 'synthetic', 'native_element_reported', 'native_explicit_ID')
    contracts = {d: contract for d in ('elmer', 'motorcad')}
    plan = {'fixed_seeds': [17, 29, 43], 'phase_encoding': 'mechanical',
            'arms': {'C': {'context_scope': 'component'}, 'G': {'context_scope': 'graph'}}}
    hashes = {'plan_sha256': 'synthetic_plan', 'model_sha256': 'synthetic_backbone',
              'context_model_sha256': 'synthetic_context'}
    graph = {'node': torch.randn(4, 14), 'edge_index': torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
             'edge_attr': torch.randn(4, 4), 'weights': torch.arange(1, 5, dtype=torch.float32),
             'batch': torch.zeros(4, dtype=torch.long), 'component_id': torch.tensor([0, 0, 1, 1])}
    design, fidelity = torch.randn(1, 12), torch.zeros(1, dtype=torch.long)
    for arm, scope in [('C', 'component'), ('G', 'graph')]:
        torch.manual_seed(17)
        model = ContextFieldModel(contracts, 12, ('MotorCAD_N45P1_formal_UNKNOWN',), scope)
        with torch.no_grad():
            model.context_projection.weight.copy_(torch.eye(32) * .1)
        meta = dict(arm=arm, seed=17, context_scope=scope, training_updates=88565,
                    phase_encoding='mechanical', old_checkpoint_reused=False,
                    parameter_count=71206, status='RUNNING', **hashes)
        storage = io.BytesIO()
        torch.save({'epoch': 30, 'metadata': meta, 'model': model.state_dict()}, storage)
        storage.seek(0)
        checkpoint = torch.load(storage, weights_only=False)
        restored = restore_context(checkpoint, contracts, 12, plan, hashes)
        reference = model.forward_fields('motorcad', graph, design, fidelity)['fields']
        actual = restored.forward_fields('motorcad', graph, design, fidelity)['fields']
        assert torch.equal(reference, actual)
        checkpoint['metadata']['context_scope'] = 'graph' if arm == 'C' else 'component'
        try:
            restore_context(checkpoint, contracts, 12, plan, hashes)
        except AssertionError:
            pass
        else:
            raise AssertionError('Wrong arm scope was accepted')
    print(json.dumps({'status': 'PASS', 'scope': 'synthetic CPU evaluator serialization gate only',
                      'C_G_nonzero_projection_bitexact_roundtrip': 'PASS', 'scope_mismatch_rejected': 'PASS',
                      'GPU_work': 0, 'new_FEA': 0, 'new_training_updates': 0}))


if __name__ == '__main__':
    main()
