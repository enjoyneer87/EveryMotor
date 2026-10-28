"""Engineering contract tests, not a field-accuracy benchmark; CPU only."""
import hashlib
import json

import torch

from dual_solver_mgn import DomainContract
from context_field_model import ContextFieldModel, weighted_context


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    contract = DomainContract(tuple(str(i) for i in range(14)), tuple(str(i) for i in range(4)),
                              ('Bx_T', 'By_T'), 'synthetic_engineering_contract_only', 'native_element_reported', 'native_explicit_ID')
    models = {}
    for scope in ('component', 'graph'):
        torch.manual_seed(17)
        models[scope] = ContextFieldModel({d: contract for d in ('elmer', 'motorcad')}, 12, ('fidelity',), scope, width=32, depth=3)
    assert state_digest(models['component']) == state_digest(models['graph'])
    assert sum(p.numel() for p in models['graph'].parameters()) == 70182 + 1024
    torch.manual_seed(19)
    graph = {'node': torch.randn(8, 14), 'edge_index': torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6]]),
             'edge_attr': torch.randn(8, 4), 'batch': torch.tensor([0] * 4 + [1] * 4),
             'component_id': torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]), 'weights': torch.arange(1, 9, dtype=torch.float32)}
    design, fidelity = torch.randn(2, 12), torch.zeros(2, dtype=torch.long)
    for model in models.values():
        reference = model.backbone.forward_fields('motorcad', graph, design, fidelity)['fields']
        actual = model.forward_fields('motorcad', graph, design, fidelity)['fields']
        assert torch.equal(reference, actual), 'Zero context must reproduce the pinned backbone exactly'
    hidden = torch.randn(8, 32)
    for scope in ('component', 'graph'):
        base = weighted_context(hidden, graph['weights'], graph['batch'], graph['component_id'], scope)
        changed = hidden.clone()
        changed[4:] += 7
        result = weighted_context(changed, graph['weights'], graph['batch'], graph['component_id'], scope)
        assert torch.equal(base[:4], result[:4]), 'Pooling leaked between physical batch graphs'
        trial = hidden.clone().requires_grad_(True)
        projected = weighted_context(trial, graph['weights'], graph['batch'], graph['component_id'], scope)
        projected[0].sum().backward()
        influence = float(trial.grad[2:4].abs().sum())
        assert (influence == 0) if scope == 'component' else (influence > 0), 'Wrong cross-component derivative path'
        model = models[scope]
        with torch.no_grad():
            model.context_projection.weight.copy_(torch.eye(32) * .1)
        model.zero_grad(set_to_none=True)
        node = graph['node'].clone().requires_grad_(True)
        fields = model.forward_fields('motorcad', {**graph, 'node': node}, design, fidelity)['fields']
        fields[0].sum().backward()
        cross = float(node.grad[2:4].abs().sum())
        assert cross == 0 if scope == 'component' else cross > 1e-12
        assert torch.count_nonzero(node.grad[4:]) == 0
    return {'status': 'PASS', 'scope': 'synthetic CPU engineering checks only, not physics qualification',
            'parameter_count_each': 71206, 'extra_parameters_each': 1024,
            'same_initial_tensor_state': 'PASS', 'zero_projection_exact_backbone': 'PASS',
            'no_cross_batch_pooling': 'PASS', 'nonzero_projection_field_cross_component_derivative': 'PASS',
            'new_training_updates': 0, 'new_FEA': 0, 'GPU_work': 0}


if __name__ == '__main__':
    print(json.dumps(check()))
