"""Strict C/G checkpoint restoration shared by CPU gate and frozen evaluator."""
from context_field_model import ContextFieldModel


def restore_context(checkpoint, contracts, design_features, plan, hashes):
    meta = checkpoint['metadata']
    assert checkpoint['epoch'] == 30 and meta['training_updates'] == 88565
    assert not meta['old_checkpoint_reused']
    assert meta.get('status', 'RUNNING') in ('RUNNING', 'COMPLETED')
    assert meta['arm'] in ('C', 'G') and meta['seed'] in plan['fixed_seeds']
    assert meta['context_scope'] == plan['arms'][meta['arm']]['context_scope']
    assert meta['phase_encoding'] == plan['phase_encoding']
    for key, value in hashes.items():
        assert meta[key] == value, 'Checkpoint provenance mismatch: ' + key
    model = ContextFieldModel(contracts, design_features,
                              ('MotorCAD_N45P1_formal_UNKNOWN',),
                              scope=meta['context_scope'], width=32, depth=3)
    model.load_state_dict(checkpoint['model'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == meta['parameter_count'] == 71206
    return model
