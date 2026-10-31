"""Explicit width-only architecture contract; contains no data or compute launcher."""
from a1_allocation_loss import require

WIDTHS = {'W64': 64, 'W128': 128}
SEEDS = (17, 29, 43)
DEPTH = 3
FIELD_PREFIXES = ('encoders.motorcad.', 'design.', 'fidelity.', 'field_condition.motorcad.')


def parameter_counts(width, depth=DEPTH):
    require(type(width) is int and width in WIDTHS.values() and type(depth) is int and depth == DEPTH,
            'unregistered width/depth')
    return {'parameters': (23 + 14 * depth) * width**2 + (77 + 12 * depth) * width + 6,
            'field_path_parameters': (7 + 7 * depth) * width**2 + (42 + 6 * depth) * width + 2}


def validate_architecture(obj, arm, width):
    require(arm in WIDTHS and type(width) is int and width == WIDTHS[arm], 'arm/CLI width differs')
    require('arm' not in obj or obj['arm'] == arm, 'architecture arm metadata differs')
    require(type(obj.get('width')) is int and obj['width'] == width
            and type(obj.get('depth')) is int and obj['depth'] == DEPTH, 'explicit width/depth differs')
    return parameter_counts(width)


def validate_model_parameters(model, width):
    expected = parameter_counts(width)
    observed = {'parameters': sum(p.numel() for p in model.parameters()),
                'field_path_parameters': sum(p.numel() for n, p in model.named_parameters() if n.startswith(FIELD_PREFIXES))}
    require(observed == expected, 'capacity parameter count differs')
    return observed


def validate_initialization(plan, seed, width, digest):
    require(digest == plan['initial_state_sha256_by_seed'][str(seed)], 'initial tensor state differs from CPU admission')


def restore_capacity(checkpoint, contracts, design_features, arm, width):
    """Restore only the declared architecture; never infer width from tensors."""
    from dual_solver_mgn import DualSolverMGN
    meta = checkpoint['metadata']
    counts = validate_architecture(meta, arm, width)
    require(design_features == 12 and all(len(c.node_features) == 14 and len(c.edge_features) == 4
            and len(c.field_outputs) == 2 for c in contracts.values()), 'frozen input/output features differ')
    model = DualSolverMGN(contracts, design_features, ('MotorCAD_N45P1_formal_UNKNOWN',), width=width, depth=DEPTH)
    validate_model_parameters(model, width)
    require(all(meta.get(k) == v for k, v in counts.items()), 'checkpoint capacity metadata differs')
    model.load_state_dict(checkpoint['model'], strict=True)
    return model
