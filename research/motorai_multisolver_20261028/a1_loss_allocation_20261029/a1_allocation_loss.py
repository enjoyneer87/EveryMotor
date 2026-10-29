"""Single-axis area/a1 loss allocation and target-free native mask binding.

The alpha=0 single-graph expression is exactly the original A0 expression,
including operation order and the denominator 2*sum(area). No model parameters,
new graph edges, target-derived masks, or adaptive coefficients are introduced.
"""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np

ALPHAS = {'L0': 0.0, 'L05': 0.5}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_mask_and_q(native_ids, a1_ids, weights, binding=None):
    """Bind one physical graph's declared native a1 IDs without any field label."""
    ids = np.asarray(native_ids)
    a1 = np.asarray(a1_ids)
    area = np.asarray(weights)
    for value, name in ((ids, 'native IDs'), (a1, 'a1 IDs')):
        require(value.ndim == 1 and value.dtype.kind in 'iu' and len(value) > 0, f'invalid {name}')
        require(len(np.unique(value)) == len(value), f'duplicate {name}')
    require(area.shape == ids.shape and area.dtype.kind == 'f' and np.isfinite(area).all()
            and (area > 0).all(), 'invalid graph area')
    mask = np.isin(ids, a1)
    require(int(mask.sum()) == len(a1), 'a1 native IDs absent from graph')
    area64 = area.astype(np.float64)
    area_all = float(area64.sum())
    area_a1 = float(area64[mask].sum())
    q = area_a1 / area_all
    require(math.isfinite(q) and 0 < q <= 1, 'invalid a1 area fraction')
    if binding is not None:
        require(binding['native_element_id_sha256'] == hashlib.sha256(ids.tobytes()).hexdigest(), 'native-ID order hash differs')
        require(binding['a1_mask_sha256'] == hashlib.sha256(mask.tobytes()).hexdigest(), 'a1 mask hash differs')
        require(binding['a1_count'] == len(a1) and binding['all_count'] == len(ids), 'native mask counts differ')
        for key, actual in (('area_a1_m2', area_a1), ('area_all_m2', area_all), ('q', q),
                            ('effective_a1_allocation_alpha05', .5 + .5 * q)):
            expected = binding[key]
            require(type(expected) in (float, int) and math.isfinite(expected)
                    and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-14), f'{key} binding differs')
    return mask, q


def allocation_loss(pred, target, weights, a1_mask, alpha, batch=None, reduction='mean'):
    """Equal physical-graph reduction; no shared batch-wide area denominator.

    Training uses a single graph exactly as A0 did. The explicit batch path is
    present to test/isolate future callers; every graph must have admitted a1.
    """
    import torch
    require(type(alpha) in (float, int) and alpha in (0, .5), 'alpha must be predeclared0 or0.5')
    require(reduction in ('mean', 'none'), 'undeclared loss reduction')
    require(pred.ndim == 2 and pred.shape[1] == 2 and len(pred) > 0 and target.shape == pred.shape,
            'pred/target must be nonempty native element Bx/By')
    require(weights.shape == (len(pred),) and a1_mask.shape == weights.shape
            and a1_mask.dtype == torch.bool, 'invalid weights or native a1 mask')
    require(pred.device == target.device == weights.device == a1_mask.device, 'mixed devices')
    require(torch.isfinite(pred).all().item() and torch.isfinite(target).all().item()
            and torch.isfinite(weights).all().item() and (weights > 0).all().item(), 'nonfinite field or invalid area')
    require(pred.dtype == target.dtype == weights.dtype, 'mixed physical floating dtypes')

    def one(p, y, w, mask):
        require(mask.any().item(), 'physical graph has no admitted a1')
        # Keep this exact expression and early branch for A0 equivalence.
        all_loss = (w[:, None] * (p - y) ** 2).sum() / (2 * w.sum())
        if alpha == 0:
            return all_loss
        a1_loss = (w[mask, None] * (p[mask] - y[mask]) ** 2).sum() / (2 * w[mask].sum())
        return (1 - alpha) * all_loss + alpha * a1_loss

    if batch is None:
        loss = one(pred, target, weights, a1_mask)
        require(torch.isfinite(loss).item(), 'nonfinite allocation loss')
        return loss if reduction == 'mean' else loss[None]
    require(batch.shape == weights.shape and batch.dtype == torch.long and batch.device == pred.device,
            'invalid physical batch IDs')
    require((batch >= 0).all().item(), 'negative physical batch ID')
    unique = torch.unique(batch, sorted=True)
    require(torch.equal(unique, torch.arange(len(unique), device=batch.device)), 'batch IDs must be contiguous')
    losses = torch.stack([one(pred[batch == i], target[batch == i], weights[batch == i], a1_mask[batch == i])
                          for i in range(len(unique))])
    require(torch.isfinite(losses).all().item(), 'nonfinite allocation loss')
    return losses.mean() if reduction == 'mean' else losses


def tensor_state_sha256(state):
    """Deterministic tensor identity, independent of torch.save container bytes."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        descriptor = json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode('utf-8')
        data = tensor.numpy().tobytes()
        for payload in (descriptor, data):
            digest.update(len(payload).to_bytes(8, 'little'))
            digest.update(payload)
    return digest.hexdigest()


def trace_update(digest, update, graph_name, learning_rate):
    require(type(update) is int and update > 0, 'invalid optimizer update index')
    require(math.isfinite(learning_rate) and learning_rate > 0, 'invalid learning rate')
    digest.update((json.dumps([update, graph_name, float(learning_rate).hex()], separators=(',', ':')) + '\n').encode('utf-8'))


def learning_rate(update, lr, per_epoch):
    """Exact existing A0 update-indexed schedule; first five updates are smoke."""
    epoch = min(max(math.ceil((update - 5) / per_epoch), 1), 30)
    return lr['end'] + .5 * (lr['start'] - lr['end']) * (1 + math.cos(math.pi * (epoch - 1) / 29))
