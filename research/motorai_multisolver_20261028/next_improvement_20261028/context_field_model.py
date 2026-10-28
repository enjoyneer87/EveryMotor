"""Parameter-matched component/global context feedback around the frozen backbone.

This adds an information path, not a repaired physical FE mesh. No prior or AC head
is used. The source backbone is composed intact so its initialization can be checked.
"""
from dataclasses import dataclass

import torch
from torch import nn

from dual_solver_mgn import DualSolverMGN


@dataclass(frozen=True)
class ContextContract:
    scope: str
    width: int = 32

    def __post_init__(self):
        if self.scope not in ('component', 'graph'):
            raise ValueError('Context scope must be component or graph')


def weighted_context(hidden, weights, batch, component, scope):
    """Pure weighted pooling without pooling across different physical samples."""
    if hidden.ndim != 2 or weights.shape != (len(hidden),):
        raise ValueError('Invalid context tensor shape')
    if batch.shape != weights.shape or component.shape != weights.shape:
        raise ValueError('Every entity needs a graph ID and a native-edge component ID')
    if batch.dtype != torch.long or component.dtype != torch.long:
        raise TypeError('Context group IDs must be int64')
    if (weights < 0).any() or not torch.isfinite(weights).all():
        raise ValueError('Invalid physical area weights')
    if (batch < 0).any() or (component < 0).any():
        raise ValueError('Context IDs must be nonnegative')
    keys = batch[:, None] if scope == 'graph' else torch.stack([batch, component], dim=1)
    if scope not in ('graph', 'component'):
        raise ValueError('Unknown context scope')
    unique, group = torch.unique(keys, dim=0, return_inverse=True)
    sums = hidden.new_zeros((len(unique), hidden.shape[1]))
    denom = weights.new_zeros(len(unique))
    sums.index_add_(0, group, hidden * weights[:, None])
    denom.index_add_(0, group, weights)
    if (denom <= 0).any():
        raise ValueError('Each pooled component must have positive area')
    return (sums / denom[:, None])[group]


def backbone_hidden(encoder, graph, condition):
    """The pinned GraphEncoder local-message equations, without its final decoder."""
    node, edges, attrs = graph['node'], graph['edge_index'], graph['edge_attr']
    batch = graph['batch']
    src, dst = edges
    if edges.dtype != torch.long or batch.dtype != torch.long:
        raise TypeError('Native edges and batch IDs must be int64')
    if src.numel() and ((edges < 0).any() or edges.max() >= len(node) or (batch[src] != batch[dst]).any()):
        raise ValueError('Invalid edges or cross-sample edges')
    hidden = encoder.node(node) + condition[batch]
    edge_hidden = encoder.edge(attrs)
    for message, update, norm in zip(encoder.messages, encoder.updates, encoder.norms):
        msg = message(torch.cat([hidden[src], hidden[dst], edge_hidden], dim=-1))
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, dst, msg)
        degree = torch.bincount(dst, minlength=len(hidden)).to(hidden).clamp_min(1)[:, None]
        hidden = norm(hidden + update(torch.cat([hidden, aggregate / degree], dim=-1)))
    return hidden


class ContextFieldModel(nn.Module):
    def __init__(self, contracts, design_features, fidelity_tokens, scope, width=32, depth=3):
        super().__init__()
        self.context_contract = ContextContract(scope, width)
        self.backbone = DualSolverMGN(contracts, design_features, fidelity_tokens, width=width, depth=depth)
        self.context_projection = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.context_projection.weight)

    def forward_fields(self, domain, graph, design_op, fidelity_id):
        if domain != 'motorcad':
            raise ValueError('This bounded experiment admits only Motor-CAD B supervision')
        if 'component_id' not in graph:
            raise ValueError('Native-edge connected-component IDs are mandatory')
        if design_op.ndim != 2 or not torch.isfinite(design_op).all():
            raise ValueError('Invalid design/OP input')
        if fidelity_id.dtype != torch.long or fidelity_id.shape != (len(design_op),):
            raise ValueError('Invalid fidelity input')
        if (fidelity_id < 0).any() or (fidelity_id >= len(self.backbone.fidelity_tokens)).any():
            raise ValueError('Unknown fidelity ID')
        design = self.backbone.design(design_op)
        fidelity = self.backbone.fidelity(fidelity_id)
        condition = self.backbone.field_condition[domain](torch.cat([design, fidelity], dim=-1))
        encoder = self.backbone.encoders[domain]
        hidden = backbone_hidden(encoder, graph, condition)
        context = weighted_context(hidden, graph['weights'], graph['batch'], graph['component_id'], self.context_contract.scope)
        fields = encoder.field(hidden + self.context_projection(context))
        return {'fields': fields, 'domain': domain, 'context_scope': self.context_contract.scope,
                'field_semantics': self.backbone.contracts[domain].field_outputs}
