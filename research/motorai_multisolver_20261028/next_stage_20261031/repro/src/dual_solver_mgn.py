"""Two solver-specific FE graph networks with a shared design/OP representation.

Experimental architecture only. This file does not load labels or train models.
Each domain keeps its own mesh entities, field semantics, normalizers and head.
Solver is categorical; numerical fidelity is an independently supplied token.
"""
from dataclasses import dataclass
import torch
from torch import nn

DOMAINS=('elmer','motorcad')

@dataclass(frozen=True)
class DomainContract:
    node_features: tuple
    edge_features: tuple
    field_outputs: tuple
    normalizer_train_group_hash: str
    entity_location: str
    id_mapping: str

def mlp(n,h,out):
    return nn.Sequential(nn.Linear(n,h),nn.SiLU(),nn.Linear(h,out))

class GraphEncoder(nn.Module):
    def __init__(self,node_in,edge_in,width,depth,field_out):
        super().__init__()
        self.node=mlp(node_in,width,width)
        self.edge=mlp(edge_in,width,width)
        self.messages=nn.ModuleList(mlp(width*3,width,width) for _ in range(depth))
        self.updates=nn.ModuleList(mlp(width*2,width,width) for _ in range(depth))
        self.norms=nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        self.field=mlp(width,width,field_out)
    def forward(self,node,edge_index,edge_attr,batch,weights,ngraphs,condition):
        if node.ndim!=2 or edge_index.shape[0]!=2:raise ValueError('Invalid graph shape')
        if edge_index.dtype!=torch.long or batch.dtype!=torch.long:raise TypeError('IDs must be int64')
        if weights.shape!=(len(node),) or (weights<0).any():raise ValueError('Invalid integration weights')
        if batch.shape!=(len(node),) or batch.min()<0 or batch.max()>=ngraphs:raise ValueError('Invalid graph IDs')
        src,dst=edge_index
        if src.numel() and (edge_index.min()<0 or edge_index.max()>=len(node)):raise ValueError('Invalid entity IDs')
        if src.numel() and (batch[src]!=batch[dst]).any():raise ValueError('Edges cross physical samples')
        h=self.node(node)+condition[batch];e=self.edge(edge_attr)
        for msg,upd,norm in zip(self.messages,self.updates,self.norms):
            m=msg(torch.cat([h[src],h[dst],e],-1));agg=torch.zeros_like(h)
            agg.index_add_(0,dst,m)
            degree=torch.bincount(dst,minlength=len(h)).to(h).clamp_min(1)[:,None]
            h=norm(h+upd(torch.cat([h,agg/degree],-1)))
        denom=weights.new_zeros(ngraphs);denom.index_add_(0,batch,weights)
        if (denom<=0).any():raise ValueError('Every graph needs positive physical weights')
        pooled=h.new_zeros((ngraphs,h.shape[-1]));pooled.index_add_(0,batch,h*weights[:,None])
        return pooled/denom[:,None],self.field(h)

class DualSolverMGN(nn.Module):
    def __init__(self,contracts,design_features,fidelity_tokens,width=64,depth=3):
        super().__init__()
        if set(contracts)!=set(DOMAINS):raise ValueError('Both domains need explicit contracts')
        for d,c in contracts.items():
            if not c.normalizer_train_group_hash or c.id_mapping!='native_explicit_ID':
                raise ValueError('Train-only normalization and native IDs are mandatory')
        self.contracts=contracts
        self.fidelity_tokens=tuple(fidelity_tokens)
        if len(set(fidelity_tokens))!=len(fidelity_tokens) or not fidelity_tokens:raise ValueError('Invalid fidelity vocabulary')
        self.encoders=nn.ModuleDict({d:GraphEncoder(len(c.node_features),len(c.edge_features),width,depth,len(c.field_outputs)) for d,c in contracts.items()})
        self.design=mlp(design_features,width,width)
        self.fidelity=nn.Embedding(len(fidelity_tokens),width)
        self.field_condition=nn.ModuleDict({d:mlp(width*2,width,width) for d in DOMAINS})
        self.alignment=nn.ModuleDict({d:mlp(width*2,width,width) for d in DOMAINS})
        self.heads=nn.ModuleDict({d:mlp(width*2,width,1) for d in DOMAINS})
        # At initialization the correction is exactly1: the supplied AF baseline
        # is reproduced. No transfer of frozen AF coefficients is done here.
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    def forward_fields(self,domain,graph,design_op,fidelity_id):
        """Solver-specific field pretraining without inventing an AC baseline.

        Physical field/normalizer/alias admission is the reader's responsibility.
        This API must never fabricate AC labels for a domain lacking a baseline.
        """
        if domain not in DOMAINS:raise ValueError('Unknown solver domain')
        if design_op.ndim!=2 or not torch.isfinite(design_op).all():raise ValueError('Invalid design/OP')
        if fidelity_id.dtype!=torch.long or fidelity_id.shape!=(len(design_op),):raise ValueError('Invalid fidelity IDs')
        if (fidelity_id<0).any() or (fidelity_id>=len(self.fidelity_tokens)).any():raise ValueError('Unknown fidelity ID')
        design=self.design(design_op);fidelity=self.fidelity(fidelity_id)
        condition=self.field_condition[domain](torch.cat([design,fidelity],-1))
        pooled,fields=self.encoders[domain](graph['node'],graph['edge_index'],graph['edge_attr'],graph['batch'],graph['weights'],len(design_op),condition)
        shared=self.alignment[domain](torch.cat([pooled,design],-1))
        return dict(shared=shared,fields=fields,fidelity=fidelity,field_semantics=self.contracts[domain].field_outputs,domain=domain)

    def forward(self,domain,graph,design_op,fidelity_id,baseline_AC_W,baseline_evaluator,baseline_contract=None):
        if domain not in DOMAINS:raise ValueError('Unknown solver domain')
        if baseline_evaluator!='AF_x_canonical_mean_square_Hybrid':raise ValueError('Mismatched AF denominator')
        if baseline_AC_W.ndim!=1 or len(baseline_AC_W)!=len(design_op) or not torch.isfinite(baseline_AC_W).all() or (baseline_AC_W<=0).any():
            raise ValueError('Positive physically matched baseline required; zero-field cases need a separate contract')
        if fidelity_id.dtype!=torch.long or fidelity_id.shape!=baseline_AC_W.shape:raise ValueError('Invalid fidelity IDs')
        validate_baseline_contract(domain,graph,baseline_contract)
        field_result=self.forward_fields(domain,graph,design_op,fidelity_id)
        residual=self.heads[domain](torch.cat([field_result['shared'],field_result['fidelity']],-1)).squeeze(-1)
        return dict(AC_W=baseline_AC_W*torch.exp(residual),log_residual=residual,shared=field_result['shared'],fields=field_result['fields'],
                    field_semantics=self.contracts[domain].field_outputs,domain=domain)

def validate_baseline_contract(domain,graph,contract):
    """Bind loss predictions to the admitted denominator artifact and samples.

    This validates binding, not the scientific truth of arbitrary caller metadata.
    The source manifest reader must verify artifact hashes and admission evidence.
    """
    if not contract or contract.get('solver_domain')!=domain:raise ValueError('Missing/mismatched baseline domain')
    expected={'AF_form':'f(omega)*kappa(I,beta)^p(omega)','denominator':'canonical_spatial_mean_square_field_Hybrid',
              'unit':'W','quantity':'active_AC_excess'}
    if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('Baseline physical contract mismatch')
    for k in ['coefficient_sha256','denominator_evaluator_sha256','calibration_groups_sha256','admission_receipt_sha256']:
        value=contract.get(k,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Unbound baseline provenance')
    keys=graph.get('sample_keys')
    if not keys or keys!=contract.get('sample_keys') or len(set(keys))!=len(keys):raise ValueError('Baseline/sample binding mismatch')
    if graph.get('average_window')!=contract.get('average_window') or not contract.get('average_window'):raise ValueError('Average window mismatch')

def validate_bridge(rows):
    """Exact physical pairs only. Coordinates or nominal R1 IDs are not enough."""
    required=['geometry','operating_point','material_temperature','waveform','quantity_scope','average_window','current_convention']
    for r in rows:
        if any(r.get(k)!='exact' for k in required):raise ValueError('Unqualified cross-solver pair')
        if r.get('left_split')!=r.get('right_split') or r.get('geometry_alias_group') in ['R1_0199','R1_0076']:
            raise ValueError('Split/reservation violation')
        if r.get('left_split')!='train':raise ValueError('Alignment cannot use development/test labels')
        if not r.get('evidence_sha256'):raise ValueError('Missing bridge evidence')

def paired_alignment_loss(left,right,bridge_rows):
    if not bridge_rows:raise ValueError('No exact bridge: paired loss must remain disabled')
    validate_bridge(bridge_rows)
    if left.shape!=right.shape or len(left)!=len(bridge_rows):raise ValueError('Pair count mismatch')
    return torch.mean((left-right)**2)

def masked_field_mse(pred,target,observed_mask):
    """Missing modalities are masked; never filled with zero pseudo-truth."""
    if pred.shape!=target.shape or observed_mask.shape!=pred.shape:raise ValueError('Field/mask shape mismatch')
    valid=observed_mask & torch.isfinite(target)
    if not valid.any():return pred.sum()*0
    return torch.mean((pred[valid]-target[valid])**2)
