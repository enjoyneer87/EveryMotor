"""CPU synthetic contract checks; not training or scientific validation."""
import json,hashlib
from pathlib import Path
import torch
from dual_solver_mgn import DomainContract,DualSolverMGN,paired_alignment_loss,masked_field_mse
def rejects(fn):
    try:fn()
    except ValueError:return True
    raise AssertionError('Invalid contract accepted')
def main():
    torch.manual_seed(17);torch.set_num_threads(2)
    contracts={d:DomainContract(('x_m_scaled','y_m_scaled','region_code'),('dx_scaled','dy_scaled'),
      ('A_normalized','Bx_normalized','By_normalized') if d=='elmer' else ('Bx_normalized','By_normalized','J_definition_pending'),
      'synthetic_train_only_contract','node' if d=='elmer' else 'element_node','native_explicit_ID') for d in ['elmer','motorcad']}
    m=DualSolverMGN(contracts,4,('N128P1','N64P2','N256P3'),width=16,depth=2)
    g=dict(node=torch.randn(3,3),edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),edge_attr=torch.randn(4,2),batch=torch.zeros(3,dtype=torch.long),weights=torch.tensor([.5,1.,.5]),sample_keys=['synthetic_1'],average_window='synthetic_P2')
    args=(g,torch.ones(1,4),torch.zeros(1,dtype=torch.long),torch.tensor([100.]),'AF_x_canonical_mean_square_Hybrid')
    def binding(domain):return dict(solver_domain=domain,AF_form='f(omega)*kappa(I,beta)^p(omega)',denominator='canonical_spatial_mean_square_field_Hybrid',unit='W',quantity='active_AC_excess',sample_keys=g['sample_keys'],average_window=g['average_window'],**{k:'0'*64 for k in ['coefficient_sha256','denominator_evaluator_sha256','calibration_groups_sha256','admission_receipt_sha256']})
    a=m('elmer',*args,baseline_contract=binding('elmer'));b=m('motorcad',*args,baseline_contract=binding('motorcad'))
    assert a['AC_W'].item()==100 and b['AC_W'].item()==100
    assert m.encoders['elmer'].node[0].weight.data_ptr()!=m.encoders['motorcad'].node[0].weight.data_ptr()
    loss=masked_field_mse(a['fields'],torch.zeros_like(a['fields']),torch.ones_like(a['fields'],dtype=torch.bool));loss.backward()
    assert any(p.grad is not None for p in m.encoders['elmer'].parameters())
    assert all(p.grad is None for p in m.encoders['motorcad'].parameters())
    assert masked_field_mse(b['fields'],torch.full_like(b['fields'],float('nan')),torch.zeros_like(b['fields'],dtype=torch.bool)).item()==0
    rejects(lambda:paired_alignment_loss(a['shared'],b['shared'],[]))
    rejects(lambda:paired_alignment_loss(a['shared'],b['shared'],[dict(geometry='nominal_only')]))
    rejects(lambda:m('elmer',*args[:-1],'GetDP_proximity'))
    rejects(lambda:m('elmer',*args))
    rejects(lambda:m('motorcad',*args,baseline_contract=binding('elmer')))
    field0=m.forward_fields('motorcad',g,args[1],args[2])['fields']
    field_op=m.forward_fields('motorcad',g,args[1]*2,args[2])['fields']
    field_fidelity=m.forward_fields('motorcad',g,args[1],args[2]+1)['fields']
    assert not torch.allclose(field0,field_op) and not torch.allclose(field0,field_fidelity)
    bad={**g,'batch':torch.tensor([0,0,1])}
    rejects(lambda:m.forward_fields('elmer',bad,torch.ones(2,4),torch.zeros(2,dtype=torch.long)))
    print(json.dumps(dict(status='PASS',checks=['separate_solver_parameters','AF_identity_at_initialization','gradient_domain_isolation','missing_modality_mask','no_exact_bridge_rejected','nominal_pair_rejected','denominator_mismatch_rejected','cross_graph_edge_rejected','missing_baseline_provenance_rejected','cross_domain_baseline_rejected','field_responds_to_OP','field_responds_to_fidelity','field_pretraining_without_scalar_baseline'],
     framework=torch.__version__,device='CPU',training_updates=0,physical_validation=False,model_sha256=hashlib.sha256(Path(__file__).with_name('dual_solver_mgn.py').read_bytes()).hexdigest())))
if __name__=='__main__':main()
