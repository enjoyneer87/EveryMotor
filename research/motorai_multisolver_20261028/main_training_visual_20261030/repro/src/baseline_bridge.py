"""Pinned source/AST and synthetic CPU bridge for the saved all45 W64 control."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

from a1_allocation_loss import require, tensor_state_sha256, trace_update

OLD_ID = 'W64_PHASE45_EQUAL_UPDATE_20261030'
NEW_ID = 'W128_PHASE45_EQUAL_UPDATE_20261030'
NUMERICAL = ('admit_inputs','load_arrays','prefetched','peak_memory','save','check','to_device',
             'forward','update','evaluate','checkpoint','assessment')


def fingerprint(node):
    value=ast.dump(node,include_attributes=False) if isinstance(node,ast.AST) else '\n'.join(ast.dump(n,include_attributes=False) for n in node)
    return hashlib.sha256(value.encode()).hexdigest()


def functions(tree):
    return {node.name:node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}


def normalized(tree, kind, candidate):
    tree=copy.deepcopy(tree);changes=[]
    if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):
        tree.body[0].value.value='DOCUMENTATION_ONLY'
        changes.append('module_documentation')
    if not candidate:return tree,changes
    arm_global='ALPHAS' if kind=='trainer' else 'ARMS'
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id==arm_global:
                require(ast.literal_eval(node.value)=={'W128':0.0},'unexpected candidate arm declaration')
                node.value=ast.parse("{'W64':0.0}",mode='eval').body;changes.append(arm_global+':W128_to_W64')
            elif node.targets[0].id=='EXPERIMENT':
                require(ast.literal_eval(node.value)==NEW_ID,'unexpected candidate experiment')
                node.value=ast.Constant(OLD_ID);changes.append('experiment_identity')
    string_map={'train_width128.py':'train_phase45.py',
        'RELEASED_FIXED_WIDTH128_RUNS':'RELEASED_FIXED_PHASE45_RUNS',
        'only W128 is admitted in phase45':'only W64 is admitted in phase45',
        'native Bx/By T; all45 training, W128/depth3/alpha0; a2 excluded':'native Bx/By T; all45 training, W64/depth3/alpha0; a2 excluded',
        'E1-oldTRAIN-exposed innerdev32; W128 all45 equal-update comparison, not independent/AC/physical-torque qualification':
        'E1-oldTRAIN-exposed innerdev32; W64 all45 equal-update comparison, not independent/AC/physical-torque qualification'}
    class Allowed(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in string_map:
                changes.append('literal:'+node.value);return ast.copy_location(ast.Constant(string_map[node.value]),node)
            return node
        def visit_Compare(self,node):
            if kind=='trainer' and isinstance(node.left,ast.Name) and node.left.id=='width' and len(node.ops)==1 and isinstance(node.ops[0],ast.Eq) and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==128:
                node.comparators[0].value=64;changes.append('width_admission:128_to_64')
            return self.generic_visit(node)
        def visit_Call(self,node):
            if kind=='trainer' and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--width':
                choices=next(k for k in node.keywords if k.arg=='choices')
                require(ast.literal_eval(choices.value)==(128,),'unexpected candidate width CLI')
                choices.value=ast.parse('(64,)',mode='eval').body;changes.append('CLI_width:128_to_64')
            return self.generic_visit(node)
    tree=Allowed().visit(tree)
    return tree,changes


def source_equivalence(old_source,new_source,kind):
    require(kind in ('trainer','evaluator'),'unknown source bridge kind')
    old_tree,_=normalized(ast.parse(old_source),kind,False)
    new_tree,changes=normalized(ast.parse(new_source),kind,True)
    require(fingerprint(old_tree)==fingerprint(new_tree),'non-whitelisted '+kind+' source AST change')
    old_fn,new_fn=functions(old_tree),functions(new_tree)
    result=dict(status='PASS_EXPLICIT_WHITELIST_FULL_MODULE_AST',normalized_module_sha256=fingerprint(old_tree),
                applied_whitelist=changes,all_other_AST_nodes_identical=True)
    if kind=='trainer':
        numerical={name:fingerprint(old_fn[name]) for name in NUMERICAL}
        require(all(fingerprint(new_fn[name])==value for name,value in numerical.items()),'numerical function AST differs')
        main=old_fn['main'];main_new=new_fn['main']
        require(fingerprint(main)==fingerprint(main_new),'outer main AST differs')
        import_index=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
        result_index=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets))
        outer_try=[n for n in main.body if isinstance(n,ast.Try)]
        require(len(outer_try)==1,'ambiguous outer training block')
        result.update(numerical_function_fingerprints=numerical,outer_main_sha256=fingerprint(main),
                      initialization_optimizer_determinism_block_sha256=fingerprint(main.body[import_index:result_index]),
                      smoke_shuffle_RNG_LR_assessment_finalization_block_sha256=fingerprint(outer_try[0]))
    return result


def load_source_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def build_source_model(module,width,seed):
    import torch
    torch.manual_seed(seed)
    contract=module.DomainContract(tuple(f'n{i}' for i in range(14)),tuple(f'e{i}' for i in range(4)),
        ('native_element_Bx_T','native_element_By_T'),'synthetic-train-only','native_element_reported','native_explicit_ID')
    return module.DualSolverMGN({d:contract for d in ('elmer','motorcad')},12,
        ('MotorCAD_N45P1_formal_UNKNOWN',),width=width,depth=3)


def same_tree(left,right):
    import torch
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(same_tree(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left) is type(right) and len(left)==len(right) and all(same_tree(a,b) for a,b in zip(left,right))
    return left==right


def actual_trainer_step(source,model,arrays):
    """Execute the pinned trainer's own extracted update on fabricated CPU arrays."""
    import torch
    function_map=functions(ast.parse(source))
    nodes=[copy.deepcopy(function_map[name]) for name in ('to_device','forward','update')]
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    env=dict(torch=torch,device=torch.device('cpu'),model=model,optimizer=optimizer,check=lambda:None,
             require=require,result={'training_updates':0},trace=hashlib.sha256(),trace_update=trace_update)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<pinned synthetic trainer update>','exec'),env)
    forward=env['forward'];observed={}
    loss_node=next(node.value for node in function_map['update'].body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in node.targets))
    loss_expression=compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(loss_node))),'<pinned trainer loss>','eval')
    def capture(graph,design):
        prediction=forward(graph,design)
        observed['prediction']=prediction.detach().clone()
        observed['loss']=eval(loss_expression,env,dict(pred=prediction,target=torch.as_tensor(arrays['target_B_T']),weights=graph['weights'])).detach().clone()
        return prediction
    env['forward']=capture
    env['update'](Path('fabricated_graph_step00.npz'),arrays)
    observed.update(clipped_gradients={name:None if p.grad is None else p.grad.detach().clone() for name,p in model.named_parameters()},
                    parameters=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
                    trace=env['trace'].hexdigest(),training_updates=env['result']['training_updates'])
    return observed


def synthetic_source_parity(old_model_path,new_model_path,old_trainer_source,new_trainer_source,seed=17):
    import numpy as np
    old=load_source_module(old_model_path,'motorai_width128_old_native_model')
    new=load_source_module(new_model_path,'motorai_width128_new_native_model')
    original,candidate=build_source_model(old,64,seed),build_source_model(new,64,seed)
    require(tensor_state_sha256(original.state_dict())==tensor_state_sha256(candidate.state_dict()),'W64 source-model fresh state differs')
    rng=np.random.default_rng(20261030)
    arrays=dict(node=rng.standard_normal((6,14)).astype(np.float32),
        edge_index=np.array([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]],np.int64),
        edge_attr=rng.standard_normal((8,4)).astype(np.float32),weights=np.arange(1,7,dtype=np.float32),
        design_op=rng.standard_normal(12).astype(np.float32),target_B_T=rng.standard_normal((6,2)).astype(np.float32))
    left=actual_trainer_step(old_trainer_source,original,arrays)
    right=actual_trainer_step(new_trainer_source,candidate,arrays)
    require(same_tree(left,right),'actual W64 trainer synthetic prediction/loss/clipped-gradient/Adam/trace differs')
    return dict(status='PASS_BIT_EXACT_SYNTHETIC_CPU',seed=seed,old_and_new_model_modules_loaded_separately=True,
        actual_source_update_functions_executed=True,prediction_equal=True,loss_equal=True,clipped_gradients_equal=True,
        post_Adam_parameters_equal=True,post_Adam_optimizer_state_equal=True,synthetic_update_trace_equal=True,
        updated_tensor_state_sha256=tensor_state_sha256(candidate.state_dict()),actual_dataset_optimizer_updates=0)
