"""Bounded engineering overfit on one fixed train graph; not accuracy evidence."""
from pathlib import Path
import datetime
import hashlib
import json
import random
import time

import numpy as np
import torch

from context_field_model import ContextFieldModel
from dual_solver_mgn import DomainContract
from train_context import load_arrays, load_component_map, sha


def main():
    root = Path(__file__).parent
    cfg = json.loads((root/'RUN_PRIVATE.json').read_text())
    plan = json.loads((root/'CONTEXT_CG_PLAN.json').read_text())
    components = load_component_map(root/'components', plan['bundle_manifest_sha256'])
    manifest = json.loads((Path(cfg['bundle'])/'BUNDLE_MANIFEST.json').read_text())
    selection = json.loads((root/'SCALE360_SELECTION_FROZEN.json').read_text())
    groups = sorted({r['geometry_group'] for r in selection['records'] if r['split']=='train'})
    contract = DomainContract(tuple(manifest['node_features']), tuple(manifest['edge_features']),
                              ('native_element_Bx_T','native_element_By_T'), hashlib.sha256(json.dumps(groups).encode()).hexdigest(),
                              'native_element_reported','native_explicit_ID')
    deadline = datetime.datetime.fromisoformat(plan['execution']['guard_deadline_UTC'])
    result = {'status':'RUNNING','scope':'single train graph engineering overfit, not physics qualification',
              'case':40,'step':0,'fixed_seed':17,'updates_each':1500,'runs':{},'new_FEA':0}
    start = time.monotonic()
    def save():
        result['elapsed_seconds'] = time.monotonic()-start
        (root/'OVERFIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for arm in ('C','G'):
        random.seed(17);np.random.seed(17);torch.manual_seed(17)
        model = ContextFieldModel({d:contract for d in ('elmer','motorcad')},12,('MotorCAD_N45P1_formal_UNKNOWN',),
                                  plan['arms'][arm]['context_scope'],32,3).cuda()
        arrays = load_arrays(Path(cfg['bundle'])/'case0040_step00.npz',plan['phase_encoding'],components)
        graph = {k:torch.as_tensor(arrays[k],device='cuda',dtype=torch.long if k in ('edge_index','component_id') else torch.float32)
                 for k in ('node','edge_index','edge_attr','weights','component_id')}
        graph['batch'] = torch.zeros(len(graph['node']),device='cuda',dtype=torch.long)
        design = torch.as_tensor(arrays['design_op'],device='cuda')[None]
        target = torch.as_tensor(arrays['target_B_T'],device='cuda')
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        def loss():
            pred = model.forward_fields('motorcad',graph,design,torch.zeros(1,device='cuda',dtype=torch.long))['fields']
            return (graph['weights'][:,None]*(pred-target)**2).sum()/(2*graph['weights'].sum())
        initial = float(loss().detach())
        for update in range(1500):
            if (root/'STOP').exists() or time.monotonic()-start>900 or datetime.datetime.now(datetime.timezone.utc)>=deadline:
                raise InterruptedError('STOP/deadline/overfit wall cap')
            if torch.cuda.max_memory_allocated()>10*1024**3:
                raise InterruptedError('Torch allocated soft memory cap')
            optimizer.zero_grad(set_to_none=True)
            current = loss()
            assert torch.isfinite(current)
            current.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
            optimizer.step()
        final = float(loss().detach())
        result['runs'][arm] = {'initial_MSE_T2':initial,'final_MSE_T2':final,'ratio':final/initial,
                               'gate_pass':bool(np.isfinite(final) and final<=.1*initial),
                               'updates':1500,'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated()}
        save()
        if not result['runs'][arm]['gate_pass']:
            result['status']='FAILED_OVERFIT';save();raise SystemExit(1)
        del model,optimizer
        torch.cuda.empty_cache()
    result['status']='PASS_ENGINEERING_ONLY'
    save()
    print(json.dumps(result))


if __name__=='__main__':
    main()
