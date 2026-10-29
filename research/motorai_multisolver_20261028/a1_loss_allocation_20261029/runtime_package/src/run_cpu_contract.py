"""CPU-only equivalence tests, fixed initial tensors and original-A0 order/LR trace."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['A1_REQUIRE_TORCH_TESTS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,hashlib,datetime as dt,time,unittest,io,sys,random
import numpy as np
import torch
from dual_solver_mgn import DomainContract,DualSolverMGN
from a1_allocation_loss import tensor_state_sha256
from reference_train_field_l12 import learning_rate
ROOT=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();spec=json.loads((ROOT/'CPU_GATE_SPEC.json').read_text())
assert sys.executable.replace('\\','/').lower()==spec['python_executable'].lower()
for row in spec['source_files']:assert sha(row['path'])==row['sha256'],row['path']
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
stream=io.StringIO();suite=unittest.TestSuite()
for module in ('test_a1_allocation','test_evaluate_a1_allocation'):
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
test=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
assert test.wasSuccessful() and not test.skipped,stream.getvalue()
sel=json.loads(Path(spec['selection']['path']).read_text())
assert sha(spec['selection']['path'])==spec['selection']['sha256']
manifest_path=Path(spec['train_bundle_dir'])/'BUNDLE_MANIFEST.json'
assert sha(manifest_path)==spec['bundle_manifest_sha256']
bundle=json.loads(manifest_path.read_text())
groups=sorted({r['geometry_group'] for r in sel['records'] if r['split']=='train'})
assert len(groups)==288
contract=DomainContract(tuple(bundle['node_features']),tuple(bundle['edge_features']),('native_element_Bx_T','native_element_By_T'),hashlib.sha256(json.dumps(groups).encode()).hexdigest(),'native_element_reported','native_explicit_ID')
initial={}
for seed in (17,29,43):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=DualSolverMGN({d:contract for d in ('elmer','motorcad')},len(bundle['design_features']),('MotorCAD_N45P1_formal_UNKNOWN',),width=32,depth=3)
    assert sum(p.numel() for p in model.parameters())==70182
    initial[str(seed)]=tensor_state_sha256(model.state_dict())
files=[]
for row in bundle['records']:
    if row['split']=='train':
        files.extend(f['name'] for f in sorted(row['files'],key=lambda f:f['name']) if int(Path(f['name']).stem.split('_step')[1]) in range(0,45,5))
assert len(files)==2952
traces={'2957':{},'88565':{}}
for seed in (17,29,43):
    h=hashlib.sha256();u=0
    def update(name,lr):
        global u
        u+=1;h.update((json.dumps([u,Path(name).name,float(lr).hex()],separators=(',',':'))+'\n').encode())
    for name in files[:5]:update(name,.001)
    rng=random.Random(seed)
    while u<88565:
        order=files.copy();rng.shuffle(order)
        for name in order:
            if u==88565:break
            update(name,learning_rate(u+1,{'start':.001,'end':.00001},2952))
            if u==2957:traces['2957'][str(seed)]=h.hexdigest()
    traces['88565'][str(seed)]=h.hexdigest()
receipt={'status':'PASS','scope':'CPU_SYNTHETIC_EQUIVALENCE_AND_FIXED_INITIAL_ORDER_CONTRACT','tests_run':test.testsRun,'failures':len(test.failures),'errors':len(test.errors),'skipped':len(test.skipped),
         'test_log':stream.getvalue(),'initial_state_sha256_by_seed':initial,'expected_data_order_LR_trace_sha256':traces,'source_files':spec['source_files'],
         'selection_sha256':spec['selection']['sha256'],'bundle_manifest_sha256':spec['bundle_manifest_sha256'],'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
         'GPU_operations':0,'real_training_updates':0,'new_FEA':0,'checked_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start}
(ROOT/'CPU_CONTRACT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','tests_run','failures','errors','skipped','elapsed_seconds','initial_state_sha256_by_seed','expected_data_order_LR_trace_sha256')}))
