"""Synthetic and JSON metadata tests; never opens actual native/graph datasets."""
import copy
import datetime as dt
import hashlib
import importlib.util
from pathlib import Path
import random
import unittest

import cpu_admission as admission
import evaluate_width128 as evaluator
import train_width128 as trainer
from a1_allocation_loss import learning_rate, trace_update
from capacity_contract import parameter_counts
from baseline_bridge import source_equivalence

HAS_TORCH = importlib.util.find_spec('torch') is not None
REFERENCE_ROOT=next((path for path in (Path(__file__).resolve().parents[2]/'phase45_20261030'/'src',
    Path('C:/work/EveryMotor_Phase45_20261030/src')) if (path/'train_phase45.py').is_file()),None)


def plan_fixture():
    return {'status':'RELEASED_ENGINEERING_ONLY','arms':{'W128':{'alpha':0.,'width':128}},'fixed_seeds':[17,29,43],
        'seen_steps':list(trainer.STEPS),'train_steps':list(range(45)),'experiment_id':trainer.EXPERIMENT,'data_arm':'A45',
        'training':{'width':128,'depth':3,'device':'cuda','expected_optimizer_updates':88565,
            'updates_per_A0_epoch':2952,'smoke_updates':5,'evaluation_updates':trainer.ASSESSMENTS,
            'learning_rate':{'start':.001,'end':.00001},'wall_seconds_max':10800,'torch_allocated_soft_limit_GiB':10,'retry':0},
        'engineering':{'max_updates':2957},'guard_deadline_UTC':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1)).isoformat()}


def bundle_fixture():
    records=[dict(case_index=i,split='train',original_split='train',geometry_group=f'train{i%288:04}',
        files=[{'name':f'case{i:04}_step{step:02}.npz'} for step in range(45)]) for i in range(328)]
    records += [dict(case_index=1000+i,split='development',original_split='train',geometry_group=f'dev{i:03}',files=[])
                for i in range(32)]
    return dict(reader_pass_cases=360,failed_cases=0,records=records)


def metadata_fixture(seed=17):
    selection={'records':[dict(case_index=i,split='train',geometry_group=f'train{i%288:04}') for i in range(328)]
        + [dict(case_index=1000+i,split='development',geometry_group=f'dev{i:03}') for i in range(32)]}
    hashes={key:format(i+1,'064x') for i,key in enumerate(evaluator.HASH_KEYS)}
    meta=dict(status='COMPLETED',arm='W128',alpha=0.,seed=seed,data_arm='A45',experiment_id=trainer.EXPERIMENT,
        phase_encoding='mechanical',width=128,depth=3,execution_device='cuda',**parameter_counts(128),
        training_updates=88565,train_steps=list(range(45)),train_diagnostic_steps=list(trainer.STEPS),
        train_graphs=14760,train_phase9_diagnostic_graphs=2952,train_conditions=328,development_conditions=32,
        old_checkpoint_reused=False,new_FEA=0,original_val_test_used=0,
        train_groups=[f'train{i:04}' for i in range(288)],development_groups=[f'dev{i:03}' for i in range(32)],**hashes)
    job=dict(id=f'W128_s{seed}',arm='W128',width=128,depth=3,alpha=0.,seed=seed,data_arm='A45',experiment_id=trainer.EXPERIMENT,
        phase_encoding='mechanical',metadata_hashes=hashes,checkpoint={'sha256':'c'*64},
        selection={'sha256':hashes['selection_sha256']},training_mask_manifest={'sha256':hashes['mask_manifest_sha256']})
    train=copy.deepcopy(meta)
    train.update(checkpoint_sha256='c'*64,metrics=[dict(epoch=30,training_updates=88565,
        development={'all':{'snapshots':288,'vector_nRMSE_percent':10.}})])
    return {'epoch':30,'epoch_equivalent':30,'metadata':meta},job,selection,train


class Phase45Tests(unittest.TestCase):
    def test_plan_all45_explicit_and_phase9_diagnostic_separate(self):
        trainer.validate_plan(plan_fixture(),'W128',17,2957,128)
        for corruption in ('steps','seen','width','arm','experiment','updates','retry','full','device'):
            plan=plan_fixture();updates=2957
            if corruption=='steps':plan['train_steps']=list(trainer.STEPS)
            elif corruption=='seen':plan['seen_steps']=list(range(45))
            elif corruption=='width':plan['training']['width']=32
            elif corruption=='arm':plan['data_arm']='A0'
            elif corruption=='experiment':plan['experiment_id']='old'
            elif corruption=='updates':plan['training']['expected_optimizer_updates']=442805
            elif corruption=='retry':plan['training']['retry']=1
            elif corruption=='full':updates=None
            else:plan['training']['device']='cpu'
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):trainer.validate_plan(plan,'W128',17,updates,128)
        plan=plan_fixture();plan['status']='RELEASED_FIXED_WIDTH128_RUNS'
        trainer.validate_plan(plan,'W128',43,None,128)

    def test_all45_metadata_names_and_missing_phase_rejected(self):
        names=admission.graph_names(bundle_fixture())
        self.assertEqual(len(names),14760)
        self.assertEqual(names[:6],[f'case0000_step{i:02}.npz' for i in range(6)])
        for corruption in ('missing','duplicate','originalsplit','reserved'):
            bundle=bundle_fixture()
            if corruption=='missing':bundle['records'][0]['files'].pop()
            elif corruption=='duplicate':bundle['records'][0]['files'][-1]=bundle['records'][0]['files'][0]
            elif corruption=='originalsplit':bundle['records'][0]['original_split']='test'
            else:bundle['records'][0]['geometry_group']='R1_0199'
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):admission.graph_names(bundle)

    def test_smoke_is_historical9_first5_then_independent_all45_shuffle(self):
        names=admission.graph_names(bundle_fixture());seed=17
        trace=hashlib.sha256();smoke=[f'case0000_step{i:02}.npz' for i in (0,5,10,15,20)]
        for update,name in enumerate(smoke,1):trace_update(trace,update,name,.001)
        first5=trace.hexdigest()
        order=names.copy();random.Random(seed).shuffle(order)
        for update,name in enumerate(order[:2952],6):trace_update(trace,update,name,learning_rate(update,admission.LR,2952))
        actual=admission.order_traces(names,seed,(5,2957))
        self.assertEqual(actual,{'5':first5,'2957':trace.hexdigest()})
        self.assertNotEqual(actual['2957'],admission.order_traces(names,29,(5,2957))['2957'])

    def test_final_update_count_is_six_balanced_full45_passes(self):
        self.assertEqual(5+6*14760,88565)
        self.assertEqual(trainer.ASSESSMENTS,{'0':5,'1':2957,'5':14765,'10':29525,'20':59045,'30':88565})
        names=admission.graph_names(bundle_fixture())
        trace=admission.order_traces(names,17)
        self.assertEqual(set(trace),{'2957','88565'})

    def test_evaluator_accepts_only_phase45_final_and_same_dev_anchor(self):
        for seed in (17,29,43):self.assertEqual(evaluator.validate_new_metadata(*metadata_fixture(seed))['data_arm'],'A45')
        for corruption in ('data','phases','graphcount','diag','epoch','source','experiment'):
            cp,job,sel,train=metadata_fixture()
            if corruption=='data':cp['metadata']['data_arm']='A0'
            elif corruption=='phases':train['train_steps']=list(trainer.STEPS)
            elif corruption=='graphcount':cp['metadata']['train_graphs']=2952
            elif corruption=='diag':train['train_phase9_diagnostic_graphs']=14760
            elif corruption=='epoch':cp['epoch']=29
            elif corruption=='source':job['metadata_hashes']['model_sha256']='bad'
            else:job['experiment_id']='old'
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):evaluator.validate_new_metadata(cp,job,sel,train)

    def test_historical_initials_use_savedW64_not_otherwidths(self):
        history={'status':'PASS','data_arm':'A45','experiment_id':'W64_PHASE45_EQUAL_UPDATE_20261030',
            'widths':{'W32':{'initial_state_sha256_by_seed':{'17':'a'*64}},
                     'W64':{'initial_state_sha256_by_seed':{'17':'b'*64}}}}
        self.assertEqual(admission.historical_initials(history),{'17':'b'*64})
        history['status']='FAILED'
        with self.assertRaises(ValueError):admission.historical_initials(history)

    def test_declared_parameter_counts_and_rejected_unregistered_width(self):
        self.assertEqual(parameter_counts(64),{'parameters':273478,'field_path_parameters':118530})
        self.assertEqual(parameter_counts(128),{'parameters':1079430,'field_path_parameters':466434})
        with self.assertRaises(ValueError):parameter_counts(32)

    @unittest.skipIf(REFERENCE_ROOT is None,'pinned prior source stage unavailable')
    def test_full_module_AST_includes_outer_order_LR_and_determinism(self):
        root=Path(__file__).parent
        old=(REFERENCE_ROOT/'train_phase45.py').read_text(encoding='utf-8-sig')
        new=(root/'train_width128.py').read_text(encoding='utf-8-sig')
        proof=source_equivalence(old,new,'trainer')
        self.assertTrue(proof['all_other_AST_nodes_identical'])
        self.assertIn('smoke_shuffle_RNG_LR_assessment_finalization_block_sha256',proof)
        for token,replacement in (("rng = random.Random(seed)","rng = random.Random(seed+1)"),
                                  ("lr = learning_rate(result['training_updates']+1", "lr = learning_rate(result['training_updates']+2"),
                                  ("torch.backends.cudnn.benchmark = False","torch.backends.cudnn.benchmark = True"),
                                  ("files['train_phase9_diagnostic'][:5]","files['train'][:5]")):
            self.assertIn(token,new)
            with self.subTest(token=token),self.assertRaises(ValueError):source_equivalence(old,new.replace(token,replacement),'trainer')
        self.assertEqual(source_equivalence((REFERENCE_ROOT/'evaluate_phase45.py').read_text(encoding='utf-8-sig'),
            (root/'evaluate_width128.py').read_text(encoding='utf-8-sig'),'evaluator')['status'],'PASS_EXPLICIT_WHITELIST_FULL_MODULE_AST')


@unittest.skipUnless(HAS_TORCH,'torch unavailable locally; must run admitted remote CPU suite before release')
class Phase45TorchTests(unittest.TestCase):
    def test_alpha0_no_mask_prediction_gradient_Adam_parity_W64(self):
        import torch
        torch.set_num_threads(1)
        if REFERENCE_ROOT is None:self.skipTest('pinned prior source stage unavailable')
        proof=admission.synthetic_step_parity(old_model_path=REFERENCE_ROOT/'dual_solver_mgn.py',
            old_trainer_source=(REFERENCE_ROOT/'train_phase45.py').read_text(encoding='utf-8-sig'))
        self.assertEqual(proof['status'],'PASS_BIT_EXACT_SYNTHETIC_CPU')
        self.assertTrue(proof['actual_source_update_functions_executed'])

    def test_explicit_W128_restore_and_state_shape_mismatch_rejected(self):
        import torch
        from capacity_contract import restore_capacity
        from dual_solver_mgn import DomainContract
        model=admission.make_model(128,17)
        contract=DomainContract(tuple(range(14)),tuple(range(4)),('Bx','By'),'synthetic-only','native_element_reported','native_explicit_ID')
        contracts={'elmer':contract,'motorcad':contract}
        checkpoint={'model':model.state_dict(),'metadata':dict(arm='W128',width=128,depth=3,**parameter_counts(128))}
        restored=restore_capacity(checkpoint,contracts,12,'W128',128)
        self.assertEqual(restored.encoders['motorcad'].node[0].weight.shape,torch.Size([128,14]))
        with self.assertRaises(ValueError):restore_capacity(checkpoint,contracts,12,'W64',64)
        bad=copy.deepcopy(checkpoint);bad['model']['encoders.motorcad.node.0.weight']=torch.zeros((64,14))
        with self.assertRaises(RuntimeError):restore_capacity(bad,contracts,12,'W128',128)

    def test_maskless_alpha0_objective_does_not_read_mask(self):
        import torch
        from a1_allocation_loss import allocation_loss
        pred=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
        target=torch.zeros_like(pred);area=torch.tensor([1.,2.])
        direct=(area[:,None]*(pred-target)**2).sum()/(2*area.sum())
        for mask in (torch.tensor([True,False]),torch.tensor([False,True])):
            self.assertTrue(torch.equal(direct,allocation_loss(pred,target,area,mask,0.)))


if __name__=='__main__':unittest.main(verbosity=2)
