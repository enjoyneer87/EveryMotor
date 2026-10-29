"""Synthetic native-mask and exact A0-loss/Adam equivalence checks.

Set A1_REQUIRE_TORCH_TESTS=1 in the admitted CPU runtime so missing torch is a
hard failure. No dataset, checkpoint, remote connection, CUDA, or model training
campaign is used; the only optimizer action is one synthetic CPU Adam step.
"""
import copy
import hashlib
import importlib.util
import math
import os
import unittest
import numpy as np
from a1_allocation_loss import allocation_loss, learning_rate, native_mask_and_q, tensor_state_sha256, trace_update

HAS_TORCH = importlib.util.find_spec('torch') is not None
if os.environ.get('A1_REQUIRE_TORCH_TESTS') == '1' and not HAS_TORCH:
    raise RuntimeError('Required CPU PyTorch equivalence tests unavailable')
if HAS_TORCH:
    import torch
    from dual_solver_mgn import DomainContract, DualSolverMGN


class NativeMaskTests(unittest.TestCase):
    def fixture(self):
        ids = np.array([88, 3, 55, 12], dtype=np.int64)
        a1 = np.array([12, 88], dtype=np.int64)
        weights = np.array([2, 4, 8, 6], dtype=np.float32)
        mask = np.array([True, False, False, True])
        binding = {'native_element_id_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                   'a1_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
                   'a1_count': 2, 'all_count': 4, 'area_a1_m2': 8., 'area_all_m2': 20.,
                   'q': .4, 'effective_a1_allocation_alpha05': .7}
        return ids, a1, weights, mask, binding

    def test_native_permutation_is_id_based(self):
        ids, a1, weights, mask, binding = self.fixture()
        actual, q = native_mask_and_q(ids, a1, weights, binding)
        np.testing.assert_array_equal(actual, mask)
        self.assertEqual(q, .4)
        order = np.array([2, 0, 3, 1])
        permuted, changed_q = native_mask_and_q(ids[order], a1[::-1], weights[order])
        np.testing.assert_array_equal(permuted, mask[order])
        self.assertEqual(changed_q, q)

    def test_order_hash_prevents_silent_reordering(self):
        ids, a1, weights, _, binding = self.fixture()
        with self.assertRaisesRegex(ValueError, 'order hash'):
            native_mask_and_q(ids[::-1], a1, weights[::-1], binding)

    def test_duplicate_missing_nonfinite_and_zero_area_rejected(self):
        ids, a1, weights, _, _ = self.fixture()
        variants = [(np.array([88, 88, 55, 12]), a1, weights), (ids, np.array([12, 12]), weights),
                    (ids, np.array([999]), weights), (ids, a1, np.array([0, 4, 8, 6.], dtype=float)),
                    (ids, a1, np.array([np.nan, 4, 8, 6.])), (ids, np.array([], dtype=np.int64), weights)]
        for triple in variants:
            with self.subTest(triple=str(triple)):
                with self.assertRaises(ValueError):
                    native_mask_and_q(*triple)

    def test_q_and_mask_pin_tampering_rejected(self):
        ids, a1, weights, _, binding = self.fixture()
        for key, value in (('a1_mask_sha256', '0'*64), ('q', .7), ('a1_count', 3), ('area_a1_m2', 9.),
                           ('effective_a1_allocation_alpha05', .5)):
            changed = dict(binding); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                native_mask_and_q(ids, a1, weights, changed)

    def test_original_update_schedule_and_smoke_count(self):
        lr = {'start': .001, 'end': .00001}
        for update in (1, 5, 6, 2957, 2958, 14765, 88565):
            epoch = min(max(math.ceil((update-5)/2952), 1), 30)
            expected = lr['end']+.5*(lr['start']-lr['end'])*(1+math.cos(math.pi*(epoch-1)/29))
            self.assertEqual(learning_rate(update, lr, 2952), expected)
        self.assertEqual(30*2952+5, 88565)
        self.assertEqual(learning_rate(88565, lr, 2952), .00001)

    def test_data_order_and_lr_trace_detects_deviation(self):
        a, b, changed = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        for index, name in enumerate(('case0040_step00.npz', 'case0040_step05.npz'), 1):
            trace_update(a, index, name, .001); trace_update(b, index, name, .001)
            trace_update(changed, index, name, .0009)
        self.assertEqual(a.hexdigest(), b.hexdigest())
        self.assertNotEqual(a.hexdigest(), changed.hexdigest())


@unittest.skipUnless(HAS_TORCH, 'CPU PyTorch unavailable in this local interpreter; must run in admitted CPU runtime')
class TorchLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(17)

    def tensors(self, dtype=torch.float32 if HAS_TORCH else None):
        p = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=dtype, requires_grad=True)
        y = torch.tensor([[.5, 1.5], [2., 1.], [1., 3.]], dtype=dtype)
        w = torch.tensor([2., 3., 5.], dtype=dtype)
        mask = torch.tensor([True, False, True])
        return p, y, w, mask

    def test_alpha0_loss_and_gradient_bit_exact(self):
        for dtype in (torch.float32, torch.float64):
            p, y, w, mask = self.tensors(dtype)
            reference = (w[:, None]*(p-y)**2).sum()/(2*w.sum())
            observed = allocation_loss(p, y, w, mask, 0.)
            self.assertTrue(torch.equal(reference, observed))
            a = torch.autograd.grad(reference, p, retain_graph=True)[0]
            b = torch.autograd.grad(observed, p)[0]
            self.assertTrue(torch.equal(a, b))

    def test_alpha05_exact_region_formula_and_gradient(self):
        p, y, w, mask = self.tensors(torch.float64)
        expected = .5*(w[:, None]*(p-y)**2).sum()/(2*w.sum()) + .5*(w[mask, None]*(p[mask]-y[mask])**2).sum()/(2*w[mask].sum())
        actual = allocation_loss(p, y, w, mask, .5)
        self.assertTrue(torch.equal(expected, actual))
        grad = torch.autograd.grad(actual, p)[0]
        analytic = .5*w[:, None]*(p.detach()-y)/w.sum()
        analytic[mask] += .5*w[mask, None]*(p.detach()[mask]-y[mask])/w[mask].sum()
        self.assertTrue(torch.allclose(grad, analytic, atol=1e-14, rtol=1e-14))

    def test_batch_isolation_and_equal_graph_reduction(self):
        p, y, w, mask = self.tensors(torch.float64)
        p = torch.cat([p.detach(), p.detach()+10]).requires_grad_(True)
        y = torch.cat([y, y]); weights = torch.cat([w, w*1000]); masks = torch.cat([mask, mask])
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        losses = allocation_loss(p, y, weights, masks, .5, batch, 'none')
        separate = torch.stack([allocation_loss(p[:3], y[:3], weights[:3], masks[:3], .5),
                                allocation_loss(p[3:], y[3:], weights[3:], masks[3:], .5)])
        self.assertTrue(torch.equal(losses, separate))
        self.assertTrue(torch.equal(allocation_loss(p, y, weights, masks, .5, batch), separate.mean()))
        grad = torch.autograd.grad(losses[0], p)[0]
        self.assertTrue(torch.equal(grad[3:], torch.zeros_like(grad[3:])))

    def test_reject_no_a1_per_graph_invalid_alpha_and_nonfinite(self):
        p, y, w, mask = self.tensors()
        for alpha in (-.5, 1., float('nan')):
            with self.assertRaises(ValueError): allocation_loss(p, y, w, mask, alpha)
        with self.assertRaises(ValueError): allocation_loss(p, y, w, torch.zeros_like(mask), 0.)
        with self.assertRaises(ValueError): allocation_loss(p, y, w, torch.tensor([True,False,False]), .5, torch.tensor([0,1,1]))
        with self.assertRaises(ValueError): allocation_loss(p*float('nan'), y, w, mask, .5)

    def model(self, seed=17):
        torch.manual_seed(seed)
        c = DomainContract(tuple(f'n{i}' for i in range(14)), tuple(f'e{i}' for i in range(4)),
                           ('native_element_Bx_T','native_element_By_T'), 'synthetic-train-only', 'native_element_reported', 'native_explicit_ID')
        return DualSolverMGN({'elmer':c,'motorcad':c}, 12, ('MotorCAD_N45P1_formal_UNKNOWN',), width=32, depth=3)

    def test_parameter_count_and_all_paired_initial_tensor_hashes(self):
        hashes = []
        for seed in (17, 29, 43):
            a, b = self.model(seed), self.model(seed)
            self.assertEqual(sum(p.numel() for p in a.parameters()), 70182)
            digest = tensor_state_sha256(a.state_dict())
            self.assertEqual(digest, tensor_state_sha256(b.state_dict()))
            hashes.append(digest)
        self.assertEqual(len(set(hashes)), 3)

    def test_actual_backbone_alpha0_adam_one_step_bit_exact(self):
        original = self.model()
        candidate = copy.deepcopy(original)
        graph = {'node': torch.randn(6,14), 'edge_index': torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]]),
                 'edge_attr': torch.randn(8,4), 'weights': torch.arange(1,7,dtype=torch.float32), 'batch': torch.zeros(6,dtype=torch.long)}
        design = torch.randn(1,12); fidelity = torch.zeros(1,dtype=torch.long); target = torch.randn(6,2)
        mask = torch.tensor([False,True,True,False,True,False])
        aopt = torch.optim.Adam(original.parameters(), lr=.001)
        bopt = torch.optim.Adam(candidate.parameters(), lr=.001)
        def step(model,opt,use_new):
            opt.zero_grad(set_to_none=True)
            pred = model.forward_fields('motorcad',graph,design,fidelity)['fields']
            w=graph['weights']
            loss = allocation_loss(pred,target,w,mask,0.) if use_new else (w[:,None]*(pred-target)**2).sum()/(2*w.sum())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
            grads = {name: None if p.grad is None else p.grad.clone() for name,p in model.named_parameters()}
            opt.step(); return loss.detach(), grads
        la, ga=step(original,aopt,False); lb, gb=step(candidate,bopt,True)
        self.assertTrue(torch.equal(la,lb))
        for name in ga:
            self.assertTrue(ga[name] is gb[name] is None or torch.equal(ga[name],gb[name]),name)
        for (name,a),(other,b) in zip(original.named_parameters(),candidate.named_parameters()):
            self.assertEqual(name,other); self.assertTrue(torch.equal(a,b),name)
        self.assertEqual(tensor_state_sha256(original.state_dict()),tensor_state_sha256(candidate.state_dict()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
