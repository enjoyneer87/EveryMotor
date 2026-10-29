"""Synthetic metric/identity tests only; no checkpoint, GPU, or native data."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

import evaluate_a1_allocation as evaluator
import evaluate_common_airgap as common
from strict_airgap import build_strict_airgap, finish_statistics
from test_strict_airgap import fixture, analytic_field


def metadata_fixture(arm="L0", seed=17):
    selection = {"records": [dict(case_index=i, split="train", geometry_group=f"train{i % 288:04}", source_sha256="a" * 64) for i in range(328)]
                            + [dict(case_index=1000+i, split="development", geometry_group=f"dev{i:03}", source_sha256="b" * 64) for i in range(32)]}
    hashes = {key: format(i+1, "064x") for i, key in enumerate(evaluator.HASH_KEYS)}
    meta = dict(status="COMPLETED", arm=arm, alpha=evaluator.ARMS[arm], seed=seed, data_arm="A0", phase_encoding="mechanical",
                training_updates=88565, train_steps=sorted(common.SEEN), train_graphs=2952, train_conditions=328,
                development_conditions=32, old_checkpoint_reused=False, new_FEA=0, original_val_test_used=0,
                train_groups=[f"train{i:04}" for i in range(288)], development_groups=[f"dev{i:03}" for i in range(32)], **hashes)
    job = dict(id=f"{arm}_s{seed}", arm=arm, alpha=evaluator.ARMS[arm], seed=seed, phase_encoding="mechanical", metadata_hashes=hashes,
               checkpoint={"sha256": "c" * 64}, selection={"sha256": hashes["selection_sha256"]},
               training_mask_manifest={"sha256": hashes["mask_manifest_sha256"]}, geometry_manifest={"sha256": "d" * 64})
    train = copy.deepcopy(meta)
    train.update(checkpoint_sha256="c" * 64,
                 metrics=[dict(epoch=30, training_updates=88565, development={"all": {"snapshots": 288, "vector_nRMSE_percent": 10.}})])
    return {"epoch": 30, "epoch_equivalent": 30, "metadata": meta}, job, selection, train


def synthetic_graph(op):
    ids, truth = analytic_field(op)
    node = np.zeros((len(ids), 14), dtype=np.float32); node[-1, 6] = 1.
    reg = np.array([1, 1, 3], dtype=np.int64)
    weights = np.r_[op.band.area_m2, 50.e-6]
    return ids, truth, weights, node, reg


class EvaluationTests(unittest.TestCase):
    def test_01_strict_new_metadata(self):
        for arm in evaluator.ARMS:
            for seed in (17, 29, 43):
                checkpoint, job, selection, train = metadata_fixture(arm, seed)
                self.assertEqual(evaluator.validate_new_metadata(checkpoint, job, selection, train)["arm"], arm)
        for corruption in ("arm", "alpha", "bool_alpha", "seed", "updates", "epoch", "status", "hash", "initial", "trace", "train_sha", "phases", "anchor_count", "old"):
            checkpoint, job, selection, train = metadata_fixture()
            if corruption == "arm": checkpoint["metadata"]["arm"] = "A0"
            elif corruption == "alpha": checkpoint["metadata"]["alpha"] = .5
            elif corruption == "bool_alpha": checkpoint["metadata"]["alpha"] = False
            elif corruption == "seed": job["seed"] = 19
            elif corruption == "updates": checkpoint["metadata"]["training_updates"] = 88564
            elif corruption == "epoch": checkpoint["epoch"] = 29
            elif corruption == "status": train["status"] = "RUNNING"
            elif corruption == "hash": checkpoint["metadata"]["loss_module_sha256"] = "wrong"
            elif corruption == "initial": train["initial_tensor_state_sha256"] = "0" * 64
            elif corruption == "trace": checkpoint["metadata"]["data_order_LR_trace_sha256"] = "0" * 64
            elif corruption == "train_sha": train["checkpoint_sha256"] = "e" * 64
            elif corruption == "phases": train["train_steps"][-1] = 44
            elif corruption == "anchor_count": train["metrics"][-1]["development"]["all"]["snapshots"] = 1440
            else: checkpoint["metadata"]["old_checkpoint_reused"] = True
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                evaluator.validate_new_metadata(checkpoint, job, selection, train)

    def test_02_mask_admission_coverage_and_bindings(self):
        _, job, selection, _ = metadata_fixture()
        mask = dict(status="ADMITTED", selection_sha256=job["selection"]["sha256"],
                    bundle_manifest_sha256=job["metadata_hashes"]["bundle_manifest_sha256"],
                    dev_geometry_manifest_sha256=job["geometry_manifest"]["sha256"], train_steps=sorted(common.SEEN))
        for split, key in (("train", "records"), ("development", "development_records")):
            mask[key] = [r | {"graphs": [{"step": s} for s in sorted(common.SEEN)]} for r in selection["records"] if r["split"] == split]
        evaluator.validate_mask_manifest(mask, job, selection)
        for kind in ("missing", "duplicate", "wrong_source", "wrong_phase", "wrong_dev_binding"):
            bad = copy.deepcopy(mask)
            if kind == "missing": bad["records"].pop()
            elif kind == "duplicate": bad["records"][-1] = bad["records"][0]
            elif kind == "wrong_source": bad["records"][0]["source_sha256"] = "z" * 64
            elif kind == "wrong_phase": bad["records"][0]["graphs"][-1]["step"] = 44
            else: bad["dev_geometry_manifest_sha256"] = "0" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError): evaluator.validate_mask_manifest(bad, job, selection)

    def test_03_common_fields_torque_unchanged_and_anchor(self):
        op = build_strict_airgap(**fixture()); ids, truth, weights, node, reg = synthetic_graph(op)
        old, new, conductors = {}, {}, {}
        for step in range(45):
            reference = common.accumulate_snapshot(old, op, ids, 1.1*truth, truth, weights, 44, step, 75.)
            actual = common.accumulate_snapshot(new, op, ids, 1.1*truth, truth, weights, 44, step, 75.)
            evaluator.accumulate_copper(new, conductors, 1.1*truth, truth, weights, node, reg, 44, step)
            for a, b in zip(actual, reference): np.testing.assert_array_equal(a, b)
        for key in old: self.assertEqual(old[key], new[key])
        train = {"metrics": [{"development": {"all": {"vector_nRMSE_percent": 10.}}}]}
        self.assertEqual(evaluator.new_reproduction_gate(new, train)["status"], "PASS_ARITHMETIC_ONLY")
        train["metrics"][-1]["development"]["all"]["vector_nRMSE_percent"] += .001
        with self.assertRaises(ValueError): evaluator.new_reproduction_gate(new, train)

    def test_04_copper_B2_matches_original_uniform_snapshot_reduction(self):
        stats, conductors = {}, {}
        raw = []
        for case, area, truth_value, pred_value in ((44, 1., 1., 2.), (45, 2., 2., 3.)):
            weights = np.array([area]); truth = np.array([[truth_value, 0.]]); pred = np.array([[pred_value, 0.]])
            node = np.zeros((1, 14)); node[0, 6] = 1; reg = np.array([12], np.int64)
            for step in range(45):
                evaluator.accumulate_copper(stats, conductors, pred, truth, weights, node, reg, case, step)
            raw.append((area, truth_value**2, pred_value**2))
        rows, result = evaluator.finish_copper(conductors, {44, 45})
        expected = 100 * sum(a*(p-t) for a,t,p in raw) / sum(a*t for a,t,p in raw)
        self.assertAlmostEqual(expected, 100*13/9)
        for phase in evaluator.PHASE_COUNTS:
            self.assertAlmostEqual(result[phase]["signed_bias_percent"], expected)
            self.assertAlmostEqual(result[phase]["area_weighted_relative_MAE_percent"], expected)
        self.assertEqual(len(rows), 10)
        self.assertEqual(stats["all45/copper/area"].count, 90)

    def test_05_copper_phase_counts_nonfinite_and_zero_reference(self):
        stats, conductors = {}, {}
        node = np.zeros((1, 14)); node[0, 6] = 1
        for step in range(45):
            evaluator.accumulate_copper(stats, conductors, np.ones((1, 2)), np.zeros((1, 2)), np.ones(1), node, np.array([12]), 44, step)
        _, result = evaluator.finish_copper(conductors, {44})
        self.assertIsNone(result["all45"]["signed_bias_percent"])
        self.assertEqual(result["all45"]["weighted_absolute_error_T2_m2"], 2.)
        bad = copy.deepcopy(conductors); bad[(44,12,"seen9")][-1] -= 1
        with self.assertRaises(ValueError): evaluator.finish_copper(bad, {44})
        for value in (np.nan, .6):
            bad_node = node.copy(); bad_node[0, 6] = value
            with self.assertRaises(ValueError): evaluator.accumulate_copper({}, {}, np.ones((1,2)), np.ones((1,2)), np.ones(1), bad_node, np.array([12]), 44, 0)

    def test_06_PA_mean_case_is_not_pooled_metric(self):
        op = build_strict_airgap(**fixture()); ids, truth, weights, node, reg = synthetic_graph(op)
        stats, conductors = {}, {}
        dev = {44:{"native_input":{"PhaseAdvance":75.}},45:{"native_input":{"PhaseAdvance":75.}},46:{"native_input":{"PhaseAdvance":20.}}}
        for case in dev:
            t = truth * (100 if case == 45 else 1); p = t * (2 if case == 44 else 1)
            for step in range(45):
                common.accumulate_snapshot(stats, op, ids, p, t, weights, case, step, dev[case]["native_input"]["PhaseAdvance"])
                evaluator.accumulate_copper(stats, conductors, p, t, weights, node, reg, case, step)
        result = evaluator.mean_case_metrics(stats, dev)
        for region in ("all", "a1", "copper"):
            self.assertAlmostEqual(result[f"all45/PA_ge_70/{region}"]["mean_case_Bvector_nrmse_percent"], 50.)
        pooled = finish_statistics(stats["all45/PA_ge_70/all/area"]).vector_nrmse_percent
        self.assertLess(pooled, 1.)

    def test_07_historical_A0_large_difference_is_report_only(self):
        op = build_strict_airgap(**fixture()); ids, truth, weights, _, _ = synthetic_graph(op)
        stats = {}
        for step in range(45): common.accumulate_snapshot(stats, op, ids, 2*truth, truth, weights, 44, step, 75.)
        ref = dict(status="COMPLETED", arm="A0", seed=17, graphs_processed=1440, checkpoint_sha256="a"*64,
                   selection_sha256="b"*64, bundle_manifest_sha256="c"*64, geometry_manifest_sha256="d"*64,
                   field_metrics={f"all45/{r}/area":{"vector_nrmse_percent":1.,"magnitude_nrmse_percent":1.} for r in ("all","a1")},
                   pooled_torque={"nrmse_percent":1.})
        job = dict(arm="L0", seed=17, selection={"sha256":"b"*64}, bundle_manifest_sha256="c"*64, geometry_manifest={"sha256":"d"*64})
        actual = evaluator.historical_reference_differences(stats,{"nrmse_percent":100.},ref,job)
        self.assertEqual(actual["status"],"REPORT_ONLY_NO_GATE")
        self.assertAlmostEqual(actual["differences_pp"]["torque_pooled_nrmse_percent"],99.)
        with self.assertRaises(ValueError): evaluator.historical_reference_differences(stats,{"nrmse_percent":100.},ref,job|{"arm":"L05"})

    def test_08_no_execution_paths_or_unreleased_output(self):
        source = inspect.getsource(evaluator)
        for forbidden in ("torch.optim", ".backward(", ".train(", "h5py.File", "subprocess.", "nodal_prior_features("):
            self.assertNotIn(forbidden, source)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/"never"
            with self.assertRaises(ValueError): evaluator.run_job(dict(deadline_UTC="2099-01-01T00:00:00+00:00",release_status="HOLD"),"a"*64,out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
