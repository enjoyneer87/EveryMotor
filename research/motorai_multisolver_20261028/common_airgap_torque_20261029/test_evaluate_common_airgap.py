"""Synthetic common-evaluator gate tests. No checkpoint or native dataset opened."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import evaluate_common_airgap as common
from strict_airgap import build_strict_airgap, finish_statistics
from test_strict_airgap import fixture, analytic_field


def sidecar_values(op):
    values = {name: getattr(op, name) for name in ("admitted_element_ids", "a1_element_ids", "a1_node_ids",
               "a1_node_xy_mm", "a1_triangle_node_ids", "a1_orientation", "axial_length_m")}
    values.update({name: getattr(op.band, name) for name in ("radius_m", "area_m2", "cos_theta", "sin_theta",
                   "r_inner_m", "r_outer_m", "sector_span_rad", "symmetry_multiplier", "region_codes")})
    return values


def checkpoint_fixture(arm="A0", seed=17):
    selection = {"records": [{"geometry_group": f"train{i:04}", "split": "train"} for i in range(288)]
                              + [{"geometry_group": f"dev{i:03}", "split": "development"} for i in range(32)]}
    hashes = {"model_sha256": "a" * 64, "selection_sha256": "b" * 64, "plan_sha256": "c" * 64}
    encoding = common.ARMS[arm]
    meta = dict(seed=seed, arm=arm, phase_encoding=encoding, training_updates=88565, old_checkpoint_reused=False,
                train_groups=[f"train{i:04}" for i in range(288)], development_groups=[f"dev{i:03}" for i in range(32)], **hashes)
    if arm == "A2": meta["pole_pairs"] = 4
    if arm in ("C", "G"): meta["context_scope"] = {"C": "component", "G": "graph"}[arm]
    job = dict(id=f"{arm}_s{seed}", arm=arm, seed=seed, phase_encoding=encoding, metadata_hashes=hashes,
               checkpoint={"sha256": "d" * 64})
    return {"epoch": 30, "metadata": meta}, job, selection


def graph_fixture(op):
    ids, field = analytic_field(op)
    return dict(native_element_id=ids, native_tri_node_ids=np.vstack([op.a1_triangle_node_ids, [80, 90, 100]]),
                native_reg_code=np.array([1, 1, 3]), node=np.zeros((3, 14), dtype=np.float32),
                weights=np.r_[op.band.area_m2, 50.e-6].astype(np.float32), target_B_T=field,
                step=np.array(0), edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                edge_attr=np.zeros((2, 4), dtype=np.float32))


class CommonEvaluatorTests(unittest.TestCase):
    def test_01_sidecar_roundtrip_and_corruption(self):
        op = build_strict_airgap(**fixture())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.npz"
            np.savez(path, **sidecar_values(op))
            actual = common.load_operator(path)
            self.assertTrue(np.array_equal(actual.admitted_element_ids, op.admitted_element_ids))
            self.assertTrue(np.array_equal(actual.a1_element_ids, op.a1_element_ids))
            self.assertTrue(np.array_equal(actual.band.area_m2, op.band.area_m2))
            bad = sidecar_values(op); bad["area_m2"] = op.band.area_m2 * 2
            np.savez(path, **bad)
            with self.assertRaises(ValueError): common.load_operator(path)

    def test_02_graph_id_permutation_and_a2_mapping(self):
        op = build_strict_airgap(**fixture()); graph = graph_fixture(op)
        np.testing.assert_array_equal(common.validate_graph(graph, op, 0), [0, 1])
        for key in ("native_element_id", "native_tri_node_ids", "native_reg_code", "node", "weights", "target_B_T"):
            graph[key] = graph[key][[2, 1, 0]]
        np.testing.assert_array_equal(common.validate_graph(graph, op, 0), [2, 1])
        graph["native_element_id"][-1] = 900
        with self.assertRaises(ValueError): common.validate_graph(graph, op, 0)

    def test_03_graph_incidence_area_and_nonfinite_guards(self):
        op = build_strict_airgap(**fixture())
        for kind in ("incidence", "area", "nonfinite", "step", "fractional_step", "edge", "region", "region_float"):
            graph = graph_fixture(op)
            if kind == "incidence": graph["native_tri_node_ids"][0] = [80, 90, 100]
            elif kind == "area": graph["weights"][0] *= 1.001
            elif kind == "nonfinite": graph["target_B_T"][0, 0] = np.nan
            elif kind == "step": graph["step"] = np.array(1)
            elif kind == "fractional_step": graph["step"] = np.array(.5)
            elif kind == "edge": graph["edge_index"][0, 0] = 3
            elif kind == "region_float": graph["native_reg_code"] = graph["native_reg_code"].astype(float)
            else: graph["native_reg_code"][0] = 3
            with self.subTest(kind=kind), self.assertRaises(ValueError): common.validate_graph(graph, op, 0)
        with self.assertRaises(ValueError): common.integer_ids(np.array([2**63], dtype=np.uint64), "native IDs")

    def test_04_all_arms_metadata_and_final_update(self):
        for arm in common.ARMS:
            checkpoint, job, selection = checkpoint_fixture(arm)
            self.assertEqual(common.validate_checkpoint_metadata(checkpoint, job, selection), [])
            checkpoint["metadata"]["training_updates"] -= 1
            with self.subTest(arm=arm), self.assertRaises(ValueError): common.validate_checkpoint_metadata(checkpoint, job, selection)

    def test_05_explicit_legacy_hash_bound_absence(self):
        checkpoint, job, selection = checkpoint_fixture("A0", 29)
        del checkpoint["metadata"]["phase_encoding"]
        with self.assertRaises(ValueError): common.validate_checkpoint_metadata(checkpoint, job, selection)
        job["legacy_absent_fields"] = ["phase_encoding"]
        evidence = dict(checkpoint_sha256="d" * 64, allowed_absent_fields=["phase_encoding"])
        self.assertEqual(common.validate_checkpoint_metadata(checkpoint, job, selection, evidence), ["phase_encoding"])
        with self.assertRaises(ValueError): common.validate_checkpoint_metadata(checkpoint, job, selection, evidence | {"checkpoint_sha256": "e" * 64})
        checkpoint, job, selection = checkpoint_fixture("A2", 29)
        del checkpoint["metadata"]["phase_encoding"]; job["legacy_absent_fields"] = ["phase_encoding"]
        with self.assertRaises(ValueError): common.validate_checkpoint_metadata(checkpoint, job, selection, evidence)

    def test_06_wrong_encoding_scope_hash_split_and_run_id(self):
        for kind in ("encoding", "scope", "hash", "split", "id"):
            checkpoint, job, selection = checkpoint_fixture("C", 17)
            if kind == "encoding": checkpoint["metadata"]["phase_encoding"] = "electrical"
            elif kind == "scope": checkpoint["metadata"]["context_scope"] = "graph"
            elif kind == "hash": checkpoint["metadata"]["model_sha256"] = "bad"
            elif kind == "split": checkpoint["metadata"]["development_groups"][0] = "train0000"
            else: job["id"] = "A0_s17"
            with self.subTest(kind=kind), self.assertRaises(ValueError): common.validate_checkpoint_metadata(checkpoint, job, selection)

    def test_07_phase_encoding_and_phase_regions(self):
        m = np.deg2rad(-18.)
        d = np.zeros(12, dtype=np.float32); d[10:] = [np.sin(m), np.cos(m)]; old = d.copy()
        np.testing.assert_array_equal(common.encoded_design(d, -18., "mechanical"), old)
        np.testing.assert_allclose(common.encoded_design(d, -18., "electrical")[10:], [np.sin(4*m), np.cos(4*m)], rtol=1e-6)
        np.testing.assert_array_equal(d, old)
        with self.assertRaises(ValueError): common.encoded_design(d, -16., "electrical")
        self.assertEqual(common.phase_scopes(40), ("all45", "seen9"))
        self.assertEqual(common.phase_scopes(41), ("all45", "unused36", "after_last_seen"))
        self.assertEqual(common.phase_scopes(39), ("all45", "unused36", "interior_unused"))

    def test_08_all_a1_statistics_and_original_score_reproduction(self):
        op = build_strict_airgap(**fixture()); graph = graph_fixture(op)
        ids, truth, weights = graph["native_element_id"], graph["target_B_T"], graph["weights"].astype(float)
        pred = truth * 1.1; stats = {}
        for step in range(45):
            common.accumulate_snapshot(stats, op, ids, pred, truth, weights, 44, step, 75.)
        original = {"metrics": {phase + "/all": {"B_vector_nRMSE_percent": 10.} for phase in ("all45", "seen9")}}
        train = {"metrics": [{"development": {"all": {"vector_nRMSE_percent": 10.}}}]}
        self.assertEqual(common.reproduction_gate(stats, original, train)["status"], "PASS_ARITHMETIC_ONLY")
        self.assertEqual(stats["all45/a1/area"].count, 90)
        self.assertEqual(stats["seen9/a1/unweighted"].count, 18)
        self.assertEqual(stats["after_last_seen/a1/area"].count, 8)
        self.assertEqual(stats["all45/PA_ge_70/all/area"].count, 135)
        bad = copy.deepcopy(original); bad["metrics"]["all45/all"]["B_vector_nRMSE_percent"] = 10.001
        with self.assertRaises(ValueError): common.reproduction_gate(stats, bad, train)

    def test_09_torque_pooling_no_cross_case_ripple(self):
        summary = common.pooled_torque([2., 4., 6.], [1., 2., 3.])
        self.assertAlmostEqual(summary["nrmse_percent"], 100.)
        self.assertEqual(summary["ripple"], "NOT_POOLED_ACROSS_CASES")
        zero = common.pooled_torque([1., 2.], [0., 0.])
        self.assertIsNone(zero["nrmse_percent"])
        self.assertAlmostEqual(zero["rmse_Nm"], np.sqrt(2.5))
        with self.assertRaises(ValueError): common.pooled_torque([np.nan], [1.])
        with self.assertRaises(FloatingPointError): common.pooled_torque([1e308], [1.])

    def test_10_pinned_file_and_root_containment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / "metadata.json"
            path.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
            spec = {"path": str(path), "sha256": common.sha256(path), "bytes": path.stat().st_size}
            self.assertTrue(common.pinned_json(spec)["synthetic"])
            with self.assertRaises(ValueError): common.pinned_json(spec | {"sha256": "0" * 64})
            nested = root / "nested"; nested.mkdir()
            with self.assertRaises(ValueError): common.child_file(nested, "../metadata.json")

    def test_11_source_pin_requires_actual_evaluator(self):
        source = Path(common.__file__).resolve()
        self.assertIn(source.name, common.verify_sources([{"path": str(source), "sha256": common.sha256(source)}], lambda: None))
        with self.assertRaises(ValueError): common.verify_sources([], lambda: None)

    def test_12_no_training_or_hdf5_execution_path(self):
        source = inspect.getsource(common)
        for forbidden in ("torch.optim", ".backward(", ".train(", "h5py.File", "nodal_prior_features(", "subprocess."):
            self.assertNotIn(forbidden, source)
        with tempfile.TemporaryDirectory() as folder:
            job = dict(deadline_UTC="2099-01-01T00:00:00+00:00", release_status="DRAFT")
            out = Path(folder) / "must_not_create"
            with self.assertRaises(ValueError): common.run_job(job, "0" * 64, out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
