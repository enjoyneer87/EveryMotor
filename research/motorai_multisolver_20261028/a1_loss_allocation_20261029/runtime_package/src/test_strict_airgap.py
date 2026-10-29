"""Synthetic-only tests; never read an experimental field or checkpoint."""
import dataclasses
import unittest

import numpy as np

from strict_airgap import (
    FieldStatistics, align_airgap_field, build_strict_airgap,
    field_sufficient_statistics, finish_statistics, merge_statistics,
    strict_arkkio_torque, summarize_torque45,
    validate_all45_stationarity, validate_snapshot_coverage,
)
from eval.torque import MU0


def fixture():
    return dict(
        node_ids=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        node_xy_mm=np.array([[100., 0.], [110., 0.], [100., 10.], [110., 10.],
                             [90., 0.], [90., 10.], [95., 0.],
                             [120., 0.], [130., 0.], [120., 10.]]),
        element_ids=np.array([900, 200, 100, 300]),
        triangle_node_ids=np.array([[50, 60, 70], [10, 20, 30], [20, 40, 30], [80, 90, 100]]),
        region_codes=np.array([2, 1, 1, 3]), region_names={1: "a1", 2: "a2", 3: "turn_1_1"},
        moving_region_codes=[2], moving_node_ids=[50, 60, 70],
        admitted_element_ids=[100, 200, 300], axial_length_m=.15, symmetry_multiplier=8,
    )


def analytic_field(op, br=2., bt=3.):
    band = op.band
    field = np.array([[br * c - bt * s, br * s + bt * c]
                      for c, s in zip(band.cos_theta, band.sin_theta)])
    return np.array([100, 200, 300]), np.vstack([field, [5., 7.]])


class StrictAirgapTests(unittest.TestCase):
    def test_01_analytic_torque_units_sign_and_stack(self):
        op = build_strict_airgap(**fixture())
        ids, field = analytic_field(op)
        expected = .15 * 8 / (MU0 * op.band.thickness_m) * np.sum(op.band.radius_m * 6 * op.band.area_m2)
        self.assertAlmostEqual(strict_arkkio_torque(op, ids, field), expected, places=10)
        np.testing.assert_allclose(op.band.area_m2, [50.e-6, 50.e-6], rtol=1e-14, atol=0)
        _, negative = analytic_field(op, bt=-3)
        self.assertAlmostEqual(strict_arkkio_torque(op, ids, negative), -expected, places=10)
        args = fixture(); args["axial_length_m"] = 1.
        one_m = build_strict_airgap(**args)
        self.assertAlmostEqual(strict_arkkio_torque(one_m, ids, field) * .15, expected, places=10)

    def test_02_a2_excluded_and_id_permutations(self):
        base = fixture(); op = build_strict_airgap(**base)
        ids, field = analytic_field(op)
        permutation = [2, 0, 1]
        self.assertEqual(strict_arkkio_torque(op, ids, field), strict_arkkio_torque(op, ids[permutation], field[permutation]))
        node_order = np.arange(len(base["node_ids"]))[::-1]
        element_order = [3, 2, 0, 1]
        base["node_ids"], base["node_xy_mm"] = base["node_ids"][node_order], base["node_xy_mm"][node_order]
        for name in ("element_ids", "triangle_node_ids", "region_codes"):
            base[name] = base[name][element_order]
        permuted = build_strict_airgap(**base)
        self.assertEqual(strict_arkkio_torque(op, ids, field), strict_arkkio_torque(permuted, ids, field))
        validate_all45_stationarity({step: op if step % 2 else permuted for step in range(45)})

    def test_03_field_missing_duplicate_unknown_ids_and_shape(self):
        op = build_strict_airgap(**fixture()); ids, field = analytic_field(op)
        for bad_ids, bad_field in ((ids[:-1], field[:-1]), ([100, 100, 300], field),
                                   ([100, 200, 999], field), ([100, 200, 300, 900], np.vstack([field, [0., 0.]])),
                                   (ids, field[:, :1])):
            with self.subTest(ids=bad_ids), self.assertRaises(ValueError):
                align_airgap_field(op, bad_ids, bad_field)

    def test_04_geometry_id_errors(self):
        changes = [dict(node_ids=[10] * 10), dict(element_ids=[900, 200, 200, 300]),
                   dict(admitted_element_ids=[100, 200, 999]), dict(admitted_element_ids=[200, 300]),
                   dict(moving_node_ids=[999]), dict(moving_region_codes=[9]),
                   dict(region_names={1: "a1", 2: "a2"})]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_strict_airgap(**(fixture() | change))
        args = fixture(); args["triangle_node_ids"][0, 0] = 999
        with self.assertRaises(ValueError): build_strict_airgap(**args)

    def test_05_no_a1_or_unproved_motion(self):
        changes = [dict(region_names={1: "air", 2: "a2", 3: "turn_1_1"}),
                   dict(moving_region_codes=None), dict(moving_node_ids=None),
                   dict(moving_region_codes=[1, 2]), dict(moving_node_ids=[10, 50, 60, 70])]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_strict_airgap(**(fixture() | change))

    def test_06_geometry_degenerate_nonfinite_overflow(self):
        for kind in ("degenerate", "nan", "overflow", "zero_radius"):
            args = fixture()
            if kind == "degenerate": args["node_xy_mm"][1] = args["node_xy_mm"][0]
            elif kind == "nan": args["node_xy_mm"][0, 0] = np.nan
            elif kind == "overflow": args["node_xy_mm"] *= 1e200
            else: args["node_xy_mm"][0] = [0., 0.]
            with self.subTest(kind=kind), self.assertRaises(ValueError): build_strict_airgap(**args)

    def test_07_length_symmetry_contract(self):
        for length in (0., -1., np.nan, np.inf, True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                build_strict_airgap(**(fixture() | dict(axial_length_m=length)))
        for symmetry in (0, -8, 8., None, True):
            with self.subTest(symmetry=symmetry), self.assertRaises(ValueError):
                build_strict_airgap(**(fixture() | dict(symmetry_multiplier=symmetry)))

    def test_08_clockwise_valid_but_orientation_change_rejected(self):
        op = build_strict_airgap(**fixture())
        args = fixture(); args["triangle_node_ids"] = args["triangle_node_ids"][:, [0, 2, 1]]
        clockwise = build_strict_airgap(**args)
        self.assertTrue(np.array_equal(op.band.area_m2, clockwise.band.area_m2))
        validate_all45_stationarity(dict.fromkeys(range(45), clockwise))
        with self.assertRaises(ValueError):
            validate_all45_stationarity(dict.fromkeys(range(45), op) | {44: clockwise})

    def test_09_stationarity_changes_and_coverage(self):
        op = build_strict_airgap(**fixture())
        validate_all45_stationarity(dict.fromkeys(range(45), op))
        with self.assertRaises(ValueError): validate_all45_stationarity(dict.fromkeys(range(44), op))
        args = fixture(); args["node_xy_mm"][0, 0] += .001
        changed = build_strict_airgap(**args)
        with self.assertRaises(ValueError): validate_all45_stationarity(dict.fromkeys(range(45), op) | {10: changed})
        args = fixture(); args["triangle_node_ids"][[1, 2]] = args["triangle_node_ids"][[2, 1]]
        changed = build_strict_airgap(**args)
        with self.assertRaises(ValueError): validate_all45_stationarity(dict.fromkeys(range(45), op) | {10: changed})

    def test_10_immutable_and_no_input_mutation(self):
        args = fixture(); xy = args["node_xy_mm"].copy(); tri = args["triangle_node_ids"].copy()
        op = build_strict_airgap(**args); ids, field = analytic_field(op); before = field.copy()
        aligned = align_airgap_field(op, ids, field); strict_arkkio_torque(op, ids, field)
        self.assertTrue(np.array_equal(xy, args["node_xy_mm"]))
        self.assertTrue(np.array_equal(tri, args["triangle_node_ids"]))
        self.assertTrue(np.array_equal(before, field))
        args["node_xy_mm"][:] = 0
        self.assertNotEqual(float(op.a1_node_xy_mm.sum()), 0)
        for array in (op.a1_node_xy_mm, op.band.area_m2, aligned):
            with self.assertRaises(ValueError): array.setflags(write=True)
        with self.assertRaises(dataclasses.FrozenInstanceError): op.axial_length_m = 2.

    def test_11_nonfinite_and_overflow_integrand(self):
        op = build_strict_airgap(**fixture()); ids, field = analytic_field(op)
        for value in (np.nan, np.inf, 1e308):
            bad = field.copy(); bad[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError): strict_arkkio_torque(op, ids, bad)
        bad = field.copy(); bad[-1, 0] = np.nan
        with self.assertRaises(ValueError): strict_arkkio_torque(op, ids, bad)

    def test_12_pooled_weighted_unweighted_statistics(self):
        pred = np.array([[2., 0.], [0., 3.], [1., 2.]])
        truth = np.array([[1., 0.], [0., 1.], [2., 2.]])
        weights = np.array([1., 2., 4.])
        direct = field_sufficient_statistics(pred, truth, weights)
        pooled = merge_statistics([field_sufficient_statistics(pred[:1], truth[:1], weights[:1]),
                                   field_sufficient_statistics(pred[1:], truth[1:], weights[1:])])
        self.assertEqual(direct, pooled)
        score = finish_statistics(pooled)
        expected = 100 * np.sqrt(np.sum(weights[:, None] * (pred - truth) ** 2) / np.sum(weights[:, None] * truth ** 2))
        self.assertAlmostEqual(score.vector_nrmse_percent, expected)
        unweighted = finish_statistics(field_sufficient_statistics(pred, truth))
        self.assertAlmostEqual(unweighted.vector_nrmse_percent, 100 * np.sqrt(np.sum((pred - truth) ** 2) / np.sum(truth ** 2)))
        self.assertNotEqual(score.vector_nrmse_percent, unweighted.vector_nrmse_percent)

    def test_13_statistics_invalid_zero_and_overflow(self):
        a = np.ones((2, 2))
        for weights in ([1., 0.], [1., -1.], [np.nan, 1.], [1.]):
            with self.subTest(weights=weights), self.assertRaises(ValueError): field_sufficient_statistics(a, a, weights)
        zero = finish_statistics(field_sufficient_statistics(a, a * 0))
        self.assertEqual(zero.normalization_status, "UNDEFINED_ZERO_REFERENCE")
        self.assertIsNone(zero.vector_nrmse_percent)
        self.assertIsNone(zero.magnitude_nrmse_percent)
        self.assertAlmostEqual(zero.vector_rmse_T, np.sqrt(2.))
        self.assertAlmostEqual(zero.magnitude_rmse_T, np.sqrt(2.))
        with self.assertRaises(ValueError): field_sufficient_statistics(a * 1e308, a)
        with self.assertRaises(ValueError): merge_statistics([])
        with self.assertRaises(ValueError): merge_statistics([FieldStatistics(2, 1., -1., 0., 1.)])
        with self.assertRaises(ValueError): merge_statistics([FieldStatistics(2, 1e308, 1e308, 0., 1e308)] * 2)

    def test_14_torque45_exact_near_zero_and_invalid(self):
        steps = np.arange(45); truth = np.linspace(-2., 2., 45); pred = 2 * truth
        summary = summarize_torque45(steps[::-1], pred[::-1], truth[::-1])
        self.assertAlmostEqual(summary.nrmse_percent, 100.)
        self.assertAlmostEqual(summary.pred_ripple_Nm, 8.)
        self.assertAlmostEqual(summary.truth_ripple_Nm, 4.)
        self.assertAlmostEqual(summary.mean_error_rms_normalized_percent, 0.)
        zero = summarize_torque45(steps, pred, np.zeros(45))
        self.assertEqual(zero.normalization_status, "UNDEFINED_ZERO_REFERENCE")
        self.assertIsNone(zero.nrmse_percent)
        self.assertIsNone(zero.mean_error_rms_normalized_percent)
        self.assertAlmostEqual(zero.rmse_Nm, np.sqrt(np.mean(pred ** 2)))
        self.assertAlmostEqual(zero.pred_ripple_Nm, 8.)
        self.assertEqual(zero.truth_ripple_Nm, 0.)
        for bad_steps, bad_pred, bad_truth in ((steps[:-1], pred[:-1], truth[:-1]),
                                              (np.zeros(45, dtype=int), pred, truth),
                                              (steps, np.full(45, 1e308), truth),
                                              (steps, np.full(45, np.nan), truth)):
            with self.subTest(), self.assertRaises(ValueError): summarize_torque45(bad_steps, bad_pred, bad_truth)

    def test_15_snapshot_coverage(self):
        cases = [44, 55]
        pairs = np.array([(case, step) for case in cases for step in range(45)])
        validate_snapshot_coverage(pairs[::-1], cases)
        for bad in (pairs[:-1], np.vstack([pairs, pairs[0]]), pairs + [1, 0]):
            with self.subTest(), self.assertRaises(ValueError): validate_snapshot_coverage(bad, cases)
        with self.assertRaises(ValueError): validate_snapshot_coverage(pairs, [44, 44])

    def test_16_excluded_a2_invalid_geometry_is_not_admitted(self):
        op = build_strict_airgap(**fixture()); ids, field = analytic_field(op)
        with self.assertRaises(ValueError):
            build_strict_airgap(**(fixture() | dict(admitted_element_ids=[100, 200, 300, 900])))
        for kind in ("degenerate", "overflow", "nonfinite"):
            args = fixture()
            if kind == "degenerate": args["node_xy_mm"][4:7] = [90., 0.]
            elif kind == "overflow": args["node_xy_mm"][4:7] *= 1e200
            else: args["node_xy_mm"][4:7] = np.nan
            other = build_strict_airgap(**args)
            self.assertEqual(strict_arkkio_torque(other, ids, field), strict_arkkio_torque(op, ids, field))
            args["admitted_element_ids"] = [100, 200, 300, 900]
            with self.subTest(kind=kind), self.assertRaises(ValueError): build_strict_airgap(**args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
