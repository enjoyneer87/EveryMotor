"""Tests for physical-unit field metrics and region grouping."""

import numpy as np
import pytest

from eval.mesh_regions import (
    REGION_GROUP_ORDER,
    build_region_grouping,
    classify_region_name,
    element_areas_m2,
    element_centroids_m,
)
from eval.metrics import aggregate_channel, channel_metric, field_metrics, region_metrics


def test_perfect_prediction_scores_zero_error_and_unit_r2():
    truth = np.array([0.1, -0.4, 1.2, 0.7])
    m = channel_metric(truth, truth, "Bx")
    assert m.rmse == pytest.approx(0.0)
    assert m.nrmse_pct == pytest.approx(0.0)
    assert m.r2 == pytest.approx(1.0)
    assert m.unit == "T"


def test_zero_prediction_scores_exactly_100_percent_nrmse():
    """The reference floor must land on 100%, so the scale is interpretable."""
    truth = np.array([0.3, -0.9, 1.4, 0.2])
    m = channel_metric(np.zeros_like(truth), truth, "Bx")
    assert m.nrmse_pct == pytest.approx(100.0)


def test_nrmse_follows_the_documented_convention():
    truth = np.array([1.0, -1.0, 1.0, -1.0])  # rms = 1
    pred = truth + 0.1
    assert channel_metric(pred, truth, "Bx").nrmse_pct == pytest.approx(10.0)


def test_metrics_are_in_physical_units_not_normalized():
    """Scaling the field scales RMSE but leaves nRMSE and R2 alone."""
    truth = np.array([0.2, 0.8, -0.5])
    pred = truth * 1.1
    small = channel_metric(pred, truth, "Bx")
    big = channel_metric(pred * 1000, truth * 1000, "Bx")

    assert big.rmse == pytest.approx(small.rmse * 1000)
    assert big.nrmse_pct == pytest.approx(small.nrmse_pct)
    assert big.r2 == pytest.approx(small.r2)


def test_non_finite_samples_are_dropped_not_zero_filled():
    """Zero-filling a NaN prediction would score it as a correct null field."""
    truth = np.array([1.0, 1.0, 1.0, 1.0])
    pred = np.array([1.0, np.nan, 1.0, np.inf])
    m = channel_metric(pred, truth, "Bx")
    assert m.n == 2
    assert m.rmse == pytest.approx(0.0)


def test_all_nan_prediction_reports_nan_not_a_good_score():
    m = channel_metric(np.full(4, np.nan), np.ones(4), "Bx")
    assert m.n == 0
    assert np.isnan(m.rmse) and np.isnan(m.nrmse_pct)


def test_zero_truth_channel_gives_nan_relative_error():
    m = channel_metric(np.array([0.1, 0.0]), np.zeros(2), "Je")
    assert np.isnan(m.nrmse_pct)
    assert m.rmse > 0


def test_bnorm_is_derived_when_both_components_are_present():
    truth = np.array([[3.0, 4.0], [0.0, 1.0]])
    pred = truth.copy()
    metrics = field_metrics(pred, truth, ("Bx", "By"))
    assert set(metrics) == {"Bx", "By", "Bnorm"}
    assert metrics["Bnorm"].rms_truth == pytest.approx(np.sqrt((25 + 1) / 2))


def test_bnorm_is_not_invented_for_a_single_component():
    truth = np.array([[1.0], [2.0]])
    assert set(field_metrics(truth, truth, ("A",))) == {"A"}


def test_channel_name_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="channel names"):
        field_metrics(np.zeros((3, 2)), np.zeros((3, 2)), ("Bx",))


def test_region_metrics_split_by_group():
    truth = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    pred = truth.copy()
    pred[2:] = 0.0  # only the airgap elements are wrong
    groups = np.array(["stator_yoke", "stator_yoke", "airgap", "airgap"])

    out = region_metrics(pred, truth, ("Bx", "By"), groups)
    assert out["stator_yoke"]["Bx"].nrmse_pct == pytest.approx(0.0)
    assert out["airgap"]["Bx"].nrmse_pct == pytest.approx(100.0)


def test_region_metrics_skip_groups_below_the_sample_floor():
    groups = np.array(["airgap", "shaft", "shaft"])
    truth = np.ones((3, 1))
    out = region_metrics(truth, truth, ("A",), groups, min_points=2)
    assert set(out) == {"shaft"}


def test_pooled_rmse_equals_the_concatenated_rmse():
    """Aggregation must not average ratios; it must pool the errors."""
    rng = np.random.default_rng(0)
    a_true, b_true = rng.normal(size=(50, 1)), rng.normal(scale=10.0, size=(30, 1))
    a_pred, b_pred = a_true + 0.1, b_true + 1.0

    per_sample = [
        field_metrics(a_pred, a_true, ("Bx",)),
        field_metrics(b_pred, b_true, ("Bx",)),
    ]
    pooled = aggregate_channel(per_sample, "Bx")

    cat_pred = np.concatenate([a_pred, b_pred])
    cat_true = np.concatenate([a_true, b_true])
    direct = channel_metric(cat_pred, cat_true, "Bx")

    assert pooled["n"] == 80
    assert pooled["rmse"] == pytest.approx(direct.rmse)
    assert pooled["nrmse_pct"] == pytest.approx(direct.nrmse_pct)


def test_aggregate_reports_tail_not_just_the_mean():
    per_sample = []
    for scale in (1.0, 1.0, 1.0, 50.0):
        truth = np.ones((10, 1))
        per_sample.append(field_metrics(truth * (1 + 0.01 * scale), truth, ("Bx",)))
    agg = aggregate_channel(per_sample, "Bx")
    assert agg["nrmse_pct_p95"] > agg["nrmse_pct_median"]


def test_aggregate_of_nothing_is_nan_not_zero():
    agg = aggregate_channel([], "Bx")
    assert agg["n"] == 0 and np.isnan(agg["nrmse_pct"])


# --- region grouping ---------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a1", "airgap"),
        ("a4", "airgap"),
        ("ArmatureSlotA6", "winding"),
        ("Turn_1_1", "winding"),
        ("Impreg_LossSlot", "winding"),
        ("StatorAir_2", "winding"),
        ("StatorWedge", "winding"),
        ("L1_1Magnet2N1", "magnet"),
        ("Rotor Pocket_3", "rotor_air"),
        ("Rotor", "rotor_iron"),
        ("Shaft", "shaft"),
        ("Stator_5", "_stator_iron"),
        ("SomethingElse", "other"),
    ],
)
def test_region_names_map_to_engineering_groups(name, expected):
    assert classify_region_name(name) == expected


def test_stator_iron_splits_into_teeth_and_yoke_at_the_slot_radius():
    reg = np.array([1, 1, 2])  # two stator iron elements, one slot
    names = {1: "Stator", 2: "ArmatureSlotA1"}
    radius = np.array([0.080, 0.095, 0.086])  # slot outer radius = 0.086

    grouping = build_region_grouping(reg, names, radius)
    assert grouping.group_of_element.tolist() == ["stator_teeth", "stator_yoke", "winding"]
    assert grouping.tooth_yoke_radius_m == pytest.approx(0.086)


def test_stator_iron_without_slots_is_reported_as_yoke_not_dropped():
    grouping = build_region_grouping([1], {1: "Stator"}, [0.09])
    assert grouping.group_of_element.tolist() == ["stator_yoke"]
    assert np.isnan(grouping.tooth_yoke_radius_m)


def test_every_group_name_is_declared_in_the_reported_order():
    reg = np.array([1, 2, 3, 4, 5, 6, 7])
    names = {1: "a1", 2: "Stator", 3: "ArmatureSlotA1", 4: "L1_1Magnet1N1",
             5: "Rotor", 6: "Rotor Pocket", 7: "Shaft"}
    radius = np.array([0.075, 0.095, 0.086, 0.070, 0.060, 0.065, 0.020])
    grouping = build_region_grouping(reg, names, radius)
    assert set(grouping.group_of_element) <= set(REGION_GROUP_ORDER)


def test_mismatched_region_and_radius_lengths_are_rejected():
    with pytest.raises(ValueError, match="align"):
        build_region_grouping([1, 2], {1: "a1"}, [0.075])


def test_element_geometry_helpers_convert_mm_to_metres():
    node_x = np.array([0.0, 3.0, 0.0])  # mm
    node_y = np.array([0.0, 0.0, 4.0])
    tri = (np.array([0]), np.array([1]), np.array([2]))

    cx, cy = element_centroids_m(node_x, node_y, tri)
    assert cx[0] == pytest.approx(1.0e-3)
    assert cy[0] == pytest.approx(4.0 / 3.0 * 1e-3)
    assert element_areas_m2(node_x, node_y, tri)[0] == pytest.approx(6.0e-6)
