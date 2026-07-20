"""Tests for the solution-derived input feature guard."""

import pytest

from eval.feature_guard import (
    FeatureLeakageError,
    assert_feature_count,
    assert_input_features_clean,
    classify_features,
    normalize_feature_name,
)

CLEAN_MGN_FEATURES = (
    "pos_x",
    "pos_y",
    "region_code",
    "time_s",
    "rotate_step",
    "Ratio_Bore",
    "Ratio_SlotDepth_ParallelSlot",
    "PeakCurrent",
    "PhaseAdvance",
)


def test_actual_training_features_pass():
    assert_input_features_clean(CLEAN_MGN_FEATURES)


@pytest.mark.parametrize("leaked", ["A", "J", "Je", "Bx", "By", "B", "Bnorm", "flux_linkage"])
def test_solution_derived_features_are_rejected(leaked):
    with pytest.raises(FeatureLeakageError, match="solution-derived"):
        assert_input_features_clean(CLEAN_MGN_FEATURES + (leaked,))


def test_the_docstring_layout_that_the_review_flagged_is_rejected():
    """The stale 11-feature docstring lists A and J as inputs; that must fail."""
    documented = (
        "pos_x", "pos_y", "A", "J", "region_code", "time_s", "rotate_step",
        "Ratio_Bore", "Ratio_SlotDepth", "PeakCurrent", "PhaseAdvance",
    )
    with pytest.raises(FeatureLeakageError) as excinfo:
        assert_input_features_clean(documented)
    assert "'A'" in str(excinfo.value) and "'J'" in str(excinfo.value)


def test_name_normalization_catches_spelling_variants():
    assert normalize_feature_name("B_norm") == normalize_feature_name("b norm") == "b_norm"
    with pytest.raises(FeatureLeakageError):
        assert_input_features_clean(["vector_potential"])
    with pytest.raises(FeatureLeakageError):
        assert_input_features_clean(["Current Density"])


def test_unknown_features_are_surfaced_but_not_fatal():
    leaking, unknown = classify_features(CLEAN_MGN_FEATURES + ("some_new_geometry_param",))
    assert leaking == ()
    assert unknown == ("some_new_geometry_param",)
    assert_input_features_clean(CLEAN_MGN_FEATURES + ("some_new_geometry_param",))


def test_feature_count_mismatch_is_rejected():
    assert_feature_count(CLEAN_MGN_FEATURES, 9)
    with pytest.raises(FeatureLeakageError, match="drifted"):
        assert_feature_count(CLEAN_MGN_FEATURES, 11)


def test_error_message_points_at_the_review():
    with pytest.raises(FeatureLeakageError, match="methodology_review_20260720"):
        assert_input_features_clean(["pos_x", "A"])
