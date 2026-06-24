"""TASK-P1B-4: Regression baseline validation.

Compares the recorded baseline metrics in results/regression_baseline.json
against tolerance thresholds. Fails when the stored best_train_total
exceeds the overfit_target by more than an allowed margin.

Run on host (no torch required):
    pytest tests/test_regression_baseline.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASELINE_PATH = (
    Path(__file__).parent.parent / "results" / "regression_baseline.json"
)

# Allowed relative slack above the recorded overfit_target.
# E.g. 0.10 → 10% slack: passes if best_train_total <= overfit_target * 1.10
REGRESSION_SLACK = 0.10

SUPPORTED_CHANNEL_ORDERS = (
    ["Bx", "By", "A", "J"],
    ["Bx", "By", "A", "J", "Je"],
)

REQUIRED_KEYS = {
    "overfit_target",
    "best_train_total",
    "epoch_at_best",
    "seed",
    "channel_order",
    "pbc_enabled",
    "pbc_rotation_deg",
}


@pytest.fixture(scope="module", name="baseline_data")
def _baseline_data_fixture():
    assert BASELINE_PATH.exists(), f"Baseline file not found: {BASELINE_PATH}"
    with BASELINE_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_baseline_file_exists():
    """Baseline JSON must exist in results/."""
    assert BASELINE_PATH.exists(), f"Missing: {BASELINE_PATH}"


def test_baseline_has_required_keys(baseline_data):
    """All required fields must be present."""
    missing = REQUIRED_KEYS - set(baseline_data.keys())
    assert not missing, f"Missing keys in baseline: {missing}"


def test_best_loss_within_target(baseline_data):
    """best_train_total must be below overfit_target."""
    best = baseline_data["best_train_total"]
    target = baseline_data["overfit_target"]
    assert best <= target, (
        f"best_train_total {best} exceeds overfit_target {target}. "
        "Re-run overfit-single and update the baseline if intentionally "
        "changed."
    )


def test_no_regression_from_baseline(baseline_data):
    """best_train_total stays within REGRESSION_SLACK of overfit_target."""
    best = baseline_data["best_train_total"]
    target = baseline_data["overfit_target"]
    threshold = target * (1.0 + REGRESSION_SLACK)
    assert best <= threshold, (
        f"best_train_total {best:.6f} exceeds regression threshold "
        f"{threshold:.6f} "
        f"(overfit_target={target} + {REGRESSION_SLACK*100:.0f}% slack). "
        "Performance regressed—check recent training changes."
    )


def test_channel_order_frozen(baseline_data):
    """Baseline channel order must be a supported frozen contract."""
    assert baseline_data["channel_order"] in SUPPORTED_CHANNEL_ORDERS, (
        "Unsupported channel order in baseline: "
        f"{baseline_data['channel_order']}. "
        f"Expected one of {SUPPORTED_CHANNEL_ORDERS}."
    )


def test_output_feature_count_matches_channel_order(baseline_data):
    """Recorded output feature count must align with stored channel order."""
    output_features = baseline_data.get("num_output_features")
    assert output_features == len(baseline_data["channel_order"]), (
        "num_output_features does not match channel_order length: "
        f"{output_features} != {len(baseline_data['channel_order'])}"
    )


def test_pbc_is_enabled(baseline_data):
    """PBC must remain enabled in the baseline run."""
    assert baseline_data["pbc_enabled"] is True, (
        "PBC was disabled in baseline—unexpected."
    )


def test_pbc_rotation_deg(baseline_data):
    """PBC rotation must be -45 degrees for 1/8 sector symmetry."""
    assert baseline_data["pbc_rotation_deg"] == -45.0, (
        "pbc_rotation_deg changed: "
        f"{baseline_data['pbc_rotation_deg']} != -45.0"
    )


def test_epoch_at_best_is_positive(baseline_data):
    """Epoch at best loss must be a positive integer."""
    epoch = baseline_data["epoch_at_best"]
    assert isinstance(epoch, int) and epoch > 0, (
        f"epoch_at_best invalid: {epoch}"
    )
