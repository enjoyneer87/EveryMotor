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

BASELINE_PATH = Path(__file__).parent.parent / "results" / "regression_baseline.json"

# Allowed relative slack above the recorded overfit_target.
# E.g. 0.10 → 10% slack: passes if best_train_total <= overfit_target * 1.10
REGRESSION_SLACK = 0.10

REQUIRED_KEYS = {
    "overfit_target",
    "best_train_total",
    "epoch_at_best",
    "seed",
    "channel_order",
    "pbc_enabled",
    "pbc_rotation_deg",
}


@pytest.fixture(scope="module")
def baseline():
    assert BASELINE_PATH.exists(), f"Baseline file not found: {BASELINE_PATH}"
    with open(BASELINE_PATH) as f:
        data = json.load(f)
    return data


def test_baseline_file_exists():
    """Baseline JSON must exist in results/."""
    assert BASELINE_PATH.exists(), f"Missing: {BASELINE_PATH}"


def test_baseline_has_required_keys(baseline):
    """All required fields must be present."""
    missing = REQUIRED_KEYS - set(baseline.keys())
    assert not missing, f"Missing keys in baseline: {missing}"


def test_best_loss_within_target(baseline):
    """best_train_total must be below overfit_target."""
    best = baseline["best_train_total"]
    target = baseline["overfit_target"]
    assert best <= target, (
        f"best_train_total {best} exceeds overfit_target {target}. "
        "Re-run overfit-single and update the baseline if intentionally changed."
    )


def test_no_regression_from_baseline(baseline):
    """best_train_total stays within REGRESSION_SLACK of overfit_target."""
    best = baseline["best_train_total"]
    target = baseline["overfit_target"]
    threshold = target * (1.0 + REGRESSION_SLACK)
    assert best <= threshold, (
        f"best_train_total {best:.6f} exceeds regression threshold {threshold:.6f} "
        f"(overfit_target={target} + {REGRESSION_SLACK*100:.0f}% slack). "
        "Performance regressed—check recent training changes."
    )


def test_channel_order_frozen(baseline):
    """Channel order must remain [Bx, By, A, J]."""
    expected = ["Bx", "By", "A", "J"]
    assert baseline["channel_order"] == expected, (
        f"Channel order changed: {baseline['channel_order']} != {expected}. "
        "This is a contract-breaking change requiring explicit approval."
    )


def test_pbc_is_enabled(baseline):
    """PBC must remain enabled in the baseline run."""
    assert baseline["pbc_enabled"] is True, "PBC was disabled in baseline—unexpected."


def test_pbc_rotation_deg(baseline):
    """PBC rotation must be -45 degrees for 1/8 sector symmetry."""
    assert baseline["pbc_rotation_deg"] == -45.0, (
        f"pbc_rotation_deg changed: {baseline['pbc_rotation_deg']} != -45.0"
    )


def test_epoch_at_best_is_positive(baseline):
    """Epoch at best loss must be a positive integer."""
    epoch = baseline["epoch_at_best"]
    assert isinstance(epoch, int) and epoch > 0, f"epoch_at_best invalid: {epoch}"
