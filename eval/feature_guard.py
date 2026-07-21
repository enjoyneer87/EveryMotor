"""Input-feature whitelist for the field surrogate.

The surrogate's whole purpose is to replace the FEM solve, so its inputs must be
things known *before* solving: node position, region code, geometry parameters
and the operating point. Anything derived from the solution — A, J, |B|, Bx, By,
Je — is an answer, not a question. Feeding one back in produces excellent
validation numbers and a model that cannot be used.

`train_doe_meshgraphnet.py` currently builds a clean 9-feature input, but its
docstring advertises 11 features including A and J. The review (F1b) flagged that
stale docstring as a live hazard: the next person to touch the file may
"restore" A/J as intended design. This module turns the convention into an
assert so the mistake fails at graph-build time instead of at review time.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence, Tuple

# Solved quantities. Present as *targets*; never as inputs.
SOLUTION_DERIVED_FEATURES = frozenset(
    {
        "a",
        "a_z",
        "az",
        "vector_potential",
        "j",
        "je",
        "j_e",
        "current_density",
        "eddy_current_density",
        "b",
        "bx",
        "by",
        "bz",
        "b_norm",
        "bnorm",
        "b_mag",
        "bmag",
        "br",
        "btheta",
        "b_theta",
        "flux",
        "flux_linkage",
        "torque",
        "loss",
        "iron_loss",
    }
)

# Features that are not solved quantities but carry no information the model
# should be using — they duplicate a physical input while being easier to fit,
# so the network latches onto them and stops learning the real mapping.
#
# `time_s` is the case that motivated this. Every DOE case runs at the same
# speed, so time_s is exactly proportional to rotor angle (corr = +1.0000 with
# |cumulative angle|) and identical across cases: a step counter wearing a
# physics label. The model made it the dominant temporal cue — perturbing only
# time_s at step 0 swung the predicted torque from -264 to +48 N*m, sign and
# all — and its error concentrated at the edge of the time_s range where the
# shortcut breaks down. It is also actively wrong the moment a DOE varies
# speed, since then equal time no longer means equal angle.
SHORTCUT_FEATURES = frozenset(
    {
        "time_s",
        "time",
        "t",
        "step_index",
        "step",
        "sample_index",
        "case_index",
        "solution_index",
    }
)

# Known-good inputs: geometry, topology and the operating point.
ALLOWED_INPUT_FEATURES = frozenset(
    {
        # geometry / topology
        "pos_x",
        "pos_y",
        "x",
        "y",
        "r",
        "theta",
        "region_code",
        "region",
        "node_degree",
        # DOE geometry parameters
        "ratio_bore",
        "ratio_slotdepth_parallelslot",
        "ratio_slotdepth",
        # operating point
        "peakcurrent",
        "peak_current",
        "ipk",
        "phaseadvance",
        "phase_advance",
        "rotate_step",
        "rotor_angle_sin",
        "rotor_angle_cos",
        "rotor_angle_deg",
        "edge_sign",
        "dt_s",
        # fidelity metadata
        "fidelity_type",
        "coupling_policy",
        "step_semantics",
    }
)

_NORMALIZE = re.compile(r"[^a-z0-9]+")

# Declared input layouts. Kept here — torch-free — so the training script, the
# eval adapters and the tests all name the same tuple instead of each carrying a
# docstring that can drift out of sync with the tensor.
MGN_NODE_FEATURES: Tuple[str, ...] = (
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

# Layout with the sector-symmetry fixes: `rotate_step` (which only ever takes
# the values {0, -2} and so cannot distinguish timesteps) is replaced by the
# cumulative rotor angle encoded at the anti-periodic frequency. `time_s` is
# gone too — see SHORTCUT_FEATURES for why it had to go.
MGN_NODE_FEATURES_V2: Tuple[str, ...] = (
    "pos_x",
    "pos_y",
    "region_code",
    "rotor_angle_sin",
    "rotor_angle_cos",
    "Ratio_Bore",
    "Ratio_SlotDepth_ParallelSlot",
    "PeakCurrent",
    "PhaseAdvance",
)

MGN_EDGE_FEATURES: Tuple[str, ...] = ("dx", "dy", "dist")

# With anti-periodic edges the sign rides alongside the geometric attributes.
MGN_EDGE_FEATURES_V2: Tuple[str, ...] = ("edge_sign", "dx", "dy", "dist")


class FeatureLeakageError(ValueError):
    """Raised when a solution-derived quantity is used as a model input."""


def normalize_feature_name(name: str) -> str:
    """Canonicalize a feature name for whitelist comparison.

    ``"Ratio_Bore"``, ``"ratio bore"`` and ``"RatioBore"`` all normalize to
    ``"ratio_bore"``-comparable form.
    """
    return _NORMALIZE.sub("_", str(name).strip().lower()).strip("_")


def classify_features(
    names: Iterable[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Split feature names into (leaking, shortcut, unrecognized).

    Unrecognized names are not an error on their own — the whitelist cannot
    anticipate every future feature — but they are surfaced so a reviewer sees
    them.
    """
    leaking, shortcut, unknown = [], [], []
    for raw in names:
        norm = normalize_feature_name(raw)
        if norm in SOLUTION_DERIVED_FEATURES:
            leaking.append(str(raw))
        elif norm in SHORTCUT_FEATURES:
            shortcut.append(str(raw))
        elif norm not in ALLOWED_INPUT_FEATURES:
            unknown.append(str(raw))
    return tuple(leaking), tuple(shortcut), tuple(unknown)


def assert_input_features_clean(
    names: Sequence[str],
    context: str = "model input",
    allow_shortcuts: bool = False,
) -> None:
    """Fail if any input feature is derived from the FEM solution.

    Call this wherever the input feature vector is assembled — graph builders,
    grid encoders, dataset adapters — passing the column names in order.

    `allow_shortcuts` exists only so historical checkpoints trained on the
    pre-diagnosis layout can still be loaded and scored. Never set it when
    building a new model.
    """
    leaking, shortcut, _ = classify_features(names)
    if leaking:
        raise FeatureLeakageError(
            f"{context} contains solution-derived feature(s) {list(leaking)}. "
            "A, J, B and Je are prediction targets and must not be fed back as inputs "
            "(see .github/plans/methodology_review_20260720.md F1b)."
        )
    if shortcut and not allow_shortcuts:
        raise FeatureLeakageError(
            f"{context} contains shortcut feature(s) {list(shortcut)}. These duplicate a "
            "physical input while being easier to fit, so the model learns them instead of "
            "the geometry-to-field mapping (see section 9 of the methodology review)."
        )


def assert_feature_count(names: Sequence[str], n_columns: int, context: str = "model input") -> None:
    """Fail if the declared feature names do not match the actual column count.

    Guards against the specific failure mode that produced the stale 11-feature
    docstring: documentation and tensor drifting apart unnoticed.
    """
    if len(names) != int(n_columns):
        raise FeatureLeakageError(
            f"{context} declares {len(names)} feature names {list(names)} but the tensor "
            f"has {n_columns} columns. Documentation and code have drifted."
        )
