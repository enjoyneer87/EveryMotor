"""Unified evaluation harness for the EveryMotor field surrogate benchmark.

Every model (MeshGraphNet / FNO / GINO / Seq2SeqRNN) is scored through this
package so that reported numbers are comparable. The harness enforces three
things the previous per-model eval scripts did not:

1. **Case-level holdout** (`case_split`) — timesteps of one geometry never
   straddle the train/test boundary.
2. **Solution-derived features are banned from the input** (`feature_guard`).
3. **Physical units + an engineering metric** (`metrics`, `torque`) — field
   nRMSE in tesla and Maxwell-stress torque error, not normalized MSE.

See `.github/plans/methodology_review_20260720.md` §R0.
"""

from eval.case_split import (
    CaseSplit,
    DEFAULT_SPLIT_SIZES,
    load_case_split,
    make_case_split,
    manifest_digest,
    save_case_split,
)
from eval.feature_guard import (
    SOLUTION_DERIVED_FEATURES,
    FeatureLeakageError,
    assert_input_features_clean,
)

__all__ = [
    "CaseSplit",
    "DEFAULT_SPLIT_SIZES",
    "FeatureLeakageError",
    "SOLUTION_DERIVED_FEATURES",
    "assert_input_features_clean",
    "load_case_split",
    "make_case_split",
    "manifest_digest",
    "save_case_split",
]
