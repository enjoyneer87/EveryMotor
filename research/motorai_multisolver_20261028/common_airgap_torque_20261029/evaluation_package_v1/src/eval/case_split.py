"""Deterministic case-level train/val/test split for the DOE benchmark.

Why this module exists
----------------------
`train_doe_meshgraphnet.py` used to shuffle the flat list of (case, timestep)
records and cut it at ``train_ratio``. With 40 geometries x N rotor angles that
places near-duplicate samples — same mesh, same geometry parameters, rotor angle
differing by 2 degrees — on both sides of the split. Every val MSE reported that
way measures *rotor-angle interpolation on a seen geometry*, not generalization
to an unseen geometry, which is the only thing a DOE screening surrogate is for.

Splits are therefore always cut at case granularity, written to a JSON manifest,
and shared verbatim by every model in the benchmark.

Design notes
------------
`CaseSplit` is an immutable dataclass; `make_case_split` is a pure function of
(case indices, sizes, seed). The manifest carries a digest of the DOE definition
so a split silently generated against a different DOE cannot be reused.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

# 40 DOE cases -> 30 train / 4 val / 6 test (methodology review R0.1).
DEFAULT_SPLIT_SIZES: Tuple[int, int, int] = (30, 4, 6)

SUBSET_NAMES: Tuple[str, str, str] = ("train", "val", "test")

SPLIT_MANIFEST_VERSION = "case_split/v1"


class SplitContractError(ValueError):
    """Raised when a split violates the case-level holdout contract."""


@dataclass(frozen=True)
class CaseSplit:
    """Immutable case-level assignment of DOE cases to benchmark subsets.

    Attributes
    ----------
    train, val, test:
        Sorted tuples of DOE case indices. Disjoint by construction.
    seed:
        RNG seed used to shuffle before cutting. Recorded so the split is
        reproducible from the manifest alone.
    digest:
        Digest of the DOE manifest the split was cut from (see `manifest_digest`).
    """

    train: Tuple[int, ...]
    val: Tuple[int, ...]
    test: Tuple[int, ...]
    seed: int
    digest: str

    @property
    def all_cases(self) -> Tuple[int, ...]:
        return tuple(sorted(self.train + self.val + self.test))

    @property
    def sizes(self) -> Tuple[int, int, int]:
        return (len(self.train), len(self.val), len(self.test))

    def subset(self, name: str) -> Tuple[int, ...]:
        """Return the case indices of one subset by name."""
        if name not in SUBSET_NAMES:
            raise KeyError(f"Unknown subset {name!r}; expected one of {SUBSET_NAMES}")
        return getattr(self, name)

    def subset_of_case(self, case_index: int) -> str:
        """Return which subset a case belongs to, or 'unassigned'."""
        for name in SUBSET_NAMES:
            if case_index in self.subset(name):
                return name
        return "unassigned"


def manifest_digest(manifest: Mapping) -> str:
    """Digest the DOE definition (case indices + geometry/electrical values).

    Only the fields that define the design of experiments are hashed, so
    re-running the DOE solver — which changes timestamps and solve times — does
    not invalidate an existing split.
    """
    payload = []
    for case in manifest.get("cases", []):
        payload.append(
            {
                "index": int(case["index"]),
                "geometry": {k: float(v) for k, v in sorted(case.get("geometry", {}).items())},
                "electrical": {k: float(v) for k, v in sorted(case.get("electrical", {}).items())},
            }
        )
    payload.sort(key=lambda c: c["index"])
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def make_case_split(
    case_indices: Iterable[int],
    sizes: Tuple[int, int, int] = DEFAULT_SPLIT_SIZES,
    seed: int = 42,
    digest: str = "",
) -> CaseSplit:
    """Cut a deterministic case-level split.

    Parameters
    ----------
    case_indices:
        All DOE case indices that have usable data.
    sizes:
        ``(n_train, n_val, n_test)``. Must sum to at most ``len(case_indices)``;
        any remainder is appended to the train subset so no case is silently
        dropped.
    seed:
        Shuffle seed. Fixed seed + fixed case list => identical split.
    digest:
        DOE manifest digest to record (see `manifest_digest`).
    """
    cases = np.asarray(sorted({int(c) for c in case_indices}), dtype=np.int64)
    n_total = int(cases.size)
    n_train, n_val, n_test = (int(s) for s in sizes)

    if min(n_train, n_val, n_test) < 0:
        raise SplitContractError(f"Split sizes must be non-negative, got {sizes}")
    requested = n_train + n_val + n_test
    if requested > n_total:
        raise SplitContractError(
            f"Split sizes {sizes} request {requested} cases but only {n_total} are available"
        )
    if n_test == 0:
        raise SplitContractError("Test subset must be non-empty — it is the reported benchmark")

    # Held-out subsets are cut first so that adding cases to the DOE later grows
    # the training set rather than reshuffling what is already held out.
    shuffled = np.random.default_rng(seed).permutation(cases)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]  # absorbs the remainder

    split = CaseSplit(
        train=tuple(int(c) for c in np.sort(train)),
        val=tuple(int(c) for c in np.sort(val)),
        test=tuple(int(c) for c in np.sort(test)),
        seed=int(seed),
        digest=str(digest),
    )
    assert_case_disjoint(split)
    return split


def assert_case_disjoint(split: CaseSplit) -> None:
    """Fail loudly if any case appears in more than one subset.

    This is the guard against the leak described in the module docstring; it is
    cheap enough to run on every split load.
    """
    train, val, test = set(split.train), set(split.val), set(split.test)
    for a_name, a, b_name, b in (
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ):
        overlap = sorted(a & b)
        if overlap:
            raise SplitContractError(
                f"Case-level holdout violated: cases {overlap} appear in both {a_name} and {b_name}"
            )
    if not test:
        raise SplitContractError("Test subset is empty")


def assign_records(case_of_record: Sequence[int], split: CaseSplit) -> Dict[str, np.ndarray]:
    """Map per-record case indices to record positions per subset.

    Parameters
    ----------
    case_of_record:
        ``case_of_record[i]`` is the DOE case index that record ``i`` came from.

    Returns
    -------
    dict with keys ``train``/``val``/``test``/``unassigned`` holding int arrays
    of record positions. Records from cases outside the split land in
    ``unassigned`` rather than being silently attached to train.
    """
    cases = np.asarray(list(case_of_record), dtype=np.int64)
    out: Dict[str, np.ndarray] = {}
    claimed = np.zeros(cases.shape, dtype=bool)
    for name in SUBSET_NAMES:
        member = np.isin(cases, np.asarray(split.subset(name), dtype=np.int64))
        out[name] = np.flatnonzero(member)
        claimed |= member
    out["unassigned"] = np.flatnonzero(~claimed)
    return out


def save_case_split(split: CaseSplit, path: Path | str) -> Path:
    """Write the split manifest as JSON (version-controlled artifact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SPLIT_MANIFEST_VERSION,
        "seed": split.seed,
        "doe_digest": split.digest,
        "sizes": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "granularity": "case",
        "note": (
            "Case-level holdout. Every model in the benchmark must consume this exact "
            "split; see .github/plans/methodology_review_20260720.md R0."
        ),
        "train": list(split.train),
        "val": list(split.val),
        "test": list(split.test),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_case_split(path: Path | str, expected_digest: str | None = None) -> CaseSplit:
    """Load and validate a split manifest.

    Raises `SplitContractError` when the manifest was cut from a different DOE
    than the one being evaluated, which would otherwise reintroduce the leak
    quietly.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = CaseSplit(
        train=tuple(int(c) for c in payload["train"]),
        val=tuple(int(c) for c in payload["val"]),
        test=tuple(int(c) for c in payload["test"]),
        seed=int(payload.get("seed", -1)),
        digest=str(payload.get("doe_digest", "")),
    )
    assert_case_disjoint(split)
    if expected_digest is not None and split.digest and split.digest != expected_digest:
        raise SplitContractError(
            f"Split manifest {path} was cut from DOE digest {split.digest!r} but the "
            f"loaded DOE has digest {expected_digest!r}. Regenerate the split."
        )
    return split


def resolve_case_split(
    manifest: Mapping,
    case_indices: Iterable[int],
    path: Path | str,
    sizes: Tuple[int, int, int] = DEFAULT_SPLIT_SIZES,
    seed: int = 42,
) -> CaseSplit:
    """Load the split at `path`, creating it on first use.

    This is the entry point training and eval scripts should call so that both
    provably consume the same holdout.
    """
    digest = manifest_digest(manifest)
    path = Path(path)
    if path.exists():
        return load_case_split(path, expected_digest=digest)
    split = make_case_split(case_indices, sizes=sizes, seed=seed, digest=digest)
    save_case_split(split, path)
    return split
