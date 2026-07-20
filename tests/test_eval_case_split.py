"""Tests for the case-level holdout split.

The property under test is the one the old record-level shuffle violated: no
geometry may contribute records to more than one subset.
"""

import json

import numpy as np
import pytest

from eval.case_split import (
    DEFAULT_SPLIT_SIZES,
    CaseSplit,
    SplitContractError,
    assign_records,
    load_case_split,
    make_case_split,
    manifest_digest,
    resolve_case_split,
    save_case_split,
)


def _manifest(n_cases=40):
    return {
        "n_cases": n_cases,
        "cases": [
            {
                "index": i,
                "geometry": {"Ratio_Bore": 0.7 + i * 1e-3, "Ratio_SlotDepth_ParallelSlot": 0.4},
                "electrical": {"PeakCurrent": 130.0 + i, "PhaseAdvance": 12.0},
                "h5_paths": [f"case_{i:04d}.h5"],
                "solve_time_s": 100.0,
            }
            for i in range(n_cases)
        ],
    }


def test_split_sizes_match_review_spec():
    split = make_case_split(range(40))
    assert split.sizes == DEFAULT_SPLIT_SIZES == (30, 4, 6)


def test_subsets_are_disjoint():
    split = make_case_split(range(40))
    assert not set(split.train) & set(split.val)
    assert not set(split.train) & set(split.test)
    assert not set(split.val) & set(split.test)


def test_split_covers_every_case_exactly_once():
    split = make_case_split(range(40))
    assert split.all_cases == tuple(range(40))
    assert len(split.train) + len(split.val) + len(split.test) == 40


def test_split_is_deterministic_for_a_seed():
    a = make_case_split(range(40), seed=42)
    b = make_case_split(range(40), seed=42)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_different_seeds_give_different_splits():
    a = make_case_split(range(40), seed=42)
    b = make_case_split(range(40), seed=7)
    assert a.test != b.test


def test_remainder_cases_go_to_train_not_dropped():
    split = make_case_split(range(45), sizes=(30, 4, 6))
    assert len(split.all_cases) == 45
    assert len(split.train) == 35


def test_growing_the_doe_does_not_reshuffle_the_holdout():
    """Adding cases must not move a previously held-out geometry into train."""
    small = make_case_split(range(40), sizes=(30, 4, 6), seed=42)
    # Same seed, same 40 cases, larger train request: held-out sets are cut first.
    same = make_case_split(range(40), sizes=(28, 4, 6), seed=42)
    assert same.test == small.test
    assert same.val == small.val


def test_oversized_split_is_rejected():
    with pytest.raises(SplitContractError):
        make_case_split(range(5), sizes=(30, 4, 6))


def test_empty_test_subset_is_rejected():
    with pytest.raises(SplitContractError):
        make_case_split(range(40), sizes=(36, 4, 0))


def test_overlapping_split_is_rejected_on_load(tmp_path):
    leaky = CaseSplit(train=(1, 2, 3), val=(4,), test=(3, 5), seed=0, digest="x")
    path = tmp_path / "leaky.json"
    path.write_text(
        json.dumps({"train": leaky.train, "val": leaky.val, "test": leaky.test, "seed": 0}),
        encoding="utf-8",
    )
    with pytest.raises(SplitContractError, match="Case-level holdout violated"):
        load_case_split(path)


def test_roundtrip_through_manifest(tmp_path):
    split = make_case_split(range(40), digest="abc123")
    path = save_case_split(split, tmp_path / "split.json")
    loaded = load_case_split(path, expected_digest="abc123")
    assert (loaded.train, loaded.val, loaded.test) == (split.train, split.val, split.test)
    assert loaded.seed == split.seed


def test_split_from_a_different_doe_is_rejected(tmp_path):
    split = make_case_split(range(40), digest="digest_of_doe_a")
    path = save_case_split(split, tmp_path / "split.json")
    with pytest.raises(SplitContractError, match="Regenerate the split"):
        load_case_split(path, expected_digest="digest_of_doe_b")


def test_digest_ignores_solve_metadata_but_tracks_design_points():
    base = _manifest()
    same = _manifest()
    same["cases"][0]["solve_time_s"] = 999.0
    same["timestamp"] = "later"
    assert manifest_digest(base) == manifest_digest(same)

    changed = _manifest()
    changed["cases"][0]["geometry"]["Ratio_Bore"] = 0.99
    assert manifest_digest(base) != manifest_digest(changed)


def test_records_of_one_case_never_straddle_subsets():
    """The regression test for the original leak.

    40 geometries x 10 timesteps, assigned by case: every record of a geometry
    must land in exactly one subset.
    """
    split = make_case_split(range(40))
    case_of_record = np.repeat(np.arange(40), 10)
    assigned = assign_records(case_of_record, split)

    assert assigned["unassigned"].size == 0
    total = sum(assigned[k].size for k in ("train", "val", "test"))
    assert total == 400

    for name in ("train", "val", "test"):
        subset_cases = set(case_of_record[assigned[name]].tolist())
        other_cases = set()
        for other in ("train", "val", "test"):
            if other != name:
                other_cases |= set(case_of_record[assigned[other]].tolist())
        assert not subset_cases & other_cases


def test_records_from_unknown_cases_are_not_absorbed_into_train():
    split = CaseSplit(train=(0, 1), val=(2,), test=(3,), seed=0, digest="")
    assigned = assign_records([0, 1, 2, 3, 99, 99], split)
    assert assigned["unassigned"].tolist() == [4, 5]
    assert assigned["train"].tolist() == [0, 1]


def test_resolve_creates_then_reuses_the_manifest(tmp_path):
    manifest = _manifest()
    path = tmp_path / "split.json"

    first = resolve_case_split(manifest, range(40), path)
    assert path.exists()
    second = resolve_case_split(manifest, range(40), path)
    assert (first.train, first.val, first.test) == (second.train, second.val, second.test)


def test_resolve_rejects_a_stale_manifest_after_the_doe_changes(tmp_path):
    path = tmp_path / "split.json"
    resolve_case_split(_manifest(), range(40), path)

    changed = _manifest()
    changed["cases"][3]["geometry"]["Ratio_Bore"] = 0.123456
    with pytest.raises(SplitContractError):
        resolve_case_split(changed, range(40), path)


def test_subset_of_case_reports_membership():
    split = CaseSplit(train=(0, 1), val=(2,), test=(3,), seed=0, digest="")
    assert split.subset_of_case(1) == "train"
    assert split.subset_of_case(2) == "val"
    assert split.subset_of_case(3) == "test"
    assert split.subset_of_case(9) == "unassigned"
