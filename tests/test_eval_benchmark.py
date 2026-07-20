"""Tests for the benchmark harness and the DOE reader.

These run against the real DOE export when it is present (it is the only place
the two Motor-CAD H5 layouts actually appear together) and are skipped otherwise
so the rest of the suite stays runnable on any machine.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from eval.benchmark import (
    BenchmarkConfig,
    CurlFloorPredictor,
    FemIdentityPredictor,
    NodeResamplingFloorPredictor,
    TrainMeanPredictor,
    ZeroFieldPredictor,
    evaluate_predictor,
    fit_train_region_mean,
    run_benchmark,
    summarize,
    to_element_support,
    truth_matrix,
)
from eval.doe_dataset import (
    STATIC_FORMAT,
    TIMESERIES_FORMAT,
    classify_source_type,
    load_doe_cases,
    read_case_file,
    resolve_h5_path,
)

DATA_DIR = Path("backup/doe_data")
MANIFEST = DATA_DIR / "doe_manifest.json"

requires_doe = pytest.mark.skipif(not MANIFEST.exists(), reason="DOE export not present")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def onload_record(manifest):
    case = manifest["cases"][0]
    path = resolve_h5_path(case["h5_paths"][0], DATA_DIR, 0)
    return read_case_file(path, 0, {**case["geometry"], **case["electrical"]})


# --- reader ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Mag_OnLoadTorque_result_1.h5", "OnLoadTorque"),
        ("Mag_StaticLoad_result_1.h5", "StaticLoad"),
        ("Mag_StaticLoadInductance_result_1.h5", "StaticLoadInductance"),
        ("Mag_StaticOC_result_1.h5", "StaticOC"),
        ("whatever.h5", "Unknown"),
    ],
)
def test_source_type_classification(name, expected):
    assert classify_source_type(name) == expected


def test_windows_paths_resolve_against_the_local_data_dir():
    recorded = r"D:\KDH\Sim_4SolverX\DOE_TrainingData\case_0000\postproc\Mag_OnLoadTorque_result_1.h5"
    resolved = resolve_h5_path(recorded, DATA_DIR, 0)
    if MANIFEST.exists():
        assert resolved is not None and resolved.name == "Mag_OnLoadTorque_result_1.h5"


@requires_doe
def test_static_mesh_files_load_instead_of_being_skipped(manifest):
    """Regression test for the 'missing steps' warning.

    StaticLoad / StaticOC use a scalar `step` and 1-D fields; the training
    reader raised on them and the caller swallowed the exception, dropping the
    data silently.
    """
    report = load_doe_cases(
        manifest, DATA_DIR, case_indices=[0],
        source_types=("OnLoadTorque", "StaticLoad", "StaticOC"),
    )
    layouts = {r.source_type: r.samples[0].layout for r in report.records}
    assert layouts["OnLoadTorque"] == TIMESERIES_FORMAT
    assert layouts["StaticLoad"] == STATIC_FORMAT
    assert layouts["StaticOC"] == STATIC_FORMAT
    assert report.skipped == ()


@requires_doe
def test_skips_are_recorded_with_a_reason(manifest):
    broken = json.loads(json.dumps(manifest))
    broken["cases"] = broken["cases"][:1]
    broken["cases"][0]["h5_paths"] = ["Mag_OnLoadTorque_nonexistent.h5"]

    report = load_doe_cases(broken, DATA_DIR, case_indices=[0])
    assert report.records == ()
    assert report.skipped[0][2] == "file_not_found"
    assert report.to_dict()["n_skipped"] == 1


@requires_doe
def test_fields_stay_on_element_support(onload_record):
    sample = onload_record.samples[0]
    assert sample.fields["bx"].shape == (onload_record.mesh.n_elements,)
    assert onload_record.mesh.n_elements != onload_record.mesh.n_nodes


@requires_doe
def test_moving_nodes_displace_between_steps(onload_record):
    first, later = onload_record.samples[0], onload_record.samples[10]
    moved = ~np.isclose(first.node_x_mm, later.node_x_mm)
    assert np.any(moved), "rotor nodes should move between timesteps"
    assert not np.all(moved), "stator nodes should stay put"


@requires_doe
def test_mesh_node_ordering_is_consistent_with_connectivity(onload_record):
    i1, i2, i3 = onload_record.mesh.tri
    n = onload_record.mesh.n_nodes
    for idx in (i1, i2, i3):
        assert idx.min() >= 0 and idx.max() < n


@requires_doe
def test_max_steps_caps_the_sample_count(manifest):
    report = load_doe_cases(manifest, DATA_DIR, case_indices=[0], max_steps=3)
    assert all(len(r) == 3 for r in report.records)


# --- harness -----------------------------------------------------------------


@requires_doe
def test_identity_predictor_scores_exactly_zero(onload_record):
    """Harness self-test: any nonzero error here means the harness is broken."""
    config = BenchmarkConfig(data_dir=DATA_DIR, split_path=Path("unused.json"))
    results, failures = evaluate_predictor(
        FemIdentityPredictor(), [onload_record], config
    )
    assert failures == ()
    assert results

    summary = summarize(results, config.channels)
    assert summary["overall"]["Bnorm"]["nrmse_pct"] == pytest.approx(0.0, abs=1e-9)
    assert summary["torque"]["rel_error_pct_mean"] == pytest.approx(0.0, abs=1e-9)


@requires_doe
def test_zero_predictor_scores_exactly_one_hundred_percent(onload_record):
    config = BenchmarkConfig(data_dir=DATA_DIR, split_path=Path("unused.json"))
    results, _ = evaluate_predictor(ZeroFieldPredictor(), [onload_record], config)
    summary = summarize(results, config.channels)
    assert summary["overall"]["Bnorm"]["nrmse_pct"] == pytest.approx(100.0)
    assert summary["torque"]["rel_error_pct_mean"] == pytest.approx(100.0)


@requires_doe
def test_region_breakdown_covers_the_engineering_groups(onload_record):
    config = BenchmarkConfig(data_dir=DATA_DIR, split_path=Path("unused.json"))
    results, _ = evaluate_predictor(ZeroFieldPredictor(), [onload_record], config)
    summary = summarize(results, config.channels)
    assert {"airgap", "stator_teeth", "stator_yoke", "magnet", "rotor_iron"} <= set(summary["by_region"])


@requires_doe
def test_a_broken_predictor_is_reported_not_silently_dropped(onload_record):
    class Broken:
        name, channels, output_support = "broken", ("Bx", "By"), "element"

        def predict(self, record, sample):
            raise RuntimeError("adapter exploded")

        def describe(self):
            return {}

    config = BenchmarkConfig(data_dir=DATA_DIR, split_path=Path("unused.json"))
    results, failures = evaluate_predictor(Broken(), [onload_record], config)
    assert results == ()
    assert len(failures) == len(onload_record.samples)
    assert "adapter exploded" in failures[0]


@requires_doe
def test_node_predictions_are_averaged_onto_elements(onload_record):
    n_nodes = onload_record.mesh.n_nodes
    node_pred = np.ones((n_nodes, 2))
    elem = to_element_support(node_pred, "node", onload_record)
    assert elem.shape == (onload_record.mesh.n_elements, 2)
    assert np.allclose(elem, 1.0)


@requires_doe
def test_wrong_node_count_is_rejected(onload_record):
    with pytest.raises(ValueError, match="nodes"):
        to_element_support(np.ones((7, 2)), "node", onload_record)


@requires_doe
def test_unknown_output_support_is_rejected(onload_record):
    with pytest.raises(ValueError, match="output_support"):
        to_element_support(np.ones((3, 2)), "vertex", onload_record)


@requires_doe
def test_train_mean_baseline_beats_zero_but_not_by_much(onload_record):
    """The geometry-blind reference: any real surrogate must clear it."""
    config = BenchmarkConfig(data_dir=DATA_DIR, split_path=Path("unused.json"))
    predictor = TrainMeanPredictor(
        region_mean=fit_train_region_mean([onload_record], config.channels)
    )
    fitted, _ = evaluate_predictor(predictor, [onload_record], config)
    zero, _ = evaluate_predictor(ZeroFieldPredictor(), [onload_record], config)

    fitted_nrmse = summarize(fitted, config.channels)["overall"]["Bnorm"]["nrmse_pct"]
    zero_nrmse = summarize(zero, config.channels)["overall"]["Bnorm"]["nrmse_pct"]
    assert fitted_nrmse < zero_nrmse


@requires_doe
def test_end_to_end_run_writes_a_self_describing_artifact(tmp_path, manifest):
    config = BenchmarkConfig(
        data_dir=DATA_DIR,
        split_path=tmp_path / "split.json",
        max_steps_per_case=2,
    )
    out = tmp_path / "benchmark.json"
    artifact = run_benchmark([FemIdentityPredictor(), ZeroFieldPredictor()], config, output_path=out)

    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["version"] == artifact["version"]

    # The artifact must carry enough provenance to be reproducible.
    assert on_disk["split"]["granularity"] == "case"
    assert on_disk["doe_digest"]
    assert on_disk["config"]["axial_length_m"] == 1.0
    assert on_disk["data"]["n_skipped"] == 0
    assert "model_comparison_summary.json" in on_disk["supersedes"]

    scored_cases = set(on_disk["split"]["evaluated_cases"])
    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    assert scored_cases == set(split["test"])
    assert not scored_cases & set(split["train"])


@requires_doe
def test_evaluated_cases_are_absent_from_the_training_subset(tmp_path, manifest):
    """The property the old record-level shuffle broke, checked end to end."""
    config = BenchmarkConfig(
        data_dir=DATA_DIR, split_path=tmp_path / "split.json", max_steps_per_case=1
    )
    artifact = run_benchmark([ZeroFieldPredictor()], config)
    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))

    per_case = artifact["models"]["zero_field"]["summary"]["by_case"]
    for case_index in per_case:
        assert int(case_index) in split["test"]
        assert int(case_index) not in split["train"]
        assert int(case_index) not in split["val"]


@requires_doe
def test_truth_matrix_rejects_an_unmapped_channel(onload_record):
    with pytest.raises(KeyError):
        truth_matrix(onload_record.samples[0], ("Bx", "Nonexistent"))


@requires_doe
def test_curl_floor_beats_the_node_roundtrip_floor(onload_record):
    """The measurement that motivates R1: element support removes the round trip.

    Predicting A at nodes and taking the P1 curl lands B on elements directly,
    so it does not pay the element -> node -> element cost that dominates the
    node-support pipeline's torque error.
    """
    from eval.benchmark import CurlFloorPredictor

    config = BenchmarkConfig(data_dir=DATA_DIR, split_path=Path("unused.json"))
    trimmed = onload_record.__class__(
        case_index=onload_record.case_index,
        source_type=onload_record.source_type,
        path=onload_record.path,
        mesh=onload_record.mesh,
        samples=onload_record.samples[:3],
        condition=onload_record.condition,
    )

    curl, curl_fail = evaluate_predictor(CurlFloorPredictor(), [trimmed], config)
    node, _ = evaluate_predictor(NodeResamplingFloorPredictor(), [trimmed], config)
    assert curl_fail == ()

    curl_s = summarize(curl, config.channels)
    node_s = summarize(node, config.channels)

    assert curl_s["overall"]["Bnorm"]["nrmse_pct"] < node_s["overall"]["Bnorm"]["nrmse_pct"]
    # The torque gap is the decisive one: ~1-3% against ~60-80%.
    assert curl_s["torque"]["nrmse_torque_pct"] < 10.0
    assert node_s["torque"]["nrmse_torque_pct"] > 40.0


@requires_doe
def test_sliding_band_is_identified_from_the_export(onload_record):
    """Motor-CAD re-meshes the rotor-side airgap layers; the export cannot describe them."""
    from eval.mesh_regions import sliding_band_codes, sliding_band_mask

    mesh = onload_record.mesh
    codes = sliding_band_codes(mesh.name_of_code, mesh.moving_reg_codes)
    names = {mesh.name_of_code[c] for c in codes}

    assert names == {"a2", "a3", "a4"}, f"expected the moving airgap layers, got {names}"
    # a1 is the stationary stator-side layer the torque band is built on.
    assert "a1" not in names

    mask = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)
    assert 0 < mask.sum() < mesh.n_elements


@requires_doe
def test_rotated_steps_tangle_the_sliding_band(onload_record):
    """Regression guard for the finding that per-step connectivity is invalid.

    Reference connectivity + per-step coordinates inverts elements once the
    rotor has turned, and every inverted element is in the sliding band.
    """
    from eval.mesh_regions import sliding_band_mask
    from phase1_static.discrete_curl import signed_double_area

    mesh = onload_record.mesh
    reference = signed_double_area(mesh.node_x_mm, mesh.node_y_mm, mesh.tri)
    band = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)

    late = onload_record.samples[10]
    area = signed_double_area(late.node_x_mm, late.node_y_mm, mesh.tri)
    inverted = np.sign(area) != np.sign(reference)

    assert inverted.any(), "expected tangled elements at a rotated step"
    assert np.all(band[inverted]), "every inverted element must lie in the sliding band"
