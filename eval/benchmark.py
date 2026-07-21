"""Single evaluation harness for the motor field surrogate benchmark.

Every model is scored here, on the same case-level holdout, in physical units,
with the same torque operator. Numbers produced by the older per-model eval
scripts are not comparable to these and are superseded (methodology review R0,
gate G0).

What this module guarantees
---------------------------
* **Case-level holdout** — the test cases come from a version-controlled split
  manifest; no timestep of a test geometry was ever seen in training.
* **Element-level comparison** — predictions are compared against the raw
  Motor-CAD element fields. Node-predicting models are averaged onto elements
  first, so a model is never scored against its own resampling of the truth.
* **Physical units** — tesla and Wb/m, never normalized MSE.
* **An engineering metric** — Maxwell-stress torque error.
* **Visible skips** — every file that failed to load is written into the result
  artifact with its reason.

This module deliberately imports no torch, so the harness runs on the host.
Model adapters that need torch live in `eval.predictors`.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from eval.case_split import CaseSplit, manifest_digest, resolve_case_split
from eval.determinism import pin_determinism
from eval.doe_dataset import CaseRecord, CaseSample, load_doe_cases
from eval.mesh_regions import REGION_GROUP_ORDER
from eval.metrics import ChannelMetric, aggregate_channel, field_metrics, metrics_to_dict, region_metrics
from eval.torque import AirgapBand, TorqueComparison, aggregate_torque, build_airgap_band, compare_torque, nodal_to_element

BENCHMARK_VERSION = "benchmark/v2"

DEFAULT_CHANNELS: Tuple[str, ...] = ("Bx", "By")

# Motor-CAD field key -> benchmark channel name.
FIELD_KEY_OF_CHANNEL: Mapping[str, str] = {"Bx": "bx", "By": "by", "A": "a", "J": "j"}


class FieldPredictor(Protocol):
    """What the harness needs from a model.

    Implementations live in `eval.predictors` (torch) or are defined inline for
    reference baselines.
    """

    name: str
    channels: Tuple[str, ...]
    output_support: str  # "node" or "element"

    def predict(self, record: CaseRecord, sample: CaseSample) -> np.ndarray:
        """Return predictions for `sample`, shaped ``(n_points, len(channels))``."""
        ...

    def describe(self) -> Dict[str, object]:
        """Serializable model description for the results artifact."""
        ...


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable benchmark run configuration."""

    data_dir: Path
    split_path: Path
    subset: str = "test"
    source_types: Tuple[str, ...] = ("OnLoadTorque",)
    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    max_steps_per_case: Optional[int] = None
    axial_length_m: float = 1.0
    compute_torque: bool = True
    # Field metrics are computed on geometrically valid, non-sliding-band
    # elements only, for every model alike. The excluded elements have no valid
    # per-step geometry in the export (see eval.mesh_regions.sliding_band_mask),
    # so scoring them would compare against a mesh that does not exist. Torque is
    # unaffected: its band is the stationary a1 layer, which is never excluded.
    exclude_invalid_elements: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "data_dir": str(self.data_dir),
            "split_path": str(self.split_path),
            "subset": self.subset,
            "source_types": list(self.source_types),
            "channels": list(self.channels),
            "max_steps_per_case": self.max_steps_per_case,
            "axial_length_m": self.axial_length_m,
            "compute_torque": self.compute_torque,
            "exclude_invalid_elements": self.exclude_invalid_elements,
        }


@dataclass(frozen=True)
class SampleResult:
    """Per-sample scoring outcome."""

    case_index: int
    source_type: str
    step_index: int
    channel_metrics: Mapping[str, ChannelMetric]
    region_metrics: Mapping[str, Mapping[str, ChannelMetric]]
    torque: Optional[TorqueComparison]
    element_coverage: float = 1.0


def scored_element_mask(record: CaseRecord, sample: CaseSample, config: BenchmarkConfig) -> np.ndarray:
    """Elements whose per-step geometry is valid, and therefore scoreable.

    Applied identically to every model so the rows of the scorecard stay
    comparable.
    """
    n = record.mesh.n_elements
    if not config.exclude_invalid_elements:
        return np.ones(n, dtype=bool)

    from eval.mesh_regions import sliding_band_mask
    from phase1_static.discrete_curl import mesh_validity_mask

    band = sliding_band_mask(
        record.mesh.reg_code, record.mesh.name_of_code, record.mesh.moving_reg_codes
    )
    return mesh_validity_mask(
        sample.node_x_mm * 1e-3, sample.node_y_mm * 1e-3, record.mesh.tri, exclude=band
    )


def truth_matrix(sample: CaseSample, channels: Sequence[str]) -> np.ndarray:
    """Ground-truth element fields for the requested channels."""
    keys = []
    for c in channels:
        key = FIELD_KEY_OF_CHANNEL.get(c)
        if key is None:
            raise KeyError(f"No Motor-CAD field mapped for channel {c!r}")
        if key not in sample.fields:
            raise KeyError(f"Sample from case {sample.case_index} has no field {key!r}")
        keys.append(key)
    return sample.field_matrix(keys)


def to_element_support(
    prediction: np.ndarray,
    output_support: str,
    record: CaseRecord,
) -> np.ndarray:
    """Bring a prediction onto element support so it can be compared to the FEM output."""
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 2:
        raise ValueError(f"Prediction must be [N, C], got {prediction.shape}")

    if output_support == "element":
        return prediction
    if output_support != "node":
        raise ValueError(f"Unknown output_support {output_support!r}; expected 'node' or 'element'")

    if prediction.shape[0] != record.mesh.n_nodes:
        raise ValueError(
            f"Node prediction has {prediction.shape[0]} rows but the mesh has "
            f"{record.mesh.n_nodes} nodes"
        )
    return np.stack(
        [nodal_to_element(prediction[:, c], record.mesh.tri) for c in range(prediction.shape[1])],
        axis=1,
    )


def evaluate_predictor(
    predictor: FieldPredictor,
    records: Sequence[CaseRecord],
    config: BenchmarkConfig,
) -> Tuple[Tuple[SampleResult, ...], Tuple[str, ...]]:
    """Score one predictor over the given case records.

    Returns the per-sample results and a list of non-fatal failures, which are
    propagated into the results artifact rather than printed and forgotten.
    """
    channels = tuple(config.channels)
    results: List[SampleResult] = []
    failures: List[str] = []

    for record in records:
        grouping = record.mesh.region_grouping()
        band: Optional[AirgapBand] = None
        if config.compute_torque and {"Bx", "By"}.issubset(set(channels)):
            try:
                band = build_airgap_band(
                    record.mesh.node_x_mm,
                    record.mesh.node_y_mm,
                    record.mesh.tri,
                    record.mesh.reg_code,
                    record.mesh.name_of_code,
                    moving_reg_codes=record.mesh.moving_reg_codes,
                )
            except ValueError as exc:
                failures.append(f"case {record.case_index} {record.path.name}: torque band: {exc}")

        has_b = {"Bx", "By"}.issubset(set(channels))
        ix = channels.index("Bx") if has_b else -1
        iy = channels.index("By") if has_b else -1

        for sample in record.samples:
            try:
                raw = predictor.predict(record, sample)
                pred = to_element_support(raw, predictor.output_support, record)
                truth = truth_matrix(sample, channels)
            except Exception as exc:  # a broken adapter must not silently shrink the test set
                failures.append(
                    f"case {record.case_index} step {sample.step_index}: predict: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            if pred.shape != truth.shape:
                failures.append(
                    f"case {record.case_index} step {sample.step_index}: shape "
                    f"{pred.shape} vs truth {truth.shape}"
                )
                continue

            # Torque first: it integrates over the stationary a1 band, which is
            # always geometrically valid, so it uses the unmasked fields.
            torque = None
            if band is not None and has_b:
                torque = compare_torque(
                    band,
                    pred[:, ix],
                    pred[:, iy],
                    truth[:, ix],
                    truth[:, iy],
                    axial_length_m=config.axial_length_m,
                )

            scored = scored_element_mask(record, sample, config)
            if not np.any(scored):
                failures.append(
                    f"case {record.case_index} step {sample.step_index}: no scoreable elements"
                )
                continue

            results.append(
                SampleResult(
                    case_index=record.case_index,
                    source_type=record.source_type,
                    step_index=sample.step_index,
                    channel_metrics=field_metrics(pred[scored], truth[scored], channels),
                    region_metrics=region_metrics(
                        pred[scored],
                        truth[scored],
                        channels,
                        grouping.group_of_element[scored],
                        groups=REGION_GROUP_ORDER,
                    ),
                    torque=torque,
                    element_coverage=float(np.count_nonzero(scored)) / float(scored.size),
                )
            )

    return tuple(results), tuple(failures)


def summarize(results: Sequence[SampleResult], channels: Sequence[str]) -> Dict[str, object]:
    """Aggregate per-sample results into the reported scorecard."""
    if not results:
        return {"n_samples": 0, "note": "no samples scored"}

    reported = list(channels)
    if "Bx" in reported and "By" in reported:
        reported.append("Bnorm")

    overall = {c: aggregate_channel([r.channel_metrics for r in results], c) for c in reported}

    per_region: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group in REGION_GROUP_ORDER:
        present = [r.region_metrics[group] for r in results if group in r.region_metrics]
        if not present:
            continue
        per_region[group] = {c: aggregate_channel(present, c) for c in reported}

    per_case: Dict[str, Dict[str, object]] = {}
    for case_index in sorted({r.case_index for r in results}):
        subset = [r for r in results if r.case_index == case_index]
        entry: Dict[str, object] = {
            "n_samples": len(subset),
            "channels": {c: aggregate_channel([r.channel_metrics for r in subset], c) for c in reported},
        }
        torques = [r.torque for r in subset if r.torque is not None]
        if torques:
            entry["torque"] = aggregate_torque(torques)
        per_case[str(case_index)] = entry

    summary: Dict[str, object] = {
        "n_samples": len(results),
        "n_cases": len({r.case_index for r in results}),
        "element_coverage": float(np.mean([r.element_coverage for r in results])),
        "overall": overall,
        "by_region": per_region,
        "by_case": per_case,
    }

    all_torque = [r.torque for r in results if r.torque is not None]
    if all_torque:
        pooled = aggregate_torque(all_torque)
        # Ripple is a within-case quantity: max-min over the rotor sweep of one
        # geometry. Pooling it across cases would measure the spread of the DOE
        # operating points instead, so the ripple fields are recomputed as the
        # mean over per-case ripples and the pooled ones dropped.
        case_ripple = [
            entry["torque"] for entry in per_case.values() if "torque" in entry
        ]
        for key in ("ripple_true", "ripple_pred", "ripple_error_pct"):
            values = [c[key] for c in case_ripple if np.isfinite(c.get(key, np.nan))]
            pooled[f"{key}_case_mean"] = float(np.mean(values)) if values else float("nan")
            pooled.pop(key, None)
        pooled["note"] = "ripple aggregated per case, then averaged; rel_error pooled over samples"
        summary["torque"] = pooled
    return summary


def models_with_no_samples(artifact: Mapping[str, object]) -> List[str]:
    """Names of models in a scorecard that scored nothing.

    `evaluate_predictor` catches per-sample exceptions so one broken sample
    cannot silently shrink the test set. The cost of that safeguard is that a
    *systematic* fault — a missing graph-op backend, a checkpoint whose graph
    never builds — also arrives as per-sample failures, leaving a scorecard that
    is structurally complete and numerically empty. Callers use this to tell the
    two apart; `main` turns a non-empty result into a non-zero exit code.
    """
    models = artifact.get("models") or {}
    return [
        name
        for name, card in models.items()
        if not (card.get("summary") or {}).get("n_samples")
    ]


def run_benchmark(
    predictors: Sequence[FieldPredictor],
    config: BenchmarkConfig,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Run the benchmark end to end and optionally write the results artifact."""
    # Before any scoring: fix TF32, cuDNN autotuning and the seeds, so a number
    # cannot move because a torch default changed underneath it. This does not
    # make the run bitwise reproducible — see eval.determinism for what it can
    # and cannot promise — and the settings it fixed go into the artifact.
    determinism = pin_determinism()

    manifest = json.loads((Path(config.data_dir) / "doe_manifest.json").read_text(encoding="utf-8"))
    digest = manifest_digest(manifest)
    all_cases = [int(c["index"]) for c in manifest.get("cases", [])]

    split = resolve_case_split(manifest, all_cases, config.split_path)
    subset_cases = split.subset(config.subset)

    report = load_doe_cases(
        manifest,
        Path(config.data_dir),
        case_indices=subset_cases,
        source_types=config.source_types,
        max_steps=config.max_steps_per_case,
    )

    scorecards: Dict[str, object] = {}
    for predictor in predictors:
        results, failures = evaluate_predictor(predictor, report.records, config)
        scorecards[predictor.name] = {
            "model": predictor.describe(),
            "summary": summarize(results, config.channels),
            "failures": list(failures),
        }

    artifact: Dict[str, object] = {
        "version": BENCHMARK_VERSION,
        "supersedes": [
            "model_comparison_summary.json",
            "comparison_results.json",
            "mgn_fem_comparison/test_summary.json",
        ],
        "note": (
            "Case-level holdout, element-support comparison, physical units. "
            "Numbers from the pre-R0 eval scripts used a record-level random split "
            "and are not comparable to these."
        ),
        "config": config.to_dict(),
        "doe_digest": digest,
        "split": {
            "granularity": "case",
            "seed": split.seed,
            "sizes": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
            "evaluated_subset": config.subset,
            "evaluated_cases": list(subset_cases),
        },
        "data": report.to_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "determinism": determinism,
        },
        "models": scorecards,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return artifact


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Reference predictors — the floor and the ceiling of the scorecard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FemIdentityPredictor:
    """Returns the ground truth. Scores 0 error; used to self-test the harness.

    If this predictor ever reports nonzero error, the harness itself is broken
    (support mismatch, channel misalignment) and no model number from that run
    should be trusted.
    """

    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    name: str = "fem_identity"
    output_support: str = "element"

    def predict(self, record: CaseRecord, sample: CaseSample) -> np.ndarray:
        return truth_matrix(sample, self.channels)

    def describe(self) -> Dict[str, object]:
        return {"kind": "harness_self_test", "params": 0}


@dataclass(frozen=True)
class ZeroFieldPredictor:
    """Predicts zero everywhere — the do-nothing reference.

    Its nRMSE is 100% by construction, which anchors the scale: a model at 60%
    nRMSE is not "somewhat accurate", it is worse than useful. Its torque error
    is the reference for how much of the torque a model actually recovers.
    """

    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    name: str = "zero_field"
    output_support: str = "element"

    def predict(self, record: CaseRecord, sample: CaseSample) -> np.ndarray:
        return np.zeros((record.mesh.n_elements, len(self.channels)), dtype=np.float64)

    def describe(self) -> Dict[str, object]:
        return {"kind": "reference_floor", "params": 0}


@dataclass(frozen=True)
class NodeResamplingFloorPredictor:
    """Ground truth pushed through the element -> node -> element round trip.

    Node-predicting models (MeshGraphNet, GINO) are trained on element fields
    that were first averaged onto nodes, and the harness averages their node
    output back onto elements to compare against the FEM answer. Both steps are
    lossy, and in the airgap they are badly lossy: the band is one or two
    elements thick, so its nodes sit on the boundary with the stator iron and
    pick up iron flux density an order of magnitude larger than the airgap value.

    This predictor isolates that loss. Its score is the best any node-support
    model could achieve on this pipeline — a model at this line has zero error of
    its own. Read every node-support row against this one, not against zero.
    """

    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    name: str = "node_resampling_floor"
    output_support: str = "element"

    def predict(self, record: CaseRecord, sample: CaseSample) -> np.ndarray:
        from eval.torque import element_to_nodal

        truth = truth_matrix(sample, self.channels)
        n_nodes = record.mesh.n_nodes
        return np.stack(
            [
                nodal_to_element(
                    element_to_nodal(truth[:, c], record.mesh.tri, n_nodes), record.mesh.tri
                )
                for c in range(truth.shape[1])
            ],
            axis=1,
        )

    def describe(self) -> Dict[str, object]:
        return {
            "kind": "pipeline_floor",
            "params": 0,
            "note": "irreducible error of the element->node->element support round trip",
        }


def _bilinear_sample(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a regular grid at scattered query points.

    `field` is indexed ``[row, col]`` with row along `gy` and column along `gx`,
    matching the ``np.meshgrid(gx, gy)`` layout the grid trainers build.
    """
    n_x, n_y = len(gx), len(gy)
    dx = (gx[-1] - gx[0]) / (n_x - 1)
    dy = (gy[-1] - gy[0]) / (n_y - 1)

    fx = np.clip((x - gx[0]) / dx, 0.0, n_x - 1)
    fy = np.clip((y - gy[0]) / dy, 0.0, n_y - 1)
    j0 = np.clip(np.floor(fx).astype(np.int64), 0, n_x - 2)
    i0 = np.clip(np.floor(fy).astype(np.int64), 0, n_y - 2)
    tx = fx - j0
    ty = fy - i0

    return (
        field[i0, j0] * (1.0 - tx) * (1.0 - ty)
        + field[i0, j0 + 1] * tx * (1.0 - ty)
        + field[i0 + 1, j0] * (1.0 - tx) * ty
        + field[i0 + 1, j0 + 1] * tx * ty
    )


@dataclass(frozen=True)
class GridResamplingFloorPredictor:
    """Ground truth pushed through the mesh -> regular grid -> mesh round trip.

    Grid models (FNO, and the RNN that consumed the same tensors) never see the
    unstructured mesh. `doe_data_utils.mesh_to_grid` scatters the element fields
    onto nodes, then linearly interpolates them onto a `grid_res` x `grid_res`
    Cartesian grid spanning the mesh bounding box, filling everything outside the
    convex hull with 0.0. Whatever such a model predicts has to come back through
    that grid to be compared against the FEM element fields.

    This predictor applies exactly that transform to the truth itself, so its
    score is the *best case* for any grid model: a perfect FNO, one with zero
    error of its own, still lands here. The measurement is why FNO/GINO/RNN were
    retired rather than tuned — the round trip alone costs far more than the
    torque gate allows, so no amount of model capacity can recover it.

    Two things make the loss large. The motor annulus is mostly empty in a square
    bounding box, so a 64x64 grid spends most of its cells on air while the
    airgap — one or two elements thick and the only place torque comes from — is
    thinner than a single cell. And the bounding box is rebuilt from each step's
    own node coordinates, so as the rotor sweeps, the grid moves with it and the
    resampling error is not even stationary across a case.
    """

    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    name: str = "grid_resampling_floor"
    output_support: str = "element"
    grid_res: int = 64

    def __post_init__(self) -> None:
        # This is the only scipy use in `eval/` — the rest of the package is
        # numpy-only — and the row is on by default. Probe it here, at
        # construction, which main() does *before* load_doe_cases: otherwise a
        # scipy-less host reads all six test cases from H5 (minutes), scores
        # fem_identity, and only then dies inside predict().
        try:
            import scipy  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise ModuleNotFoundError(
                "grid_resampling_floor needs scipy, the only scipy dependency in "
                "eval/. Install scipy, or pass --skip-grid-floor to score without "
                "this row."
            ) from exc

    def predict(self, record: CaseRecord, sample: CaseSample) -> np.ndarray:
        # scipy.interpolate.griddata(method="linear") *is* LinearNDInterpolator on
        # a Delaunay triangulation of the points; constructing it once and reusing
        # it across channels is numerically identical and avoids re-triangulating
        # ~20k nodes per channel per step.
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay

        from eval.mesh_regions import element_centroids_m
        from eval.torque import element_to_nodal

        truth = truth_matrix(sample, self.channels)
        mesh = record.mesh
        node_x = np.asarray(sample.node_x_mm, dtype=np.float64) * 1e-3
        node_y = np.asarray(sample.node_y_mm, dtype=np.float64) * 1e-3

        # The grid spans THIS step's node bounding box, as mesh_to_grid does; the
        # rotor sweep moves it, so the floor must be measured over the full step
        # range, not the first few steps.
        res = int(self.grid_res)
        gx = np.linspace(node_x.min(), node_x.max(), res)
        gy = np.linspace(node_y.min(), node_y.max(), res)
        grid_x, grid_y = np.meshgrid(gx, gy)
        query = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        triangulation = Delaunay(np.column_stack([node_x, node_y]))
        centroid_x, centroid_y = element_centroids_m(sample.node_x_mm, sample.node_y_mm, mesh.tri)

        out = np.empty((mesh.n_elements, len(self.channels)), dtype=np.float64)
        for c in range(truth.shape[1]):
            node_values = element_to_nodal(truth[:, c], mesh.tri, mesh.n_nodes)
            interpolator = LinearNDInterpolator(triangulation, node_values, fill_value=0.0)
            on_grid = interpolator(query).reshape(res, res).astype(np.float32)
            out[:, c] = _bilinear_sample(on_grid, gx, gy, centroid_x, centroid_y)
        return out

    def describe(self) -> Dict[str, object]:
        return {
            "kind": "pipeline_floor",
            "params": 0,
            "grid_res": int(self.grid_res),
            "note": (
                f"irreducible error of the element->node->{self.grid_res}x{self.grid_res} "
                "regular grid->element round trip that every grid model (FNO/RNN) is "
                "subject to; grid rebuilt per step from that step's node bounding box, "
                "outside-hull fill 0.0"
            ),
        }


@dataclass(frozen=True)
class CurlFloorPredictor:
    """Best B field a nodal-A model could produce through the P1 curl.

    Solves for the nodal A whose element-wise curl best fits the FEM B field, and
    returns that curl. It is the A-and-curl counterpart of
    `NodeResamplingFloorPredictor`: the representation floor of the pipeline the
    review's R1 proposes, so the two rows make the pipelines directly comparable.

    On the DOE export this floor is ~5-8% |B| nRMSE and ~1-3% torque nRMSE,
    against ~17% and ~60% for the node round trip. That gap — not any model
    change — is what puts gate G2 within reach.

    Elements of the re-meshed sliding band are excluded (see
    `eval.mesh_regions.sliding_band_mask`); on those the FEM value is passed
    through unchanged so the comparison is not credited for them either way.
    """

    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    name: str = "curl_representation_floor"
    output_support: str = "element"
    exclude_sliding_band: bool = True

    def predict(self, record: CaseRecord, sample: CaseSample) -> np.ndarray:
        from eval.mesh_regions import sliding_band_mask
        from phase1_static.discrete_curl import (
            build_p1_curl_operator,
            curl_a_to_b,
            fit_nodal_a,
            mesh_validity_mask,
        )

        truth = truth_matrix(sample, self.channels)
        ix, iy = self.channels.index("Bx"), self.channels.index("By")

        mesh = record.mesh
        exclude = (
            sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)
            if self.exclude_sliding_band
            else None
        )
        valid = mesh_validity_mask(
            sample.node_x_mm * 1e-3, sample.node_y_mm * 1e-3, mesh.tri, exclude=exclude
        )
        operator = build_p1_curl_operator(
            sample.node_x_mm * 1e-3, sample.node_y_mm * 1e-3, mesh.tri, valid=valid
        )

        target = np.stack(
            [truth[operator.element_index, ix], truth[operator.element_index, iy]], axis=1
        )
        a_nodal, _ = fit_nodal_a(operator, target)

        out = truth.copy()
        fitted = curl_a_to_b(operator, a_nodal)
        out[operator.element_index, ix] = fitted[:, 0]
        out[operator.element_index, iy] = fitted[:, 1]
        return out

    def describe(self) -> Dict[str, object]:
        return {
            "kind": "pipeline_floor",
            "params": 0,
            "note": (
                "best achievable B from a nodal-A model via P1 element curl; "
                "sliding band excluded" if self.exclude_sliding_band else "best achievable B via P1 curl"
            ),
        }


@dataclass(frozen=True)
class TrainMeanPredictor:
    """Predicts the per-region mean field of the training cases.

    A geometry-blind baseline: it captures "the airgap has more flux than the
    shaft" and nothing else. A learned surrogate that does not clearly beat this
    has not learned the geometry-to-field mapping, whatever its MSE says.
    """

    region_mean: Mapping[str, np.ndarray]
    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    name: str = "train_region_mean"
    output_support: str = "element"

    def predict(self, record: CaseRecord, sample: CaseSample) -> np.ndarray:
        groups = record.mesh.region_grouping().group_of_element
        out = np.zeros((record.mesh.n_elements, len(self.channels)), dtype=np.float64)
        for group, values in self.region_mean.items():
            mask = groups == group
            if np.any(mask):
                out[mask] = values
        return out

    def describe(self) -> Dict[str, object]:
        return {"kind": "reference_baseline", "params": int(len(self.region_mean) * len(self.channels))}


def fit_train_region_mean(
    records: Sequence[CaseRecord],
    channels: Sequence[str] = DEFAULT_CHANNELS,
) -> Dict[str, np.ndarray]:
    """Fit `TrainMeanPredictor` on training-subset records."""
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, float] = {}
    for record in records:
        groups = record.mesh.region_grouping().group_of_element
        for sample in record.samples:
            truth = truth_matrix(sample, channels)
            for group in np.unique(groups):
                mask = groups == group
                sums[group] = sums.get(group, np.zeros(len(channels))) + truth[mask].sum(axis=0)
                counts[group] = counts.get(group, 0.0) + float(np.count_nonzero(mask))
    return {g: sums[g] / counts[g] for g in sums if counts.get(g, 0.0) > 0}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    parser.add_argument("--split", type=Path, default=Path("eval/splits/doe40_case_split.json"))
    parser.add_argument("--subset", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out", type=Path, default=Path("results/benchmark_v2_baseline.json"))
    parser.add_argument("--max-steps-per-case", type=int, default=None)
    parser.add_argument("--axial-length-m", type=float, default=1.0,
                        help="Stack length; default 1.0 reports torque per metre of stack")
    parser.add_argument("--source-types", nargs="+", default=["OnLoadTorque"])
    parser.add_argument("--skip-curl-floor", action="store_true",
                        help="Skip the P1-curl representation floor (it solves a "
                             "least-squares problem per sample; ~1.5 s each)")
    parser.add_argument("--skip-grid-floor", action="store_true",
                        help="Skip the mesh->grid->mesh representation floor (it "
                             "triangulates the mesh per sample; ~0.3 s each)")
    parser.add_argument("--grid-floor-res", type=int, default=64,
                        help="Resolution of the grid representation floor; default "
                             "64, the resolution train_doe_fno.py trained at")
    parser.add_argument("--mgn-ckpt", type=Path, nargs="+", default=None,
                        help="MeshGraphNet checkpoint(s) to score (requires torch)")
    parser.add_argument("--curl-ckpt", type=Path, nargs="+", default=None,
                        help="Checkpoint(s) from train_doe_curl_mgn.py (nodal A + P1 curl)")
    parser.add_argument("--mgn-output-channels", nargs="+", default=None,
                        help="Native output channel order of the checkpoint "
                             "(default: inferred from the output width)")
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        data_dir=args.data_dir,
        split_path=args.split,
        subset=args.subset,
        source_types=tuple(args.source_types),
        max_steps_per_case=args.max_steps_per_case,
        axial_length_m=args.axial_length_m,
    )

    predictors: List[FieldPredictor] = [
        FemIdentityPredictor(channels=config.channels),
        NodeResamplingFloorPredictor(channels=config.channels),
        ZeroFieldPredictor(channels=config.channels),
    ]
    if not args.skip_grid_floor:
        predictors.insert(
            2, GridResamplingFloorPredictor(channels=config.channels, grid_res=args.grid_floor_res)
        )
    if not args.skip_curl_floor:
        predictors.insert(1, CurlFloorPredictor(channels=config.channels))

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    all_cases = [int(c["index"]) for c in manifest.get("cases", [])]
    split = resolve_case_split(manifest, all_cases, config.split_path)

    train_report = load_doe_cases(
        manifest, args.data_dir, case_indices=split.train,
        source_types=config.source_types, max_steps=args.max_steps_per_case,
    )
    predictors.append(
        TrainMeanPredictor(
            region_mean=fit_train_region_mean(train_report.records, config.channels),
            channels=config.channels,
        )
    )

    if args.mgn_ckpt:
        from eval.predictors import CheckpointFeatureMismatch, MeshGraphNetPredictor

        for ckpt in args.mgn_ckpt:
            try:
                predictors.append(
                    MeshGraphNetPredictor.from_checkpoint(
                        ckpt,
                        channels=config.channels,
                        name=f"mgn:{Path(ckpt).stem}",
                        output_channel_order=args.mgn_output_channels,
                    )
                )
            except CheckpointFeatureMismatch as exc:
                # A rejected checkpoint is a result, not a crash: report why and
                # keep scoring the rest.
                print(f"[REJECTED] {ckpt}\n  {exc}\n")

    if args.curl_ckpt:
        from eval.predictors import CheckpointFeatureMismatch, CurlMeshGraphNetPredictor

        for ckpt in args.curl_ckpt:
            try:
                predictors.append(
                    CurlMeshGraphNetPredictor.from_checkpoint(
                        ckpt, channels=config.channels, name=f"curl:{Path(ckpt).stem}"
                    )
                )
            except CheckpointFeatureMismatch as exc:
                print(f"[REJECTED] {ckpt}\n  {exc}\n")

    artifact = run_benchmark(predictors, config, output_path=args.out)

    print(f"Benchmark written to {args.out}")
    print(f"  split: {config.subset} cases {list(split.subset(config.subset))}")
    print(f"  samples scored: {artifact['data']['n_samples']} "
          f"(skipped {artifact['data']['n_skipped']})")
    scored_nothing = models_with_no_samples(artifact)
    for name, card in artifact["models"].items():
        summary = card["summary"]
        if not summary.get("n_samples"):
            print(f"  {name:22s} NO SAMPLES SCORED")
            if card["failures"]:
                print(f"    {len(card['failures'])} failure(s), first: {card['failures'][0]}")
            continue
        bn = summary["overall"].get("Bnorm", {})
        tq = summary.get("torque", {})
        print(
            f"  {name:22s} |B| nRMSE {bn.get('nrmse_pct', float('nan')):7.3f}%"
            f"   torque nRMSE {tq.get('nrmse_torque_pct', float('nan')):8.3f}%"
        )
        provenance = card["model"].get("split_provenance", {})
        if provenance.get("contaminated"):
            print(f"    [!] {provenance.get('warning', 'checkpoint split provenance unknown')}")
        if card["failures"]:
            print(f"    {len(card['failures'])} failure(s), first: {card['failures'][0]}")

    # A model that scored nothing is a broken run, not a result. The harness
    # catches per-sample exceptions so one bad sample cannot shrink the test
    # set, which means a systematic fault — a missing graph-op backend, a
    # checkpoint that cannot build its graph — reaches this point looking like
    # a success. Only the exit code separates the two, so it has to say so.
    if scored_nothing:
        print(
            f"\nFAIL: {len(scored_nothing)} model(s) scored no samples: "
            f"{', '.join(scored_nothing)}\n"
            "  The scorecard was still written, but it holds no numbers for these.\n"
            "  Check the failure lines above; a systematic cause repeats identically "
            "on every sample."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
