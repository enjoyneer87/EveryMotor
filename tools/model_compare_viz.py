#!/usr/bin/env python3
"""Side-by-side field comparison: FEM against one or more surrogate checkpoints.

Renders a row per model — FEM, prediction, and (prediction - FEM) — as flat
`tripcolor` panels on the true mesh connectivity, for one rotor step of one DOE
case.

Two rules shape this tool:

* **It computes no metrics.** R0 rule 1 puts all model comparison in `eval/`.
  The numbers in the panel titles are *read* from a `results/benchmark_v2_*.json`
  scorecard produced by `python -m eval.benchmark`. If a model has no row in the
  scorecard its title says so rather than showing a number computed here.
  Note the consequence: those numbers are **test-set aggregates over every case
  and step in the scorecard**, not the error of the single step being drawn. The
  figure labels them that way; do not read them as per-step.
* **Element support.** Motor-CAD stores B per element and the harness scores per
  element, so the panels use `tripcolor(facecolors=...)` — one flat colour per
  triangle. A node-support checkpoint is brought onto elements with
  `eval.benchmark.to_element_support`, the same round trip the scorecard scores,
  so the picture matches the number in its title.

Mesh connectivity comes from `eval.doe_dataset`, which already reads it from the
same H5 (`mesh/node_1..3`, remapped through `mesh/node_id`). The retired version
of this file carried its own `load_mesh_triangles_from_h5` doing the identical
remap; a second mesh reader is how the pre-R0 pipeline drifted out of sync with
itself, so this one uses the loader.

Sliding-band elements have no valid per-step geometry and are masked out rather
than drawn from stale connectivity; they appear as a blank wedge in the airgap.
The rotor is folded back into the modelled sector for display only (see
`tools/make_field_gif.fold_into_sector`).

Usage:
    python tools/model_compare_viz.py --case 4 --step 10 \
        --model "curl-notime=results/mgn_nodeB_notime_ep55_HPC134.pt" \
        --scorecard results/benchmark_v2_nodeB_notime.json \
        --out results/viz/case0004_step10_compare.png

Several models on one figure, mixing supports:
    python tools/model_compare_viz.py --case 4 --step 10 \
        --model "long=results/mgn_nodeB_long.pt" \
        --model "notime=results/mgn_nodeB_notime_ep55_HPC134.pt" \
        --scorecard results/benchmark_v2_nodeB_long.json \
        --scorecard results/benchmark_v2_nodeB_notime.json \
        --out results/viz/compare_two.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

# Running this file directly puts tools/ on sys.path, not the repo root, so the
# eval/ and phase1_static/ packages would not resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from eval.doe_dataset import load_doe_cases  # noqa: E402
from eval.mesh_regions import sliding_band_mask  # noqa: E402
from phase1_static.discrete_curl import mesh_validity_mask  # noqa: E402
from tools.make_field_gif import fold_into_sector  # noqa: E402


# {model label -> element-support field array of shape (n_elements, 2) as [bx, by]}.
# "GT" is the FEM reference and is required.
FieldMap = Mapping[str, np.ndarray]

GT_KEY = "GT"

# Panel quantity -> the channel name the scorecard reports it under.
SCORECARD_CHANNEL = {"bx": "Bx", "by": "By", "bmag": "Bnorm"}


# --------------------------------------------------------------------------
# scorecard
# --------------------------------------------------------------------------


def load_scorecard_metrics(paths: Iterable[str | Path]) -> Dict[str, dict]:
    """Index every model row across one or more benchmark_v2_*.json scorecards.

    Returns ``{scorecard_key: {"overall": ..., "checkpoint": ..., "source": ...}}``.
    Nothing is recomputed here; this only reads what `eval.benchmark` wrote.
    """
    index: Dict[str, dict] = {}
    for path in paths:
        path = Path(path)
        card = json.loads(path.read_text(encoding="utf-8"))
        for key, entry in (card.get("models") or {}).items():
            summary = entry.get("summary") or {}
            model = entry.get("model") or {}
            index[key] = {
                "overall": summary.get("overall") or {},
                "checkpoint": model.get("checkpoint"),
                "source": path.name,
            }
    return index


def resolve_metrics(
    label: str,
    ckpt: Optional[Path],
    index: Mapping[str, dict],
) -> Optional[dict]:
    """Find the scorecard row belonging to a checkpoint.

    Matches on the checkpoint filename first (the scorecard stores the path it
    scored), then falls back to the label appearing in the row key. Returns None
    when there is no match — the caller must then say "no scorecard row" rather
    than invent a number.
    """
    if ckpt is not None:
        stem = Path(ckpt).name
        for entry in index.values():
            recorded = entry.get("checkpoint")
            if recorded and Path(str(recorded).replace("\\", "/")).name == stem:
                return entry
    for key, entry in index.items():
        if label in key:
            return entry
    return None


def metric_caption(metrics: Optional[dict], quantity: str) -> str:
    """Title fragment carrying the scorecard's numbers for this quantity."""
    if metrics is None:
        return "no scorecard row"
    channel = SCORECARD_CHANNEL.get(quantity.lower().strip(), "Bnorm")
    stats = (metrics.get("overall") or {}).get(channel)
    if not stats:
        return f"no {channel} row in {metrics.get('source', 'scorecard')}"
    return (
        f"test-set nRMSE {stats['nrmse_pct']:.2f}%  RMSE {stats['rmse']:.3f} T"
    )


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def build_predictor(ckpt: Path, node_support: bool):
    """Instantiate a live predictor from `eval/predictors.py`."""
    if node_support:
        from eval.predictors import MeshGraphNetPredictor

        return MeshGraphNetPredictor.from_checkpoint(ckpt)

    from eval.predictors import CurlMeshGraphNetPredictor

    return CurlMeshGraphNetPredictor.from_checkpoint(ckpt)


def predict_field_map(
    record,
    step: int,
    models: Sequence[Tuple[str, Path, bool]],
) -> Dict[str, np.ndarray]:
    """Run each checkpoint on one step and return {label: (n_elements, 2)}.

    The FEM truth is included under ``"GT"``. Node-support predictions go
    through `eval.benchmark.to_element_support`, so what is drawn sits on the
    same support the scorecard scored.
    """
    from eval.benchmark import to_element_support

    sample = record.samples[step]
    field_map: Dict[str, np.ndarray] = {
        GT_KEY: np.stack([sample.fields["bx"], sample.fields["by"]], axis=1)
    }
    for label, ckpt, node_support in models:
        predictor = build_predictor(ckpt, node_support)
        raw = predictor.predict(record, sample)
        field_map[label] = to_element_support(raw, predictor.output_support, record)
    return field_map


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _field(values: np.ndarray, quantity: str) -> np.ndarray:
    """Pick bx, by or |B| out of an (n_elements, 2) array."""
    q = quantity.lower().strip()
    if q == "bx":
        return values[:, 0]
    if q == "by":
        return values[:, 1]
    return np.hypot(values[:, 0], values[:, 1])


def plot_model_comparison_grid(
    node_x: np.ndarray,
    node_y: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray],
    field_map: FieldMap,
    *,
    valid: Optional[np.ndarray] = None,
    captions: Optional[Mapping[str, str]] = None,
    quantity: str = "bmag",
    models: Optional[Iterable[str]] = None,
    title: str = "",
    save_path: Optional[str | Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """One row per model: FEM, prediction, error — flat-shaded on the true mesh.

    `field_map` maps a label to an element-support ``(n_elements, 2)`` array of
    ``[bx, by]``; the FEM reference lives under ``"GT"``. `captions` supplies the
    scorecard text for each label. No metric is computed in this module.
    """
    if GT_KEY not in field_map:
        raise ValueError(f"field_map needs a {GT_KEY!r} entry holding the FEM reference")
    labels = [m for m in (models if models is not None else field_map) if m != GT_KEY]
    labels = [m for m in labels if m in field_map]
    if not labels:
        raise ValueError("field_map holds no model entries besides GT")

    captions = captions or {}
    v_gt = _field(np.asarray(field_map[GT_KEY], dtype=np.float64), quantity)

    if valid is None:
        valid = np.ones(v_gt.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    preds = {m: _field(np.asarray(field_map[m], dtype=np.float64), quantity) for m in labels}
    errors = {m: preds[m] - v_gt for m in labels}

    # One field scale shared by FEM and every prediction, so the panels are
    # comparable; a separate symmetric scale for the error. Only valid, finite
    # elements set the limits — masked band elements are NaN.
    def _finite(a: np.ndarray) -> np.ndarray:
        sel = a[valid]
        return sel[np.isfinite(sel)]

    pool = np.concatenate([_finite(v_gt)] + [_finite(preds[m]) for m in labels])
    vmin, vmax = float(np.min(pool)), float(np.max(pool))

    err_pool = np.concatenate([np.abs(_finite(errors[m])) for m in labels])
    err_lim = max(float(np.max(err_pool)) if err_pool.size else 0.0, 1e-12)

    triang = Triangulation(node_x, node_y, np.stack(tri, axis=1))
    triang.set_mask(~valid)

    qty_label = "|B| [T]" if quantity.lower() == "bmag" else f"{quantity.upper()} [T]"

    fig, axes = plt.subplots(
        len(labels), 3, figsize=(13.5, 4.2 * len(labels)), dpi=dpi, layout="constrained"
    )
    axes = np.atleast_2d(axes)

    for i, m in enumerate(labels):
        ax_gt, ax_pred, ax_err = axes[i, 0], axes[i, 1], axes[i, 2]

        c0 = ax_gt.tripcolor(
            triang, facecolors=v_gt, shading="flat", cmap="viridis", vmin=vmin, vmax=vmax
        )
        ax_gt.set_title(f"FEM  {qty_label}", fontsize=10)
        fig.colorbar(c0, ax=ax_gt, fraction=0.046, pad=0.04)

        c1 = ax_pred.tripcolor(
            triang, facecolors=preds[m], shading="flat", cmap="viridis", vmin=vmin, vmax=vmax
        )
        caption = captions.get(m, "no scorecard row")
        ax_pred.set_title(f"{m}  {qty_label}\n{caption}", fontsize=9)
        fig.colorbar(c1, ax=ax_pred, fraction=0.046, pad=0.04)

        c2 = ax_err.tripcolor(
            triang, facecolors=errors[m], shading="flat", cmap="RdBu_r",
            vmin=-err_lim, vmax=err_lim,
        )
        ax_err.set_title(f"{m} - FEM   (scale ±{err_lim:.3f} T)", fontsize=9)
        fig.colorbar(c2, ax=ax_err, fraction=0.046, pad=0.04)

        for ax in (ax_gt, ax_pred, ax_err):
            ax.set_aspect("equal")
            ax.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=10)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")

    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_model_spec(spec: str) -> Tuple[str, Path]:
    """Parse ``label=path``. Split on the first '=' so Windows paths survive."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--model expects 'label=path/to/ckpt.pt', got {spec!r}"
        )
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"--model has an empty label: {spec!r}")
    return label, Path(path.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument(
        "--model", action="append", required=True, metavar="LABEL=CKPT",
        help="Checkpoint to render, repeatable. Label is what appears in the panel title.",
    )
    parser.add_argument(
        "--node-support", action="append", default=[], metavar="LABEL",
        help="Mark a label as a node-support MeshGraphNet checkpoint (default: curl, "
             "which is element-support natively).",
    )
    parser.add_argument(
        "--scorecard", action="append", default=[], type=Path,
        help="results/benchmark_v2_*.json to read panel-title metrics from, repeatable.",
    )
    parser.add_argument("--quantity", choices=["bx", "by", "bmag"], default="bmag")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--no-wrap-display", action="store_true",
        help="Draw the rotor at its raw exported (unwrapped) position, which "
             "separates it from the stator on screen.",
    )
    args = parser.parse_args()

    node_support = {label.strip() for label in args.node_support}
    specs = [parse_model_spec(s) for s in args.model]
    models = [(label, path, label in node_support) for label, path in specs]

    missing = [str(p) for _, p, _ in models if not p.exists()]
    if missing:
        parser.error("checkpoint not found: " + ", ".join(missing))

    index = load_scorecard_metrics(args.scorecard)
    if not index:
        # ASCII only: this console is cp949 on the Windows host and an em-dash
        # here raises UnicodeEncodeError. Figure text goes through matplotlib
        # and is unaffected.
        print("no scorecard given - panel titles will say so (no metric is computed here)")

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    record = load_doe_cases(manifest, args.data_dir, case_indices=[args.case]).records[0]
    if not 0 <= args.step < len(record.samples):
        parser.error(f"--step {args.step} out of range; case has {len(record.samples)} steps")

    print(f"case {args.case}: step {args.step} of {len(record.samples)}, "
          f"{record.mesh.n_elements} elements, {len(models)} model(s)")

    field_map = predict_field_map(record, args.step, models)

    captions = {}
    for label, ckpt, _ in models:
        metrics = resolve_metrics(label, ckpt, index)
        captions[label] = metric_caption(metrics, args.quantity)
        print(f"  {label}: {captions[label]}"
              + (f"  [{metrics['source']}]" if metrics else ""))

    mesh = record.mesh
    sample = record.samples[args.step]
    band = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)
    valid = mesh_validity_mask(
        sample.node_x_mm * 1e-3, sample.node_y_mm * 1e-3, mesh.tri, exclude=band
    )

    wrap = not args.no_wrap_display
    if wrap:
        from phase1_static.sector_symmetry import (
            cumulative_rotor_angle,
            rigid_rotor_node_mask,
            wrap_rotor_coordinates,
        )

        rigid = rigid_rotor_node_mask(
            mesh.tri, mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes, mesh.n_nodes
        )
        cum = cumulative_rotor_angle([s.rotate_step for s in record.samples])
        wx, wy, _ = wrap_rotor_coordinates(
            sample.node_x_mm, sample.node_y_mm, rigid, float(cum[args.step])
        )
        draw_x, draw_y, draw_tri = fold_into_sector(wx, wy, mesh.tri)
    else:
        draw_x, draw_y, draw_tri = sample.node_x_mm, sample.node_y_mm, mesh.tri

    sources = ", ".join(sorted({e["source"] for e in index.values()})) or "none"
    title = (
        f"DOE case {args.case:04d}, step {args.step} — {args.quantity}\n"
        f"panel metrics are test-set aggregates read from {sources}, "
        f"not the error of this step"
        + ("   (rotor wrapped into the sector for display)" if wrap else "")
    )

    fig = plot_model_comparison_grid(
        draw_x, draw_y, draw_tri, field_map,
        valid=valid,
        captions=captions,
        quantity=args.quantity,
        models=[label for label, _, _ in models],
        title=title,
        save_path=args.out,
        dpi=args.dpi,
    )
    plt.close(fig)

    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB, "
          f"{int(valid.sum())}/{valid.size} elements drawn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
