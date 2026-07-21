#!/usr/bin/env python3
"""Animate |B| contours and the torque waveform, FEM against surrogate.

Renders one frame per rotor step: three element-wise |B| panels (FEM,
prediction, absolute error) over the mesh, plus the torque waveform with a
marker on the current step. Frames are assembled into a GIF with PIL.

Two conventions matter for the plot to mean anything:

* **Element support.** Motor-CAD stores B per element and the surrogate is
  scored per element, so the panels use `tripcolor(facecolors=...)` — one flat
  colour per triangle. Interpolating to nodes for a smoother picture would show
  a field neither model produced.
* **Validity mask.** The sliding-band elements have no valid per-step geometry
  (the solver re-meshes them and the export keeps only the reference
  connectivity), so they are masked out rather than drawn from stale
  connectivity. They appear as a blank wedge in the airgap; that is honest.

Colour scales are fixed across every frame and every panel, otherwise the
animation shows the colormap rescaling rather than the field changing.

Usage (inside the PhysicsNeMo container):
    python tools/make_field_gif.py --case 4 --ckpt results/mgn_nodeB_long.pt \
        --out results/viz/case0004_field_torque.gif
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Running this file directly puts tools/ on sys.path, not the repo root, so the
# eval/ and phase1_static/ packages would not resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402
from PIL import Image  # noqa: E402

from eval.doe_dataset import load_doe_cases  # noqa: E402
from eval.mesh_regions import sliding_band_mask  # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band  # noqa: E402
from phase1_static.discrete_curl import mesh_validity_mask  # noqa: E402
from phase1_static.sector_symmetry import (  # noqa: E402
    cumulative_rotor_angle,
    rigid_rotor_node_mask,
    wrap_rotor_coordinates,
)


def compute_frames(record, predictor, steps: List[int], wrap_display: bool = True,
                   axial_length_m: float = 0.150):
    """Per-step FEM/predicted |B|, the validity mask and both torque curves.

    `wrap_display` folds the rigid rotor back into the modelled sector **for
    drawing only**. The export reports the rotor unwrapped — it walks out to
    -153 deg while the stator stays at [-45, 0] — so the raw coordinates draw
    two disjoint wedges instead of a machine. Wrapping moves where each element
    is drawn; it does not touch the value plotted in it, and |B| is a scalar so
    no vector rotation is involved. Torque is computed from the unwrapped
    fields either way, on the stationary a1 band.
    """
    mesh = record.mesh
    band = build_airgap_band(
        mesh.node_x_mm, mesh.node_y_mm, mesh.tri, mesh.reg_code,
        mesh.name_of_code, moving_reg_codes=mesh.moving_reg_codes,
    )
    sliding = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)

    rigid = cum = None
    if wrap_display:
        rigid = rigid_rotor_node_mask(
            mesh.tri, mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes, mesh.n_nodes
        )
        cum = cumulative_rotor_angle([s.rotate_step for s in record.samples])

    frames = []
    for si in steps:
        sample = record.samples[si]
        valid = mesh_validity_mask(
            sample.node_x_mm * 1e-3, sample.node_y_mm * 1e-3, mesh.tri, exclude=sliding
        )
        bx_t, by_t = sample.fields["bx"], sample.fields["by"]
        pred = predictor.predict(record, sample)

        draw_tri = None
        if wrap_display:
            # Rigid wrap first (keeps the rotor coherent), then fold whatever
            # still sits outside the sector — the two together give a picture
            # that reads as one machine.
            wx, wy, _ = wrap_rotor_coordinates(
                sample.node_x_mm, sample.node_y_mm, rigid, float(cum[si])
            )
            draw_x, draw_y, draw_tri = fold_into_sector(wx, wy, mesh.tri)
        else:
            draw_x, draw_y = sample.node_x_mm, sample.node_y_mm

        frames.append(
            {
                "step": si,
                "node_x": draw_x,
                "node_y": draw_y,
                "draw_tri": draw_tri,
                "valid": valid,
                "b_fem": np.hypot(bx_t, by_t),
                "b_pred": np.hypot(pred[:, 0], pred[:, 1]),
                "t_fem": arkkio_torque(band, bx_t, by_t, axial_length_m),
                "t_pred": arkkio_torque(band, pred[:, 0], pred[:, 1], axial_length_m),
            }
        )
    return frames


def fold_into_sector(node_x, node_y, tri, sector_deg: float = 45.0):
    """Explode the mesh per element and fold every element into one sector.

    A rigid rotation cannot make this mesh look like a machine: the exported
    rotor sector occupies [-65.5, -20.5] deg while the stator occupies
    [-45, 0], an offset of 20.5 deg that no multiple of 45 removes. The FEM
    closes that gap through the anti-periodic interface, so the display does the
    same — each element is rotated by whatever multiple of the sector puts its
    centroid inside [-sector, 0].

    This is the anti-periodic map, which carries a sign flip; |B| is a scalar so
    the plotted value is unaffected. Elements are given their own vertices
    because neighbours can need different multiples.

    Returns ``(x, y, tri)`` for a triangulation with ``3 * n_elements`` vertices.
    """
    i1, i2, i3 = (np.asarray(a, dtype=np.int64) for a in tri)
    n_elem = i1.size

    vx = np.stack([node_x[i1], node_x[i2], node_x[i3]], axis=1)
    vy = np.stack([node_y[i1], node_y[i2], node_y[i3]], axis=1)

    cx, cy = vx.mean(axis=1), vy.mean(axis=1)
    theta = np.degrees(np.arctan2(cy, cx))
    # Multiples of the sector needed to bring the centroid into [-sector, 0).
    k = np.ceil(-theta / sector_deg) - 1.0
    k = np.where(np.isclose(theta, 0.0), 0.0, k)

    angle = np.radians(k * sector_deg)[:, None]
    ca, sa = np.cos(angle), np.sin(angle)
    rx = ca * vx - sa * vy
    ry = sa * vx + ca * vy

    verts = np.arange(3 * n_elem).reshape(n_elem, 3)
    return rx.ravel(), ry.ravel(), (verts[:, 0], verts[:, 1], verts[:, 2])


def _panel(ax, triang, values, vmin, vmax, cmap, title):
    art = ax.tripcolor(triang, facecolors=values, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    return art


def render(
    frames, case_index: int, out_path: Path, fps: float, label: str, wrap_note: str = "",
    axial_length_m: float = 0.150,
) -> Tuple[Path, float, float]:
    torque_unit = ("torque  [N·m per m of stack]" if axial_length_m == 1.0
                   else f"torque  [N·m]  (stack {axial_length_m * 1e3:.0f} mm)")
    # One scale for FEM and prediction so the two panels are comparable, and a
    # separate one for the error. p99 rather than max keeps a handful of
    # saturated tooth-tip elements from flattening the rest of the picture.
    all_fem = np.concatenate([f["b_fem"][f["valid"]] for f in frames])
    vmax = float(np.ceil(np.percentile(all_fem, 99) * 10) / 10)
    all_err = np.concatenate(
        [np.abs(f["b_pred"] - f["b_fem"])[f["valid"]] for f in frames]
    )
    emax = float(np.ceil(np.percentile(all_err[np.isfinite(all_err)], 99) * 20) / 20)

    t_fem = np.array([f["t_fem"] for f in frames])
    t_pred = np.array([f["t_pred"] for f in frames])
    steps = np.array([f["step"] for f in frames])

    images = []
    for k, f in enumerate(frames):
        fig = plt.figure(figsize=(11.0, 6.4), dpi=100)
        grid = fig.add_gridspec(2, 3, height_ratios=[2.05, 1.0], hspace=0.18, wspace=0.06)

        triang = Triangulation(f["node_x"], f["node_y"], np.stack(f["draw_tri"], axis=1))
        triang.set_mask(~f["valid"])

        err = np.abs(f["b_pred"] - f["b_fem"])
        ax0 = fig.add_subplot(grid[0, 0])
        a0 = _panel(ax0, triang, f["b_fem"], 0.0, vmax, "viridis", "FEM  |B|")
        ax1 = fig.add_subplot(grid[0, 1])
        _panel(ax1, triang, f["b_pred"], 0.0, vmax, "viridis", "Surrogate  |B|")
        ax2 = fig.add_subplot(grid[0, 2])
        a2 = _panel(ax2, triang, err, 0.0, emax, "magma", "|error|")

        cb0 = fig.colorbar(a0, ax=[ax0, ax1], fraction=0.030, pad=0.01)
        cb0.set_label("|B|  [T]", fontsize=9)
        cb2 = fig.colorbar(a2, ax=ax2, fraction=0.046, pad=0.01)
        cb2.set_label("|B| error  [T]", fontsize=9)

        axt = fig.add_subplot(grid[1, :])
        axt.plot(steps, t_fem, color="#1f77b4", lw=1.8, label="FEM")
        axt.plot(steps, t_pred, color="#d62728", lw=1.8, ls="--", label="Surrogate")
        axt.axvline(f["step"], color="0.35", lw=1.0)
        axt.plot([f["step"]], [t_fem[k]], "o", color="#1f77b4", ms=6)
        axt.plot([f["step"]], [t_pred[k]], "o", color="#d62728", ms=6)
        axt.set_xlabel("rotor step", fontsize=9)
        axt.set_ylabel(torque_unit, fontsize=9)
        axt.legend(loc="upper right", fontsize=8, ncol=2)
        axt.grid(alpha=0.25)
        axt.tick_params(labelsize=8)

        fig.suptitle(
            f"DOE case {case_index:04d} (holdout) — {label}\n"
            f"step {f['step']:02d}/{steps.max():02d}   "
            f"T_FEM {t_fem[k]:7.1f}   T_pred {t_pred[k]:7.1f}   |   "
            f"|B| 0–{vmax:.1f} T, error 0–{emax:.2f} T{wrap_note}",
            fontsize=9.5,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000.0 / fps),
        loop=0,
        optimize=True,
    )
    return out_path, vmax, emax


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--axial-length-m", type=float, default=0.150,
                        help="Stack length for the torque axis. 0.150 m is "
                             "Stator_Lam_Length from the Motor-CAD .mot for this design; "
                             "pass 1.0 to plot torque per metre of stack instead")
    parser.add_argument("--no-wrap-display", action="store_true",
                        help="Draw the rotor at its raw exported (unwrapped) position, "
                             "which separates it from the stator on screen")
    args = parser.parse_args()

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    record = load_doe_cases(manifest, args.data_dir, case_indices=[args.case]).records[0]

    from eval.predictors import CurlMeshGraphNetPredictor

    predictor = CurlMeshGraphNetPredictor.from_checkpoint(args.ckpt)

    steps = list(range(0, len(record.samples), max(1, args.step_stride)))
    if args.max_steps:
        steps = steps[: args.max_steps]
    print(f"case {args.case}: {len(steps)} frames from {len(record.samples)} steps")

    wrap = not args.no_wrap_display
    frames = compute_frames(record, predictor, steps, wrap_display=wrap,
                            axial_length_m=args.axial_length_m)
    for f in frames:
        if f.get("draw_tri") is None:
            f["draw_tri"] = record.mesh.tri

    out, vmax, emax = render(
        frames, args.case, args.out, args.fps,
        args.label or Path(args.ckpt).stem,
        axial_length_m=args.axial_length_m,
        wrap_note="   (rotor wrapped into the sector for display)" if wrap else "",
    )
    size_mb = out.stat().st_size / 1e6
    t_fem = np.array([f["t_fem"] for f in frames])
    t_pred = np.array([f["t_pred"] for f in frames])
    print(f"wrote {out}  ({size_mb:.2f} MB, {len(frames)} frames)")
    print(f"  |B| scale 0-{vmax:.1f} T, error scale 0-{emax:.2f} T")
    print(f"  mean torque  FEM {t_fem.mean():.1f}  pred {t_pred.mean():.1f}  "
          f"(stack {args.axial_length_m * 1e3:.0f} mm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
