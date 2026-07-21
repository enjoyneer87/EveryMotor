#!/usr/bin/env python3
"""MeshGraphNet that predicts nodal A and is supervised on element B via the curl.

What is different from `train_doe_meshgraphnet.py`
--------------------------------------------------
That script predicts (Bx, By) at nodes. Motor-CAD stores B per element, so its
targets are element fields averaged onto nodes and its predictions are averaged
back onto elements to be scored. Measured on the DOE holdout, that round trip
alone costs 17.0% |B| nRMSE and **59.5% torque nRMSE** — before the model
predicts anything. Gate G2 (torque < 3%) is unreachable through it.

Here the network predicts the scalar potential A at nodes, and B is obtained on
elements by the exact P1 element-wise curl (`phase1_static.discrete_curl`). Node
values go in, element values come out, so there is no round trip. The
representation floor of this path is 5.3% |B| nRMSE and **1.58% torque nRMSE**.

Supervision (review R1 "loss v2")
---------------------------------
    L = w_B * MSE(curl(A_pred), B_fem)  +  w_gauge * mean(A_pred)^2
        [+ w_A * MSE(A_pred, A_export)  — off by default, see below]

B is supervised directly through the curl. A is deliberately **not** supervised
against the exported potential: the export stores A already averaged onto
elements, and differentiating that reconstruction scores 38.7% |B| nRMSE — worse
than the round trip it is meant to replace. Fitting A to it would drag the model
down to that floor. `--w-a` exists to test the claim, not because it helps.

A is a potential, so it is only determined up to an additive constant; the gauge
term pins that constant instead of leaving it to drift.

Excluded elements
-----------------
The Motor-CAD moving-mesh export writes one reference connectivity plus per-step
node coordinates, but the solver re-meshes the airgap sliding band every step.
Combining them tangles 180-246 elements (all in layer a2) from step 3 onward, so
those elements carry no valid geometry and are dropped from the curl and the
loss. They are **kept in the message-passing graph**: they are the only thing
connecting rotor to stator, and removing them would split the graph in two. The
edge vectors there pair nodes that are not really adjacent at that step, which
makes them long-range edges rather than wrong ones.

Usage (inside Docker):
    python /workspace/app/train_doe_curl_mgn.py \
        --data-dir backup/doe_data --epochs 40 --step-stride 3

Dependencies: torch, torch_geometric, physicsnemo, h5py, numpy, scipy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from eval.case_split import assign_records, resolve_case_split
from eval.doe_dataset import CaseRecord, CaseSample, load_doe_cases
from eval.feature_guard import (
    MGN_EDGE_FEATURES,
    MGN_EDGE_FEATURES_V2,
    MGN_NODE_FEATURES,
    MGN_NODE_FEATURES_V2,
    assert_feature_count,
    assert_input_features_clean,
)
from eval.mesh_regions import AIRGAP_NAME_PATTERN, sliding_band_mask
from phase1_static.discrete_curl import build_p1_curl_operator, mesh_validity_mask
from phase1_static.sector_symmetry import (
    DEFAULT_SECTOR_DEG,
    anti_periodic_edges,
    cumulative_rotor_angle,
    cut_plane_pairs,
    rigid_rotor_node_mask,
    rotor_angle_features,
    wrap_rotor_coordinates,
)

SOURCE_TYPES_FOR_TRAINING = ("OnLoadTorque",)


class CurlData(Data):
    """Graph carrying its own P1 curl operator.

    `curl_node_index` holds node ids, so PyG must shift it by the node offset
    when graphs are batched — the same treatment `edge_index` gets. Without the
    `__inc__` override, batching would silently point every graph's operator at
    the first graph's nodes.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("curl_node_index", "cut_pairs"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("curl_node_index", "cut_pairs"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def build_curl_graph(
    record: CaseRecord,
    sample: CaseSample,
    band_mask: np.ndarray,
    rigid_mask: np.ndarray,
    cumulative_deg: float,
    sector_deg: float = DEFAULT_SECTOR_DEG,
    wrap_rotor: bool = True,
    anti_periodic: bool = True,
    legacy_features: bool = False,
    airgap_weight: float = 1.0,
) -> Optional[CurlData]:
    """Build one training graph: clean node features + a curl operator + element B.

    Node features are the 10 in `MGN_NODE_FEATURES_V2` — position, region code,
    the operating point, the DOE geometry parameters and the cumulative rotor
    angle. No solved quantity is used as an input; `assert_input_features_clean`
    enforces it here.

    With `wrap_rotor` the rigid rotor is folded back into the modelled sector and
    the B target is multiplied by the anti-periodic sign, so every sample shows
    the network a configuration that was actually solved.
    """
    mesh = record.mesh
    n_nodes = mesh.n_nodes
    i1, i2, i3 = mesh.tri
    if n_nodes == 0 or i1.size == 0:
        return None

    if wrap_rotor:
        node_x_mm, node_y_mm, field_sign = wrap_rotor_coordinates(
            sample.node_x_mm, sample.node_y_mm, rigid_mask, cumulative_deg, sector_deg
        )
    else:
        node_x_mm, node_y_mm, field_sign = sample.node_x_mm, sample.node_y_mm, 1.0

    node_x_m = node_x_mm * 1e-3
    node_y_m = node_y_mm * 1e-3
    pos = np.column_stack([node_x_mm, node_y_mm]).astype(np.float64)

    # Region code per node: majority vote over incident elements, counted with a
    # single flattened bincount rather than a per-node mask.
    reg = np.asarray(mesh.reg_code, dtype=np.int64)
    idx3 = np.concatenate([i1, i2, i3])
    reg3 = np.tile(reg, 3)
    max_reg = int(reg3.max()) + 1
    counts = np.bincount(
        idx3 * max_reg + reg3, minlength=n_nodes * max_reg
    ).reshape(n_nodes, max_reg)
    node_reg = counts.argmax(axis=1).astype(np.float64)

    cond = record.condition
    geom_and_drive = [
        float(cond.get("Ratio_Bore", 0.0)),
        float(cond.get("Ratio_SlotDepth_ParallelSlot", 0.0)),
        float(cond.get("PeakCurrent", 0.0)),
        float(cond.get("PhaseAdvance", 0.0)),
    ]
    if legacy_features:
        # Pre-fix layout, kept only so checkpoints trained before the sector
        # symmetry work can still be scored as a baseline.
        scalars = [sample.time_s, sample.rotate_step] + geom_and_drive
        node_names, edge_names = MGN_NODE_FEATURES, MGN_EDGE_FEATURES
    else:
        angle_sin, angle_cos = rotor_angle_features(cumulative_deg, sector_deg)
        scalars = [sample.time_s, angle_sin, angle_cos] + geom_and_drive
        node_names, edge_names = MGN_NODE_FEATURES_V2, MGN_EDGE_FEATURES_V2

    x = np.column_stack(
        [pos, node_reg[:, None]] + [np.full((n_nodes, 1), s) for s in scalars]
    ).astype(np.float32)
    assert_input_features_clean(node_names, context="curl trainer node features")
    assert_feature_count(node_names, x.shape[1], context="curl trainer node features")

    # Curl operator on geometrically valid, non-sliding-band elements only.
    valid = mesh_validity_mask(node_x_m, node_y_m, mesh.tri, exclude=band_mask)
    if not np.any(valid):
        return None
    operator = build_p1_curl_operator(node_x_m, node_y_m, mesh.tri, valid=valid)

    # Per-element loss weight. The airgap band carries the torque — it is 6.4%
    # of the scoreable elements but sets the acceptance metric — so it can be
    # emphasised without letting it dominate. After the sliding band is
    # excluded, the only airgap layer left is the stationary `a1`, which is
    # exactly the band the Arkkio integral runs on.
    elem_weight = np.ones(operator.n_valid, dtype=np.float64)
    if airgap_weight != 1.0:
        is_airgap = np.array(
            [
                bool(AIRGAP_NAME_PATTERN.match(str(mesh.name_of_code.get(int(c), ""))))
                for c in np.asarray(mesh.reg_code)[operator.element_index]
            ]
        )
        elem_weight[is_airgap] = float(airgap_weight)

    # The wrapped configuration's field is `field_sign` times the exported one.
    b_true = field_sign * np.stack(
        [
            sample.fields["bx"][operator.element_index],
            sample.fields["by"][operator.element_index],
        ],
        axis=1,
    ).astype(np.float32)

    # Edges span every element, including the sliding band, so the rotor stays
    # connected to the stator for message passing.
    src = np.concatenate([i1, i2, i3, i2, i3, i1])
    dst = np.concatenate([i2, i3, i1, i1, i2, i3])
    edge_index = np.unique(np.stack([src, dst], axis=1), axis=0).T.astype(np.int64)
    edge_sign = np.ones(edge_index.shape[1], dtype=np.float64)

    # Anti-periodic edges close the sector: without them message passing cannot
    # cross the cut, and a tooth at theta=0 never sees its physical neighbour at
    # theta=-45 even though they are one sector apart.
    pairs = cut_plane_pairs(node_x_mm, node_y_mm, sector_deg)
    if anti_periodic and pairs.size:
        pbc_index, pbc_sign = anti_periodic_edges(pairs)
        edge_index = np.concatenate([edge_index, pbc_index], axis=1)
        edge_sign = np.concatenate([edge_sign, pbc_sign])

    dxy = pos[edge_index[1]] - pos[edge_index[0]]
    dist = np.linalg.norm(dxy, axis=1, keepdims=True)
    parts = [dxy, dist] if legacy_features else [edge_sign[:, None], dxy, dist]
    edge_attr = np.concatenate(parts, axis=1).astype(np.float32)
    assert_feature_count(edge_names, edge_attr.shape[1], context="curl trainer edge features")

    return CurlData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        pos=torch.from_numpy(pos.astype(np.float32)),
        curl_node_index=torch.from_numpy(operator.node_index.astype(np.int64)),
        # Which elements the operator produces B on, so the evaluator can place
        # the prediction back into the full element array.
        curl_element_index=torch.from_numpy(operator.element_index.astype(np.int64)),
        curl_coef_bx=torch.from_numpy(operator.coef_bx.astype(np.float32)),
        curl_coef_by=torch.from_numpy(operator.coef_by.astype(np.float32)),
        b_true=torch.from_numpy(b_true),
        elem_weight=torch.from_numpy(elem_weight.astype(np.float32)),
        a_export=torch.from_numpy(
            (field_sign * _element_to_nodal(sample.fields["a"], mesh.tri, n_nodes)).astype(np.float32)
        ).unsqueeze(1),
        cut_pairs=torch.from_numpy(pairs.astype(np.int64)) if pairs.size
        else torch.zeros((0, 2), dtype=torch.long),
        n_elements_total=int(mesh.n_elements),
        num_nodes=n_nodes,
    )


def _element_to_nodal(values: np.ndarray, tri, n_nodes: int) -> np.ndarray:
    i1, i2, i3 = tri
    idx = np.concatenate([i1, i2, i3])
    total = np.bincount(idx, weights=np.tile(np.asarray(values, np.float64), 3), minlength=n_nodes)
    count = np.bincount(idx, minlength=n_nodes).astype(np.float64)
    return total / np.clip(count, 1.0, None)


def curl_from_batch(batch, a_nodal: torch.Tensor) -> torch.Tensor:
    """Apply the batched P1 curl operator to nodal A. Differentiable in `a_nodal`."""
    a = a_nodal.view(-1)
    gathered = a[batch.curl_node_index]                      # (K_total, 3)
    bx = (batch.curl_coef_bx * gathered).sum(dim=1)
    by = (batch.curl_coef_by * gathered).sum(dim=1)
    return torch.stack([bx, by], dim=1)


def average_nodes_to_elements(batch, b_nodal: torch.Tensor) -> torch.Tensor:
    """Average a node-wise vector field onto the operator's elements.

    This is the node-support baseline's path to element support — the round trip
    whose cost the curl route avoids. It reuses `curl_node_index`, so both
    targets are compared on exactly the same element subset.
    """
    return b_nodal[batch.curl_node_index].mean(dim=1)         # (K_total, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    parser.add_argument("--split", type=str, default="eval/splits/doe40_case_split.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--processor-size", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--step-stride", type=int, default=3,
                        help="Keep every Nth timestep; samples the whole rotor sweep evenly")
    parser.add_argument("--max-steps-per-case", type=int, default=None)
    parser.add_argument("--w-b", type=float, default=1.0, help="Weight on MSE(curl(A), B_fem)")
    parser.add_argument("--w-gauge", type=float, default=1e-3,
                        help="Weight pinning the additive gauge of A")
    parser.add_argument("--w-a", type=float, default=0.0,
                        help="Weight on MSE(A, exported A). Default 0: the exported "
                             "potential is element-averaged and fitting it caps the "
                             "model at a 38.7%% |B| nRMSE floor")
    parser.add_argument("--airgap-weight", type=float, default=1.0,
                        help="Loss weight on airgap-band elements. They are 6.4%% of the "
                             "scoreable elements but determine the torque; weight 5 lifts "
                             "their share of the loss to ~26%%. 1.0 disables the weighting")
    parser.add_argument("--w-pbc", type=float, default=1e-2,
                        help="Weight on the anti-periodicity penalty "
                             "MSE(A(theta=0) + A(theta=-sector)) at the cut planes")
    parser.add_argument("--sector-deg", type=float, default=DEFAULT_SECTOR_DEG,
                        help="Modelled sector span; 45 deg for the 1/8 DOE model")
    parser.add_argument("--no-wrap-rotor", action="store_true",
                        help="Keep the exported (unwrapped) rotor coordinates. The "
                             "export reports the rotor out to -153 deg, which is not "
                             "the domain that was solved")
    parser.add_argument("--no-anti-periodic-edges", action="store_true",
                        help="Leave the two cut planes unconnected in the graph")
    parser.add_argument("--target", choices=("A", "B"), default="A",
                        help="'A': predict nodal A, derive element B by the P1 curl. "
                             "'B': predict nodal Bx,By and average onto elements — the "
                             "node-support baseline, run through this same script so "
                             "features, split and scoring differ in nothing but the "
                             "output representation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="results/mgn_curl_caseholdout.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    print(f"Manifest: {manifest['n_cases']} cases")

    report = load_doe_cases(
        manifest,
        args.data_dir,
        source_types=SOURCE_TYPES_FOR_TRAINING,
        max_steps=None,
    )
    if report.skipped:
        print(f"  [WARN] {len(report.skipped)} file(s) skipped; first: {report.skipped[0]}")

    stride = max(1, int(args.step_stride))
    graphs: List[CurlData] = []
    graph_case: List[int] = []
    n_wrapped = 0
    t_load = time.time()
    for record in report.records:
        mesh = record.mesh
        band = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)
        rigid = rigid_rotor_node_mask(
            mesh.tri, mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes, mesh.n_nodes
        )
        # rotate_step is the per-step increment, so the absolute rotor angle has
        # to be accumulated over the sweep.
        cum_deg = cumulative_rotor_angle([s.rotate_step for s in record.samples])

        chosen = list(range(0, len(record.samples), stride))
        if args.max_steps_per_case:
            chosen = chosen[: int(args.max_steps_per_case)]
        for si in chosen:
            g = build_curl_graph(
                record,
                record.samples[si],
                band,
                rigid,
                cumulative_deg=float(cum_deg[si]),
                sector_deg=args.sector_deg,
                wrap_rotor=not args.no_wrap_rotor,
                anti_periodic=not args.no_anti_periodic_edges,
                airgap_weight=args.airgap_weight,
            )
            if g is not None:
                graphs.append(g)
                graph_case.append(record.case_index)
                if abs(float(cum_deg[si])) >= args.sector_deg:
                    n_wrapped += 1
    print(f"Total graphs: {len(graphs)}  (built in {time.time() - t_load:.0f}s)")
    if len(graphs) < 2:
        print("ERROR: need at least 2 graphs. Exiting.")
        return 1

    # How much of the mesh the curl loss actually covers. Reported rather than
    # left implicit: the excluded fraction is the sliding band plus whatever the
    # per-step geometry inverted, and a sudden jump here means the export changed.
    covered = np.array([int(g.b_true.shape[0]) for g in graphs], dtype=np.float64)
    total = np.array([int(g.n_elements_total) for g in graphs], dtype=np.float64)
    frac = covered / np.clip(total, 1.0, None)
    print(f"Curl loss covers {100 * frac.mean():.1f}% of elements "
          f"(min {100 * frac.min():.1f}%, max {100 * frac.max():.1f}%)")
    if args.airgap_weight != 1.0:
        w0 = graphs[0].elem_weight.numpy()
        n_air = int((w0 > 1.0).sum())
        share = float(w0[w0 > 1.0].sum() / w0.sum()) if n_air else 0.0
        print(f"Airgap weighting: {args.airgap_weight:g}x on {n_air}/{w0.size} elements "
              f"({100 * n_air / w0.size:.1f}% of elements -> {100 * share:.1f}% of the loss)")

    n_pbc = int(graphs[0].cut_pairs.shape[0])
    print(f"Sector symmetry: wrap_rotor={not args.no_wrap_rotor} "
          f"({n_wrapped} of {len(graphs)} samples past one sector), "
          f"anti_periodic_edges={not args.no_anti_periodic_edges} ({n_pbc} cut-plane pairs)")

    split = resolve_case_split(manifest, sorted(set(graph_case)), Path(args.split), seed=args.seed)
    assigned = assign_records(graph_case, split)
    train_idx, val_idx = assigned["train"], assigned["val"]
    if train_idx.size == 0 or val_idx.size == 0:
        print("ERROR: case-level split produced an empty train or val subset.")
        return 1

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    train_cases = {graph_case[i] for i in train_idx}
    val_cases = {graph_case[i] for i in val_idx}
    assert not (train_cases & val_cases), "case-level holdout violated"

    print(f"Split manifest: {args.split}")
    print(f"Train: {len(train_graphs)} graphs / {len(train_cases)} cases, "
          f"Val: {len(val_graphs)} graphs / {len(val_cases)} cases "
          f"(test {len(split.test)} cases held back)")

    # ---- Normalization from train statistics only ----
    x_cat = torch.cat([g.x for g in train_graphs], dim=0)
    e_cat = torch.cat([g.edge_attr for g in train_graphs], dim=0)
    b_cat = torch.cat([g.b_true for g in train_graphs], dim=0)
    a_cat = torch.cat([g.a_export for g in train_graphs], dim=0)

    x_mean, x_std = x_cat.mean(0, keepdim=True), x_cat.std(0, keepdim=True).clamp_min(1e-6)
    e_mean, e_std = e_cat.mean(0, keepdim=True), e_cat.std(0, keepdim=True).clamp_min(1e-6)
    b_std = b_cat.std().clamp_min(1e-9)
    # Output scale for A: the network emits an order-1 value, this maps it into
    # the physical range so that curl(A) starts near the right magnitude.
    a_scale = a_cat.std().clamp_min(1e-9)

    for g in train_graphs + val_graphs:
        g.x = torch.nan_to_num((g.x - x_mean) / x_std)
        g.edge_attr = torch.nan_to_num((g.edge_attr - e_mean) / e_std)
        g.b_true = torch.nan_to_num(g.b_true)

    target_desc = "1 (nodal A -> curl)" if args.target == "A" else "2 (nodal Bx,By -> average)"
    print(f"Node features: {train_graphs[0].x.shape[1]}, "
          f"Edge features: {train_graphs[0].edge_attr.shape[1]}, Output: {target_desc}")
    print(f"b_std = {float(b_std):.4f} T, a_scale = {float(a_scale):.4e} Wb/m")
    print(f"Curl elements per graph: {train_graphs[0].b_true.shape[0]} "
          f"(sliding band and inverted elements excluded)")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    from physicsnemo.models.meshgraphnet import MeshGraphNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    predicts_a = args.target == "A"
    out_dim = 1 if predicts_a else 2
    # A is scaled into the potential range so its curl starts near the right
    # magnitude; B is scaled directly by the field spread.
    out_scale = a_scale if predicts_a else b_std

    model = MeshGraphNet(
        input_dim_nodes=train_graphs[0].x.shape[1],
        input_dim_edges=train_graphs[0].edge_attr.shape[1],
        output_dim=out_dim,
        processor_size=args.processor_size,
        hidden_dim_processor=args.hidden_dim,
        hidden_dim_node_encoder=args.hidden_dim,
        hidden_dim_edge_encoder=args.hidden_dim,
        hidden_dim_node_decoder=args.hidden_dim,
        aggregation="sum",
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    b_std_d = b_std.to(device)
    a_scale_d = a_scale.to(device)
    out_scale_d = out_scale.to(device)

    def run_epoch(loader, training: bool):
        model.train() if training else model.eval()
        totals = {"loss": 0.0, "b": 0.0, "nrmse": 0.0, "pbc": 0.0}
        n_batch = 0
        for batch in loader:
            batch = batch.to(device)
            with torch.set_grad_enabled(training):
                raw = model(batch.x, batch.edge_attr, batch)             # (N, out_dim)
                if predicts_a:
                    b_pred = curl_from_batch(batch, raw * out_scale_d)   # (K, 2) tesla
                else:
                    b_pred = average_nodes_to_elements(batch, raw * out_scale_d)

                # Loss in normalized B units so it is scale-free.
                # Weighted MSE, normalized by the weight sum so the effective
                # step size does not change with the weighting — a plain
                # sum-of-weighted-errors would silently raise the learning rate.
                sq = ((b_pred - batch.b_true) / b_std_d).pow(2)          # (K, 2)
                w = batch.elem_weight.unsqueeze(1)
                b_loss = (w * sq).sum() / (w.sum() * sq.shape[1])
                loss = args.w_b * b_loss

                # The gauge and anti-periodicity terms are properties of the
                # scalar potential. Bx is not a potential — the anti-periodic
                # relation for a vector field carries a rotation as well as the
                # sign — so they are applied only when A is the output.
                pbc = raw.new_zeros(())
                if predicts_a:
                    loss = loss + args.w_gauge * raw.mean().pow(2)
                    if args.w_pbc > 0.0 and batch.cut_pairs.numel():
                        flat = raw.view(-1)
                        pbc = (flat[batch.cut_pairs[:, 0]] + flat[batch.cut_pairs[:, 1]]).pow(2).mean()
                        loss = loss + args.w_pbc * pbc
                if args.w_a > 0.0:
                    # Normalized by a_scale^2 so w_a is comparable to w_b.
                    a_loss = torch.nn.functional.mse_loss(raw * out_scale_d, batch.a_export) / a_scale_d**2
                    loss = loss + args.w_a * a_loss

                if not torch.isfinite(loss):
                    continue
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            with torch.no_grad():
                # Physical-unit |B| nRMSE, the quantity the benchmark reports.
                err = (b_pred - batch.b_true).pow(2).sum(1).mean().sqrt()
                ref = batch.b_true.pow(2).sum(1).mean().sqrt().clamp_min(1e-12)
                totals["nrmse"] += float(100.0 * err / ref)
            totals["loss"] += float(loss)
            totals["b"] += float(b_loss)
            totals["pbc"] += float(pbc)
            n_batch += 1
        return {k: v / max(n_batch, 1) for k, v in totals.items()}

    best_val = float("inf")
    train_hist, val_hist = [], []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        scheduler.step()
        train_hist.append(tr)
        val_hist.append(va)

        if va["b"] < best_val:
            best_val = va["b"]
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "x_mean": x_mean, "x_std": x_std,
                    "e_mean": e_mean, "e_std": e_std,
                    "b_std": b_std, "a_scale": a_scale, "out_scale": out_scale,
                    "train_hist": train_hist, "val_hist": val_hist,
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "output_kind": (
                        "nodal_A_curl_to_element_B" if predicts_a
                        else "nodal_B_average_to_element_B"
                    ),
                    "node_features": list(MGN_NODE_FEATURES_V2),
                    "edge_features": list(MGN_EDGE_FEATURES_V2),
                    "sector_symmetry": {
                        "sector_deg": args.sector_deg,
                        "wrap_rotor": not args.no_wrap_rotor,
                        "anti_periodic_edges": not args.no_anti_periodic_edges,
                        "w_pbc": args.w_pbc,
                    },
                    "airgap_weight": args.airgap_weight,
                    "split": {
                        "granularity": "case",
                        "manifest": str(args.split),
                        "seed": split.seed,
                        "doe_digest": split.digest,
                        "train_cases": list(split.train),
                        "val_cases": list(split.val),
                        "test_cases": list(split.test),
                    },
                },
                args.ckpt,
            )

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f"  ep {ep:04d}/{args.epochs} | train B {tr['b']:.6f} nRMSE {tr['nrmse']:6.2f}% "
                  f"| val B {va['b']:.6f} nRMSE {va['nrmse']:6.2f}% pbc {va['pbc']:.2e} "
                  f"| best {best_val:.6f} | {time.time() - t0:.0f}s")

    print(f"\nTraining complete: {args.epochs} epochs in {time.time() - t0:.1f}s")
    print(f"Best val B MSE (normalized): {best_val:.6f}")
    print(f"Checkpoint: {args.ckpt}")
    print("Score it with:  python -m eval.benchmark --curl-ckpt " + str(args.ckpt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
