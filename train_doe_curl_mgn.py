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
import os
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
    MGN_NODE_FEATURES_V3,
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
        if key in ("curl_node_index", "cut_pairs", "world_edge_index"):
            # world_edge_index holds node ids like edge_index — shift by num_nodes.
            return self.num_nodes
        if key == "band_row_index":
            # Positions into this graph's OPERATOR ROWS (b_true), not its nodes.
            # PyG's default would shift any '*index*' key by num_nodes, silently
            # pointing every batched band at the first graph's elements.
            return self.b_true.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("curl_node_index", "cut_pairs", "band_row_index"):
            return 0
        if key == "world_edge_index":
            return 1              # (2, E_world): concatenate along the edge axis
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
    node_feature_names: Optional[Tuple[str, ...]] = None,
    edge_feature_names: Optional[Tuple[str, ...]] = None,
    airgap_weight: float = 1.0,
    world_edges: bool = False,
    prior_features: bool = False,
) -> Optional[CurlData]:
    """Build one training graph: clean node features + a curl operator + element B.

    Node features are the 9 in `MGN_NODE_FEATURES_V2` — position, region code,
    the operating point, the DOE geometry parameters and the cumulative rotor
    angle. No solved quantity and no shortcut feature is used as an input;
    `assert_input_features_clean` enforces both here. `time_s` used to be in this
    list and was removed: it is proportional to rotor angle in this DOE, and the
    model was using it as a shortcut (methodology review section 9).

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
    angle_sin, angle_cos = rotor_angle_features(cumulative_deg, sector_deg)

    # Every scalar this builder can supply, keyed by its declared feature name.
    # Selecting by name rather than by a layout flag is what lets a checkpoint
    # trained on an older feature set still be rebuilt exactly as it was trained:
    # dropping `time_s` from the current layout silently broke every existing
    # checkpoint until this became name-driven.
    available = {
        "time_s": sample.time_s,
        "rotate_step": sample.rotate_step,
        "rotor_angle_sin": angle_sin,
        "rotor_angle_cos": angle_cos,
        "Ratio_Bore": float(cond.get("Ratio_Bore", 0.0)),
        "Ratio_SlotDepth_ParallelSlot": float(cond.get("Ratio_SlotDepth_ParallelSlot", 0.0)),
        "PeakCurrent": float(cond.get("PeakCurrent", 0.0)),
        "PhaseAdvance": float(cond.get("PhaseAdvance", 0.0)),
    }
    if node_feature_names is not None:
        node_names = tuple(node_feature_names)
        edge_names = tuple(edge_feature_names) if edge_feature_names else MGN_EDGE_FEATURES_V2
    elif legacy_features:
        node_names, edge_names = MGN_NODE_FEATURES, MGN_EDGE_FEATURES
    elif prior_features:
        node_names, edge_names = MGN_NODE_FEATURES_V3, MGN_EDGE_FEATURES_V2
    else:
        node_names, edge_names = MGN_NODE_FEATURES_V2, MGN_EDGE_FEATURES_V2

    # pos_x, pos_y and region_code are positional; prior_* features are per-node
    # arrays appended last; everything else is a scalar broadcast to every node,
    # looked up by name.
    scalar_names = [n for n in node_names
                    if n not in ("pos_x", "pos_y", "region_code")
                    and not n.startswith("prior_")]
    prior_names = [n for n in node_names if n.startswith("prior_")]
    missing = [n for n in scalar_names if n not in available]
    if missing:
        raise ValueError(f"build_curl_graph cannot supply node feature(s) {missing}")
    scalars = [available[n] for n in scalar_names]

    cols = [pos, node_reg[:, None]] + [np.full((n_nodes, 1), s) for s in scalars]
    if prior_names:
        # Column order is positional + scalars + priors; a layout that interleaves
        # them cannot be assembled and must fail loudly rather than silently
        # permute columns under a checkpoint's declared names.
        expected = ("pos_x", "pos_y", "region_code") + tuple(scalar_names) + tuple(prior_names)
        if tuple(node_names) != expected:
            raise ValueError(
                f"prior features must come last: declared {tuple(node_names)}, "
                f"buildable order is {expected}")
        if wrap_rotor:
            # The prior is computed in the export (unwrapped) frame; feeding it to
            # a wrapped graph would recreate the section-18 train/eval asymmetry.
            raise ValueError("prior features require wrap_rotor=False")
        from fem_warmstart.prior import nodal_prior_features

        prior = nodal_prior_features(record, sample)
        cols += [np.asarray(prior[n], dtype=np.float64)[:, None] for n in prior_names]
    x = np.column_stack(cols).astype(np.float32)
    # Shortcut features are forbidden when *building* a model and tolerated when
    # *replaying* one: an explicit layout means we are reconstructing exactly what
    # some existing checkpoint was trained on, which is scoring, not training.
    replaying = legacy_features or node_feature_names is not None
    assert_input_features_clean(node_names, context="curl trainer node features",
                                allow_shortcuts=replaying)
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

    # Long-range "world" edges for the hybrid backbone (R3 fallback): connect
    # airgap-band nodes at slot-pitch angular stride so the band couples across a
    # slot in one hop instead of ~65 message-passing hops. Same 4-dim edge layout
    # as the mesh edges (edge_sign=+1; these do not cross the anti-periodic cut).
    world_edge_index, world_edge_attr = _build_world_edges(
        mesh, node_x_mm, node_y_mm, pos, sector_deg, edge_names, legacy_features
    ) if world_edges else (
        np.zeros((2, 0), dtype=np.int64),
        np.zeros((0, len(edge_names)), dtype=np.float32),
    )

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
        world_edge_index=torch.from_numpy(world_edge_index),
        world_edge_attr=torch.from_numpy(world_edge_attr),
        n_elements_total=int(mesh.n_elements),
        num_nodes=n_nodes,
    )


def _build_world_edges(mesh, node_x_mm, node_y_mm, pos, sector_deg, edge_names,
                       legacy_features, slot_mults=(1, 2)):
    """Long-range angular skip edges over the airgap-band nodes.

    The Arkkio band (where torque and the residual |B| error live) is an arc of
    ~`sector_deg` spanning ~6 slots for the 1/8 DOE sector. We sort its nodes by
    angle and connect each to the node ~k slot-pitches away (k in `slot_mults`),
    within the arc (no wrap across the anti-periodic cut). Returns
    ``(edge_index (2,E), edge_attr (E, D_edge))`` matching the mesh edge layout;
    empty when the mesh has no usable band.
    """
    from eval.torque import AirgapBandError, build_airgap_band

    empty = (np.zeros((2, 0), dtype=np.int64),
             np.zeros((0, len(edge_names)), dtype=np.float32))
    try:
        band = build_airgap_band(
            mesh.node_x_mm, mesh.node_y_mm, mesh.tri, mesh.reg_code,
            mesh.name_of_code, moving_reg_codes=mesh.moving_reg_codes,
        )
    except AirgapBandError:
        return empty

    i1, i2, i3 = mesh.tri
    el = band.element_index
    bnodes = np.unique(np.concatenate([i1[el], i2[el], i3[el]]))
    if bnodes.size < 12:
        return empty

    theta = np.arctan2(node_y_mm[bnodes], node_x_mm[bnodes])
    ring = bnodes[np.argsort(theta)]                    # band nodes in angular order
    n_slots_per_sector = 6                              # 48 slots / 8 sectors
    step = max(1, int(round(ring.size / n_slots_per_sector)))

    src_list, dst_list = [], []
    for k in slot_mults:
        s = step * k
        if s >= ring.size:
            continue
        src_list.append(ring[:-s]); dst_list.append(ring[s:])   # within-arc, no wrap
    if not src_list:
        return empty
    a = np.concatenate(src_list); b = np.concatenate(dst_list)
    w_src = np.concatenate([a, b]); w_dst = np.concatenate([b, a])   # undirected
    world_ei = np.stack([w_src, w_dst], axis=0).astype(np.int64)

    wdxy = pos[world_ei[1]] - pos[world_ei[0]]
    wdist = np.linalg.norm(wdxy, axis=1, keepdims=True)
    wsign = np.ones((world_ei.shape[1], 1))
    parts = [wdxy, wdist] if legacy_features else [wsign, wdxy, wdist]
    world_attr = np.concatenate(parts, axis=1).astype(np.float32)
    return world_ei, world_attr


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


# Spatial orders admissible on the anti-periodic 1/8 sector: odd multiples of 4
# cycles per revolution (4 = fundamental of the 8-pole machine, 12/20/28 = MMF
# harmonics, 44/52 and 92/100 and 140/148 = slotting sidebands of 48/96/144).
# The band projection study (methodology review §14b) showed the model's torque
# error lives ENTIRELY in the coefficients of these orders — filtering removed
# nothing — so this loss supervises those coefficients directly instead of
# letting them be diluted across ~13k per-element errors.
BAND_SPECTRAL_ORDERS: Tuple[int, ...] = tuple(4 * (2 * m + 1) for m in range(24))


def band_spectral_operator(mesh) -> Optional[Dict[str, np.ndarray]]:
    """Coefficient-extraction operator for the stationary Arkkio band.

    Returns the band element ids, the radial unit vectors and the extraction
    matrix ``ext`` with ``ext.T @ values = coefficients`` — i.e. rows of
    ``pinv(design)`` transposed so batching can concatenate along elements.
    ``None`` when the mesh has no usable band (the graph then trains without
    the spectral term rather than failing).
    """
    from eval.torque import AirgapBandError, build_airgap_band

    try:
        band = build_airgap_band(
            mesh.node_x_mm, mesh.node_y_mm, mesh.tri, mesh.reg_code,
            mesh.name_of_code, moving_reg_codes=mesh.moving_reg_codes,
        )
    except AirgapBandError:
        return None
    phi = np.arctan2(band.sin_theta, band.cos_theta)
    cols = []
    for k in BAND_SPECTRAL_ORDERS:
        cols.append(np.cos(k * phi))
        cols.append(np.sin(k * phi))
    design = np.column_stack(cols)
    return {
        "element_index": band.element_index.astype(np.int64),
        "cos": band.cos_theta.astype(np.float32),
        "sin": band.sin_theta.astype(np.float32),
        "ext": np.linalg.pinv(design).T.astype(np.float32),   # (n_band, 2*orders)
    }


def band_coefficients(batch, values: torch.Tensor) -> torch.Tensor:
    """Per-graph harmonic coefficients of a band scalar (segmented matmul).

    ``values`` holds one scalar per batched band element; the return is
    ``(num_graphs, 2*orders)`` with ``c[g, j] = sum_i ext[i, j] * values[i]``
    over graph *g*'s band rows. Differentiable in ``values``.
    """
    gid = torch.repeat_interleave(
        torch.arange(batch.num_graphs, device=values.device), batch.band_count
    )
    out = values.new_zeros((batch.num_graphs, batch.band_ext.shape[1]))
    out.index_add_(0, gid, batch.band_ext * values.unsqueeze(1))
    return out


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
    parser.add_argument("--model", choices=("mgn", "transolver", "hybrid"), default="mgn",
                        help="Field backbone. 'mgn': physicsnemo MeshGraphNet "
                             "(local message passing). 'transolver': R3 physics-"
                             "attention graph transformer (global receptive field; "
                             "the --processor-size flag is ignored, --n-layers/"
                             "--n-head/--slice-num apply). 'hybrid': MGN + long-range "
                             "airgap-band 'world' edges (keeps mesh MP, adds slot-"
                             "pitch coupling; --hidden-dim/--processor-size apply, "
                             "h208/p15 ~9.5M matches h256 MGN). Use --batch-size 1 "
                             "with transolver (global attention must not cross graphs)")
    parser.add_argument("--n-layers", type=int, default=12,
                        help="transolver: number of Transolver blocks (~9M params "
                             "at hidden 256 / 12 layers, matching h256 MGN)")
    parser.add_argument("--n-head", type=int, default=8,
                        help="transolver: attention heads")
    parser.add_argument("--slice-num", type=int, default=32,
                        help="transolver: physics-attention slices (soft tokens); "
                             "attention is O(N*slice_num)")
    parser.add_argument("--emb-dim", type=int, default=64,
                        help="transolver: positional-embedding width from (pos_x,pos_y)")
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
    parser.add_argument("--band-spectral-weight", type=float, default=0.0,
                        help="Weight on the band harmonic-coefficient loss. The projection "
                             "study (methodology review §14b) measured that the torque "
                             "error is entirely in-band — the model gets the admissible "
                             "harmonic coefficients themselves wrong — so this supervises "
                             "those coefficients directly (Br/Btheta of the Arkkio band "
                             "projected on the anti-periodic orders). 0 disables (default; "
                             "the winning recipes were trained without it)")
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
    parser.add_argument("--prior-features", action="store_true",
                        help="R7-A: append the linear-solve physics prior "
                             "(prior_a, prior_bx, prior_by from fem_warmstart/prior.py) "
                             "to the node features — layout MGN_NODE_FEATURES_V3. "
                             "Requires --no-wrap-rotor; priors are cached under "
                             "results/prior_cache/ (~1 s/graph to build cold)")
    parser.add_argument("--target", choices=("A", "B"), default="A",
                        help="'A': predict nodal A, derive element B by the P1 curl. "
                             "'B': predict nodal Bx,By and average onto elements — the "
                             "node-support baseline, run through this same script so "
                             "features, split and scoring differ in nothing but the "
                             "output representation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="results/mgn_curl_caseholdout.pt")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume from a checkpoint written by this script: restores "
                             "weights, optimizer moments, LR schedule position and the "
                             "loss history, and continues at its epoch+1. Run-defining "
                             "flags must match the original run or this aborts.")
    parser.add_argument("--ckpt-every", type=int, default=5,
                        help="also write <ckpt>.last every N epochs regardless of val "
                             "improvement, so a host restart costs at most N epochs "
                             "(the best-val checkpoint alone can be many epochs stale)")
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
    n_band_missing = 0
    for record in report.records:
        mesh = record.mesh
        band = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)
        rigid = rigid_rotor_node_mask(
            mesh.tri, mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes, mesh.n_nodes
        )
        # The spectral operator is a property of the mesh (the Arkkio band is
        # stationary), so it is built once per case and shared by every step.
        spec = band_spectral_operator(mesh) if args.band_spectral_weight > 0.0 else None
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
                world_edges=(args.model == "hybrid"),
                prior_features=args.prior_features,
            )
            if g is not None:
                if args.band_spectral_weight > 0.0:
                    if spec is None:
                        n_band_missing += 1
                        g.band_row_index = torch.zeros(0, dtype=torch.long)
                        g.band_cos = torch.zeros(0)
                        g.band_sin = torch.zeros(0)
                        g.band_ext = torch.zeros((0, 2 * len(BAND_SPECTRAL_ORDERS)))
                        g.band_count = torch.zeros(1, dtype=torch.long)
                    else:
                        # Map full-array band element ids onto this step's
                        # operator rows. The a1 band is stationary, so normally
                        # every band element is covered; if a step drops some,
                        # the extraction matrix is recomputed on the survivors
                        # so it stays an exact pseudo-inverse.
                        lut = np.full(int(g.n_elements_total), -1, dtype=np.int64)
                        lut[g.curl_element_index.numpy()] = np.arange(
                            g.curl_element_index.shape[0], dtype=np.int64)
                        rows = lut[spec["element_index"]]
                        keep = rows >= 0
                        if keep.all():
                            ext = spec["ext"]
                        else:
                            phi = np.arctan2(spec["sin"][keep], spec["cos"][keep])
                            cols = []
                            for k in BAND_SPECTRAL_ORDERS:
                                cols.append(np.cos(k * phi))
                                cols.append(np.sin(k * phi))
                            ext = np.linalg.pinv(np.column_stack(cols)).T.astype(np.float32)
                        g.band_row_index = torch.from_numpy(rows[keep])
                        g.band_cos = torch.from_numpy(spec["cos"][keep])
                        g.band_sin = torch.from_numpy(spec["sin"][keep])
                        g.band_ext = torch.from_numpy(ext)
                        g.band_count = torch.tensor([int(keep.sum())], dtype=torch.long)
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

    if args.band_spectral_weight > 0.0:
        nb = int(graphs[0].band_count)
        print(f"Band spectral loss: weight {args.band_spectral_weight:g}, "
              f"{nb} band elements, {len(BAND_SPECTRAL_ORDERS)} orders "
              f"-> {4 * len(BAND_SPECTRAL_ORDERS)} coefficients (Br+Btheta, cos+sin)"
              + (f"  [WARN: {n_band_missing} graphs without a band]" if n_band_missing else ""))

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
        # World edges (hybrid) share the mesh edge stats — normalize them the same
        # way, or the model would train on raw-scale world features and be scored
        # on normalized ones (a silent train/eval mismatch that wrecks the band).
        if getattr(g, "world_edge_attr", None) is not None and g.world_edge_attr.numel():
            g.world_edge_attr = torch.nan_to_num((g.world_edge_attr - e_mean) / e_std)
        g.b_true = torch.nan_to_num(g.b_true)

    target_desc = "1 (nodal A -> curl)" if args.target == "A" else "2 (nodal Bx,By -> average)"
    print(f"Node features: {train_graphs[0].x.shape[1]}, "
          f"Edge features: {train_graphs[0].edge_attr.shape[1]}, Output: {target_desc}")
    print(f"b_std = {float(b_std):.4f} T, a_scale = {float(a_scale):.4e} Wb/m")
    print(f"Curl elements per graph: {train_graphs[0].b_true.shape[0]} "
          f"(sliding band and inverted elements excluded)")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    from r3_models import build_curl_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    predicts_a = args.target == "A"
    out_dim = 1 if predicts_a else 2
    # A is scaled into the potential range so its curl starts near the right
    # magnitude; B is scaled directly by the field spread.
    out_scale = a_scale if predicts_a else b_std

    # Construction goes through the shared factory so the trainer and the eval
    # harness build the identical module (the "mgn" branch reproduces the exact
    # kwargs used before the factory existed; the gate re-score guards it).
    model = build_curl_model(
        args.model,
        in_node=train_graphs[0].x.shape[1],
        in_edge=train_graphs[0].edge_attr.shape[1],
        out_dim=out_dim,
        hp=vars(args),
        device=device,
    )
    print(f"Model[{args.model}] params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- resume -------------------------------------------------------------
    # A host restart mid-run (2026-08-06: R7-A died 2 min short of epoch 50) used to
    # mean a full retrain, because the state was on disk but the code path to reload it
    # was not. Restores weights + optimizer moments + LR position + loss history.
    start_epoch, best_val = 1, float("inf")
    resume_hist = ([], [])
    if args.resume:
        rck = torch.load(args.resume, map_location=device, weights_only=False)
        prev = rck.get("args", {})
        # Flags that change what is being learned. A mismatch means the resumed run
        # would not be the run it claims to continue, so fail loudly rather than
        # produce a checkpoint whose provenance is a blend of two configurations.
        run_defining = ("data_dir", "split", "target", "model", "hidden_dim",
                        "processor_size", "prior_features", "band_spectral_weight",
                        "airgap_weight", "no_wrap_rotor", "no_anti_periodic_edges",
                        "w_pbc", "lr", "weight_decay", "batch_size", "step_stride",
                        "epochs", "seed")
        drift = {k: (prev.get(k), getattr(args, k)) for k in run_defining
                 if k in prev and str(prev.get(k)) != str(getattr(args, k, None))}
        if drift:
            for k, (was, now) in drift.items():
                print(f"  RESUME MISMATCH  {k}: checkpoint={was!r}  now={now!r}")
            raise SystemExit("--resume refused: run-defining flags differ from the checkpoint")

        model.load_state_dict(rck["model_state_dict"])
        optimizer.load_state_dict(rck["optimizer_state_dict"])
        resume_hist = (list(rck.get("train_hist", [])), list(rck.get("val_hist", [])))
        start_epoch = int(rck["epoch"]) + 1
        best_val = min((float(v["b"]) for v in resume_hist[1]), default=float("inf"))

        # Replay the LR schedule instead of trusting the restored param-group lr: the
        # cosine is defined from the initial lr, and stepping it start_epoch-1 times
        # from a fresh scheduler reproduces the original trajectory exactly.
        for g in optimizer.param_groups:
            g["lr"] = args.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
        for _ in range(start_epoch - 1):
            scheduler.step()

        if start_epoch > args.epochs:
            raise SystemExit(f"--resume: checkpoint is already at epoch {rck['epoch']} "
                             f"of {args.epochs} -- nothing to do")
        print(f"Resumed {args.resume} at epoch {rck['epoch']} -> continuing "
              f"{start_epoch}..{args.epochs} | best val B {best_val:.6f} | "
              f"lr {optimizer.param_groups[0]['lr']:.3e} | "
              f"hist {len(resume_hist[0])} train / {len(resume_hist[1])} val")

    b_std_d = b_std.to(device)
    a_scale_d = a_scale.to(device)
    out_scale_d = out_scale.to(device)

    def run_epoch(loader, training: bool):
        model.train() if training else model.eval()
        totals = {"loss": 0.0, "b": 0.0, "nrmse": 0.0, "pbc": 0.0, "spec": 0.0}
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

                spec_loss = raw.new_zeros(())
                if args.band_spectral_weight > 0.0 and int(batch.band_count.sum()) > 0:
                    # Harmonic coefficients of the band field, prediction vs FEM,
                    # in normalized units so the weight is comparable to w_b.
                    # §14b measured the torque error to live exactly here.
                    pn = b_pred[batch.band_row_index] / b_std_d          # (M, 2)
                    tn = batch.b_true[batch.band_row_index] / b_std_d
                    br_p = pn[:, 0] * batch.band_cos + pn[:, 1] * batch.band_sin
                    bt_p = -pn[:, 0] * batch.band_sin + pn[:, 1] * batch.band_cos
                    br_t = tn[:, 0] * batch.band_cos + tn[:, 1] * batch.band_sin
                    bt_t = -tn[:, 0] * batch.band_sin + tn[:, 1] * batch.band_cos
                    c_pred = torch.cat(
                        [band_coefficients(batch, br_p), band_coefficients(batch, bt_p)], dim=1)
                    c_true = torch.cat(
                        [band_coefficients(batch, br_t), band_coefficients(batch, bt_t)], dim=1)
                    spec_loss = torch.nn.functional.mse_loss(c_pred, c_true)
                    loss = loss + args.band_spectral_weight * spec_loss

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
            totals["spec"] += float(spec_loss)
            n_batch += 1
        return {k: v / max(n_batch, 1) for k, v in totals.items()}

    train_hist, val_hist = resume_hist
    t0 = time.time()

    def checkpoint_payload(ep: int) -> dict:
        return (
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "x_mean": x_mean, "x_std": x_std,
                    "e_mean": e_mean, "e_std": e_std,
                    "b_std": b_std, "a_scale": a_scale, "out_scale": out_scale,
                    "train_hist": train_hist, "val_hist": val_hist,
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "model_arch": args.model,
                    "output_kind": (
                        "nodal_A_curl_to_element_B" if predicts_a
                        else "nodal_B_average_to_element_B"
                    ),
                    "node_features": list(MGN_NODE_FEATURES_V3 if args.prior_features
                                          else MGN_NODE_FEATURES_V2),
                    "edge_features": list(MGN_EDGE_FEATURES_V2),
                    "sector_symmetry": {
                        "sector_deg": args.sector_deg,
                        "wrap_rotor": not args.no_wrap_rotor,
                        "anti_periodic_edges": not args.no_anti_periodic_edges,
                        "w_pbc": args.w_pbc,
                    },
                    "airgap_weight": args.airgap_weight,
                    "band_spectral": {
                        "weight": args.band_spectral_weight,
                        "orders": list(BAND_SPECTRAL_ORDERS),
                    },
                    "split": {
                        "granularity": "case",
                        "manifest": str(args.split),
                        "seed": split.seed,
                        "doe_digest": split.digest,
                        "train_cases": list(split.train),
                        "val_cases": list(split.val),
                        "test_cases": list(split.test),
                    },
                }
        )

    last_ckpt = str(args.ckpt) + ".last"

    def save_atomic(payload: dict, path: str) -> None:
        # A ~110 MB torch.save takes seconds; a host reboot mid-write would leave a
        # truncated file at the FINAL path -- fatal when that file is the resume source.
        # Write to a pid-unique sibling and os.replace, so the final path only ever
        # holds a complete checkpoint.
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    for ep in range(start_epoch, args.epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        scheduler.step()
        train_hist.append(tr)
        val_hist.append(va)

        if va["b"] < best_val:
            best_val = va["b"]
            save_atomic(checkpoint_payload(ep), args.ckpt)

        # Crash insurance: the best-val checkpoint can be many epochs stale by the time
        # a host restart lands, and --resume from a stale epoch silently re-does work.
        if args.ckpt_every > 0 and (ep % args.ckpt_every == 0 or ep == args.epochs):
            save_atomic(checkpoint_payload(ep), last_ckpt)

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            spec_part = (f"spec {va['spec']:.2e} "
                         if args.band_spectral_weight > 0.0 else "")
            print(f"  ep {ep:04d}/{args.epochs} | train B {tr['b']:.6f} nRMSE {tr['nrmse']:6.2f}% "
                  f"| val B {va['b']:.6f} nRMSE {va['nrmse']:6.2f}% pbc {va['pbc']:.2e} {spec_part}"
                  f"| best {best_val:.6f} | {time.time() - t0:.0f}s", flush=True)

    print(f"\nTraining complete: epochs {start_epoch}..{args.epochs} in {time.time() - t0:.1f}s")
    print(f"Best val B MSE (normalized): {best_val:.6f}")
    print(f"Checkpoint: {args.ckpt}")
    print("Score it with:  python -m eval.benchmark --curl-ckpt " + str(args.ckpt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
