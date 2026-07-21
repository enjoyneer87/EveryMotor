"""Torch-backed model adapters for the benchmark harness.

Kept separate from `eval.benchmark` so the harness, its reference baselines and
its tests all run on the host without torch or the PhysicsNeMo container.

Checkpoint loading and PyTorch 2.6
----------------------------------
`torch.load` flipped its `weights_only` default to ``True`` in PyTorch 2.6, which
is why `eval_all_models.py` died partway through the four-model comparison. The
DOE checkpoints legitimately carry non-tensor payload (normalization statistics
and the argparse namespace), so they need ``weights_only=False``. That is safe
here — these are our own training artifacts — and `load_checkpoint` says so at
the call site rather than leaving a bare flag for the next reader to worry about.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from eval.benchmark import DEFAULT_CHANNELS
from eval.doe_dataset import CaseRecord, CaseSample
from eval.feature_guard import (
    MGN_EDGE_FEATURES,
    MGN_NODE_FEATURES,
    assert_feature_count,
    assert_input_features_clean,
)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, object]:
    """Load a training checkpoint.

    ``weights_only=False`` is required because our checkpoints store the
    normalization statistics and run arguments alongside the state dict. These
    files are produced by this repository's own training scripts; do not point
    this at an untrusted checkpoint.
    """
    return torch.load(Path(path), map_location=map_location, weights_only=False)


class CheckpointFeatureMismatch(ValueError):
    """Raised when a checkpoint's tensor interface does not match the clean contract."""


# Historical output layouts, keyed by channel count. The 4-channel form is what
# `train_doe_meshgraphnet_aj.py` wrote; the 2-channel form is the Bx/By-only run.
_DEFAULT_OUTPUT_ORDERS: Mapping[int, Tuple[str, ...]] = {
    2: ("Bx", "By"),
    4: ("Bx", "By", "A", "J"),
    5: ("Bx", "By", "A", "J", "Je"),
}


def default_output_order(n_channels: int) -> Tuple[str, ...]:
    order = _DEFAULT_OUTPUT_ORDERS.get(int(n_channels))
    if order is None:
        raise CheckpointFeatureMismatch(
            f"No known channel order for a {n_channels}-output checkpoint; "
            "pass output_channel_order explicitly."
        )
    return order


def split_provenance(ckpt: Mapping[str, object]) -> Dict[str, object]:
    """Describe how the checkpoint's own train/val split was cut.

    Checkpoints written before R0 carry `train_ratio` and no `split` block: they
    were trained on a record-level random shuffle over *all* DOE cases, so every
    geometry — including the ones now held out — contributed ~80% of its
    timesteps to training. Scoring such a checkpoint on the case holdout gives an
    optimistic upper bound, not a generalization measure, and the result must say
    so rather than being filed next to clean numbers.
    """
    split = ckpt.get("split")
    if isinstance(split, dict) and split.get("granularity") == "case":
        return {
            "granularity": "case",
            "contaminated": False,
            "train_cases": list(split.get("train_cases", [])),
            "val_cases": list(split.get("val_cases", [])),
            "test_cases": list(split.get("test_cases", [])),
            "seed": split.get("seed"),
            "doe_digest": split.get("doe_digest"),
        }

    args = ckpt.get("args") or {}
    ratio = args.get("train_ratio") if isinstance(args, dict) else None
    return {
        "granularity": "record",
        "contaminated": True,
        "train_ratio": ratio,
        "warning": (
            "Checkpoint predates the case-level split (no 'split' block; "
            f"train_ratio={ratio}). It was trained on a random shuffle of all "
            "records, so the evaluation cases were seen during training. Treat "
            "these scores as an optimistic upper bound, NOT as gate-G0 "
            "case-holdout results. Retrain with the current "
            "train_doe_meshgraphnet.py for a clean number."
        ),
    }


def build_mgn_inputs(
    record: CaseRecord,
    sample: CaseSample,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the node features, edges and edge features for one sample.

    Mirrors `train_doe_meshgraphnet.build_graph`, minus the targets. The feature
    whitelist is asserted here so a future edit that reintroduces A or J as an
    input fails at evaluation time.
    """
    assert_input_features_clean(MGN_NODE_FEATURES, context="MeshGraphNet node features",
                                allow_shortcuts=True)

    i1, i2, i3 = record.mesh.tri
    n_nodes = record.mesh.n_nodes
    pos = np.column_stack([sample.node_x_mm, sample.node_y_mm]).astype(np.float64)

    # Region code per node: majority vote over incident elements, matching the
    # training graph builder.
    reg = np.asarray(record.mesh.reg_code, dtype=np.int64)
    idx3 = np.concatenate([i1, i2, i3])
    reg3 = np.tile(reg, 3)
    node_reg = np.zeros(n_nodes, dtype=np.float64)
    order = np.lexsort((reg3, idx3))
    idx_sorted, reg_sorted = idx3[order], reg3[order]
    boundaries = np.flatnonzero(np.diff(idx_sorted)) + 1
    for chunk_start, chunk_end in zip(
        np.concatenate([[0], boundaries]), np.concatenate([boundaries, [idx_sorted.size]])
    ):
        node = idx_sorted[chunk_start]
        vals, counts = np.unique(reg_sorted[chunk_start:chunk_end], return_counts=True)
        node_reg[node] = float(vals[np.argmax(counts)])

    cond = record.condition
    scalars = [
        sample.time_s,
        sample.rotate_step,
        float(cond.get("Ratio_Bore", 0.0)),
        float(cond.get("Ratio_SlotDepth_ParallelSlot", 0.0)),
        float(cond.get("PeakCurrent", 0.0)),
        float(cond.get("PhaseAdvance", 0.0)),
    ]
    x = np.column_stack(
        [pos, node_reg[:, None]] + [np.full((n_nodes, 1), s, dtype=np.float64) for s in scalars]
    )
    assert_feature_count(MGN_NODE_FEATURES, x.shape[1], context="MeshGraphNet node features")

    src = np.concatenate([i1, i2, i3, i2, i3, i1])
    dst = np.concatenate([i2, i3, i1, i1, i2, i3])
    edge_pairs = np.unique(np.stack([src, dst], axis=1), axis=0)
    edge_index = edge_pairs.T.astype(np.int64)

    dxy = pos[edge_index[1]] - pos[edge_index[0]]
    dist = np.linalg.norm(dxy, axis=1, keepdims=True)
    edge_attr = np.concatenate([dxy, dist], axis=1)

    return x, edge_index, edge_attr, node_reg


@dataclass
class MeshGraphNetPredictor:
    """Scores a trained DOE MeshGraphNet checkpoint.

    De-normalizes the model output back to tesla using the statistics stored in
    the checkpoint, so the harness compares physical units.
    """

    model: torch.nn.Module
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    e_mean: torch.Tensor
    e_std: torch.Tensor
    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    # Channel order the checkpoint itself predicts; `channels` is the subset scored.
    native_channels: Tuple[str, ...] = DEFAULT_CHANNELS
    name: str = "meshgraphnet"
    output_support: str = "node"
    device: torch.device = torch.device("cpu")
    checkpoint_path: Optional[Path] = None
    n_params: int = 0
    provenance: Optional[Dict[str, object]] = None
    train_args: Optional[Dict[str, object]] = None

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        channels: Sequence[str] = DEFAULT_CHANNELS,
        device: Optional[torch.device] = None,
        name: str = "meshgraphnet",
        output_channel_order: Optional[Sequence[str]] = None,
    ) -> "MeshGraphNetPredictor":
        from physicsnemo.models.meshgraphnet import MeshGraphNet

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_checkpoint(Path(path), map_location=str(device))
        args = ckpt.get("args", {})

        y_mean = ckpt["y_mean"]
        n_out = int(y_mean.shape[-1])

        expected_in = int(ckpt["x_mean"].shape[-1])
        if expected_in != len(MGN_NODE_FEATURES):
            raise CheckpointFeatureMismatch(
                f"{Path(path).name} expects {expected_in} node input features but the clean "
                f"layout has {len(MGN_NODE_FEATURES)}: {list(MGN_NODE_FEATURES)}.\n"
                "An 11-feature checkpoint was trained by the pre-R0 graph builder, which fed "
                "A (vector potential) and J (current density) in as node inputs. Since B = curl A, "
                "that hands the model most of its own target: any score it produces is not a "
                "surrogate result and this harness will not reproduce it.\n"
                "Score a checkpoint trained with the current builder instead, or retrain."
            )

        native = tuple(output_channel_order) if output_channel_order else default_output_order(n_out)
        if len(native) != n_out:
            raise CheckpointFeatureMismatch(
                f"{Path(path).name} outputs {n_out} channels but {len(native)} names were given: {native}"
            )
        missing = [c for c in channels if c not in native]
        if missing:
            raise CheckpointFeatureMismatch(
                f"{Path(path).name} predicts {list(native)}; cannot score requested {missing}"
            )

        model = MeshGraphNet(
            input_dim_nodes=len(MGN_NODE_FEATURES),
            input_dim_edges=len(MGN_EDGE_FEATURES),
            output_dim=n_out,
            processor_size=int(args.get("processor_size", 15)),
            hidden_dim_processor=int(args.get("hidden_dim", 128)),
            hidden_dim_node_encoder=int(args.get("hidden_dim", 128)),
            hidden_dim_edge_encoder=int(args.get("hidden_dim", 128)),
            hidden_dim_node_decoder=int(args.get("hidden_dim", 128)),
            aggregation="sum",
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        return cls(
            model=model,
            x_mean=ckpt["x_mean"].to(device),
            x_std=ckpt["x_std"].to(device),
            y_mean=y_mean.to(device),
            y_std=ckpt["y_std"].to(device),
            e_mean=ckpt["e_mean"].to(device),
            e_std=ckpt["e_std"].to(device),
            channels=tuple(channels),
            native_channels=native,
            name=name,
            device=device,
            checkpoint_path=Path(path),
            n_params=sum(p.numel() for p in model.parameters()),
            provenance=split_provenance(ckpt),
            train_args={k: v for k, v in args.items() if k != "ckpt"} if isinstance(args, dict) else {},
        )

    @torch.no_grad()
    def predict(self, record: CaseRecord, sample: CaseSample) -> np.ndarray:
        from torch_geometric.data import Data

        x, edge_index, edge_attr, _ = build_mgn_inputs(record, sample)

        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        et = torch.as_tensor(edge_attr, dtype=torch.float32, device=self.device)
        xt = torch.nan_to_num((xt - self.x_mean) / self.x_std)
        et = torch.nan_to_num((et - self.e_mean) / self.e_std)

        graph = Data(
            x=xt,
            edge_index=torch.as_tensor(edge_index, dtype=torch.long, device=self.device),
            edge_attr=et,
        )
        pred = self.model(graph.x, graph.edge_attr, graph)
        # Back to tesla — the harness compares physical units only.
        pred = pred * self.y_std + self.y_mean
        out = pred.detach().cpu().numpy().astype(np.float64)

        # A checkpoint may predict more channels than the benchmark scores.
        if tuple(self.native_channels) != tuple(self.channels):
            cols = [self.native_channels.index(c) for c in self.channels]
            out = out[:, cols]
        return out

    def describe(self) -> Dict[str, object]:
        return {
            "kind": "MeshGraphNet",
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "params": int(self.n_params),
            "node_features": list(MGN_NODE_FEATURES),
            "edge_features": list(MGN_EDGE_FEATURES),
            "predicted_channels": list(self.native_channels),
            "scored_channels": list(self.channels),
            "output_support": self.output_support,
            "train_args": self.train_args or {},
            "split_provenance": self.provenance or {},
        }


def _wrap_sign(cumulative_deg: float, sector_deg: float, wrapped: bool) -> Tuple[float, int, float]:
    """Anti-periodic sign that was applied to the target during training.

    Returns ``(wrapped_deg, k, sign)``; `sign` is 1.0 when wrapping is off, so
    the caller can apply it unconditionally.
    """
    if not wrapped:
        return float(cumulative_deg), 0, 1.0
    from phase1_static.sector_symmetry import wrap_rotor_angle

    return wrap_rotor_angle(cumulative_deg, sector_deg)


@dataclass
class CurlMeshGraphNetPredictor:
    """Scores a checkpoint that predicts nodal A and derives B by the P1 curl.

    Output support is element, natively: the curl of a nodal field lands on
    elements, so nothing is averaged and the round-trip cost the node-support
    models pay does not apply here.

    Elements with no valid per-step geometry (the re-meshed sliding band, plus
    anything inverted) get NaN rather than a fabricated value. The harness masks
    them out of the field metrics, and `arkkio_torque` drops non-finite entries —
    but the torque band is the stationary a1 layer, which is never excluded, so
    the torque integral is unaffected either way.
    """

    model: torch.nn.Module
    x_mean: torch.Tensor
    x_std: torch.Tensor
    e_mean: torch.Tensor
    e_std: torch.Tensor
    a_scale: float
    predicts_a: bool = True
    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    name: str = "mgn_curl"
    output_support: str = "element"
    device: torch.device = torch.device("cpu")
    checkpoint_path: Optional[Path] = None
    n_params: int = 0
    provenance: Optional[Dict[str, object]] = None
    train_args: Optional[Dict[str, object]] = None
    sector_symmetry: Optional[Dict[str, object]] = None
    node_features: Tuple[str, ...] = ()
    edge_features: Tuple[str, ...] = ()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        channels: Sequence[str] = DEFAULT_CHANNELS,
        device: Optional[torch.device] = None,
        name: str = "mgn_curl",
    ) -> "CurlMeshGraphNetPredictor":
        from physicsnemo.models.meshgraphnet import MeshGraphNet

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_checkpoint(Path(path), map_location=str(device))

        kind = ckpt.get("output_kind")
        if kind not in ("nodal_A_curl_to_element_B", "nodal_B_average_to_element_B"):
            raise CheckpointFeatureMismatch(
                f"{Path(path).name} has output_kind {kind!r}; this adapter expects one of "
                "'nodal_A_curl_to_element_B' / 'nodal_B_average_to_element_B' "
                "(written by train_doe_curl_mgn.py)."
            )
        predicts_a = kind == "nodal_A_curl_to_element_B"

        args = ckpt.get("args", {}) or {}
        # Feature layout comes from the checkpoint, not from a constant here:
        # the sector-symmetry fixes changed both the node and edge widths.
        node_features = tuple(ckpt.get("node_features") or MGN_NODE_FEATURES)
        edge_features = tuple(ckpt.get("edge_features") or MGN_EDGE_FEATURES)
        # Historical checkpoints predate the shortcut-feature finding; they are
        # still scoreable, just not a template for new models.
        assert_input_features_clean(node_features, context=f"{Path(path).name} node features",
                                    allow_shortcuts=True)

        model = MeshGraphNet(
            input_dim_nodes=len(node_features),
            input_dim_edges=len(edge_features),
            output_dim=1 if predicts_a else 2,
            processor_size=int(args.get("processor_size", 15)),
            hidden_dim_processor=int(args.get("hidden_dim", 128)),
            hidden_dim_node_encoder=int(args.get("hidden_dim", 128)),
            hidden_dim_edge_encoder=int(args.get("hidden_dim", 128)),
            hidden_dim_node_decoder=int(args.get("hidden_dim", 128)),
            aggregation="sum",
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        return cls(
            model=model,
            x_mean=ckpt["x_mean"].to(device),
            x_std=ckpt["x_std"].to(device),
            e_mean=ckpt["e_mean"].to(device),
            e_std=ckpt["e_std"].to(device),
            a_scale=float(ckpt.get("out_scale", ckpt["a_scale"])),
            predicts_a=predicts_a,
            channels=tuple(channels),
            name=name,
            device=device,
            checkpoint_path=Path(path),
            n_params=sum(p.numel() for p in model.parameters()),
            provenance=split_provenance(ckpt),
            train_args={k: v for k, v in args.items() if k != "ckpt"} if isinstance(args, dict) else {},
            sector_symmetry=dict(ckpt.get("sector_symmetry") or {}),
            node_features=node_features,
            edge_features=edge_features,
        )

    @torch.no_grad()
    def predict(self, record: CaseRecord, sample: CaseSample) -> np.ndarray:
        from torch_geometric.data import Data

        # The graph is built by the *training* module, not re-implemented here.
        # A second implementation is exactly how the pre-R0 pipeline ended up
        # with a 9-feature builder and an 11-feature docstring.
        from train_doe_curl_mgn import build_curl_graph

        from eval.mesh_regions import sliding_band_mask
        from phase1_static.sector_symmetry import cumulative_rotor_angle, rigid_rotor_node_mask

        mesh = record.mesh
        symmetry = self.sector_symmetry or {}
        sector_deg = float(symmetry.get("sector_deg", 45.0))

        band = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)
        rigid = rigid_rotor_node_mask(
            mesh.tri, mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes, mesh.n_nodes
        )
        cum = cumulative_rotor_angle([s.rotate_step for s in record.samples])
        step = next(
            (i for i, s in enumerate(record.samples) if s.step_index == sample.step_index), 0
        )

        # A checkpoint with no sector_symmetry block predates the fixes and was
        # trained on the 9-feature layout; build it the way it was trained.
        legacy = not symmetry
        graph = build_curl_graph(
            record,
            sample,
            band,
            rigid,
            cumulative_deg=float(cum[step]),
            sector_deg=sector_deg,
            wrap_rotor=bool(symmetry.get("wrap_rotor", True)) and not legacy,
            anti_periodic=bool(symmetry.get("anti_periodic_edges", True)) and not legacy,
            legacy_features=legacy,
            # Rebuild the exact layout this checkpoint was trained on, whatever
            # the current default happens to be.
            node_feature_names=self.node_features or None,
            edge_feature_names=self.edge_features or None,
        )
        if graph is None:
            return np.full((mesh.n_elements, len(self.channels)), np.nan, dtype=np.float64)

        xt = graph.x.to(self.device)
        et = graph.edge_attr.to(self.device)
        batch = Data(
            x=torch.nan_to_num((xt - self.x_mean) / self.x_std),
            edge_index=graph.edge_index.to(self.device),
            edge_attr=torch.nan_to_num((et - self.e_mean) / self.e_std),
        )
        raw = self.model(batch.x, batch.edge_attr, batch) * self.a_scale
        idx = graph.curl_node_index.to(self.device)

        if self.predicts_a:
            gathered = raw.view(-1).double()[idx]                 # (K, 3)
            cbx = graph.curl_coef_bx.to(self.device).double()
            cby = graph.curl_coef_by.to(self.device).double()
            bx = (cbx * gathered).sum(1).cpu().numpy()
            by = (cby * gathered).sum(1).cpu().numpy()
        else:
            # Node-support baseline: average the three nodal values per element.
            elem = raw.double()[idx].mean(dim=1)                  # (K, 2)
            bx = elem[:, 0].cpu().numpy()
            by = elem[:, 1].cpu().numpy()

        # Undo the anti-periodic sign so the prediction is comparable to the
        # exported FEM field, which is reported in the unwrapped convention.
        _, _, sign = _wrap_sign(
            float(cum[step]), sector_deg, bool(symmetry.get("wrap_rotor", True)) and not legacy
        )
        bx, by = sign * bx, sign * by

        out = np.full((mesh.n_elements, len(self.channels)), np.nan, dtype=np.float64)
        ix, iy = self.channels.index("Bx"), self.channels.index("By")
        elements = graph.curl_element_index.cpu().numpy()
        out[elements, ix] = bx
        out[elements, iy] = by
        return out

    def describe(self) -> Dict[str, object]:
        return {
            "kind": "MeshGraphNet + P1 curl",
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "params": int(self.n_params),
            "node_features": list(self.node_features),
            "edge_features": list(self.edge_features),
            "sector_symmetry": self.sector_symmetry or {},
            "predicts": "nodal A" if self.predicts_a else "nodal Bx, By",
            "derives": "element B = curl(A)" if self.predicts_a
            else "element B = mean of nodal B (round trip)",
            "output_support": self.output_support,
            "a_scale": self.a_scale,
            "train_args": self.train_args or {},
            "split_provenance": self.provenance or {},
        }
