"""Training script for Phase 1 static 1/8 motor model with hybrid physics loss."""

from __future__ import annotations

import argparse
import collections
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler
try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except ImportError:  # tensorboard optional
    _SummaryWriter = None

from .contracts import (
    ChannelZScoreStats,
    apply_channel_zscore,
    build_channel_zscore_stats,
    normalize_channels_to_bx_by_a_je,
    resolve_loss_routing_from_codes,
    validate_batch_fidelity_policy,
    validate_graph_batch_contract,
)
from .loss import hybrid_physics_loss
from .motor_dataset import StaticMotorDataset, build_samples_from_doe_manifest, build_samples_from_npz
from .physics_operators import PhysicsOperator, build_physics_operator

LOG = logging.getLogger(__name__)

# PhysicsNeMo emits DGL backend warnings even when the PyG execution path is used.
warnings.filterwarnings(
    "ignore",
    message=r"MeshGraphNet \(DGL version\) requires the DGL library\.",
    category=UserWarning,
)


class StepAwareBatchSampler(Sampler[List[int]]):
    """Batch sampler that keeps one step_index per batch."""

    def __init__(
        self,
        step_indices: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._groups: Dict[int, List[int]] = collections.defaultdict(list)
        for sample_idx, step_idx in enumerate(step_indices):
            self._groups[int(step_idx)].append(sample_idx)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        step_keys = list(self._groups.keys())
        if self.shuffle:
            rng.shuffle(step_keys)

        all_batches: List[List[int]] = []
        for step_key in step_keys:
            indices = list(self._groups[step_key])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for indices in self._groups.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic seeds for reproducible smoke and overfit harnesses."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_contract_gate(loader: DataLoader, spatial_dim: int) -> None:
    """Validate graph/batch contracts at the loader boundary."""
    first_batch = next(iter(loader), None)
    if first_batch is None:
        raise RuntimeError("Contract gate failed: loader is empty")
    schema = validate_graph_batch_contract(first_batch, spatial_dim)

    LOG.info(
        "Contract gate passed: x=%s pos=%s edge_index=%s edge_attr=%s y_schema=%s",
        tuple(first_batch.x.shape),
        tuple(first_batch.pos.shape),
        tuple(first_batch.edge_index.shape),
        tuple(first_batch.edge_attr.shape),
        schema,
    )


def build_model(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """Build PhysicsNeMo MeshGraphNet. Must run inside Docker PhysicsNeMo container.

    Raises RuntimeError if physicsnemo is not available — do NOT silently fall back
    to a lighter model (would invalidate training comparisons).
    Run via: docker exec physicsnemo python -m phase1_static.train ...
    """
    try:
        from physicsnemo.models.meshgraphnet import MeshGraphNet
    except ImportError as exc:
        raise RuntimeError(
            "physicsnemo.models.meshgraphnet is required but not installed.\n"
            "Run training inside the PhysicsNeMo Docker container:\n"
            "  docker exec physicsnemo python -m phase1_static.train ..."
        ) from exc

    return MeshGraphNet(
        input_dim_nodes=input_dim,
        input_dim_edges=1,
        output_dim=output_dim,
        processor_size=10,
        hidden_dim_processor=hidden_dim,
        hidden_dim_node_encoder=hidden_dim,
        hidden_dim_edge_encoder=hidden_dim,
        hidden_dim_node_decoder=hidden_dim,
        aggregation="sum",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 static 1/8 training")
    p.add_argument("--input-format", choices=("auto", "npz", "doe"), default="auto")
    p.add_argument("--data", help="Path to npz/pt bundle (used for --input-format npz/auto)")
    p.add_argument("--data-dir", default="doe_data", help="DOE root dir containing train_manifest.json (or doe_manifest.json fallback)")
    p.add_argument("--max-steps-per-case", type=int, default=None, help="Optional cap for DOE timesteps per case")
    p.add_argument(
        "--step-index",
        type=int,
        default=None,
        help="If set, keep only samples where sample.step_index == this value",
    )
    p.add_argument(
        "--case-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit DOE case indices to include during training",
    )
    p.add_argument(
        "--source-file-types",
        nargs="+",
        default=None,
        help="Optional MotorCAD source file classes to include, e.g. OnLoadTorque StaticLoad",
    )
    p.add_argument(
        "--include-temporal-features",
        action="store_true",
        help="Append [time_s, rotate_step, dt_s, step_index] to node input features",
    )
    p.add_argument("--spatial-dim", type=int, default=2, help="Spatial coordinate dimension (2 supported now)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--step-aware-batching",
        action="store_true",
        help="Group training batches by step_index so one batch contains one step only",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--lr-decay-factor",
        type=float,
        default=1.0,
        help="ReduceLROnPlateau factor (<1.0 enables decay when --lr-decay-patience > 0)",
    )
    p.add_argument(
        "--lr-decay-patience",
        type=int,
        default=0,
        help="ReduceLROnPlateau patience in epochs (0 disables scheduler)",
    )
    p.add_argument(
        "--lr-decay-min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for ReduceLROnPlateau",
    )
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42, help="Deterministic seed for smoke/overfit reproducibility")
    p.add_argument("--smoke-max-samples", type=int, default=0, help="If >0, run a deterministic smoke slice")
    p.add_argument("--contract-check-only", action="store_true", help="Validate contracts and exit")
    p.add_argument("--weight-a", type=float, default=1.0, help="Supervised A loss weight")
    p.add_argument("--weight-b", type=float, default=1.0, help="Supervised Bx/By loss weight")
    p.add_argument("--weight-current", type=float, default=1.0, help="Supervised J/Je loss weight")
    p.add_argument(
        "--current-focus-alpha",
        type=float,
        default=2.0,
        help="Magnitude-aware Je loss focus strength (0 disables sparse-focus weighting)",
    )
    p.add_argument(
        "--current-focus-gamma",
        type=float,
        default=1.0,
        help="Exponent for Je magnitude-aware loss weighting",
    )
    p.add_argument(
        "--target-normalization",
        choices=("none", "zscore"),
        default="zscore",
        help="Channel-wise normalization for [Bx, By, A, Je] at loss boundary",
    )
    p.add_argument(
        "--target-normalization-eps",
        type=float,
        default=1e-6,
        help="Numerical floor for channel std during z-score normalization",
    )
    p.add_argument("--overfit-single", action="store_true", help="Repeat the first batch to check overfitting")
    p.add_argument("--overfit-target", type=float, default=1e-4, help="Pass threshold for overfit-single train loss")
    p.add_argument(
        "--sequence-len",
        type=int,
        default=1,
        help=(
            "Temporal sequence length. "
            "1 = static Phase 1 (default, backward-compatible). "
            ">1 = Phase 2 temporal mode (TemporalMotorSample, dynamic edge refresh per step)."
        ),
    )
    p.add_argument(
        "--ckpt-out",
        default=None,
        help="Save trained SymMGN checkpoint to this path (.pt). If omitted, no checkpoint is saved.",
    )
    p.add_argument(
        "--ckpt-interval",
        type=int,
        default=0,
        help="Save intermediate checkpoint every N epochs (0 = disabled). Saved as <ckpt-out>.ep<N>.",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Append epoch metrics to this text file (one JSON line per epoch). Enables progress monitoring.",
    )
    p.add_argument(
        "--tb-log-dir",
        default=None,
        help="TensorBoard log directory. If omitted, TensorBoard logging is disabled.",
    )
    return p.parse_args()


def forward_model(model: nn.Module, batch) -> torch.Tensor:
    try:
        return model(batch.x, batch.edge_attr, batch)
    except TypeError:
        try:
            return model(batch.x, batch.edge_attr, batch.edge_index)
        except TypeError:
            return model(batch)


def _normalize_target_channels(y: torch.Tensor) -> Tuple[torch.Tensor, str]:
    return normalize_channels_to_bx_by_a_je(y)


def _normalize_prediction_channels(pred: torch.Tensor) -> torch.Tensor:
    norm, _ = normalize_channels_to_bx_by_a_je(pred)
    return norm


def _compute_channel_zscore_stats(
    samples: Sequence[dict],
    *,
    eps: float,
) -> ChannelZScoreStats:
    """Compute canonical [Bx, By, A, Je] channel statistics from sample targets."""
    channel_sum = torch.zeros((4,), dtype=torch.float64)
    channel_sq_sum = torch.zeros((4,), dtype=torch.float64)
    sample_count = 0

    for sample in samples:
        y_raw = torch.as_tensor(sample["y"], dtype=torch.float32)
        y, _ = normalize_channels_to_bx_by_a_je(y_raw)
        if y.numel() == 0:
            continue
        y64 = y.to(dtype=torch.float64)
        channel_sum += y64.sum(dim=0)
        channel_sq_sum += (y64 ** 2).sum(dim=0)
        sample_count += int(y64.shape[0])

    return build_channel_zscore_stats(
        channel_sum=channel_sum,
        channel_sq_sum=channel_sq_sum,
        sample_count=sample_count,
        eps=eps,
    )


def _resolve_sample_weight(batch) -> float:
    """Multi-fidelity hook: per-batch scalar weight from sample metadata."""
    sample_weight = getattr(batch, "sample_weight", None)
    if sample_weight is None:
        return 1.0
    if isinstance(sample_weight, torch.Tensor):
        if sample_weight.numel() == 0:
            return 1.0
        return float(sample_weight.float().mean().item())
    return float(sample_weight)


def _resolve_loss_routing_from_batch(
    batch,
    schema: str,
    w_a: float,
    w_b: float,
    w_current: float,
) -> Tuple[float, float, float, float]:
    """Resolve effective loss weights from schema and batch fidelity metadata."""
    fidelity_code, coupling_code = validate_batch_fidelity_policy(batch)
    a_scale, b_scale, _curl_scale, sample_weight_mult = resolve_loss_routing_from_codes(
        fidelity_code=fidelity_code,
        coupling_code=coupling_code,
    )

    w_a_eff = float(w_a) * a_scale
    if schema == "bx_by":
        w_a_eff = 0.0

    w_b_eff = float(w_b) * b_scale
    w_current_eff = float(w_current) * b_scale
    if schema in {"a_bx_by", "bx_by", "a_only"}:
        w_current_eff = 0.0

    sample_weight_eff = _resolve_sample_weight(batch) * sample_weight_mult
    return w_a_eff, w_b_eff, w_current_eff, sample_weight_eff


def _attach_pos_to_x(batch) -> None:
    """Ensure x coordinate channels reference differentiable pos tensor."""
    d = int(batch.pos.shape[1])
    if batch.x.shape[1] < d:
        raise ValueError(
            f"x feature dim must be >= spatial dim, got x={batch.x.shape} pos={batch.pos.shape}"
        )
    batch.x = torch.cat([batch.pos, batch.x[:, d:]], dim=1)


def _load_samples(args: argparse.Namespace):
    fmt = args.input_format
    if fmt == "auto":
        if args.data:
            return build_samples_from_npz(args.data), "npz"
        return build_samples_from_doe_manifest(
            args.data_dir,
            args.max_steps_per_case,
            case_indices=args.case_indices,
            source_file_types=args.source_file_types,
        ), "doe"
    if fmt == "npz":
        if not args.data:
            raise ValueError("--data is required when --input-format=npz")
        return build_samples_from_npz(args.data), "npz"
    if fmt == "doe":
        return build_samples_from_doe_manifest(
            args.data_dir,
            args.max_steps_per_case,
            case_indices=args.case_indices,
            source_file_types=args.source_file_types,
        ), "doe"
    raise ValueError(f"Unsupported input format: {fmt}")


def _filter_samples_by_step_index(samples: Sequence[dict], step_index: Optional[int]) -> List[dict]:
    if step_index is None:
        return list(samples)
    target = int(step_index)
    return [sample for sample in samples if int(sample.get("step_index", -1)) == target]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    operator: PhysicsOperator,
    w_a: float,
    w_b: float,
    w_current: float,
    current_focus_alpha: float,
    current_focus_gamma: float,
    target_stats: Optional[ChannelZScoreStats],
) -> Dict[str, float]:
    model.train()
    sums = {"total_loss": 0.0, "a_loss": 0.0, "b_loss": 0.0, "current_loss": 0.0}
    steps = 0
    for batch in loader:
        batch = batch.to(device)
        _attach_pos_to_x(batch)

        pred_raw = _normalize_prediction_channels(forward_model(model, batch))
        target_raw, schema = _normalize_target_channels(batch.y)
        pred = apply_channel_zscore(pred_raw, target_stats)
        target = apply_channel_zscore(target_raw, target_stats)
        w_a_eff, w_b_eff, w_current_eff, sample_weight = _resolve_loss_routing_from_batch(
            batch=batch,
            schema=schema,
            w_a=w_a,
            w_b=w_b,
            w_current=w_current,
        )

        total_loss, metrics = hybrid_physics_loss(
            pred=pred,
            target=target,
            coords=batch.pos,
            operator=operator,
            w_a=w_a_eff,
            w_b=w_b_eff,
            w_current=w_current_eff,
            current_focus_alpha=current_focus_alpha,
            current_focus_gamma=current_focus_gamma,
        )
        total_loss = total_loss * sample_weight

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sums["total_loss"] += float(total_loss.detach().item())
        sums["a_loss"] += float(metrics["a_loss"].item())
        sums["b_loss"] += float(metrics["b_loss"].item())
        sums["current_loss"] += float(metrics["current_loss"].item())
        steps += 1
    denom = max(steps, 1)
    return {k: v / denom for k, v in sums.items()}


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    operator: PhysicsOperator,
    w_a: float,
    w_b: float,
    w_current: float,
    current_focus_alpha: float,
    current_focus_gamma: float,
    target_stats: Optional[ChannelZScoreStats],
) -> Dict[str, float]:
    model.eval()
    sums = {"total_loss": 0.0, "a_loss": 0.0, "b_loss": 0.0, "current_loss": 0.0}
    steps = 0
    for batch in loader:
        batch = batch.to(device)
        _attach_pos_to_x(batch)

        with torch.no_grad():
            pred_raw = _normalize_prediction_channels(forward_model(model, batch))
            target_raw, schema = _normalize_target_channels(batch.y)
            pred = apply_channel_zscore(pred_raw, target_stats)
            target = apply_channel_zscore(target_raw, target_stats)
            w_a_eff, w_b_eff, w_current_eff, sample_weight = _resolve_loss_routing_from_batch(
                batch=batch,
                schema=schema,
                w_a=w_a,
                w_b=w_b,
                w_current=w_current,
            )
            _, metrics = hybrid_physics_loss(
                pred=pred,
                target=target,
                coords=batch.pos,
                operator=operator,
                w_a=w_a_eff,
                w_b=w_b_eff,
                w_current=w_current_eff,
                current_focus_alpha=current_focus_alpha,
                current_focus_gamma=current_focus_gamma,
            )
        sums["total_loss"] += float(metrics["total_loss"].item() * sample_weight)
        sums["a_loss"] += float(metrics["a_loss"].item())
        sums["b_loss"] += float(metrics["b_loss"].item())
        sums["current_loss"] += float(metrics["current_loss"].item())
        steps += 1
    denom = max(steps, 1)
    return {k: v / denom for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_deterministic_seed(args.seed)

    if args.sequence_len != 1:
        LOG.warning(
            "--sequence-len=%d requested. Phase 2 temporal mode is not yet implemented in train.py. "
            "Falling back to static (sequence_len=1). "
            "Implement TemporalMotorDataset in phase2_dynamic/ and wire it here.",
            args.sequence_len,
        )

    samples, resolved_fmt = _load_samples(args)
    samples = _filter_samples_by_step_index(samples, args.step_index)
    if args.smoke_max_samples and args.smoke_max_samples > 0:
        samples = samples[: args.smoke_max_samples]
        LOG.info("Smoke slice enabled: using first %d samples", len(samples))

    if len(samples) == 0:
        raise RuntimeError(
            "No samples available after filtering. "
            f"input_format={resolved_fmt} step_index={args.step_index}"
        )

    dataset = StaticMotorDataset(samples, include_temporal_features=args.include_temporal_features)

    if args.overfit_single and args.step_aware_batching:
        LOG.warning("--step-aware-batching is ignored when --overfit-single is enabled")

    if args.step_aware_batching and not args.overfit_single:
        sampler = StepAwareBatchSampler(
            [int(sample.get("step_index", -1)) for sample in samples],
            args.batch_size,
            shuffle=True,
            drop_last=False,
            seed=args.seed,
        )
        train_loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        train_gen = torch.Generator()
        train_gen.manual_seed(args.seed)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=not args.overfit_single,
            generator=train_gen,
        )

    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    operator = build_physics_operator(spatial_dim=args.spatial_dim)
    run_contract_gate(train_loader, args.spatial_dim)
    if args.contract_check_only:
        LOG.info("Contract-check-only mode complete. Exiting without training.")
        return

    LOG.info(
        "Loaded %d samples (format=%s, temporal_features=%s, operator=%s)",
        len(samples),
        resolved_fmt,
        args.include_temporal_features,
        operator.name,
    )

    target_stats: Optional[ChannelZScoreStats] = None
    if args.target_normalization == "zscore":
        target_stats = _compute_channel_zscore_stats(
            samples,
            eps=float(args.target_normalization_eps),
        )
        LOG.info(
            "Target z-score stats enabled: mean=%s std=%s",
            [round(float(v), 6) for v in target_stats.mean.tolist()],
            [round(float(v), 6) for v in target_stats.std.tolist()],
        )
    else:
        LOG.info("Target normalization disabled")

    input_dim = dataset[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=4,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_decay_patience > 0 and args.lr_decay_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            min_lr=args.lr_decay_min_lr,
        )

    first_batch: Optional[DataLoader] = None
    if args.overfit_single:
        first_batch = DataLoader([dataset[0]], batch_size=1, shuffle=False)

    # ── TensorBoard ──────────────────────────────────────────────────────────
    tb_writer = None
    if getattr(args, "tb_log_dir", None) and _SummaryWriter is not None:
        tb_writer = _SummaryWriter(log_dir=args.tb_log_dir)
        LOG.info("TensorBoard logging: %s", args.tb_log_dir)
    elif getattr(args, "tb_log_dir", None):
        LOG.warning("tensorboard not installed — TB logging disabled")

    # ── 파일 로그 핸들러 ─────────────────────────────────────────────────────
    _log_fh = None
    if args.log_file:
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(args.log_file)), exist_ok=True)
        _log_fh = open(args.log_file, "a", buffering=1)  # line-buffered
        LOG.info("Epoch log file: %s", args.log_file)

    best_train_total = float("inf")
    best_val_total = float("inf")
    for epoch in range(1, args.epochs + 1):
        if args.step_aware_batching and not args.overfit_single:
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)

        train_loader_epoch = first_batch if first_batch is not None else train_loader
        train_metrics = train_epoch(
            model=model,
            loader=train_loader_epoch,
            optimizer=optimizer,
            device=device,
            operator=operator,
            w_a=args.weight_a,
            w_b=args.weight_b,
            w_current=args.weight_current,
            current_focus_alpha=args.current_focus_alpha,
            current_focus_gamma=args.current_focus_gamma,
            target_stats=target_stats,
        )
        best_train_total = min(best_train_total, train_metrics["total_loss"])
        val_metrics = eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
            operator=operator,
            w_a=args.weight_a,
            w_b=args.weight_b,
            w_current=args.weight_current,
            current_focus_alpha=args.current_focus_alpha,
            current_focus_gamma=args.current_focus_gamma,
            target_stats=target_stats,
        )
        if scheduler is not None:
            scheduler.step(val_metrics["total_loss"])

        current_lr = float(optimizer.param_groups[0]["lr"])

        LOG.info(
            "epoch=%d/%d lr=%.6e train(total=%.6f,a=%.6f,b=%.6f,current=%.6f) "
            "val(total=%.6f,a=%.6f,b=%.6f,current=%.6f)",
            epoch,
            args.epochs,
            current_lr,
            train_metrics["total_loss"],
            train_metrics["a_loss"],
            train_metrics["b_loss"],
            train_metrics["current_loss"],
            val_metrics["total_loss"],
            val_metrics["a_loss"],
            val_metrics["b_loss"],
            val_metrics["current_loss"],
        )

        best_val_total = min(best_val_total, val_metrics["total_loss"])

        # ── 파일 로그: JSON 한 줄 ─────────────────────────────────────────────
        if _log_fh is not None:
            import json as _json
            _log_fh.write(_json.dumps({
                "epoch": epoch, "epochs": args.epochs, "lr": current_lr,
                "train_total": train_metrics["total_loss"],
                "train_a": train_metrics["a_loss"],
                "train_b": train_metrics["b_loss"],
                "val_total": val_metrics["total_loss"],
                "best_val": best_val_total,
            }) + "\n")

        # ── 중간 체크포인트 ───────────────────────────────────────────────────
        if args.ckpt_interval and args.ckpt_out and (epoch % args.ckpt_interval == 0):
            from infer_phase1_pbc import save_checkpoint as _sc
            _ep_path = Path(args.ckpt_out).with_suffix("") / f"ep{epoch:04d}.pt"
            _ep_path.parent.mkdir(parents=True, exist_ok=True)
            _sc(path=_ep_path, model=model, epoch=epoch, best_loss=best_val_total,
                args_dict={
                    "hidden_dim": args.hidden_dim,
                    "output_dim": 4,
                    "target_normalization": args.target_normalization,
                    "target_normalization_eps": args.target_normalization_eps,
                    "current_focus_alpha": args.current_focus_alpha,
                    "current_focus_gamma": args.current_focus_gamma,
                    "target_norm_mean": (
                        [float(v) for v in target_stats.mean.tolist()]
                        if target_stats is not None else None
                    ),
                    "target_norm_std": (
                        [float(v) for v in target_stats.std.tolist()]
                        if target_stats is not None else None
                    ),
                })
            LOG.info("Intermediate checkpoint saved: %s", _ep_path)

        # TensorBoard: 매 epoch 기록
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss_total",   train_metrics["total_loss"],   epoch)
            tb_writer.add_scalar("train/loss_a",       train_metrics["a_loss"],       epoch)
            tb_writer.add_scalar("train/loss_b",       train_metrics["b_loss"],       epoch)
            tb_writer.add_scalar("train/loss_current", train_metrics["current_loss"], epoch)
            tb_writer.add_scalar("val/loss_total",     val_metrics["total_loss"],     epoch)
            tb_writer.add_scalar("val/loss_a",         val_metrics["a_loss"],         epoch)
            tb_writer.add_scalar("val/loss_b",         val_metrics["b_loss"],         epoch)
            tb_writer.add_scalar("val/loss_current",   val_metrics["current_loss"],   epoch)
            tb_writer.add_scalar("train/lr",           current_lr,                    epoch)
            tb_writer.flush()

    if args.overfit_single:
        if best_train_total > float(args.overfit_target):
            raise RuntimeError(
                f"Overfit-single gate failed: best train total {best_train_total:.6f} "
                f"> target {float(args.overfit_target):.6f}"
            )
        LOG.info(
            "Overfit-single gate passed: best train total %.6f <= target %.6f",
            best_train_total,
            float(args.overfit_target),
        )

    if _log_fh is not None:
        _log_fh.close()

    if tb_writer is not None:
        tb_writer.close()
        LOG.info("TensorBoard writer closed.")

    if args.ckpt_out:
        from infer_phase1_pbc import save_checkpoint
        ckpt_path = Path(args.ckpt_out)
        save_checkpoint(
            path=ckpt_path,
            model=model,
            epoch=args.epochs,
            best_loss=best_train_total,
            args_dict={
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "output_dim": 4,
                "epochs": args.epochs,
                "lr": args.lr,
                "lr_decay_factor": args.lr_decay_factor,
                "lr_decay_patience": args.lr_decay_patience,
                "seed": args.seed,
                "step_index": args.step_index,
                "step_aware_batching": bool(args.step_aware_batching),
                "pbc_rotation_deg": -45.0,
                "target_normalization": args.target_normalization,
                "target_normalization_eps": args.target_normalization_eps,
                "current_focus_alpha": args.current_focus_alpha,
                "current_focus_gamma": args.current_focus_gamma,
                "target_norm_mean": (
                    [float(v) for v in target_stats.mean.tolist()]
                    if target_stats is not None else None
                ),
                "target_norm_std": (
                    [float(v) for v in target_stats.std.tolist()]
                    if target_stats is not None else None
                ),
            },
        )
        LOG.info("SymMGN checkpoint saved: %s", ckpt_path)

    LOG.info("Training complete.")


if __name__ == "__main__":
    main()
