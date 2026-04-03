"""Training script for Phase 1 static 1/8 motor model with hybrid physics loss."""

from __future__ import annotations

import argparse
import logging
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .contracts import (
    normalize_channels_to_bx_by_a_j,
    resolve_loss_routing_from_codes,
    validate_batch_fidelity_policy,
    validate_graph_batch_contract,
)
from .custom_mgn import AntiPeriodicMessagePassing
from .loss import hybrid_physics_loss, lambda_anneal
from .motor_dataset import StaticMotorDataset, build_samples_from_doe_manifest, build_samples_from_npz
from .physics_operators import PhysicsOperator, build_physics_operator

LOG = logging.getLogger(__name__)


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


class SimpleAntiPeriodicNet(nn.Module):
    """Lightweight anti-periodic GNN fallback when PhysicsNeMo is unavailable."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 4) -> None:
        super().__init__()
        self.mp1 = AntiPeriodicMessagePassing(in_channels=input_dim * 2 + 1, out_channels=hidden_dim)
        self.mp2 = AntiPeriodicMessagePassing(in_channels=hidden_dim * 2 + 1, out_channels=hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, data) -> torch.Tensor:
        h1 = self.mp1(data.x, data.edge_index, data.edge_attr)
        h2 = self.mp2(h1, data.edge_index, data.edge_attr)
        return self.out(h2)


def build_model(input_dim: int, hidden_dim: int, output_dim: int, use_physicsnemo: bool):
    if use_physicsnemo:
        try:
            from physicsnemo.models.meshgraphnet import MeshGraphNet
        except Exception as exc:  # pragma: no cover - optional dependency
            LOG.warning("physicsnemo not available, falling back to SimpleAntiPeriodicNet: %s", exc)
            use_physicsnemo = False

    if use_physicsnemo:
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

    return SimpleAntiPeriodicNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 static 1/8 training")
    p.add_argument("--input-format", choices=("auto", "npz", "doe"), default="auto")
    p.add_argument("--data", help="Path to npz/pt bundle (used for --input-format npz/auto)")
    p.add_argument("--data-dir", default="doe_data", help="DOE root dir containing doe_manifest.json")
    p.add_argument("--max-steps-per-case", type=int, default=None, help="Optional cap for DOE timesteps per case")
    p.add_argument(
        "--include-temporal-features",
        action="store_true",
        help="Append [time_s, rotate_step, dt_s, step_index] to node input features",
    )
    p.add_argument("--spatial-dim", type=int, default=2, help="Spatial coordinate dimension (2 supported now)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42, help="Deterministic seed for smoke/overfit reproducibility")
    p.add_argument("--smoke-max-samples", type=int, default=0, help="If >0, run a deterministic smoke slice")
    p.add_argument("--contract-check-only", action="store_true", help="Validate contracts and exit")
    p.add_argument("--weight-a", type=float, default=1.0, help="Supervised A loss weight")
    p.add_argument("--weight-b", type=float, default=1.0, help="Supervised Bx/By loss weight")
    p.add_argument("--curl-base", type=float, default=0.05, help="Initial curl consistency loss weight")
    p.add_argument("--curl-max", type=float, default=1.0, help="Maximum curl consistency loss weight")
    p.add_argument("--curl-warmup", type=int, default=10, help="Warmup epochs for curl weight annealing")
    p.add_argument("--overfit-single", action="store_true", help="Repeat the first batch to check overfitting")
    p.add_argument("--overfit-target", type=float, default=1e-4, help="Pass threshold for overfit-single train loss")
    p.add_argument("--use-physicsnemo", action="store_true", help="Force PhysicsNeMo MeshGraphNet")
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
    return normalize_channels_to_bx_by_a_j(y)


def _normalize_prediction_channels(pred: torch.Tensor) -> torch.Tensor:
    norm, _ = normalize_channels_to_bx_by_a_j(pred)
    return norm


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
    w_curl: float,
) -> Tuple[float, float, float, float]:
    """Resolve effective loss weights from schema and batch fidelity metadata."""
    fidelity_code, coupling_code = validate_batch_fidelity_policy(batch)
    a_scale, b_scale, curl_scale, sample_weight_mult = resolve_loss_routing_from_codes(
        fidelity_code=fidelity_code,
        coupling_code=coupling_code,
    )

    w_a_eff = float(w_a) * a_scale
    if schema == "bx_by":
        w_a_eff = 0.0

    w_b_eff = float(w_b) * b_scale
    w_curl_eff = float(w_curl) * curl_scale
    sample_weight_eff = _resolve_sample_weight(batch) * sample_weight_mult
    return w_a_eff, w_b_eff, w_curl_eff, sample_weight_eff


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
        return build_samples_from_doe_manifest(args.data_dir, args.max_steps_per_case), "doe"
    if fmt == "npz":
        if not args.data:
            raise ValueError("--data is required when --input-format=npz")
        return build_samples_from_npz(args.data), "npz"
    if fmt == "doe":
        return build_samples_from_doe_manifest(args.data_dir, args.max_steps_per_case), "doe"
    raise ValueError(f"Unsupported input format: {fmt}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    operator: PhysicsOperator,
    w_a: float,
    w_b: float,
    w_curl: float,
) -> Dict[str, float]:
    model.train()
    sums = {"total_loss": 0.0, "a_loss": 0.0, "b_loss": 0.0, "curl_loss": 0.0}
    steps = 0
    for batch in loader:
        batch = batch.to(device)
        batch.pos = batch.pos.detach().requires_grad_(True)
        _attach_pos_to_x(batch)

        pred = _normalize_prediction_channels(forward_model(model, batch))
        target, schema = _normalize_target_channels(batch.y)
        w_a_eff, w_b_eff, w_curl_eff, sample_weight = _resolve_loss_routing_from_batch(
            batch=batch,
            schema=schema,
            w_a=w_a,
            w_b=w_b,
            w_curl=w_curl,
        )

        total_loss, metrics = hybrid_physics_loss(
            pred=pred,
            target=target,
            coords=batch.pos,
            operator=operator,
            w_a=w_a_eff,
            w_b=w_b_eff,
            w_curl=w_curl_eff,
            retain_graph=True,
        )
        total_loss = total_loss * sample_weight

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sums["total_loss"] += float(total_loss.detach().item())
        sums["a_loss"] += float(metrics["a_loss"].item())
        sums["b_loss"] += float(metrics["b_loss"].item())
        sums["curl_loss"] += float(metrics["curl_loss"].item())
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
    w_curl: float,
) -> Dict[str, float]:
    # curl(A) requires gradient wrt coordinates, so we cannot wrap with torch.no_grad().
    model.eval()
    sums = {"total_loss": 0.0, "a_loss": 0.0, "b_loss": 0.0, "curl_loss": 0.0}
    steps = 0
    for batch in loader:
        batch = batch.to(device)
        batch.pos = batch.pos.detach().requires_grad_(True)
        _attach_pos_to_x(batch)

        with torch.set_grad_enabled(True):
            pred = _normalize_prediction_channels(forward_model(model, batch))
            target, schema = _normalize_target_channels(batch.y)
            w_a_eff, w_b_eff, w_curl_eff, sample_weight = _resolve_loss_routing_from_batch(
                batch=batch,
                schema=schema,
                w_a=w_a,
                w_b=w_b,
                w_curl=w_curl,
            )
            _, metrics = hybrid_physics_loss(
                pred=pred,
                target=target,
                coords=batch.pos,
                operator=operator,
                w_a=w_a_eff,
                w_b=w_b_eff,
                w_curl=w_curl_eff,
                retain_graph=False,
            )
        sums["total_loss"] += float(metrics["total_loss"].item() * sample_weight)
        sums["a_loss"] += float(metrics["a_loss"].item())
        sums["b_loss"] += float(metrics["b_loss"].item())
        sums["curl_loss"] += float(metrics["curl_loss"].item())
        steps += 1
    denom = max(steps, 1)
    return {k: v / denom for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_deterministic_seed(args.seed)

    samples, resolved_fmt = _load_samples(args)
    if args.smoke_max_samples and args.smoke_max_samples > 0:
        samples = samples[: args.smoke_max_samples]
        LOG.info("Smoke slice enabled: using first %d samples", len(samples))

    dataset = StaticMotorDataset(samples, include_temporal_features=args.include_temporal_features)
    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=not args.overfit_single, generator=train_gen)
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

    input_dim = dataset[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=4,
        use_physicsnemo=args.use_physicsnemo,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    first_batch: Optional[DataLoader] = None
    if args.overfit_single:
        first_batch = DataLoader([dataset[0]], batch_size=1, shuffle=False)

    best_train_total = float("inf")
    for epoch in range(1, args.epochs + 1):
        curl_w = lambda_anneal(epoch, args.curl_warmup, args.curl_base, args.curl_max)
        train_loader_epoch = first_batch if first_batch is not None else train_loader
        train_metrics = train_epoch(
            model=model,
            loader=train_loader_epoch,
            optimizer=optimizer,
            device=device,
            operator=operator,
            w_a=args.weight_a,
            w_b=args.weight_b,
            w_curl=curl_w,
        )
        best_train_total = min(best_train_total, train_metrics["total_loss"])
        val_metrics = eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
            operator=operator,
            w_a=args.weight_a,
            w_b=args.weight_b,
            w_curl=curl_w,
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            LOG.info(
                "epoch=%d curl_w=%.3f train(total=%.6f,a=%.6f,b=%.6f,curl=%.6f) "
                "val(total=%.6f,a=%.6f,b=%.6f,curl=%.6f)",
                epoch,
                curl_w,
                train_metrics["total_loss"],
                train_metrics["a_loss"],
                train_metrics["b_loss"],
                train_metrics["curl_loss"],
                val_metrics["total_loss"],
                val_metrics["a_loss"],
                val_metrics["b_loss"],
                val_metrics["curl_loss"],
            )

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

    LOG.info("Training complete.")


if __name__ == "__main__":
    main()
