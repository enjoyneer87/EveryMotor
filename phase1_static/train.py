"""Training script for Phase 1 static 1/8 motor model with hybrid physics loss."""

from __future__ import annotations

import argparse
import logging
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .custom_mgn import AntiPeriodicMessagePassing
from .loss import hybrid_physics_loss, lambda_anneal
from .motor_dataset import StaticMotorDataset, build_samples_from_npz

LOG = logging.getLogger(__name__)


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
    p.add_argument("--data", required=True, help="Path to npz/pt bundle containing samples")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--weight-a", type=float, default=1.0, help="Supervised A loss weight")
    p.add_argument("--weight-b", type=float, default=1.0, help="Supervised Bx/By loss weight")
    p.add_argument("--curl-base", type=float, default=0.05, help="Initial curl consistency loss weight")
    p.add_argument("--curl-max", type=float, default=1.0, help="Maximum curl consistency loss weight")
    p.add_argument("--curl-warmup", type=int, default=10, help="Warmup epochs for curl weight annealing")
    p.add_argument("--overfit-single", action="store_true", help="Repeat the first batch to check overfitting")
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
    """Normalize target channels to [Bx, By, A, J]."""
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    if y.dim() != 2:
        raise ValueError(f"Expected y with shape [N, C], got {tuple(y.shape)}")

    channels = y.shape[1]
    if channels >= 4:
        return y[:, 0:4], "bx_by_a_j"
    if channels == 3:
        # Backward-compatible fallback: [A, Bx, By] -> [Bx, By, A, J=0]
        a = y[:, 0:1]
        bx = y[:, 1:2]
        by = y[:, 2:3]
        j = torch.zeros_like(a)
        return torch.cat([bx, by, a, j], dim=1), "a_bx_by"
    if channels == 2:
        # Fallback: [Bx, By] -> [Bx, By, A=0, J=0]
        bx = y[:, 0:1]
        by = y[:, 1:2]
        z = torch.zeros_like(bx)
        return torch.cat([bx, by, z, z], dim=1), "bx_by"
    raise ValueError(f"Unsupported y channel count: {channels}")


def _normalize_prediction_channels(pred: torch.Tensor) -> torch.Tensor:
    """Normalize model output channels to [Bx, By, A, J]."""
    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
    if pred.dim() != 2:
        raise ValueError(f"Expected prediction with shape [N, C], got {tuple(pred.shape)}")

    channels = pred.shape[1]
    if channels >= 4:
        return pred[:, 0:4]
    if channels == 3:
        a = pred[:, 0:1]
        bx = pred[:, 1:2]
        by = pred[:, 2:3]
        j = torch.zeros_like(a)
        return torch.cat([bx, by, a, j], dim=1)
    if channels == 2:
        bx = pred[:, 0:1]
        by = pred[:, 1:2]
        z = torch.zeros_like(bx)
        return torch.cat([bx, by, z, z], dim=1)
    if channels == 1:
        a = pred[:, 0:1]
        z = torch.zeros_like(a)
        return torch.cat([z, z, a, z], dim=1)
    raise ValueError(f"Unsupported prediction channel count: {channels}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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

        pred = _normalize_prediction_channels(forward_model(model, batch))
        target, schema = _normalize_target_channels(batch.y)

        w_a_eff = float(w_a) if schema != "bx_by" else 0.0

        total_loss, metrics = hybrid_physics_loss(
            pred=pred,
            target=target,
            coords=batch.pos,
            w_a=w_a_eff,
            w_b=float(w_b),
            w_curl=float(w_curl),
            retain_graph=False,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for key in sums:
            sums[key] += float(metrics[key].item())
        steps += 1
    denom = max(steps, 1)
    return {k: v / denom for k, v in sums.items()}


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
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

        with torch.set_grad_enabled(True):
            pred = _normalize_prediction_channels(forward_model(model, batch))
            target, schema = _normalize_target_channels(batch.y)
            w_a_eff = float(w_a) if schema != "bx_by" else 0.0
            _, metrics = hybrid_physics_loss(
                pred=pred,
                target=target,
                coords=batch.pos,
                w_a=w_a_eff,
                w_b=float(w_b),
                w_curl=float(w_curl),
                retain_graph=False,
            )
        for key in sums:
            sums[key] += float(metrics[key].item())
        steps += 1
    denom = max(steps, 1)
    return {k: v / denom for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    samples = build_samples_from_npz(args.data)
    dataset = StaticMotorDataset(samples)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=not args.overfit_single)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

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

    for epoch in range(1, args.epochs + 1):
        curl_w = lambda_anneal(epoch, args.curl_warmup, args.curl_base, args.curl_max)
        train_loader_epoch = first_batch if first_batch is not None else train_loader
        train_metrics = train_epoch(
            model=model,
            loader=train_loader_epoch,
            optimizer=optimizer,
            device=device,
            w_a=args.weight_a,
            w_b=args.weight_b,
            w_curl=curl_w,
        )
        val_metrics = eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
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

    LOG.info("Training complete.")


if __name__ == "__main__":
    main()
