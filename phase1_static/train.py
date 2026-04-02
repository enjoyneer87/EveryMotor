"""Training script for Phase 1 static 1/8 motor model with physics-informed loss."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .custom_mgn import AntiPeriodicMessagePassing
from .loss import lambda_anneal, physics_informed_loss
from .motor_dataset import StaticMotorDataset, build_samples_from_npz

LOG = logging.getLogger(__name__)


class SimpleAntiPeriodicNet(nn.Module):
    """Lightweight anti-periodic GNN fallback when PhysicsNeMo is unavailable."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mp1 = AntiPeriodicMessagePassing(in_channels=input_dim * 2 + 1, out_channels=hidden_dim)
        self.mp2 = AntiPeriodicMessagePassing(in_channels=hidden_dim * 2 + 1, out_channels=hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        h1 = self.mp1(data.x, data.edge_index, data.edge_attr)
        h2 = self.mp2(h1, data.edge_index, data.edge_attr)
        return self.out(h2)


def build_model(input_dim: int, hidden_dim: int, use_physicsnemo: bool):
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
            output_dim=1,
            processor_size=10,
            hidden_dim_processor=hidden_dim,
            hidden_dim_node_encoder=hidden_dim,
            hidden_dim_edge_encoder=hidden_dim,
            hidden_dim_node_decoder=hidden_dim,
            aggregation="sum",
        )

    return SimpleAntiPeriodicNet(input_dim=input_dim, hidden_dim=hidden_dim)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 static 1/8 training")
    p.add_argument("--data", required=True, help="Path to npz/pt bundle containing samples")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lambda-base", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--lambda-warmup", type=int, default=10)
    p.add_argument("--overfit-single", action="store_true", help="Repeat the first batch to check overfitting")
    p.add_argument("--use-physicsnemo", action="store_true", help="Force PhysicsNeMo MeshGraphNet")
    return p.parse_args()


def forward_model(model: nn.Module, batch) -> torch.Tensor:
    if hasattr(model, "forward") and "edge_attr" in model.forward.__code__.co_varnames:  # crude signature check
        try:
            return model(batch.x, batch.edge_attr, batch)
        except TypeError:
            return model(batch.x, batch.edge_attr, batch.edge_index)
    return model(batch)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_w: float,
) -> float:
    model.train()
    running = 0.0
    steps = 0
    for batch in loader:
        batch = batch.to(device)
        batch.pos.requires_grad_(True)
        pred_a = forward_model(model, batch).reshape(batch.y.shape[0], 1)

        total_loss, _ = physics_informed_loss(
            pred_a=pred_a,
            target_a=batch.y[:, 0:1],
            target_b=batch.y[:, 1:3],
            coords=batch.pos,
            lambda_weight=lambda_w,
            retain_graph=False,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running += total_loss.item()
        steps += 1
    return running / max(steps, 1)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, lambda_w: float) -> float:
    model.eval()
    running = 0.0
    steps = 0
    for batch in loader:
        batch = batch.to(device)
        batch.pos.requires_grad_(True)
        pred_a = forward_model(model, batch).reshape(batch.y.shape[0], 1)
        total_loss, _ = physics_informed_loss(
            pred_a=pred_a,
            target_a=batch.y[:, 0:1],
            target_b=batch.y[:, 1:3],
            coords=batch.pos,
            lambda_weight=lambda_w,
            retain_graph=False,
        )
        running += total_loss.item()
        steps += 1
    return running / max(steps, 1)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    samples = build_samples_from_npz(args.data)
    dataset = StaticMotorDataset(samples)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=not args.overfit_single)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = dataset[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(input_dim=input_dim, hidden_dim=args.hidden_dim, use_physicsnemo=args.use_physicsnemo)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    first_batch: Optional[DataLoader] = None
    if args.overfit_single:
        first_batch = DataLoader([dataset[0]], batch_size=1, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        lambda_w = lambda_anneal(epoch, args.lambda_warmup, args.lambda_base, args.lambda_max)
        train_loader_epoch = first_batch if first_batch is not None else train_loader
        train_loss = train_epoch(model, train_loader_epoch, optimizer, device, lambda_w)
        val_loss = eval_epoch(model, val_loader, device, lambda_w)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            LOG.info(
                "epoch=%d lambda=%.3f train=%.6f val=%.6f",
                epoch,
                lambda_w,
                train_loss,
                val_loss,
            )

    LOG.info("Training complete.")


if __name__ == "__main__":
    main()
