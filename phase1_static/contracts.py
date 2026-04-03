"""Phase 1 contract definitions and pure normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


TARGET_CHANNEL_ORDER: Tuple[str, str, str, str] = ("Bx", "By", "A", "J")

STEP_SEMANTICS_TO_CODE = {
    "explicit_time_step": 0,
    "implicit_single_step": 1,
    "derived_static_step": 2,
}

FIDELITY_TYPE_TO_CODE = {
    "OnLoadTorque": 0,
    "StaticLoad": 1,
    "StaticOC": 2,
    "StaticLoadInductance": 3,
    "LossElement_OnLoadLoss": 4,
    "Unknown": 99,
}

COUPLING_POLICY_TO_CODE = {
    "transient_coupled": 0,
    "weak_coupled": 1,
    "decoupled": 2,
}

COUPLING_POLICY_CURL_SCALE: Dict[int, float] = {
    0: 1.0,
    1: 0.5,
    2: 0.0,
}

FIDELITY_ROUTING_PROFILE: Dict[int, Dict[str, float]] = {
    FIDELITY_TYPE_TO_CODE["OnLoadTorque"]: {
        "a_scale": 1.0,
        "b_scale": 1.0,
        "sample_weight_mult": 1.0,
    },
    FIDELITY_TYPE_TO_CODE["StaticLoad"]: {
        "a_scale": 1.0,
        "b_scale": 1.0,
        "sample_weight_mult": 1.0,
    },
    FIDELITY_TYPE_TO_CODE["StaticOC"]: {
        "a_scale": 0.9,
        "b_scale": 1.0,
        "sample_weight_mult": 1.0,
    },
    FIDELITY_TYPE_TO_CODE["StaticLoadInductance"]: {
        "a_scale": 0.8,
        "b_scale": 1.0,
        "sample_weight_mult": 1.0,
    },
    FIDELITY_TYPE_TO_CODE["LossElement_OnLoadLoss"]: {
        "a_scale": 1.0,
        "b_scale": 1.1,
        "sample_weight_mult": 1.0,
    },
    FIDELITY_TYPE_TO_CODE["Unknown"]: {
        "a_scale": 1.0,
        "b_scale": 1.0,
        "sample_weight_mult": 1.0,
    },
}


@dataclass(frozen=True)
class Phase1Contract:
    """Immutable contract declaration for Phase 1 tensor interfaces."""

    target_channel_order: Tuple[str, str, str, str] = TARGET_CHANNEL_ORDER
    interior_edge_sign: float = 1.0
    pbc_anti_periodic_sign: float = -1.0
    spatial_dim: int = 2
    temporal_feature_order: Tuple[str, str, str, str] = ("time_s", "rotate_step", "dt_s", "step_index")


@dataclass(frozen=True)
class FidelityObservationContract:
    """Immutable contract for multi-fidelity sample semantics."""

    default_step_semantics: str = "explicit_time_step"
    default_fidelity_type: str = "Unknown"
    default_coupling_policy: str = "weak_coupled"


@dataclass(frozen=True)
class BatchBoundaryContract:
    """Immutable loader-to-training boundary contract."""

    edge_index_rows: int = 2
    target_channels: int = 4
    allowed_edge_signs: Tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True)
class LossBoundaryContract:
    """Immutable training-to-loss boundary contract."""

    min_channels: int = 3
    expected_rank: int = 2


@dataclass(frozen=True)
class BatchRoutingPolicyContract:
    """Immutable routing policy at the batch boundary."""

    strict_batch_homogeneous_metadata: bool = True


def validate_graph_batch_contract(
    batch,
    spatial_dim: int,
    contract: BatchBoundaryContract = BatchBoundaryContract(),
) -> str:
    """Validate graph batch shape/order contract and return detected target schema."""
    if getattr(batch, "x", None) is None or batch.x.dim() != 2:
        raise ValueError(f"Contract gate failed: x must be [N,F], got {tuple(batch.x.shape)}")
    if getattr(batch, "pos", None) is None or batch.pos.dim() != 2:
        raise ValueError(f"Contract gate failed: pos must be [N,D], got {tuple(batch.pos.shape)}")
    if getattr(batch, "edge_index", None) is None:
        raise ValueError("Contract gate failed: edge_index is missing")
    if batch.edge_index.dim() != 2 or batch.edge_index.shape[0] != contract.edge_index_rows:
        raise ValueError(
            f"Contract gate failed: edge_index must be [2,E], got {tuple(batch.edge_index.shape)}"
        )
    if getattr(batch, "edge_attr", None) is None:
        raise ValueError("Contract gate failed: edge_attr is missing")
    if batch.edge_attr.dim() != 2 or batch.edge_attr.shape[0] != batch.edge_index.shape[1]:
        raise ValueError(
            "Contract gate failed: edge_attr must be [E,C] and aligned with edge_index, "
            f"got edge_attr={tuple(batch.edge_attr.shape)} edge_index={tuple(batch.edge_index.shape)}"
        )
    if getattr(batch, "y", None) is None:
        raise ValueError("Contract gate failed: y is missing")

    y_norm, schema = normalize_channels_to_bx_by_a_j(batch.y)
    if y_norm.shape[1] != contract.target_channels:
        raise ValueError(
            "Contract gate failed: canonical y channel count mismatch, "
            f"expected {contract.target_channels} got {y_norm.shape[1]}"
        )
    if batch.pos.shape[1] != int(spatial_dim):
        raise ValueError(
            f"Contract gate failed: pos spatial_dim mismatch, expected {spatial_dim}, "
            f"got {batch.pos.shape[1]}"
        )

    unique_edge_signs = torch.unique(batch.edge_attr[:, 0]).detach().cpu().tolist()
    allowed = set(float(v) for v in contract.allowed_edge_signs)
    got = set(float(v) for v in unique_edge_signs)
    if not got.issubset(allowed):
        raise ValueError(
            "Contract gate failed: edge_attr contains unsupported signs, "
            f"got={sorted(got)} allowed={sorted(allowed)}"
        )
    return schema


def validate_loss_boundary_inputs(
    pred: torch.Tensor,
    target: torch.Tensor,
    coords: torch.Tensor,
    spatial_dim: int,
    contract: LossBoundaryContract = LossBoundaryContract(),
) -> None:
    """Validate tensors at the training-to-loss boundary."""
    if pred.dim() != contract.expected_rank or target.dim() != contract.expected_rank:
        raise ValueError("pred/target must be rank-2 tensors [N, C]")
    if pred.shape[0] != target.shape[0]:
        raise ValueError("pred and target must share the same N dimension")
    if pred.shape[1] < contract.min_channels or target.shape[1] < contract.min_channels:
        raise ValueError(
            f"pred/target must have at least {contract.min_channels} channels at loss boundary"
        )
    if coords.dim() != contract.expected_rank:
        raise ValueError("coords must be rank-2 tensor [N, D]")
    if coords.shape[0] != pred.shape[0]:
        raise ValueError("coords and pred must share the same N dimension")
    if coords.shape[1] != int(spatial_dim):
        raise ValueError(
            f"coords spatial_dim mismatch at loss boundary, expected {spatial_dim} got {coords.shape[1]}"
        )


def validate_batch_fidelity_policy(
    batch,
    contract: BatchRoutingPolicyContract = BatchRoutingPolicyContract(),
) -> Tuple[int, int]:
    """Validate batch-level fidelity/coupling routing policy and return codes.

    In strict mode, one batch must contain a single fidelity_type and coupling_policy.
    """

    fidelity = getattr(batch, "fidelity_type", None)
    coupling = getattr(batch, "coupling_policy", None)

    if fidelity is None:
        fidelity_codes = torch.tensor([FIDELITY_TYPE_TO_CODE["Unknown"]], dtype=torch.long)
    elif isinstance(fidelity, torch.Tensor):
        fidelity_codes = fidelity.long().view(-1)
    else:
        fidelity_codes = torch.tensor([int(fidelity)], dtype=torch.long)

    if coupling is None:
        coupling_codes = torch.tensor([COUPLING_POLICY_TO_CODE["weak_coupled"]], dtype=torch.long)
    elif isinstance(coupling, torch.Tensor):
        coupling_codes = coupling.long().view(-1)
    else:
        coupling_codes = torch.tensor([int(coupling)], dtype=torch.long)

    if fidelity_codes.numel() == 0:
        fidelity_codes = torch.tensor([FIDELITY_TYPE_TO_CODE["Unknown"]], dtype=torch.long)
    if coupling_codes.numel() == 0:
        coupling_codes = torch.tensor([COUPLING_POLICY_TO_CODE["weak_coupled"]], dtype=torch.long)

    fidelity_unique = torch.unique(fidelity_codes)
    coupling_unique = torch.unique(coupling_codes)

    if contract.strict_batch_homogeneous_metadata:
        if fidelity_unique.numel() > 1:
            raise ValueError(
                "Batch routing policy violation: mixed fidelity_type in one batch is not allowed in strict mode"
            )
        if coupling_unique.numel() > 1:
            raise ValueError(
                "Batch routing policy violation: mixed coupling_policy in one batch is not allowed in strict mode"
            )

    return int(fidelity_unique[0].item()), int(coupling_unique[0].item())


def resolve_loss_routing_from_codes(
    fidelity_code: int,
    coupling_code: int,
) -> Tuple[float, float, float, float]:
    """Resolve fidelity/coupling metadata into routing scales.

    Returns:
        (a_scale, b_scale, curl_scale, sample_weight_mult)
    """

    profile = FIDELITY_ROUTING_PROFILE.get(
        int(fidelity_code),
        FIDELITY_ROUTING_PROFILE[FIDELITY_TYPE_TO_CODE["Unknown"]],
    )
    a_scale = float(profile["a_scale"])
    b_scale = float(profile["b_scale"])
    sample_weight_mult = float(profile["sample_weight_mult"])
    curl_scale = float(COUPLING_POLICY_CURL_SCALE.get(int(coupling_code), 1.0))
    return a_scale, b_scale, curl_scale, sample_weight_mult


def encode_fidelity_metadata(
    step_semantics: str,
    fidelity_type: str,
    coupling_policy: str,
) -> Tuple[int, int, int]:
    """Encode fidelity metadata strings into stable integer codes."""
    step_code = int(STEP_SEMANTICS_TO_CODE.get(step_semantics, STEP_SEMANTICS_TO_CODE["derived_static_step"]))
    fidelity_code = int(FIDELITY_TYPE_TO_CODE.get(fidelity_type, FIDELITY_TYPE_TO_CODE["Unknown"]))
    coupling_code = int(COUPLING_POLICY_TO_CODE.get(coupling_policy, COUPLING_POLICY_TO_CODE["weak_coupled"]))
    return step_code, fidelity_code, coupling_code


def normalize_channels_to_bx_by_a_j(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """Normalize tensor channels to canonical order [Bx, By, A, J].

    Supported source layouts:
    - 4+ channels: assumes first four are already [Bx, By, A, J]
    - 3 channels:  [A, Bx, By] -> [Bx, By, A, J=0]
    - 2 channels:  [Bx, By]     -> [Bx, By, A=0, J=0]
    - 1 channel:   [A]          -> [Bx=0, By=0, A, J=0]
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise ValueError(f"Expected tensor with shape [N, C], got {tuple(tensor.shape)}")

    channels = tensor.shape[1]
    if channels >= 4:
        return tensor[:, 0:4], "bx_by_a_j"
    if channels == 3:
        a = tensor[:, 0:1]
        bx = tensor[:, 1:2]
        by = tensor[:, 2:3]
        j = torch.zeros_like(a)
        return torch.cat([bx, by, a, j], dim=1), "a_bx_by"
    if channels == 2:
        bx = tensor[:, 0:1]
        by = tensor[:, 1:2]
        z = torch.zeros_like(bx)
        return torch.cat([bx, by, z, z], dim=1), "bx_by"
    if channels == 1:
        a = tensor[:, 0:1]
        z = torch.zeros_like(a)
        return torch.cat([z, z, a, z], dim=1), "a_only"
    raise ValueError("Channel count must be >= 1")
