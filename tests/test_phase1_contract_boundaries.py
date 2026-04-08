import pytest
import torch
from types import SimpleNamespace

from phase1_static.contracts import (
    encode_fidelity_metadata,
    normalize_channels_to_bx_by_a_j,
    resolve_loss_routing_from_codes,
    validate_batch_fidelity_policy,
    validate_graph_batch_contract,
    validate_loss_boundary_inputs,
)


def test_channel_normalization_contract_a_bx_by_to_bx_by_a_j() -> None:
    y = torch.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], dtype=torch.float32)
    norm, schema = normalize_channels_to_bx_by_a_j(y)

    assert schema == "a_bx_by"
    assert norm.shape == (2, 5)
    assert torch.allclose(norm[:, 0], torch.tensor([1.0, 4.0]))
    assert torch.allclose(norm[:, 1], torch.tensor([2.0, 5.0]))
    assert torch.allclose(norm[:, 2], torch.tensor([3.0, 6.0]))
    assert torch.allclose(norm[:, 3], torch.zeros(2))
    assert torch.allclose(norm[:, 4], torch.zeros(2))


def test_loader_to_training_boundary_contract_accepts_valid_batch() -> None:
    tg_data = pytest.importorskip("torch_geometric.data")
    Data = tg_data.Data

    batch = Data(
        x=torch.randn(3, 5),
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0], [-1.0], [1.0]], dtype=torch.float32),
        y=torch.randn(3, 5),
    )

    schema = validate_graph_batch_contract(batch, spatial_dim=2)
    assert schema == "bx_by_a_j_je"


def test_training_to_loss_boundary_contract_rejects_bad_coords() -> None:
    pred = torch.randn(4, 5)
    target = torch.randn(4, 5)
    coords = torch.randn(3, 2)

    with pytest.raises(ValueError, match="same N dimension"):
        validate_loss_boundary_inputs(
            pred=pred,
            target=target,
            coords=coords,
            spatial_dim=2,
        )


def test_encode_fidelity_metadata_contract_codes() -> None:
    step_code, fidelity_code, coupling_code = encode_fidelity_metadata(
        step_semantics="implicit_single_step",
        fidelity_type="StaticOC",
        coupling_policy="weak_coupled",
    )

    assert step_code == 1
    assert fidelity_code == 2
    assert coupling_code == 1


def test_fidelity_routing_contract_changes_a_scale_by_fidelity_type() -> None:
    _, fidelity_onload, coupling_weak = encode_fidelity_metadata(
        step_semantics="explicit_time_step",
        fidelity_type="OnLoadTorque",
        coupling_policy="weak_coupled",
    )
    _, fidelity_static_ind, _ = encode_fidelity_metadata(
        step_semantics="implicit_single_step",
        fidelity_type="StaticLoadInductance",
        coupling_policy="weak_coupled",
    )

    a_on, b_on, curl_on, _ = resolve_loss_routing_from_codes(fidelity_onload, coupling_weak)
    a_si, b_si, curl_si, _ = resolve_loss_routing_from_codes(fidelity_static_ind, coupling_weak)

    assert a_on == 1.0
    assert a_si == 0.8
    assert b_on == b_si == 1.0
    assert curl_on == curl_si == 0.5


def test_coupling_policy_contract_maps_curl_scale() -> None:
    _, fidelity_code, cp_transient = encode_fidelity_metadata(
        step_semantics="explicit_time_step",
        fidelity_type="StaticLoad",
        coupling_policy="transient_coupled",
    )
    _, _, cp_decoupled = encode_fidelity_metadata(
        step_semantics="explicit_time_step",
        fidelity_type="StaticLoad",
        coupling_policy="decoupled",
    )

    _, _, curl_transient, _ = resolve_loss_routing_from_codes(fidelity_code, cp_transient)
    _, _, curl_decoupled, _ = resolve_loss_routing_from_codes(fidelity_code, cp_decoupled)

    assert curl_transient == 1.0
    assert curl_decoupled == 0.0


def test_strict_batch_policy_rejects_mixed_fidelity_codes() -> None:
    mixed_batch = SimpleNamespace(
        fidelity_type=torch.tensor([0, 2], dtype=torch.long),
        coupling_policy=torch.tensor([1, 1], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="mixed fidelity_type"):
        validate_batch_fidelity_policy(mixed_batch)


def test_strict_batch_policy_rejects_mixed_coupling_codes() -> None:
    mixed_batch = SimpleNamespace(
        fidelity_type=torch.tensor([1, 1], dtype=torch.long),
        coupling_policy=torch.tensor([0, 2], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="mixed coupling_policy"):
        validate_batch_fidelity_policy(mixed_batch)
