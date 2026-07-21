"""Tests for the batched curl operator in the A-prediction trainer.

`curl_node_index` holds node ids, so PyG must offset it per graph when batching.
If the `__inc__` override is wrong or missing, every graph's operator silently
reads the first graph's nodes: no error, no NaN, just wrong physics. These tests
compare the batched result against per-graph evaluation.

Requires torch + torch_geometric, so they run in the PhysicsNeMo container:
    docker run ... python -m pytest tests/test_curl_trainer_batching.py
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.loader import DataLoader  # noqa: E402

from phase1_static.discrete_curl import build_p1_curl_operator, curl_a_to_b  # noqa: E402
from train_doe_curl_mgn import CurlData, curl_from_batch  # noqa: E402


def make_graph(seed: int, n_points: int = 40):
    """A small triangulated patch wrapped as a CurlData graph."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n_points, 2))
    pts = np.vstack([pts, [[0, 0], [1, 0], [0, 1], [1, 1]]])
    simp = Delaunay(pts).simplices
    tri = (simp[:, 0], simp[:, 1], simp[:, 2])

    node_x, node_y = pts[:, 0].copy(), pts[:, 1].copy()
    op = build_p1_curl_operator(node_x, node_y, tri)

    # A distinct A field per graph so a mix-up cannot pass by coincidence.
    a = np.sin((seed + 1) * 2.0 * node_x) + np.cos((seed + 2) * node_y)

    graph = CurlData(
        x=torch.zeros((node_x.size, 1), dtype=torch.float32),
        edge_index=torch.tensor(
            np.stack([np.concatenate(tri[:2]), np.concatenate(tri[1:])], axis=0), dtype=torch.long
        ),
        curl_node_index=torch.from_numpy(op.node_index.astype(np.int64)),
        curl_coef_bx=torch.from_numpy(op.coef_bx.astype(np.float64)),
        curl_coef_by=torch.from_numpy(op.coef_by.astype(np.float64)),
        a_nodal=torch.from_numpy(a),
        num_nodes=int(node_x.size),
    )
    return graph, op, a


def test_batched_curl_matches_per_graph_curl():
    graphs, ops, fields = [], [], []
    for seed in range(4):
        g, op, a = make_graph(seed, n_points=30 + 7 * seed)
        graphs.append(g)
        ops.append(op)
        fields.append(a)

    batch = next(iter(DataLoader(graphs, batch_size=4, shuffle=False)))
    batched = curl_from_batch(batch, batch.a_nodal).numpy()

    expected = np.concatenate([curl_a_to_b(op, a) for op, a in zip(ops, fields)], axis=0)
    assert batched.shape == expected.shape
    np.testing.assert_allclose(batched, expected, rtol=1e-9, atol=1e-12)


def test_node_index_is_offset_per_graph():
    """The offset is what makes graph 2 read graph 2's nodes."""
    g0, _, _ = make_graph(0, n_points=30)
    g1, _, _ = make_graph(1, n_points=35)
    batch = next(iter(DataLoader([g0, g1], batch_size=2, shuffle=False)))

    n0 = int(g0.num_nodes)
    rows0 = g0.curl_node_index.shape[0]

    # First graph's operator is untouched; the second is shifted by n0.
    assert torch.equal(batch.curl_node_index[:rows0], g0.curl_node_index)
    assert torch.equal(batch.curl_node_index[rows0:], g1.curl_node_index + n0)
    assert int(batch.curl_node_index.max()) < int(batch.num_nodes)


def test_batching_is_not_a_no_op_that_hides_a_bug():
    """Guard the guard: without the offset the result would actually differ."""
    g0, op0, a0 = make_graph(0, n_points=30)
    g1, op1, a1 = make_graph(1, n_points=35)
    batch = next(iter(DataLoader([g0, g1], batch_size=2, shuffle=False)))

    rows0 = g0.curl_node_index.shape[0]
    wrong = curl_from_batch(
        type(batch)(
            curl_node_index=torch.cat([g0.curl_node_index, g1.curl_node_index]),
            curl_coef_bx=batch.curl_coef_bx,
            curl_coef_by=batch.curl_coef_by,
        ),
        batch.a_nodal,
    ).numpy()
    right = curl_from_batch(batch, batch.a_nodal).numpy()

    assert not np.allclose(wrong[rows0:], right[rows0:]), (
        "un-offset indices produced the same answer; the test cannot detect the bug"
    )


def test_single_graph_batch_matches_direct_application():
    g, op, a = make_graph(7, n_points=50)
    batch = next(iter(DataLoader([g], batch_size=1, shuffle=False)))
    np.testing.assert_allclose(
        curl_from_batch(batch, batch.a_nodal).numpy(), curl_a_to_b(op, a), rtol=1e-9
    )


def test_gradients_reach_every_graph_in_the_batch():
    graphs = [make_graph(s, n_points=30)[0] for s in range(3)]
    batch = next(iter(DataLoader(graphs, batch_size=3, shuffle=False)))

    a = batch.a_nodal.clone().requires_grad_(True)
    curl_from_batch(batch, a).pow(2).sum().backward()

    assert a.grad is not None
    offsets = np.cumsum([0] + [int(g.num_nodes) for g in graphs])
    for i in range(len(graphs)):
        chunk = a.grad[offsets[i] : offsets[i + 1]]
        assert torch.isfinite(chunk).all()
        assert chunk.abs().sum() > 0, f"graph {i} received no gradient"


def test_node_average_matches_per_element_mean():
    """The --target B path: element value is the mean of its three nodal values."""
    from train_doe_curl_mgn import average_nodes_to_elements

    graphs, fields = [], []
    for seed in range(3):
        g, _, _ = make_graph(seed, n_points=30 + 5 * seed)
        n = int(g.num_nodes)
        b = torch.stack(
            [torch.linspace(0, 1, n, dtype=torch.float64) * (seed + 1),
             torch.linspace(1, 2, n, dtype=torch.float64) * (seed + 1)],
            dim=1,
        )
        g.b_nodal = b
        graphs.append(g)
        fields.append(b)

    batch = next(iter(DataLoader(graphs, batch_size=3, shuffle=False)))
    got = average_nodes_to_elements(batch, batch.b_nodal).numpy()

    expected = []
    for g, b in zip(graphs, fields):
        idx = g.curl_node_index.numpy()
        expected.append(b.numpy()[idx].mean(axis=1))
    np.testing.assert_allclose(got, np.concatenate(expected, axis=0), rtol=1e-12)


def test_both_targets_score_the_same_elements():
    """A/B fairness: curl and node-average produce values on one element set."""
    from train_doe_curl_mgn import average_nodes_to_elements

    g, op, a = make_graph(4, n_points=45)
    n = int(g.num_nodes)
    g.b_nodal = torch.zeros((n, 2), dtype=torch.float64)
    batch = next(iter(DataLoader([g], batch_size=1, shuffle=False)))

    curl_out = curl_from_batch(batch, batch.a_nodal)
    avg_out = average_nodes_to_elements(batch, batch.b_nodal)
    assert curl_out.shape == avg_out.shape == (op.n_valid, 2)


def test_uniform_nodal_field_averages_to_itself():
    from train_doe_curl_mgn import average_nodes_to_elements

    g, _, _ = make_graph(1, n_points=25)
    n = int(g.num_nodes)
    g.b_nodal = torch.full((n, 2), 0.75, dtype=torch.float64)
    batch = next(iter(DataLoader([g], batch_size=1, shuffle=False)))
    np.testing.assert_allclose(average_nodes_to_elements(batch, batch.b_nodal).numpy(), 0.75)


def _weighted_mse(pred, true, weight):
    """Reference weighted MSE matching the trainer's normalization."""
    sq = (pred - true) ** 2
    w = weight[:, None]
    return float((w * sq).sum() / (w.sum() * sq.shape[1]))


def test_uniform_weights_reduce_to_plain_mse():
    """airgap_weight=1 must leave the loss numerically unchanged."""
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(50, 2))
    true = rng.normal(size=(50, 2))
    w = np.ones(50)
    assert _weighted_mse(pred, true, w) == pytest.approx(float(((pred - true) ** 2).mean()))


def test_weighting_moves_the_loss_toward_the_weighted_region():
    """Errors inside the weighted region must count more, not just more total."""
    pred = np.zeros((10, 2))
    true = np.zeros((10, 2))
    true[:2] = 1.0                       # error only in the "airgap" rows
    w_flat = np.ones(10)
    w_air = np.ones(10); w_air[:2] = 5.0

    flat = _weighted_mse(pred, true, w_flat)
    weighted = _weighted_mse(pred, true, w_air)
    assert weighted > flat

    # And the reverse: error outside the weighted region counts less.
    true2 = np.zeros((10, 2)); true2[5:7] = 1.0
    assert _weighted_mse(pred, true2, w_air) < _weighted_mse(pred, true2, w_flat)


def test_weighted_loss_is_scale_normalized():
    """Scaling every weight must not change the loss (guards the /w.sum())."""
    rng = np.random.default_rng(3)
    pred, true = rng.normal(size=(30, 2)), rng.normal(size=(30, 2))
    w = rng.uniform(1.0, 5.0, size=30)
    assert _weighted_mse(pred, true, w) == pytest.approx(_weighted_mse(pred, true, 7.5 * w))


def test_batched_element_weights_concatenate():
    graphs = []
    for seed in range(3):
        g, op, _ = make_graph(seed, n_points=30)
        g.elem_weight = torch.full((op.n_valid,), float(seed + 1), dtype=torch.float32)
        graphs.append(g)

    batch = next(iter(DataLoader(graphs, batch_size=3, shuffle=False)))
    assert batch.elem_weight.shape[0] == sum(g.elem_weight.shape[0] for g in graphs)
    assert batch.elem_weight.shape[0] == batch.curl_node_index.shape[0]
    assert set(batch.elem_weight.unique().tolist()) == {1.0, 2.0, 3.0}
