"""Tests for mesh-edge curl post-processing utility."""
import numpy as np
import pytest

from phase1_static.curl_postproc import curl_consistency_metrics, mesh_edge_curl_b


def _make_triangle_mesh():
    """Simple 3-node triangle mesh with known A field."""
    # Triangle: (0,0), (1,0), (0,1)
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Undirected edges: 0↔1, 1↔2, 0↔2
    edge_index = np.array([
        [0, 1, 1, 2, 0, 2],
        [1, 0, 2, 1, 2, 0],
    ])
    return pos, edge_index


def test_uniform_a_gives_zero_b():
    """Uniform A → dA/dx = dA/dy = 0 → B = (0, 0)."""
    pos, edge_index = _make_triangle_mesh()
    a_values = np.array([5.0, 5.0, 5.0])
    b_curl = mesh_edge_curl_b(pos, a_values, edge_index)
    assert b_curl.shape == (3, 2)
    np.testing.assert_allclose(b_curl, 0.0, atol=1e-12)


def test_linear_a_x_gives_by():
    """A = x → dA/dx = 1, dA/dy = 0 → Bx = 0, By = -1."""
    pos, edge_index = _make_triangle_mesh()
    a_values = pos[:, 0].copy()  # A = x
    b_curl = mesh_edge_curl_b(pos, a_values, edge_index)
    # Bx = dA/dy = 0, By = -dA/dx = -1
    for i in range(3):
        assert abs(b_curl[i, 0]) < 0.3, f"Bx at node {i} should be ~0, got {b_curl[i, 0]}"
        assert b_curl[i, 1] < -0.3, f"By at node {i} should be ~-1, got {b_curl[i, 1]}"


def test_linear_a_y_gives_bx():
    """A = y → dA/dx = 0, dA/dy = 1 → Bx = 1, By = 0."""
    pos, edge_index = _make_triangle_mesh()
    a_values = pos[:, 1].copy()  # A = y
    b_curl = mesh_edge_curl_b(pos, a_values, edge_index)
    # Bx = dA/dy = 1, By = -dA/dx = 0
    for i in range(3):
        assert b_curl[i, 0] > 0.3, f"Bx at node {i} should be ~1, got {b_curl[i, 0]}"
        assert abs(b_curl[i, 1]) < 0.3, f"By at node {i} should be ~0, got {b_curl[i, 1]}"


def test_curl_consistency_metrics_shape():
    """Verify metrics dict keys and curl_b shape."""
    pos, edge_index = _make_triangle_mesh()
    a_pred = np.array([0.0, 1.0, 0.5])
    b_pred = np.zeros((3, 2))
    result = curl_consistency_metrics(pos, a_pred, b_pred, edge_index)
    assert "curl_b" in result
    assert result["curl_b"].shape == (3, 2)
    assert "rmse_bx" in result
    assert "rmse_by" in result
    assert "rmse_total" in result
    assert "max_abs_error" in result
    assert result["rmse_total"] >= 0


def test_regular_grid_curl_accuracy():
    """On a regular grid, A = x*y → exact Bx = x, By = -y."""
    # 5x5 grid
    xs = np.linspace(0, 1, 5)
    ys = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(xs, ys)
    pos = np.stack([xx.ravel(), yy.ravel()], axis=1)
    n = pos.shape[0]

    # Build grid edges (4-connected)
    edges_src, edges_dst = [], []
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if j < 4:  # right
                edges_src.append(idx)
                edges_dst.append(idx + 1)
                edges_src.append(idx + 1)
                edges_dst.append(idx)
            if i < 4:  # down
                edges_src.append(idx)
                edges_dst.append(idx + 5)
                edges_src.append(idx + 5)
                edges_dst.append(idx)
    edge_index = np.array([edges_src, edges_dst])

    # A = x * y → dA/dx = y, dA/dy = x → Bx = x, By = -y
    a_values = pos[:, 0] * pos[:, 1]
    b_curl = mesh_edge_curl_b(pos, a_values, edge_index)

    # Interior nodes (not on boundary) should be fairly accurate
    interior = (pos[:, 0] > 0.1) & (pos[:, 0] < 0.9) & (pos[:, 1] > 0.1) & (pos[:, 1] < 0.9)
    bx_exact = pos[interior, 0]  # dA/dy = x
    by_exact = -pos[interior, 1]  # -dA/dx = -y

    np.testing.assert_allclose(b_curl[interior, 0], bx_exact, atol=0.25)
    np.testing.assert_allclose(b_curl[interior, 1], by_exact, atol=0.25)


def test_input_validation():
    """Bad shapes should raise ValueError."""
    with pytest.raises(ValueError):
        mesh_edge_curl_b(np.zeros((3, 3)), np.zeros(3), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        mesh_edge_curl_b(np.zeros((3, 2)), np.zeros(3), np.zeros((3, 3)))
