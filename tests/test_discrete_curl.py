"""Tests for the P1 element-wise curl operator.

The operator is exact for linear A fields, so most of these are equalities to
machine precision rather than tolerances.
"""

import numpy as np
import pytest

from phase1_static.discrete_curl import (
    CurlOperatorError,
    build_p1_curl_operator,
    curl_a_to_b,
    fit_nodal_a,
    mesh_validity_mask,
    signed_double_area,
)


def unit_square_mesh():
    """Two triangles covering the unit square (metres)."""
    node_x = np.array([0.0, 1.0, 0.0, 1.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0])
    tri = (np.array([0, 1]), np.array([1, 3]), np.array([2, 2]))
    return node_x, node_y, tri


def random_mesh(n=60, seed=0):
    """A Delaunay triangulation of scattered points, for the fit tests."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n, 2))
    pts = np.vstack([pts, [[0, 0], [1, 0], [0, 1], [1, 1]]])
    d = Delaunay(pts)
    simp = d.simplices
    return pts[:, 0].copy(), pts[:, 1].copy(), (simp[:, 0], simp[:, 1], simp[:, 2])


@pytest.mark.parametrize(
    "a_of_xy,expected",
    [
        (lambda x, y: np.zeros_like(x), (0.0, 0.0)),          # uniform A -> no field
        (lambda x, y: y, (1.0, 0.0)),                          # Bx =  dA/dy
        (lambda x, y: x, (0.0, -1.0)),                         # By = -dA/dx
        (lambda x, y: 3.0 * y - 2.0 * x, (3.0, 2.0)),
        (lambda x, y: -0.5 * x + 0.25 * y, (0.25, 0.5)),
    ],
)
def test_linear_a_is_reproduced_exactly(a_of_xy, expected):
    node_x, node_y, tri = unit_square_mesh()
    op = build_p1_curl_operator(node_x, node_y, tri)
    b = curl_a_to_b(op, a_of_xy(node_x, node_y))
    assert b.shape == (2, 2)
    np.testing.assert_allclose(b[:, 0], expected[0], atol=1e-12)
    np.testing.assert_allclose(b[:, 1], expected[1], atol=1e-12)


def test_operator_is_linear_in_a():
    node_x, node_y, tri = random_mesh()
    op = build_p1_curl_operator(node_x, node_y, tri)
    a1 = np.sin(3 * node_x) * node_y
    a2 = np.cos(2 * node_y)

    lhs = curl_a_to_b(op, 2.5 * a1 - 1.5 * a2)
    rhs = 2.5 * curl_a_to_b(op, a1) - 1.5 * curl_a_to_b(op, a2)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_adding_a_constant_to_a_changes_nothing():
    """A is a potential: only its gradient is physical."""
    node_x, node_y, tri = random_mesh()
    op = build_p1_curl_operator(node_x, node_y, tri)
    a = np.sin(4 * node_x) + node_y
    np.testing.assert_allclose(curl_a_to_b(op, a), curl_a_to_b(op, a + 17.3), rtol=1e-10)


def test_result_is_independent_of_triangle_node_order():
    """Signed 2S must cancel the orientation of the b/c coefficients."""
    node_x, node_y, tri = unit_square_mesh()
    i1, i2, i3 = tri
    a = 2.0 * node_y - node_x

    forward = curl_a_to_b(build_p1_curl_operator(node_x, node_y, (i1, i2, i3)), a)
    swapped = curl_a_to_b(build_p1_curl_operator(node_x, node_y, (i1, i3, i2)), a)
    np.testing.assert_allclose(forward, swapped, atol=1e-12)


def test_millimetre_coordinates_are_a_thousand_times_off():
    """Guards the documented unit requirement rather than hiding it."""
    node_x, node_y, tri = unit_square_mesh()
    a = node_y.copy()
    metres = curl_a_to_b(build_p1_curl_operator(node_x, node_y, tri), a)
    millis = curl_a_to_b(build_p1_curl_operator(node_x * 1e3, node_y * 1e3, tri), a)
    np.testing.assert_allclose(millis, metres * 1e-3, rtol=1e-10)


def test_output_is_element_support_not_node_support():
    """The whole point: B lands on elements, so no node round trip is needed."""
    node_x, node_y, tri = random_mesh()
    op = build_p1_curl_operator(node_x, node_y, tri)
    b = curl_a_to_b(op, node_y.copy())
    assert b.shape[0] == op.n_valid == tri[0].size
    assert b.shape[0] != op.n_nodes


# --- validity masking --------------------------------------------------------


def test_degenerate_triangles_are_dropped():
    node_x = np.array([0.0, 1.0, 2.0, 0.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0])  # first triangle is collinear
    tri = (np.array([0, 0]), np.array([1, 1]), np.array([2, 3]))
    mask = mesh_validity_mask(node_x, node_y, tri)
    assert mask.tolist() == [False, True]

    op = build_p1_curl_operator(node_x, node_y, tri, valid=mask)
    assert op.n_valid == 1
    assert op.n_elements == 2
    assert op.element_index.tolist() == [1]
    assert op.coverage == pytest.approx(0.5)


def test_inverted_elements_are_detected_against_a_reference():
    node_x, node_y, tri = unit_square_mesh()
    reference = signed_double_area(node_x, node_y, tri)

    # Drag node 2 across the first triangle's edge so it flips orientation.
    moved_y = node_y.copy()
    moved_y[2] = -1.0
    mask = mesh_validity_mask(node_x, moved_y, tri, reference_sign=reference)
    assert mask[0] is np.False_ or not mask[0]


def test_explicit_exclusion_is_honoured():
    node_x, node_y, tri = unit_square_mesh()
    mask = mesh_validity_mask(node_x, node_y, tri, exclude=np.array([True, False]))
    assert mask.tolist() == [False, True]


def test_operator_with_no_valid_elements_is_an_error():
    node_x, node_y, tri = unit_square_mesh()
    with pytest.raises(CurlOperatorError, match="No valid elements"):
        build_p1_curl_operator(node_x, node_y, tri, valid=np.zeros(2, dtype=bool))


def test_wrong_nodal_count_is_rejected():
    node_x, node_y, tri = unit_square_mesh()
    op = build_p1_curl_operator(node_x, node_y, tri)
    with pytest.raises(CurlOperatorError, match="nodal values"):
        curl_a_to_b(op, np.zeros(3))


# --- least-squares fit -------------------------------------------------------


def test_fit_recovers_a_field_that_the_operator_can_represent():
    node_x, node_y, tri = random_mesh(n=120, seed=3)
    op = build_p1_curl_operator(node_x, node_y, tri)
    a_true = 0.7 * node_x - 1.3 * node_y + 0.2
    b_target = curl_a_to_b(op, a_true)

    # Iterative least squares: the residual is relative, so 1e-6 already means
    # the fitted A reproduces the target field to one part in a million.
    a_fit, residual = fit_nodal_a(op, b_target)
    assert residual < 1e-6
    np.testing.assert_allclose(curl_a_to_b(op, a_fit), b_target, atol=1e-6)


def test_fit_is_invariant_to_the_gauge_constant():
    """Only curl(A) is recovered; the additive constant is unconstrained."""
    node_x, node_y, tri = random_mesh(n=80, seed=5)
    op = build_p1_curl_operator(node_x, node_y, tri)
    b_target = curl_a_to_b(op, 0.4 * node_x + 0.9 * node_y)

    a_fit, _ = fit_nodal_a(op, b_target)
    np.testing.assert_allclose(curl_a_to_b(op, a_fit), b_target, atol=1e-6)


def test_fit_rejects_mismatched_b_shape():
    node_x, node_y, tri = random_mesh(n=40)
    op = build_p1_curl_operator(node_x, node_y, tri)
    with pytest.raises(CurlOperatorError, match="to match the operator"):
        fit_nodal_a(op, np.zeros((op.n_valid, 3)))


# --- torch parity ------------------------------------------------------------


def test_torch_path_matches_numpy_and_is_differentiable():
    torch = pytest.importorskip("torch")
    from phase1_static.discrete_curl import curl_a_to_b_torch

    node_x, node_y, tri = random_mesh(n=90, seed=11)
    op = build_p1_curl_operator(node_x, node_y, tri)
    a = np.sin(5 * node_x) * np.cos(3 * node_y)

    expected = curl_a_to_b(op, a)
    a_t = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    got = curl_a_to_b_torch(op, a_t)

    np.testing.assert_allclose(got.detach().numpy(), expected, rtol=1e-10)

    got.pow(2).sum().backward()
    assert a_t.grad is not None
    assert torch.isfinite(a_t.grad).all()
    assert a_t.grad.abs().sum() > 0


def test_torch_accepts_a_column_vector():
    torch = pytest.importorskip("torch")
    from phase1_static.discrete_curl import curl_a_to_b_torch

    node_x, node_y, tri = unit_square_mesh()
    op = build_p1_curl_operator(node_x, node_y, tri)
    a = torch.tensor(node_y, dtype=torch.float64).unsqueeze(1)
    got = curl_a_to_b_torch(op, a)
    np.testing.assert_allclose(got.numpy()[:, 0], 1.0, atol=1e-12)
