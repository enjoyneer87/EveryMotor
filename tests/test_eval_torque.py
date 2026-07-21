"""Tests for the Arkkio airgap torque operator.

The operator is checked against the closed-form Maxwell stress result on a
synthetic annular band, and against the physical invariants a torque must obey
(no tangential field means no torque; torque is quadratic in B; reversing the
tangential field reverses the sign).
"""

import numpy as np
import pytest

from eval.torque import (
    MU0,
    AirgapBandError,
    aggregate_torque,
    arkkio_torque,
    build_airgap_band,
    compare_torque,
    nodal_to_element,
)


def make_annular_band(
    r_inner=0.0750,
    r_outer=0.0755,
    span_rad=2 * np.pi,
    n_theta=720,
    n_r=2,
    region_code=93,
    region_name="a1",
    extra_region=False,
):
    """Triangulated annular sector shaped like the Motor-CAD airgap layers."""
    closed = np.isclose(span_rad, 2 * np.pi)
    n_theta_nodes = n_theta if closed else n_theta + 1
    theta = np.linspace(0.0, span_rad, n_theta_nodes, endpoint=not closed)
    radii = np.linspace(r_inner, r_outer, n_r + 1)

    xs, ys = [], []
    for r in radii:
        xs.append(r * np.cos(theta) * 1e3)  # mm
        ys.append(r * np.sin(theta) * 1e3)
    node_x = np.concatenate(xs)
    node_y = np.concatenate(ys)

    def nid(ri, ti):
        return ri * n_theta_nodes + (ti % n_theta_nodes)

    i1, i2, i3, reg = [], [], [], []
    n_quads = n_theta if closed else n_theta
    for ri in range(n_r):
        for ti in range(n_quads):
            a, b = nid(ri, ti), nid(ri, ti + 1)
            c, d = nid(ri + 1, ti), nid(ri + 1, ti + 1)
            i1 += [a, b]
            i2 += [b, d]
            i3 += [c, c]
            reg += [region_code, region_code]

    if extra_region:
        # A second, rotor-side layer that must be excluded from the band.
        offset = node_x.size
        r2 = np.linspace(r_inner - 0.001, r_inner - 0.0005, 2)
        for r in r2:
            node_x = np.concatenate([node_x, r * np.cos(theta) * 1e3])
            node_y = np.concatenate([node_y, r * np.sin(theta) * 1e3])
        for ti in range(n_quads):
            a = offset + (ti % n_theta_nodes)
            b = offset + ((ti + 1) % n_theta_nodes)
            c = offset + n_theta_nodes + (ti % n_theta_nodes)
            i1 += [a]
            i2 += [b]
            i3 += [c]
            reg += [94]

    tri = (np.array(i1), np.array(i2), np.array(i3))
    return node_x, node_y, tri, np.array(reg), {region_code: region_name, 94: "a2"}


def uniform_radial_tangential(node_x, node_y, tri, b_r, b_theta):
    """Element-wise (Bx, By) for a field with constant radial/tangential parts."""
    i1, i2, i3 = tri
    cx = (node_x[i1] + node_x[i2] + node_x[i3]) / 3.0
    cy = (node_y[i1] + node_y[i2] + node_y[i3]) / 3.0
    r = np.hypot(cx, cy)
    ct, st = cx / r, cy / r
    return b_r * ct - b_theta * st, b_r * st + b_theta * ct


def analytic_torque(r_inner, r_outer, b_r, b_theta, axial_length=1.0):
    """Closed-form Arkkio result for a uniform field over a full annulus.

    T = L/(mu0 (r_out - r_in)) * integral(r * Br * Bt) dS
      = L * Br * Bt * 2*pi*(r_out^3 - r_in^3) / (3 * mu0 * (r_out - r_in))
    """
    return (
        axial_length
        * b_r
        * b_theta
        * 2.0
        * np.pi
        * (r_outer**3 - r_inner**3)
        / (3.0 * MU0 * (r_outer - r_inner))
    )


def _band(**kwargs):
    node_x, node_y, tri, reg, names = make_annular_band(**kwargs)
    band = build_airgap_band(node_x, node_y, tri, reg, names, moving_reg_codes=[94])
    return node_x, node_y, tri, band


def test_matches_closed_form_on_a_full_annulus():
    r_in, r_out = 0.0750, 0.0755
    node_x, node_y, tri, band = _band(r_inner=r_in, r_outer=r_out, n_theta=2000)
    bx, by = uniform_radial_tangential(node_x, node_y, tri, b_r=0.9, b_theta=0.35)

    got = arkkio_torque(band, bx, by)
    want = analytic_torque(r_in, r_out, 0.9, 0.35)
    assert got == pytest.approx(want, rel=2e-3)


def test_converges_to_the_closed_form_as_the_mesh_refines():
    r_in, r_out = 0.0750, 0.0755
    want = analytic_torque(r_in, r_out, 0.8, 0.4)

    errors = []
    for n_theta in (120, 480, 1920):
        node_x, node_y, tri, band = _band(r_inner=r_in, r_outer=r_out, n_theta=n_theta)
        bx, by = uniform_radial_tangential(node_x, node_y, tri, 0.8, 0.4)
        errors.append(abs(arkkio_torque(band, bx, by) - want) / abs(want))

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-3


def test_purely_radial_field_produces_no_torque():
    node_x, node_y, tri, band = _band()
    bx, by = uniform_radial_tangential(node_x, node_y, tri, b_r=1.2, b_theta=0.0)
    assert arkkio_torque(band, bx, by) == pytest.approx(0.0, abs=1e-6)


def test_purely_tangential_field_produces_no_torque():
    node_x, node_y, tri, band = _band()
    bx, by = uniform_radial_tangential(node_x, node_y, tri, b_r=0.0, b_theta=1.2)
    assert arkkio_torque(band, bx, by) == pytest.approx(0.0, abs=1e-6)


def test_reversing_the_tangential_field_reverses_the_torque():
    node_x, node_y, tri, band = _band()
    bx_p, by_p = uniform_radial_tangential(node_x, node_y, tri, 0.9, 0.3)
    bx_m, by_m = uniform_radial_tangential(node_x, node_y, tri, 0.9, -0.3)
    assert arkkio_torque(band, bx_p, by_p) == pytest.approx(-arkkio_torque(band, bx_m, by_m))


def test_torque_is_quadratic_in_field_magnitude():
    node_x, node_y, tri, band = _band()
    bx, by = uniform_radial_tangential(node_x, node_y, tri, 0.5, 0.2)
    base = arkkio_torque(band, bx, by)
    doubled = arkkio_torque(band, 2 * bx, 2 * by)
    assert doubled == pytest.approx(4.0 * base, rel=1e-9)


def test_torque_scales_linearly_with_stack_length():
    node_x, node_y, tri, band = _band()
    bx, by = uniform_radial_tangential(node_x, node_y, tri, 0.7, 0.25)
    assert arkkio_torque(band, bx, by, axial_length_m=0.15) == pytest.approx(
        0.15 * arkkio_torque(band, bx, by, axial_length_m=1.0)
    )


def test_moving_airgap_layers_are_excluded_from_the_band():
    node_x, node_y, tri, reg, names = make_annular_band(extra_region=True)
    band = build_airgap_band(node_x, node_y, tri, reg, names, moving_reg_codes=[94])
    assert band.region_codes == (93,)
    assert np.all(reg[band.element_index] == 93)


def test_sector_symmetry_is_derived_from_the_span():
    _, _, _, band = _band(span_rad=np.pi / 4, n_theta=180)
    assert band.symmetry_multiplier == 8


def test_a_sector_reproduces_the_full_annulus_torque():
    """A 45-degree sector times 8 must equal the full-annulus result."""
    r_in, r_out = 0.0750, 0.0755
    nx_f, ny_f, tri_f, band_f = _band(r_inner=r_in, r_outer=r_out, n_theta=1440)
    bx_f, by_f = uniform_radial_tangential(nx_f, ny_f, tri_f, 0.9, 0.3)

    nx_s, ny_s, tri_s, band_s = _band(
        r_inner=r_in, r_outer=r_out, span_rad=np.pi / 4, n_theta=180
    )
    bx_s, by_s = uniform_radial_tangential(nx_s, ny_s, tri_s, 0.9, 0.3)

    assert band_s.symmetry_multiplier == 8
    assert arkkio_torque(band_s, bx_s, by_s) == pytest.approx(
        arkkio_torque(band_f, bx_f, by_f), rel=5e-3
    )


def test_non_dividing_sector_span_is_rejected():
    with pytest.raises(AirgapBandError, match="does not divide 360"):
        _band(span_rad=1.0, n_theta=100)


def test_missing_airgap_regions_is_an_explicit_error():
    node_x, node_y, tri, reg, _ = make_annular_band()
    with pytest.raises(AirgapBandError, match="No airgap regions"):
        build_airgap_band(node_x, node_y, tri, reg, {93: "Stator"})


def test_band_accepts_prescliced_arrays():
    node_x, node_y, tri, band = _band(extra_region=True)
    bx, by = uniform_radial_tangential(node_x, node_y, tri, 0.9, 0.3)
    full = arkkio_torque(band, bx, by)
    sliced = arkkio_torque(band, bx[band.element_index], by[band.element_index])
    assert full == pytest.approx(sliced)


def test_identical_fields_report_zero_error():
    node_x, node_y, tri, band = _band()
    bx, by = uniform_radial_tangential(node_x, node_y, tri, 0.9, 0.3)
    cmp = compare_torque(band, bx, by, bx, by)
    assert cmp.abs_error == pytest.approx(0.0)
    assert cmp.rel_error_pct == pytest.approx(0.0)


def test_relative_error_is_invariant_to_stack_length():
    """The metric that gates G2 must not depend on the unknown stack length."""
    node_x, node_y, tri, band = _band()
    bx_t, by_t = uniform_radial_tangential(node_x, node_y, tri, 0.9, 0.30)
    bx_p, by_p = uniform_radial_tangential(node_x, node_y, tri, 0.9, 0.33)

    a = compare_torque(band, bx_p, by_p, bx_t, by_t, axial_length_m=1.0)
    b = compare_torque(band, bx_p, by_p, bx_t, by_t, axial_length_m=0.15)
    assert a.rel_error_pct == pytest.approx(b.rel_error_pct)
    assert a.rel_error_pct == pytest.approx(10.0, rel=1e-6)


def test_nodal_to_element_averages_over_the_triangle():
    tri = (np.array([0]), np.array([1]), np.array([2]))
    assert nodal_to_element(np.array([1.0, 2.0, 6.0]), tri)[0] == pytest.approx(3.0)


def test_aggregate_separates_mean_torque_from_ripple():
    """A prediction with the right mean but no ripple must be visible as such."""

    class _Cmp:
        def __init__(self, p, t):
            self.torque_pred, self.torque_true = p, t
            self.abs_error = abs(p - t)
            self.rel_error_pct = 100.0 * self.abs_error / abs(t)

    true = [100.0, 110.0, 90.0, 100.0]
    flat = [100.0, 100.0, 100.0, 100.0]
    agg = aggregate_torque([_Cmp(p, t) for p, t in zip(flat, true)])

    assert agg["mean_torque_error_pct"] == pytest.approx(0.0, abs=1e-9)
    assert agg["ripple_true"] == pytest.approx(20.0)
    assert agg["ripple_pred"] == pytest.approx(0.0)
    assert agg["ripple_error_pct"] == pytest.approx(100.0)


class _Cmp:
    """Minimal stand-in for TorqueComparison in aggregation tests."""

    def __init__(self, pred, true):
        self.torque_pred, self.torque_true = pred, true
        self.abs_error = abs(pred - true)
        self.rel_error_pct = 100.0 * self.abs_error / abs(true) if true else float("nan")


def test_torque_nrmse_is_the_headline_metric():
    true = [100.0, 110.0, 90.0, 100.0]
    pred = [103.0, 113.0, 93.0, 103.0]  # uniform +3 offset
    agg = aggregate_torque([_Cmp(p, t) for p, t in zip(pred, true)])
    rms_true = float(np.sqrt(np.mean(np.square(true))))
    assert agg["nrmse_torque_pct"] == pytest.approx(100.0 * 3.0 / rms_true)


def test_near_zero_mean_torque_does_not_explode_the_metric():
    """DOE case 32 (phase advance 89 deg) averages ~0 torque over the sweep."""
    true = [700.0, -710.0, 690.0, -685.0]  # mean ~ -1.25, rms ~ 696
    pred = [710.0, -700.0, 700.0, -675.0]
    agg = aggregate_torque([_Cmp(p, t) for p, t in zip(pred, true)])

    assert agg["mean_torque_is_significant"] is False
    assert np.isnan(agg["mean_torque_error_pct"])
    # The robust normalizations stay small and usable.
    assert agg["nrmse_torque_pct"] < 5.0
    assert agg["mean_torque_error_norm_pct"] < 5.0


def test_mean_torque_error_is_reported_when_the_mean_is_significant():
    true = [100.0, 102.0, 98.0, 100.0]
    pred = [105.0, 107.0, 103.0, 105.0]
    agg = aggregate_torque([_Cmp(p, t) for p, t in zip(pred, true)])
    assert agg["mean_torque_is_significant"] is True
    assert agg["mean_torque_error_pct"] == pytest.approx(5.0, rel=1e-3)


def test_instantaneous_relative_error_is_the_fragile_one():
    """Documents why nrmse_torque_pct, not rel_error, gates G2."""
    true = [500.0, 0.5, -500.0]  # sweep crosses zero
    pred = [505.0, 5.5, -495.0]  # uniform +5 error
    agg = aggregate_torque([_Cmp(p, t) for p, t in zip(pred, true)])

    assert agg["rel_error_pct_max"] > 500.0  # meaningless near the zero crossing
    assert agg["nrmse_torque_pct"] < 5.0  # the honest number


def test_perfect_prediction_scores_zero_torque_nrmse():
    true = [100.0, 110.0, 90.0]
    agg = aggregate_torque([_Cmp(t, t) for t in true])
    assert agg["nrmse_torque_pct"] == pytest.approx(0.0)
    assert agg["ripple_error_pct"] == pytest.approx(0.0)


def test_aggregate_of_nothing_reports_nan():
    agg = aggregate_torque([])
    assert agg["n"] == 0 and np.isnan(agg["nrmse_torque_pct"])


def test_sign_follows_the_plus_theta_convention():
    """Torque is positive about +theta; a clockwise-motoring machine reads negative.

    Locks the convention that the Motor-CAD comparison relies on: our number is
    -1x Motor-CAD's positive-motoring magnitude. See eval/torque.py.
    """
    node_x, node_y, tri, band = _band()

    # B_r > 0 with B_theta > 0 drives the rotor counter-clockwise -> +T.
    bx, by = uniform_radial_tangential(node_x, node_y, tri, b_r=0.9, b_theta=0.3)
    assert arkkio_torque(band, bx, by) > 0

    # Reversing the tangential component reverses the drive direction -> -T.
    bx_cw, by_cw = uniform_radial_tangential(node_x, node_y, tri, b_r=0.9, b_theta=-0.3)
    assert arkkio_torque(band, bx_cw, by_cw) < 0


def test_a_dc_dominated_waveform_cannot_be_sign_flipped_by_a_phase_shift():
    """Why the Motor-CAD phase alignment cannot be masking a sign error.

    The measured waveform has mean ~368 N*m and ripple ~+-28, so it never
    crosses zero. Any circular shift preserves the mean, so no shift can turn
    +mean into -mean.
    """
    phase = np.linspace(0.0, 2 * np.pi, 45, endpoint=False)
    wave = 368.0 + 28.0 * np.sin(12 * phase)

    assert wave.min() > 0.0  # never crosses zero
    for shift in range(45):
        rolled = np.roll(wave, shift)
        assert rolled.mean() == pytest.approx(wave.mean())
        # Best achievable RMSE against the negated waveform stays enormous.
        assert np.sqrt(np.mean((rolled - (-wave)) ** 2)) > 500.0
