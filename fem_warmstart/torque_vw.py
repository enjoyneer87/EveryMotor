"""Virtual-work torque, two ways, on the same domain and the same nodal A.

Both routines evaluate the SAME quantity — the virtual work principle,

    T = dW'/dtheta  at constant current,   W' = integral over V of (integral H.dB)

— and differ only in how the derivative is taken:

``torque_vw_finite_difference``
    The textbook/global route. Coenergy is evaluated at two *separately solved*
    rotor positions and differenced. Needs >= 2 solves, and inherits the mesh
    difference between them (each step is its own mesh: the rotor moved).

``torque_vw_coulomb``
    Coulomb's local virtual work. ONE solve. The rotor is given an infinitesimal
    *virtual* rotation, the air-gap layer absorbs the distortion, and the
    coenergy of the discretised system is differentiated with the nodal A held
    fixed. Holding A fixed is legitimate, not an approximation: at the FE
    solution the functional is stationary with respect to A, so the implicit
    term dW'/dA . dA/dtheta vanishes and only the explicit geometric derivative
    survives. That is exactly why one solve is enough.

So the two are not competing principles. They are the numerical derivative of
two solutions versus the analytic derivative of one, and they should agree to
within discretisation. Disagreement means a bug in one of them (or a rotor step
too coarse for the finite difference) -- which is the point of shipping both.

This module deliberately reuses `fem_warmstart.domain` rather than re-deriving
materials: magnet remanence, easy axes, the BH curves and the shear-layer
remesh are already resolved and audited there (section 23). The exported nodal
A is lifted into the domain with `domain.lift_guess`, so both routines score
the solver's own field on the solver's own geometry.

Nothing here touches the eMach submodule; eMach has no virtual-work
implementation at all (it only reads Motor-CAD's TorqueVW output).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from fem_warmstart.bh_curve import MU0, NU0, Reluctivity
from fem_warmstart.domain import AIR, MAGNET, STEEL, FemDomain, build_domain
from fem_warmstart.prior import case_current_scale, default_bh_path


# --------------------------------------------------------------------------- #
# coenergy
# --------------------------------------------------------------------------- #
def _steel_coenergy_density(curve: Reluctivity, b_mag: np.ndarray) -> np.ndarray:
    """Exact integral of the piecewise-linear H(B) interpolant, J/m^3.

    `Reluctivity` interpolates H linearly between table points and continues with
    the fully-saturated slope NU0 above Bmax. Integrating that interpolant in
    closed form (rather than quadrature) keeps the coenergy consistent with the
    reluctivity the solver itself used, to machine precision.
    """
    b_tab, h_tab = curve.B, curve.H
    # cumulative trapezoid at each table point
    seg = 0.5 * (h_tab[1:] + h_tab[:-1]) * np.diff(b_tab)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    b = np.asarray(b_mag, dtype=np.float64)
    out = np.empty_like(b)

    inside = b <= curve.Bmax
    if np.any(inside):
        bi = b[inside]
        i = np.clip(np.searchsorted(b_tab, bi, side="right") - 1, 0, b_tab.size - 2)
        db = bi - b_tab[i]
        slope = (h_tab[i + 1] - h_tab[i]) / (b_tab[i + 1] - b_tab[i])
        out[inside] = cum[i] + h_tab[i] * db + 0.5 * slope * db * db
    tail = ~inside
    if np.any(tail):
        db = b[tail] - curve.Bmax
        out[tail] = cum[-1] + curve.Hmax * db + 0.5 * NU0 * db * db
    return out


def coenergy(domain: FemDomain, b_elem: np.ndarray,
             br_override: Optional[np.ndarray] = None) -> float:
    """Total coenergy of the modelled sector, J per metre of stack.

    Constitutive laws match the assembler exactly:
      air / non-steel linear:  H = nu*B                -> w' = nu|B|^2/2
      magnet:                  H = nu*(B - Br)         -> w' = nu(|B|^2/2 - Br.B)
      steel:                   H = H(|B|) from the BH curve, isotropic
    """
    b_elem = np.asarray(b_elem, dtype=np.float64)
    b_mag = np.hypot(b_elem[:, 0], b_elem[:, 1])
    w = np.zeros(domain.n_elements, dtype=np.float64)

    lin = domain.material != STEEL
    if np.any(lin):
        w[lin] = 0.5 * domain.nu_linear[lin] * b_mag[lin] ** 2
    mag = domain.material == MAGNET
    if np.any(mag):
        br = domain.br if br_override is None else br_override
        w[mag] -= domain.nu_linear[mag] * (
            br[mag, 0] * b_elem[mag, 0] + br[mag, 1] * b_elem[mag, 1])
    steel = domain.material == STEEL
    if np.any(steel):
        for idx, curve in enumerate(domain.curves):
            sel = steel & (domain.curve_index == idx)
            if np.any(sel):
                w[sel] = _steel_coenergy_density(curve, b_mag[sel])
    return float(np.sum(w * domain.area))


# --------------------------------------------------------------------------- #
# geometry helpers -- B from A on ARBITRARY node coordinates
# --------------------------------------------------------------------------- #
def _element_b_on_coords(x: np.ndarray, y: np.ndarray, tri: np.ndarray,
                         a_nodal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(B (E,2), area (E,)) for P1 triangles at the given node coordinates.

    Same convention as `FemDomain.element_b`: Bx = dA/dy, By = -dA/dx. Recomputed
    from scratch because the whole point of the virtual displacement is that the
    coordinates -- and therefore the shape-function gradients and the areas --
    have moved.
    """
    xe, ye = x[tri], y[tri]                                   # (E,3)
    b0 = ye[:, 1] - ye[:, 2]
    b1 = ye[:, 2] - ye[:, 0]
    b2 = ye[:, 0] - ye[:, 1]
    c0 = xe[:, 2] - xe[:, 1]
    c1 = xe[:, 0] - xe[:, 2]
    c2 = xe[:, 1] - xe[:, 0]
    two_a = xe[:, 0] * b0 + xe[:, 1] * b1 + xe[:, 2] * b2      # = 2*signed area
    ae = a_nodal[tri]
    dady = (c0 * ae[:, 0] + c1 * ae[:, 1] + c2 * ae[:, 2]) / two_a
    dadx = (b0 * ae[:, 0] + b1 * ae[:, 1] + b2 * ae[:, 2]) / two_a
    return np.stack([dady, -dadx], axis=1), np.abs(0.5 * two_a)


@dataclass(frozen=True)
class VirtualDisplacement:
    """Per-node rotation weight for Coulomb's virtual displacement.

    ``weight`` is 1 on everything that moves rigidly with the rotor, 0 on the
    stator, and blends across the air-gap annulus. Virtual work is invariant to
    the choice of blend, which `torque_vw_coulomb(..., self_test=True)` checks by
    re-running with a different profile.
    """
    weight: np.ndarray             # (N,)
    rotor_element: np.ndarray      # (E,) bool -- centroid inside the rotor
    r_inner_m: float
    r_outer_m: float
    n_moving_nodes: int


def build_virtual_displacement(domain: FemDomain, profile: str = "linear"
                               ) -> VirtualDisplacement:
    """Rotor-side rigid rotation, blended to zero across the air-gap layer.

    The layer is every air element the mesh calls ``a<k>`` plus the elements the
    shear-layer remesh created (region code -1) -- i.e. exactly the annulus
    between rotor and stator. Radius classifies the rest: inside the annulus is
    rotor, outside is stator. The rotor must move RIGIDLY (weight exactly 1) or
    its own elements would distort and inject spurious energy.
    """
    names = domain.name_of_code
    is_layer = np.zeros(domain.n_elements, dtype=bool)
    for e, code in enumerate(np.asarray(domain.reg_code)):
        if int(code) < 0:                       # created by the band remesh
            is_layer[e] = True
            continue
        nm = str(names.get(int(code), ""))
        if len(nm) > 1 and nm[0] == "a" and nm[1:].isdigit():
            is_layer[e] = True
    is_layer &= domain.material == AIR
    if not np.any(is_layer):
        raise ValueError("no air-gap layer found; cannot build a virtual displacement")

    r_node = np.hypot(domain.x, domain.y)
    layer_nodes = np.unique(domain.tri[is_layer].ravel())
    r_in = float(r_node[layer_nodes].min())
    r_out = float(r_node[layer_nodes].max())
    if not (r_out > r_in):
        raise ValueError("degenerate air-gap annulus")

    t = np.clip((r_out - r_node) / (r_out - r_in), 0.0, 1.0)   # 1 inside, 0 outside
    if profile == "linear":
        weight = t
    elif profile == "smoothstep":
        weight = t * t * (3.0 - 2.0 * t)       # different blend, same torque
    else:
        raise ValueError(f"unknown profile {profile!r}")
    weight[r_node <= r_in] = 1.0
    weight[r_node >= r_out] = 0.0

    r_cent = np.hypot(domain.x[domain.tri].mean(1), domain.y[domain.tri].mean(1))
    rotor_element = r_cent < r_in
    return VirtualDisplacement(weight, rotor_element, r_in, r_out,
                               int(np.count_nonzero(weight > 0.0)))


# --------------------------------------------------------------------------- #
# method 1 -- Coulomb local virtual work, ONE solve
# --------------------------------------------------------------------------- #
def torque_vw_coulomb(domain: FemDomain, a_nodal: np.ndarray,
                      axial_length_m: float = 1.0,
                      symmetry_multiplier: Optional[int] = None,
                      eps_rad: float = 1e-6,
                      profile: str = "linear",
                      self_test: bool = False) -> Dict[str, float]:
    """T = dW'/dtheta from a single solved field, by virtual rotor rotation.

    The derivative is taken numerically in the VIRTUAL angle, which is not the
    same thing as a finite difference between solves: there is no second solve
    and no second mesh, only an exact re-evaluation of the discrete coenergy at
    perturbed coordinates with the nodal A frozen. It is therefore free of
    solver noise, and `eps_rad` can be pushed to where round-off, not truncation,
    dominates. A central difference makes the truncation error O(eps^2).
    """
    a = np.asarray(a_nodal, dtype=np.float64).reshape(-1)
    if a.size != domain.n_nodes:
        raise ValueError(f"a_nodal has {a.size} entries, domain has {domain.n_nodes} nodes")
    vd = build_virtual_displacement(domain, profile=profile)
    if symmetry_multiplier is None:
        symmetry_multiplier = _symmetry_from_span(domain)

    def coenergy_at(delta: float) -> float:
        ang = vd.weight * delta
        ca, sa = np.cos(ang), np.sin(ang)
        xr = domain.x * ca - domain.y * sa
        yr = domain.x * sa + domain.y * ca
        b_elem, area = _element_b_on_coords(xr, yr, domain.tri, a)
        # The rotor carries its magnets: their easy axis rotates with the body,
        # otherwise a rigid rotation would fake a change in the Br.B term.
        br = domain.br
        if np.any(vd.rotor_element):
            br = br.copy()
            c, s = np.cos(delta), np.sin(delta)
            bx, by = br[vd.rotor_element, 0].copy(), br[vd.rotor_element, 1].copy()
            br[vd.rotor_element, 0] = c * bx - s * by
            br[vd.rotor_element, 1] = s * bx + c * by
        # coenergy() uses domain.area; swap in the perturbed areas for this call
        moved = _DomainAreaView(domain, area)
        return coenergy(moved, b_elem, br_override=br)

    w_plus = coenergy_at(+eps_rad)
    w_minus = coenergy_at(-eps_rad)
    dwd = (w_plus - w_minus) / (2.0 * eps_rad)
    torque = axial_length_m * symmetry_multiplier * dwd

    out = {
        "torque": float(torque),
        "coenergy_J_per_m": float(coenergy_at(0.0)),
        "eps_rad": float(eps_rad),
        "symmetry_multiplier": int(symmetry_multiplier),
        "n_moving_nodes": vd.n_moving_nodes,
        "r_inner_m": vd.r_inner_m,
        "r_outer_m": vd.r_outer_m,
        "profile": profile,
    }
    if self_test:
        # Virtual work does not care HOW the gap absorbs the displacement. If it
        # does here, the displacement field is touching something it should not.
        alt = torque_vw_coulomb(domain, a, axial_length_m, symmetry_multiplier,
                                eps_rad, profile="smoothstep", self_test=False)
        out["torque_alt_profile"] = alt["torque"]
        denom = max(abs(torque), 1e-12)
        out["profile_invariance_pct"] = float(
            100.0 * abs(alt["torque"] - torque) / denom)
    return out


class _DomainAreaView:
    """A FemDomain with its element areas replaced, for coenergy re-evaluation."""

    __slots__ = ("_d", "area")

    def __init__(self, domain: FemDomain, area: np.ndarray):
        self._d = domain
        self.area = area

    def __getattr__(self, item):
        return getattr(self._d, item)


# --------------------------------------------------------------------------- #
# method 2 -- global virtual work by finite difference over rotor steps
# --------------------------------------------------------------------------- #
def _symmetry_from_span(domain: FemDomain) -> int:
    """Sectors in the full machine, measured on the STATIONARY air-gap layer.

    Do not measure this on the raw node cloud. Two things inflate it: the
    anti-periodic image nodes `build_domain` appends, and -- much larger -- the
    sliding band, which the export draws over a wider arc than the modelled
    sector (65.5 deg against the true 44.9 deg on case 4). Either one silently
    scales every torque this module returns. The stationary `a1` layer is the
    one region whose span is the sector, which is exactly why
    `eval.torque.build_airgap_band` measures it there too.
    """
    names = domain.name_of_code
    codes = np.asarray(domain.reg_code)
    stationary = np.zeros(domain.n_elements, dtype=bool)
    for e, code in enumerate(codes):
        if int(code) < 0:
            continue                                   # remeshed = the moving band
        nm = str(names.get(int(code), ""))
        if nm == "a1":                                 # the stationary Arkkio layer
            stationary[e] = True
    if not np.any(stationary):
        raise ValueError("cannot measure sector symmetry: no stationary 'a1' layer")
    nodes = np.unique(domain.tri[stationary].ravel())
    ang = np.arctan2(domain.y[nodes], domain.x[nodes])
    span = float(ang.max() - ang.min())
    if span <= 0:
        raise ValueError("degenerate sector span")
    ratio = 2.0 * np.pi / span
    n = int(round(ratio))
    if abs(ratio - n) > 0.05:
        raise ValueError(f"sector span {np.degrees(span):.3f} deg does not divide 360 "
                         f"(got {ratio:.4f} sectors); pass symmetry_multiplier explicitly")
    return max(1, n)


def torque_vw_finite_difference(record, step: int,
                                bh_path: Optional[Path] = None,
                                axial_length_m: float = 1.0,
                                pole_pairs: Optional[int] = None,
                                elec_deg_per_step: float = 8.0,
                                ) -> Dict[str, float]:
    """T = dW'/dtheta by central-differencing coenergy across adjacent solves.

    WARNING -- this is invalid on a synchronous DOE, and it is kept because that
    fact is worth demonstrating rather than asserting. Virtual work needs
    dW'/dtheta **at constant current**. In this dataset the stator current wave
    advances together with the rotor (that is what an on-load synchronous sweep
    is), so differencing adjacent steps measures the derivative ALONG THE
    SYNCHRONOUS TRAJECTORY -- mechanical and electrical rate of change together
    -- and the two contributions very nearly cancel. Measured on case 4 it
    returns O(10) N*m/m where the true torque is O(2500).

    Making it valid requires re-solving at a displaced rotor angle with the
    currents frozen, which the export does not contain. Coulomb's route has no
    such problem: it freezes the currents by construction, which is the practical
    reason to prefer it here and not merely a cost argument.
    """
    n = len(record.samples)
    if not (1 <= step < n - 1):
        raise ValueError(f"central difference needs 1 <= step < {n - 1}, got {step}")

    w = {}
    sym = None
    for k in (step - 1, step + 1):
        # Two independent Newton solves on two different meshes -- that IS the
        # method, and the reason it is noisier than Coulomb's single-solve route.
        dom, a_full, _ = domain_and_a(record, k, bh_path)
        w[k] = coenergy(dom, dom.element_b(a_full))
        sym = _symmetry_from_span(dom) if sym is None else sym
        if pole_pairs is None:
            pole_pairs = max(1, sym // 2)

    d_theta_mech = np.radians(elec_deg_per_step / float(pole_pairs)) * 2.0   # two steps
    torque = axial_length_m * sym * (w[step + 1] - w[step - 1]) / d_theta_mech
    return {
        "torque": float(torque),
        "coenergy_prev_J_per_m": float(w[step - 1]),
        "coenergy_next_J_per_m": float(w[step + 1]),
        "d_theta_mech_rad": float(d_theta_mech),
        "pole_pairs": int(pole_pairs),
        "symmetry_multiplier": int(sym),
    }


def domain_and_a(record, step: int, bh_path: Optional[Path] = None,
                 tol: float = 1e-8, max_iter: int = 60
                 ) -> Tuple[FemDomain, np.ndarray, Dict[str, float]]:
    """The domain for one step plus a NODAL A to differentiate.

    The Motor-CAD export has no nodal A -- its ``a`` field is element-wise, like
    ``bx``/``by`` (one value per triangle). Virtual work needs the nodal degrees
    of freedom, because the whole method is the derivative of the discrete
    functional whose unknown is exactly that vector. Averaging element A onto
    nodes would fabricate a field the solver never produced and smear precisely
    the air-gap gradients the torque is made of.

    So we solve. `fem_warmstart` is the licence-free reference solver validated
    in section 23 to 0.06-0.18% of Motor-CAD's torque over a full rotor sweep;
    its converged nodal A is a legitimate field for BOTH operators, and using one
    field for both is what makes the comparison an operator comparison rather
    than a field comparison.
    """
    from fem_warmstart.newton import newton_solve

    dom = build_domain(record, step, bh_path or default_bh_path(),
                       j_source="export", magnet_polarity="from_field",
                       current_scale=1.0)
    res = newton_solve(dom, tol=tol, max_iter=max_iter)
    info = {"newton_iterations": int(res.iterations),
            "newton_residual": float(res.residual_history[-1]) if res.residual_history else float("nan"),
            "converged": bool(res.converged)}
    if not res.converged:
        raise RuntimeError(f"Newton did not converge at step {step}: {res.message}")
    return dom, np.asarray(res.a_nodal, dtype=np.float64), info
