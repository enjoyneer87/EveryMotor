"""P1 assembly, residual and Newton tangent for 2D nonlinear magnetostatics.

Solves  -div( nu(|B|) grad a_z ) = j_z + curl(nu_m B_r)|_z  with  B = curl(a_z z^).

Weak form on a P1 triangulation, with ``dphi_i/dx = b_i/(2A)`` and
``dphi_i/dy = c_i/(2A)`` (the same coefficients `phase1_static/discrete_curl.py`
uses, so a solved ``a`` can be pushed through the existing curl operator without
a convention change):

    K^e_ij = nu * (b_i b_j + c_i c_j) / (4A)
    f^e_i  = j_z * A / 3                                 (source current)
           + nu_m * (B_rx * c_i - B_ry * b_i) / 2        (magnet remanence)

The magnet term is the ``+ int nu_m ( B_rx dw/dy - B_ry dw/dx )`` that appears
when ``H = nu_m (B - B_r)`` is substituted into the weak form.

The constraint map from `domain.py` is applied as ``T`` (one column per reduced
dof, entries +-1), so the reduced system is ``T^T K T`` -- symmetric positive
definite, and anti-periodicity is enforced exactly rather than penalised.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from fem_warmstart.domain import MAGNET, STEEL, FemDomain


def element_nu(domain: FemDomain, b_mag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(nu, dnu/dB) per element: the matching BH curve in steel, constant elsewhere."""
    nu = domain.nu_linear.copy()
    dnu = np.zeros_like(nu)
    for idx, curve in enumerate(domain.curves):
        sel = domain.curve_index == idx
        if not np.any(sel):
            continue
        nu_s, dnu_s = curve.nu_and_dnu(b_mag[sel])
        nu[sel] = nu_s
        dnu[sel] = dnu_s
    return nu, dnu


def constraint_matrix(domain: FemDomain) -> csr_matrix:
    """``T`` with ``a_full = T @ a_reduced``."""
    free = np.flatnonzero(domain.dof_of_node >= 0)
    return coo_matrix(
        (domain.dof_sign[free], (free, domain.dof_of_node[free])),
        shape=(domain.n_nodes, domain.n_dof),
    ).tocsr()


def _pairs(domain: FemDomain):
    """Row/col index pattern of the 3x3 element blocks."""
    tri = domain.tri
    rows = np.repeat(tri, 3, axis=1).ravel()          # i i i j j j k k k
    cols = np.tile(tri, (1, 3)).ravel()               # i j k i j k i j k
    return rows, cols


def stiffness(domain: FemDomain, nu: np.ndarray) -> csr_matrix:
    """Full (unconstrained) stiffness for a given element-wise reluctivity."""
    gb, gc, area = domain.grad_b, domain.grad_c, domain.area
    # dphi_i . dphi_j * A * nu
    block = (gb[:, :, None] * gb[:, None, :] + gc[:, :, None] * gc[:, None, :])
    block *= (nu * area)[:, None, None]
    rows, cols = _pairs(domain)
    return coo_matrix((block.ravel(), (rows, cols)),
                      shape=(domain.n_nodes, domain.n_nodes)).tocsr()


def load_vector(domain: FemDomain) -> np.ndarray:
    """Source term: winding current plus magnet remanence."""
    f = np.zeros(domain.n_nodes, dtype=np.float64)
    # currents: j_z * A / 3 on each of the three nodes
    contrib = (domain.j_z * domain.area / 3.0)[:, None] * np.ones((1, 3))
    np.add.at(f, domain.tri, contrib)
    # magnets
    mag = np.flatnonzero(domain.material == MAGNET)
    if mag.size:
        nu_m = domain.nu_linear[mag][:, None]
        brx = domain.br[mag, 0][:, None]
        bry = domain.br[mag, 1][:, None]
        two_a = (2.0 * domain.area[mag])[:, None]
        b = domain.grad_b[mag] * two_a          # back to raw b_i
        c = domain.grad_c[mag] * two_a          # back to raw c_i
        np.add.at(f, domain.tri[mag], nu_m * (brx * c - bry * b) / 2.0)
    return f


def residual(domain: FemDomain, a_nodal: np.ndarray,
             f: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(R, nu, dnu)`` with ``R = K(nu(|B(a)|)) a - f`` on full nodes."""
    if f is None:
        f = load_vector(domain)
    b = domain.element_b(a_nodal)
    b_mag = np.hypot(b[:, 0], b[:, 1])
    nu, dnu = element_nu(domain, b_mag)
    k = stiffness(domain, nu)
    return k @ a_nodal - f, nu, dnu


def tangent(domain: FemDomain, a_nodal: np.ndarray, nu: np.ndarray,
            dnu: np.ndarray) -> csr_matrix:
    """Newton tangent ``dR/da``.

    ``R^e_i = nu(|B|) (G a)_i`` with ``G_ij = (b_i b_j + c_i c_j)/(4A)``, so

        dR^e_i/da_k = nu G_ik + (G a)_i * dnu/dB * d|B|/da_k,
        d|B|/da_k   = ( Bx c_k - By b_k ) / (2A |B|).
    """
    gb, gc, area, tri = domain.grad_b, domain.grad_c, domain.area, domain.tri
    a_e = a_nodal[tri]                                            # (E,3)
    g = (gb[:, :, None] * gb[:, None, :] + gc[:, :, None] * gc[:, None, :]) * area[:, None, None]
    block = nu[:, None, None] * g                                  # material tangent part 1
    ga = np.einsum("eij,ej->ei", g, a_e)                           # (E,3)

    bvec = domain.element_b(a_nodal)
    b_mag = np.hypot(bvec[:, 0], bvec[:, 1])
    safe = b_mag > 1e-12
    dB_da = np.zeros_like(a_e)
    if np.any(safe):
        dB_da[safe] = (
            bvec[safe, 0][:, None] * gc[safe] - bvec[safe, 1][:, None] * gb[safe]
        ) / b_mag[safe][:, None]
    block = block + ga[:, :, None] * (dnu[:, None, None] * dB_da[:, None, :])

    rows, cols = _pairs(domain)
    return coo_matrix((block.ravel(), (rows, cols)),
                      shape=(domain.n_nodes, domain.n_nodes)).tocsr()
