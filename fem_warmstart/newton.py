"""Damped Newton solve for the nonlinear magnetostatic problem.

Convergence is measured the way the PoC design fixed it: ``||R|| / ||f||`` on the
*reduced* (constrained) system, so the anti-periodic and Dirichlet rows do not
contribute a residual they cannot reduce.

The iteration count this reports is the quantity the warm-start study compares
across initial guesses, so two things are deliberate:

* the tolerance and the damping rule are identical for every initial guess --
  only ``a0`` changes;
* the first residual is evaluated *before* any solve, so an initial guess that is
  already converged costs zero iterations rather than one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.sparse.linalg import spsolve

from fem_warmstart.assemble import (
    constraint_matrix,
    load_vector,
    residual,
    stiffness,
    tangent,
)
from fem_warmstart.domain import FemDomain


@dataclass
class NewtonResult:
    converged: bool
    iterations: int
    a_nodal: np.ndarray
    residual_history: List[float] = field(default_factory=list)
    step_history: List[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    message: str = ""


def solve_linear_guess(domain: FemDomain) -> np.ndarray:
    """Air-everywhere linear solve -- a cheap, physics-aware alternative to zero.

    Not one of the three benchmarked initialisations; used to sanity-check the
    assembly, since it must produce a finite field with the right topology.
    """
    t = constraint_matrix(domain)
    f = load_vector(domain)
    nu = domain.nu_linear.copy()
    k = stiffness(domain, nu)
    a_red = spsolve((t.T @ k @ t).tocsc(), t.T @ f)
    return t @ a_red


def newton_solve(
    domain: FemDomain,
    a0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
    min_step: float = 1.0 / 64.0,
    callback: Optional[Callable[[int, float], None]] = None,
) -> NewtonResult:
    """Damped Newton on the constrained system. ``a0`` is a full nodal vector."""
    t0 = time.perf_counter()
    tmat = constraint_matrix(domain)
    f_full = load_vector(domain)
    f_red = tmat.T @ f_full
    f_norm = float(np.linalg.norm(f_red))
    if f_norm == 0.0:
        raise ValueError("zero load vector: nothing to solve")

    a = np.zeros(domain.n_nodes) if a0 is None else np.asarray(a0, float).copy()
    # project the guess onto the constraint manifold so anti-periodicity holds exactly
    a = tmat @ domain.restrict(a)

    hist: List[float] = []
    steps: List[float] = []
    for it in range(max_iter + 1):
        r_full, nu, dnu = residual(domain, a, f_full)
        r_red = tmat.T @ r_full
        rel = float(np.linalg.norm(r_red)) / f_norm
        hist.append(rel)
        if callback is not None:
            callback(it, rel)
        if rel < tol:
            return NewtonResult(True, it, a, hist, steps,
                                time.perf_counter() - t0, "converged")
        if it == max_iter:
            break
        j_full = tangent(domain, a, nu, dnu)
        j_red = (tmat.T @ j_full @ tmat).tocsc()
        try:
            delta = spsolve(j_red, -r_red)
        except Exception as exc:                      # singular tangent
            return NewtonResult(False, it, a, hist, steps,
                                time.perf_counter() - t0, f"linear solve failed: {exc}")
        if not np.all(np.isfinite(delta)):
            return NewtonResult(False, it, a, hist, steps,
                                time.perf_counter() - t0, "non-finite Newton step")
        d_full = tmat @ delta
        # backtracking on the residual norm
        step = 1.0
        while True:
            trial = a + step * d_full
            r_try = tmat.T @ residual(domain, trial, f_full)[0]
            if np.linalg.norm(r_try) < np.linalg.norm(r_red) or step <= min_step:
                break
            step *= 0.5
        a = a + step * d_full
        steps.append(step)

    return NewtonResult(False, max_iter, a, hist, steps,
                        time.perf_counter() - t0, "max iterations reached")
