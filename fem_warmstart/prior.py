"""R7-A: a physics prior for the surrogate, computed from inputs only.

The prior is a *linear* magnetostatic solve of the exact case geometry —
initial-permeability steel, template magnets, and the section-24 synthetic
winding current — turned into three per-node features:

    prior_a         nodal A of the linear solve (the flux function)
    prior_bx/by     nodal average of the linear solve's element B

Why a linear FEM prior instead of a closed-form slotless model: this is an
*interior*-magnet rotor. Surface-PM analytical models do not represent flux
barriers or bridges at all, while the linear solve captures the exact geometry,
slotting and barriers and misses only saturation — which is precisely the
correction the network is left to learn.

Leakage status: everything here is computable before the FEM solve the
surrogate is meant to replace. The two solution-derived inputs the solver
benchmarks use are switched off (`j_source="synthetic"`,
`magnet_polarity="radial_outward"`); ``sample.fields`` is never read. The BH
file and the winding template are per-template constants, like the mesh itself.

Determinism: sparse direct solve (SuperLU) on fixed matrices — bit-stable for
a given scipy build. Results are cached per (dataset, case, step) under
``results/prior_cache`` so a training run pays the ~1 s/step solve once.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_BH_CANDIDATES = (
    Path(__file__).resolve().parent / "data" / "steel_bh_autofile.bh",
    Path("D:/KDH/Sim_4SolverX/DOE_Ext/case_0000/TestCAD1/FEResultsData/"
         "Steel_Material_BH_Magnetic_Properties_Autofile.bh"),
)
CACHE_ROOT = _REPO / "results" / "prior_cache"

PRIOR_FEATURE_NAMES = ("prior_a", "prior_bx", "prior_by")


def default_bh_path() -> Path:
    for p in _BH_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "no BH autofile found; copy the template's "
        "Steel_Material_BH_Magnetic_Properties_Autofile.bh to fem_warmstart/data/steel_bh_autofile.bh"
    )


def _step_index(record, sample) -> int:
    for i, s in enumerate(record.samples):
        if s is sample or s.step_index == sample.step_index:
            return i
    raise ValueError(f"sample step_index {sample.step_index} not found in record")


# Effective relative permeability of the steel in the linear prior solve.
# nu_init (mu_r ~2500) ignores saturation: the flux overshoots ~3.8x and, worse,
# short-circuits through the rotor bridges that the real machine saturates shut,
# so the SHAPE is wrong too (rescaled nRMSE 72%). A low effective mu_r mimics
# the saturated bridges; the sweep over {nu_init, 1000, 300, 100, 50, 30, 20,
# 15, 10, 7, 5} on cases 4/7 x steps 0/20 has a flat optimum at mu_r 20-30
# (52.4-52.6% rescaled nRMSE, pooled correlation ~0.85). The absolute scale is
# irrelevant to the network — node features are standardized per column — so
# only this shape number matters. Measured 2026-08-04, validate_prior --sweep-mu.
STEEL_MU_EFF_R = 20.0

# Section 26: the template excitation is 460 A RMS = 650.538 A peak, and
# winding_template.json's ampere_turns was calibrated against exactly that. The
# synthetic J is therefore the template current unless it is scaled, so the prior
# for a case solved at a different current must carry
# current_scale = I_pk(case) / TEMPLATE_IPK_A.
#
# This matters from R8 on and not before. Every pre-R8 case truly ran at 650.538 A
# whatever its nominal label said (that WAS the section-24 defect), so their scale is
# 1.0 and their cached priors stay byte-identical -- which keeps R7-A comparable.
# Leaving it unwired for the R8 set would reproduce the section-24 disease inside the
# prior itself: the current axis would move while the prior the network sees did not.
TEMPLATE_IPK_A = 650.538238691624


def case_current_scale(record) -> float:
    """I_pk(case) / template I_pk for cases whose current axis is real; else 1.0.

    The gate is ``excitation_wired``, not the presence of a PeakCurrent value. Every
    pre-R8 manifest carries a nominal PeakCurrent that the solve never consumed (the
    section-24 defect: e.g. case_0000 is labelled 224.05 A but was solved at 650.538 A
    like every other one). Scaling by that label would be worse than not scaling at
    all -- it would drive the prior with a current the case never saw, and silently
    invalidate the existing prior cache that R7-A was trained against. So: scale only
    when the manifest asserts the excitation was actually wired (R8 onward, and the v2
    manifest which flags the re-labelled 240 as wired=0 with the true 650.538 A).
    """
    cond = getattr(record, "condition", None) or {}
    if not cond.get("excitation_wired"):
        return 1.0
    try:
        val = float(cond.get("PeakCurrent"))
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(val) or val <= 0.0:
        return 1.0
    return val / TEMPLATE_IPK_A


def compute_prior(record, step: int, bh_path: Optional[Path] = None,
                  mu_eff_r: float = STEEL_MU_EFF_R,
                  current_scale: Optional[float] = None) -> np.ndarray:
    """(n_export_nodes, 3) float64: [prior_a, prior_bx, prior_by]."""
    from scipy.sparse.linalg import spsolve

    from fem_warmstart.assemble import constraint_matrix, load_vector, stiffness
    from fem_warmstart.bh_curve import MU0
    from fem_warmstart.domain import STEEL, build_domain

    if current_scale is None:
        current_scale = case_current_scale(record)
    domain = build_domain(
        record, step, bh_path or default_bh_path(),
        j_source="synthetic", magnet_polarity="radial_outward",
        current_scale=current_scale,
    )

    # Linear solve: fixed effective permeability in steel (or the curves'
    # initial permeability when mu_eff_r is None), the already-linear values
    # elsewhere (air nu0, magnets nu0/1.05).
    nu = domain.nu_linear.copy()
    if mu_eff_r is None:
        for idx, curve in enumerate(domain.curves):
            sel = domain.curve_index == idx
            if np.any(sel):
                nu[sel] = curve.nu_init
    else:
        nu[domain.material == STEEL] = 1.0 / (MU0 * float(mu_eff_r))
    t = constraint_matrix(domain)
    k = stiffness(domain, nu)
    f = load_vector(domain)
    a_full = t @ spsolve((t.T @ k @ t).tocsc(), t.T @ f)

    b_elem = domain.to_export_elements(domain.element_b(a_full), fill=0.0)

    # node average of the element field on the EXPORT mesh
    mesh = record.mesh
    i1, i2, i3 = mesh.tri
    n_nodes = mesh.n_nodes
    idx3 = np.concatenate([i1, i2, i3]).astype(np.int64)
    counts = np.bincount(idx3, minlength=n_nodes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    node_b = np.empty((n_nodes, 2), dtype=np.float64)
    for c in range(2):
        vals = np.tile(b_elem[:, c], 3)
        node_b[:, c] = np.bincount(idx3, weights=vals, minlength=n_nodes) / counts

    a_nodes = a_full[: domain.n_original_nodes]
    return np.column_stack([a_nodes, node_b])


def nodal_prior_features(record, sample, bh_path: Optional[Path] = None,
                         cache: bool = True) -> Dict[str, np.ndarray]:
    """The builder-facing API: {feature_name: (n_nodes,) float64}.

    Cache layout: results/prior_cache/<dataset>/case_XXXX.npz with one
    ``step_<k>`` array of shape (n_nodes, 3) per rotor position. The dataset
    key is derived from the RESOLVED h5 path's grandparent directory — note
    that the doe240 manifest references cases 40-119's files inside
    backup/doe_data_120, so one training set legitimately spans two cache
    directories (doe_data_120 + doe_data_240). Keys resolve identically on
    host and in the container because both see the same repo-relative paths;
    build_prior_cache populates whatever keys the resolution produces.
    """
    step = _step_index(record, sample)
    scale = case_current_scale(record)
    arr: Optional[np.ndarray] = None
    cache_file: Optional[Path] = None
    if cache and not os.environ.get("PRIOR_CACHE_DISABLE"):
        dataset = Path(record.path).parent.parent.parent.name or "dataset"
        cache_file = CACHE_ROOT / dataset / f"case_{record.case_index:04d}.npz"
        if cache_file.exists():
            with np.load(cache_file) as z:
                key = f"step_{step}"
                # The current the entry was computed at. Entries written before the
                # R8 current axis existed carry no scale and were, correctly, all at
                # the template current -- so absent means 1.0. A mismatch must
                # recompute rather than serve a prior for the wrong excitation.
                cached_scale = float(z[f"cs_{step}"]) if f"cs_{step}" in z else 1.0
                if key in z and abs(cached_scale - scale) <= 1e-9 * max(1.0, abs(scale)):
                    cand = np.asarray(z[key], dtype=np.float64)
                    if cand.shape == (record.mesh.n_nodes, 3):
                        arr = cand
    if arr is None:
        arr = compute_prior(record, step, bh_path, current_scale=scale)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if cache_file.exists():
                with np.load(cache_file) as z:
                    existing = {k: z[k] for k in z.files}
            existing[f"step_{step}"] = arr.astype(np.float32)
            existing[f"cs_{step}"] = np.asarray(scale, dtype=np.float64)
            tmp = cache_file.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **existing)
            os.replace(tmp, cache_file)
    return {name: np.ascontiguousarray(arr[:, i])
            for i, name in enumerate(PRIOR_FEATURE_NAMES)}
