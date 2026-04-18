"""Post-processing utility for computing current density J from motor conditions.

J is not included in the 4-channel training target [Bx, By, A, Je].
Use these pure functions to derive J from operating-point conditions for
comparison against j_raw values stored in sample dicts.

Motor condition feature vector (cond_vec) layout in motor_dataset.py:
    index 0 : rotor_angle_norm
    index 1 : time_norm  (or similar temporal feature)
    index 2 : PeakCurrent [A]    <- I_phase
    index 3 : PhaseAdvance [deg] <- beta

Usage in a notebook or post-processing script::

    from phase1_static.postproc_j import compute_j_from_conditions

    I_phase_A  = sample["cond_vec"][2]   # PeakCurrent [A]
    beta_deg   = sample["cond_vec"][3]   # PhaseAdvance [deg]
    j_nodes    = compute_j_from_conditions(
        I_phase_A=I_phase_A,
        phase_advance_deg=beta_deg,
        turns=turns,
        slot_area_m2=slot_area_m2,
    )
    j_raw      = sample["j_raw"]         # FEM-computed J [A/m^2]
"""

from __future__ import annotations

import math

import numpy as np


def compute_j_from_conditions(
    I_phase_A: float,
    phase_advance_deg: float,
    turns: int,
    slot_area_m2: float,
) -> float:
    """Compute slot-average current density J [A/m^2] from motor operating conditions.

    This is a scalar function — the result is uniform over all nodes in the
    same stator slot.  Use it to create a node-level array by broadcasting.

    Parameters
    ----------
    I_phase_A:
        Peak phase current [A] (PeakCurrent DOE parameter).
    phase_advance_deg:
        Current phase advance (beta) [degrees] from q-axis (PhaseAdvance DOE parameter).
    turns:
        Number of series turns per phase per slot.
    slot_area_m2:
        Effective conductor area of one slot [m^2].

    Returns
    -------
    float
        Slot-average current density magnitude [A/m^2].

    Notes
    -----
    The instantaneous slot current at the peak torque angle is:

        I_slot = sqrt(2) * I_phase * sin(beta_rad) * turns

    where beta_rad = phase_advance_deg * pi / 180.
    Magnitude is returned (unsigned); sign depends on slot polarity which
    is stored separately in the mesh region data.
    """
    if slot_area_m2 <= 0.0:
        raise ValueError(f"slot_area_m2 must be positive, got {slot_area_m2}")
    beta_rad = math.radians(phase_advance_deg)
    I_slot = math.sqrt(2.0) * float(I_phase_A) * math.sin(beta_rad) * int(turns)
    return abs(I_slot) / float(slot_area_m2)


def compute_j_node_array(
    I_phase_A: float,
    phase_advance_deg: float,
    turns: int,
    slot_area_m2: float,
    n_nodes: int,
) -> np.ndarray:
    """Return a 1-D numpy array of length n_nodes filled with J [A/m^2].

    All nodes receive the same scalar value.  For sign-aware J, multiply
    by the per-node polarity mask from the mesh region data.

    Parameters
    ----------
    I_phase_A, phase_advance_deg, turns, slot_area_m2:
        See :func:`compute_j_from_conditions`.
    n_nodes:
        Number of mesh nodes.

    Returns
    -------
    np.ndarray, shape (n_nodes,), dtype float32
    """
    j_scalar = compute_j_from_conditions(I_phase_A, phase_advance_deg, turns, slot_area_m2)
    return np.full(n_nodes, j_scalar, dtype=np.float32)
