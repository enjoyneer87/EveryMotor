"""PBC boundary condition encoding for grid-based models (FNO, RNN).

Graph-based models (MGN, SymMGN, GINO) use PBC edges directly.
Grid-based models encode PBC as an additional input channel:
  - pbc_mask channel: 1.0 at boundary nodes/pixels, 0.0 elsewhere
  - sign channel:     +1.0 (periodic) or -1.0 (anti-periodic) at boundary

This module provides helpers to encode PBC constraints into grid tensors
for FNO/RNN training and inference.
"""

from __future__ import annotations

import numpy as np


def build_pbc_mask_channel(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sector_angle_deg: float = 45.0,
    anti_periodic: bool = True,
    boundary_width_frac: float = 0.02,
) -> np.ndarray:
    """Build a 2-channel PBC boundary mask for grid inputs.

    Channel 0: boundary presence mask (1.0 at sector edges, 0.0 elsewhere)
    Channel 1: PBC sign (+1.0 periodic, -1.0 anti-periodic at boundary, 0.0 elsewhere)

    Args:
        grid_x:              [H, W] x-coordinates of grid pixels
        grid_y:              [H, W] y-coordinates of grid pixels
        sector_angle_deg:    Sector span in degrees (default 45.0 for 1/8)
        anti_periodic:       If True, sign channel = -1.0 at boundary
        boundary_width_frac: Fraction of grid width to mark as boundary

    Returns:
        mask: [2, H, W] float32
    """
    H, W = grid_x.shape
    theta = np.arctan2(grid_y, grid_x)  # [H, W], radians in [-pi, pi]

    sector_rad = np.deg2rad(sector_angle_deg)
    w_pixels = max(1, int(boundary_width_frac * W))

    # Boundary at theta ≈ 0 (master) and theta ≈ sector_angle (slave)
    at_master = np.abs(theta) < np.deg2rad(sector_angle_deg * boundary_width_frac * 2)
    at_slave = np.abs(theta - sector_rad) < np.deg2rad(sector_angle_deg * boundary_width_frac * 2)
    boundary_mask = (at_master | at_slave).astype(np.float32)

    sign_val = -1.0 if anti_periodic else 1.0
    sign_channel = boundary_mask * sign_val

    return np.stack([boundary_mask, sign_channel], axis=0)  # [2, H, W]


def augment_grid_input_with_pbc(
    grid_input: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sector_angle_deg: float = 45.0,
    anti_periodic: bool = True,
) -> np.ndarray:
    """Append PBC mask channels to a grid input tensor.

    Args:
        grid_input:  [C, H, W] existing input channels
        grid_x:      [H, W] grid x-coords
        grid_y:      [H, W] grid y-coords

    Returns:
        augmented: [C+2, H, W]
    """
    pbc_channels = build_pbc_mask_channel(
        grid_x, grid_y, sector_angle_deg, anti_periodic
    )
    return np.concatenate([grid_input, pbc_channels], axis=0)
