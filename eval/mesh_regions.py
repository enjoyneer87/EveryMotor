"""Motor-CAD mesh region grouping for region-resolved error reporting.

A single whole-mesh nRMSE hides the errors a motor designer cares about. Most of
the mesh is shaft and back-iron where |B| is small and easy to fit; the airgap
and tooth tips carry the torque and saturate. This module folds the ~67 raw
Motor-CAD region codes into a handful of engineering groups so metrics can be
reported per group.

Group assignment is by region *name* (Motor-CAD names are stable across the DOE),
with one geometric refinement: stator iron is split into teeth and yoke at the
outer radius of the slot regions, which is exactly the tooth/back-iron boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

REGION_GROUP_ORDER: Tuple[str, ...] = (
    "airgap",
    "stator_teeth",
    "stator_yoke",
    "winding",
    "magnet",
    "rotor_iron",
    "rotor_air",
    "shaft",
    "other",
)

# Airgap mesh layers are named a1, a2, ... in the Motor-CAD export.
AIRGAP_NAME_PATTERN = re.compile(r"^a\d+$", re.IGNORECASE)

_NAME_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    (AIRGAP_NAME_PATTERN, "airgap"),
    (re.compile(r"^armatureslot", re.IGNORECASE), "winding"),
    (re.compile(r"^turn_", re.IGNORECASE), "winding"),
    (re.compile(r"^impreg", re.IGNORECASE), "winding"),
    (re.compile(r"magnet", re.IGNORECASE), "magnet"),
    (re.compile(r"^shaft", re.IGNORECASE), "shaft"),
    (re.compile(r"^rotor\s*pocket", re.IGNORECASE), "rotor_air"),
    (re.compile(r"^rotor", re.IGNORECASE), "rotor_iron"),
    # StatorAir / StatorWedge are slot-opening fillers, not iron — group them
    # with the winding slot they belong to rather than polluting tooth metrics.
    (re.compile(r"^statorair", re.IGNORECASE), "winding"),
    (re.compile(r"^statorwedge", re.IGNORECASE), "winding"),
    (re.compile(r"^stator", re.IGNORECASE), "_stator_iron"),
)


@dataclass(frozen=True)
class RegionGrouping:
    """Immutable per-element assignment of mesh elements to engineering groups.

    Attributes
    ----------
    group_of_element:
        ``(n_elements,)`` array of group names.
    tooth_yoke_radius_m:
        Radius used to split stator iron into teeth and yoke, or ``nan`` when no
        slot regions were found and the split was skipped.
    """

    group_of_element: np.ndarray
    tooth_yoke_radius_m: float

    def mask(self, group: str) -> np.ndarray:
        """Boolean element mask for one group."""
        return self.group_of_element == group

    def counts(self) -> Dict[str, int]:
        return {g: int(np.count_nonzero(self.mask(g))) for g in REGION_GROUP_ORDER}


def classify_region_name(name: str) -> str:
    """Map one Motor-CAD region name to a coarse group.

    Stator iron is returned as the private marker ``"_stator_iron"``; it is
    refined into teeth/yoke by `build_region_grouping`, which has the radii.
    """
    text = str(name).strip()
    for pattern, group in _NAME_RULES:
        if pattern.search(text):
            return group
    return "other"


def build_region_grouping(
    reg_code_of_element: Sequence[int],
    name_of_code: Mapping[int, str],
    centroid_radius_m: Sequence[float],
) -> RegionGrouping:
    """Assign every element to an engineering group.

    Parameters
    ----------
    reg_code_of_element:
        ``mesh/reg_code`` — one Motor-CAD region code per element.
    name_of_code:
        ``regions/reg_code`` -> ``regions/name`` mapping from the H5.
    centroid_radius_m:
        Element centroid radius, used only for the tooth/yoke split.
    """
    codes = np.asarray(reg_code_of_element, dtype=np.int64)
    radius = np.asarray(centroid_radius_m, dtype=np.float64)
    if codes.shape != radius.shape:
        raise ValueError(
            f"reg_code and centroid radius must align, got {codes.shape} vs {radius.shape}"
        )

    coarse = np.array(
        [classify_region_name(name_of_code.get(int(c), "")) for c in codes],
        dtype=object,
    )

    # Tooth/yoke boundary = outer edge of the slots. Everything radially inside
    # that is tooth; outside is back-iron.
    slot_mask = coarse == "winding"
    if np.any(slot_mask):
        boundary = float(radius[slot_mask].max())
    else:
        boundary = float("nan")

    groups = coarse.copy()
    iron = coarse == "_stator_iron"
    if np.any(iron):
        if np.isfinite(boundary):
            groups[iron & (radius <= boundary)] = "stator_teeth"
            groups[iron & (radius > boundary)] = "stator_yoke"
        else:
            groups[iron] = "stator_yoke"

    return RegionGrouping(
        group_of_element=groups.astype(str),
        tooth_yoke_radius_m=boundary,
    )


def sliding_band_codes(
    name_of_code: Mapping[int, str],
    moving_reg_codes: Sequence[int],
) -> Tuple[int, ...]:
    """Airgap layer codes that rotate with the rotor — the sliding band.

    The Motor-CAD moving-mesh export writes one reference connectivity plus
    per-step node coordinates. That works for rigidly rotating regions, but the
    solver *re-meshes* the airgap sliding band at every step, so the reference
    connectivity does not describe the band at any rotated position: combining
    them tangles 180-246 elements from step 3 onward (all of them in layer a2).

    Any per-step geometric quantity on these elements — element gradients, edge
    vectors, areas — is therefore invalid and must be excluded. The stationary
    stator-side layer (a1) is unaffected, which is why the torque band is built
    on it.
    """
    moving = {int(c) for c in moving_reg_codes}
    return tuple(
        sorted(
            code
            for code in moving
            if AIRGAP_NAME_PATTERN.match(str(name_of_code.get(int(code), "")))
        )
    )


def sliding_band_mask(
    reg_code_of_element: Sequence[int],
    name_of_code: Mapping[int, str],
    moving_reg_codes: Sequence[int],
) -> np.ndarray:
    """Boolean element mask selecting the re-meshed sliding band."""
    codes = sliding_band_codes(name_of_code, moving_reg_codes)
    if not codes:
        return np.zeros(len(reg_code_of_element), dtype=bool)
    return np.isin(np.asarray(reg_code_of_element, dtype=np.int64), np.asarray(codes, dtype=np.int64))


def read_region_names(h5_file) -> Dict[int, str]:
    """Read the ``regions/`` name table from an open Motor-CAD H5 file.

    Returns an empty mapping when the file has no region table (the static-mesh
    exports do not carry one), which degrades region metrics to ``other``
    rather than failing the run.
    """
    if "regions/name" not in h5_file or "regions/reg_code" not in h5_file:
        return {}
    names = h5_file["regions/name"][:]
    codes = h5_file["regions/reg_code"][:]
    decoded = [n.decode("utf-8", "replace") if isinstance(n, bytes) else str(n) for n in names]
    return {int(c): n for c, n in zip(codes, decoded)}


def element_centroids_m(
    node_x_mm: np.ndarray,
    node_y_mm: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Element centroid coordinates in metres from millimetre node coordinates."""
    i1, i2, i3 = tri
    cx = (node_x_mm[i1] + node_x_mm[i2] + node_x_mm[i3]) / 3.0
    cy = (node_y_mm[i1] + node_y_mm[i2] + node_y_mm[i3]) / 3.0
    return cx * 1e-3, cy * 1e-3


def element_areas_m2(
    node_x_mm: np.ndarray,
    node_y_mm: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Triangle areas in m^2 from millimetre node coordinates."""
    i1, i2, i3 = tri
    x = np.stack([node_x_mm[i1], node_x_mm[i2], node_x_mm[i3]], axis=1) * 1e-3
    y = np.stack([node_y_mm[i1], node_y_mm[i2], node_y_mm[i3]], axis=1) * 1e-3
    cross = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    return 0.5 * np.abs(cross)
