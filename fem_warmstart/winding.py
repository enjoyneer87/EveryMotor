"""Synthesize the winding current density J from the operating point alone.

Why this is possible, and why it is needed
------------------------------------------
The section-24 audit measured that the exported OnLoadTorque excitation is a
*template constant*: every conductor region in every DOE case carries the same
~325.3 ampere-turns (0.03% spread across cases whose nominal PeakCurrent spans
18-646 A), advanced in phase by exactly the case's PhaseAdvance, and evolving as
an exact cosine in the rotor step (held-out reproduction error 0.06-0.27%,
which is the export's own 0.005 A/mm^2 quantization).

That makes J a pure function of (region, PhaseAdvance, step) — computable
*before* solving. Anything that needs a current source without touching the
exported ``fields/j`` (which the feature guard rightly blacklists as
solution-adjacent) can use this module instead: the R7-A analytical prior, the
licence-free solver on synthetic operating points, warm-start domains.

The template table
------------------
Conductor region NAMES vary between cases (case_0000 has 34 conductor regions,
case_0004 has 36 — the mesher merges/splits turn regions per geometry), so the
table is keyed by the PHYSICAL rule, not by name: the phase group of a region
is determined by its winding layer —

    layers 5-6                  -> group "56"
    layers 3-4                  -> group "34"
    layer 2 and every Turn_*    -> group "2t"

(the three groups sit 120 elec deg apart; measured within-group phase spread
0.05 deg across all regions of two cases). ``winding_template.json`` stores one
template-wide ``ampere_turns`` and one ``phi0_deg`` per group (PhaseAdvance = 0
baseline), calibrated ONCE from a reference case's exported j
(`calibrate_template`). Using one training case's solution to calibrate a
template-wide constant is the same move as reading the template's .bh file —
per-template data, not per-sample leakage; the held-out check in
`fem_warmstart/validate_prior.py` demonstrates the rule transfers across cases,
region-name variants and operating points.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

# 48-slot / 8-pole template: 2 mech deg per step x 4 pole pairs.
ELEC_DEG_PER_STEP = 8.0

TEMPLATE_PATH = Path(__file__).resolve().parent / "data" / "winding_template.json"

_CONDUCTOR_PREFIXES = ("armatureslot", "turn_")
_SLOT_RE = re.compile(r"^armatureslot[a-f](\d)$")


def is_conductor(name: str) -> bool:
    return str(name).strip().lower().startswith(_CONDUCTOR_PREFIXES)


def phase_group(name: str) -> Optional[str]:
    """Map a conductor region name to its phase group, by the layer rule."""
    n = str(name).strip().lower()
    if n.startswith("turn_"):
        return "2t"
    m = _SLOT_RE.match(n)
    if not m:
        return None
    layer = int(m.group(1))
    if layer in (5, 6):
        return "56"
    if layer in (3, 4):
        return "34"
    if layer == 2:
        return "2t"
    return None


def _region_areas_m2(record, step: int) -> Dict[str, float]:
    from eval.mesh_regions import element_areas_m2

    mesh = record.mesh
    s = record.samples[step]
    area = element_areas_m2(s.node_x_mm, s.node_y_mm, mesh.tri)
    reg = np.asarray(mesh.reg_code)
    out: Dict[str, float] = {}
    for code, name in mesh.name_of_code.items():
        if is_conductor(str(name)):
            out[str(name)] = float(area[reg == int(code)].sum())
    return out


def calibrate_template(record, steps=(0, 5, 11, 17, 23, 29, 35, 41)) -> Dict[str, dict]:
    """Fit the template constants (ampere_turns, per-group phi0) from ONE case.

    j_k = amp * cos(omega*k + phi) is fitted per region by linear least squares
    on cos/sin (several steps, so the export quantization averages out), then
    aggregated per phase group. Aggregation is also the self-check: if the
    layer rule were wrong, the within-group phase spread would be ~120 deg, not
    the asserted < 0.5 deg. phi0 subtracts the reference case's PhaseAdvance,
    making the stored phase the PhaseAdvance = 0 baseline.
    """
    mesh = record.mesh
    reg = np.asarray(mesh.reg_code)
    adv_ref = float(record.condition.get("PhaseAdvance", 0.0))
    areas = _region_areas_m2(record, 0)

    omega = np.radians(ELEC_DEG_PER_STEP) * np.asarray(steps, dtype=np.float64)
    design = np.column_stack([np.cos(omega), -np.sin(omega)])

    per_group: Dict[str, list] = {"56": [], "34": [], "2t": []}
    ampere_turns: list = []
    for code, name in mesh.name_of_code.items():
        name = str(name)
        group = phase_group(name)
        if group is None:
            continue
        mask = reg == int(code)
        jj = np.array([float(np.asarray(record.samples[k].fields["j"])[mask][0])
                       for k in steps])
        (c, s), *_ = np.linalg.lstsq(design, jj, rcond=None)
        amp = float(np.hypot(c, s))
        phi = (float(np.degrees(np.arctan2(s, c))) - adv_ref) % 360.0
        ampere_turns.append(amp * 1e6 * areas[name])       # j is A/mm^2 in the export
        per_group[group].append(phi)

    table: Dict[str, dict] = {"ampere_turns": float(np.mean(ampere_turns)),
                              "ampere_turns_spread": float(np.ptp(ampere_turns)),
                              "phi0_deg": {}, "phi0_spread_deg": {}}
    for group, phis in per_group.items():
        if not phis:
            raise ValueError(f"phase group {group!r} has no regions in the reference case")
        ref = phis[0]
        unwrapped = [ref + (((p - ref) + 180.0) % 360.0 - 180.0) for p in phis]
        spread = float(np.ptp(unwrapped))
        if spread > 0.5:
            raise ValueError(
                f"phase group {group!r} spread {spread:.2f} deg — the layer rule "
                "does not hold on this template; do not use the table")
        table["phi0_deg"][group] = float(np.mean(unwrapped)) % 360.0
        table["phi0_spread_deg"][group] = spread
    return table


def load_template(path: Optional[Path] = None) -> Dict[str, dict]:
    p = Path(path) if path is not None else TEMPLATE_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def synthesize_j(
    mesh,
    area_m2: np.ndarray,
    phase_advance_deg: float,
    step_index: int,
    template: Optional[Mapping[str, dict]] = None,
    current_scale: float = 1.0,
) -> np.ndarray:
    """Element current density (A/m^2) on the EXPORT mesh, from inputs only.

    ``current_scale`` multiplies the template ampere-turns — the hook for
    synthetic operating points at other current levels (section 24: the
    dataset itself has no current variation, so scale 1.0 reproduces it).
    """
    tpl = template if template is not None else load_template()
    at = float(tpl["ampere_turns"])
    phi0 = tpl["phi0_deg"]
    reg = np.asarray(mesh.reg_code)
    j = np.zeros(reg.size, dtype=np.float64)
    theta = ELEC_DEG_PER_STEP * float(step_index)
    for code, name in mesh.name_of_code.items():
        group = phase_group(str(name))
        if group is None:
            continue
        mask = reg == int(code)
        area = float(area_m2[mask].sum())
        if area <= 0:
            continue
        amp = current_scale * at / area                                # A/m^2
        phi = float(phi0[group]) + float(phase_advance_deg)
        j[mask] = amp * np.cos(np.radians(theta + phi))
    return j
