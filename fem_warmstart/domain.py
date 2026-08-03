"""Build a solvable 2D magnetostatic domain from a Motor-CAD DOE export.

Why this module exists at all
----------------------------
The PoC design assumed the export is a clean 45-degree wedge with two cut planes.
It is not. Measured on `backup/doe_data/case_0004`, step 0:

    stator regions (r > 68.68 mm)   theta in [-45.000,   0.000]
    rotor  regions (r < 68.68 mm)   theta in [-65.546, -20.497]

The rotor is drawn at its physical angle (already -20.497 deg at step 0), so the
union spans 65.5 deg and is *not* a sector. The node graph is nevertheless a
single connected component: stator and rotor share the r = 68.68 mm arc over
their angular overlap, theta in [-45, -20.497]. What is missing is the coupling
for the rotor arc beyond -45 deg, whose physical partner is the stator arc in
[-20.497, 0] one sector away, with the anti-periodic sign.

So the domain is closed with three families of linear constraints (all of the
form ``a_i = s * a_j``), not by moving any geometry:

    (C1) stator cut planes      a(theta=0, r)        = -a(theta=-45, r)
    (C2) rotor cut planes       a(theta=-20.497, r)  = -a(theta=-65.546, r)
    (C3) sliding-band wrap      a_stator(phi)        = -a_rotor(phi - 45)
                                for phi in [-20.497, 0] on the r=68.68 arc
    (D)  Dirichlet              a = 0 on the outer stator arc (r = 99 mm)

Folding the rotor geometrically is unnecessary: a P1 stiffness matrix is
invariant under a rigid rotation of the element, and flipping the sign of all
three nodal values leaves the quadratic form unchanged. Only the coupling is
missing, and that is exactly what (C3) supplies.

The constraints are resolved by a **signed union-find**, so a node that sits on
two of them at once (the r=68.68, theta=-20.497 corner sits on C2 and C3) is
merged consistently instead of being constrained twice.

Materials
---------
Read from the solve, not hard-coded:

* steel BH from the `.bh` autofile (`bh_curve.py`), nu(B) with a saturated tail;
* magnet remanence from the same file's ``Method:PB_Brem_var`` blocks -- the
  demagnetisation curve's B at H=0 is Br at the solve temperature (1.21568 T for
  N42EH at 80 C, matching Magnet_Br_at_20=1.31 with TBr=-0.12 %/K);
* magnet easy axis from the region's geometry (PCA minor axis of the element
  centroids: these are elongated blocks, singular-value ratio 3.6-4.3), with the
  polarity taken from the sign of the exported <B> along that axis. All four
  magnets of case_0004 come out pointing outward, i.e. one pole per sector, which
  is the expected V-shape IPM arrangement;
* source current density is the exported ``fields/j``, which is in **A/mm^2**:
  integrating it over the sector as A/mm^2 gives 241.7 A against a 224 A peak
  phase current, while reading it as A/m^2 gives 2e-4 A.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from fem_warmstart.bh_curve import MU0, NU0, Reluctivity, load_steel_reluctivity

SECTOR_DEG = 45.0

AIR = 0
STEEL = 1
MAGNET = 2

# Region-name -> material. Order matters: the first matching pattern wins.
_MATERIAL_PATTERNS: Tuple[Tuple[re.Pattern, int], ...] = (
    (re.compile(r"^rotor\s+pocket", re.I), AIR),      # before the bare "rotor" rule
    (re.compile(r"magnet", re.I), MAGNET),
    (re.compile(r"^stator(_\d+)?$", re.I), STEEL),
    (re.compile(r"^rotor$", re.I), STEEL),
    (re.compile(r"^shaft$", re.I), STEEL),
)

_MAGNET_BLOCK_RE = re.compile(
    r"Code:\s*\d+\s+Regions:\s*(\S+)\s+Method:\s*(\S+)\s+Material:\s*(\S+)"
)


class DomainError(ValueError):
    """Raised when the export cannot be turned into a solvable domain."""


# --------------------------------------------------------------------------- #
# magnet remanence from the .bh autofile
# --------------------------------------------------------------------------- #
def parse_magnet_remanence(path: str | Path) -> Dict[str, float]:
    """``{region_name: Br[T]}`` from the ``Method:PB_Brem_var`` blocks.

    Each block is a demagnetisation curve ``idx H B`` ending at ``H = 0``; that
    last point is the remanence at the solve temperature, so no temperature
    correction has to be re-derived here.
    """
    text = Path(path).read_bytes().decode("latin-1", errors="replace")
    lines = text.splitlines()
    out: Dict[str, float] = {}
    i = 0
    while i < len(lines):
        m = _MAGNET_BLOCK_RE.search(lines[i])
        if not m:
            i += 1
            continue
        region = m.group(1)
        try:
            n = int(lines[i + 1].strip())
        except (ValueError, IndexError):
            i += 1
            continue
        last_h = None
        last_b = None
        for k in range(n):
            parts = lines[i + 2 + k].split()
            if len(parts) < 3:
                break
            last_h, last_b = float(parts[1]), float(parts[2])
        if last_h is not None and abs(last_h) < 1e-6:
            out[region] = float(last_b)
        i += 2 + n
    if not out:
        raise DomainError(f"no PB_Brem_var magnet blocks found in {path}")
    return out


# --------------------------------------------------------------------------- #
# signed union-find
# --------------------------------------------------------------------------- #
class SignedUnionFind:
    """Union-find over ``a_i = sign_i * a_root``.

    Merging two nodes already in the same class with an *inconsistent* sign would
    force ``a = -a``, i.e. ``a = 0``. That is a real modelling error rather than a
    degenerate case, so it is recorded and reported instead of being absorbed.
    """

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.sign = np.ones(n, dtype=np.float64)
        self.conflicts = 0

    def find(self, i: int) -> Tuple[int, float]:
        s = 1.0
        root = i
        while self.parent[root] != root:
            s *= self.sign[root]
            root = self.parent[root]
        # path compression
        node = i
        acc = s
        while self.parent[node] != node:
            nxt = int(self.parent[node])
            nxt_sign = self.sign[node]
            self.parent[node] = root
            self.sign[node] = acc
            acc /= nxt_sign
            node = nxt
        return int(root), float(s)

    def union(self, i: int, j: int, relative_sign: float) -> bool:
        """Impose ``a_i = relative_sign * a_j``. Returns False on a conflict."""
        ri, si = self.find(i)
        rj, sj = self.find(j)
        if ri == rj:
            if abs(si - relative_sign * sj) > 1e-9:
                self.conflicts += 1
                return False
            return True
        # a_i = si*a_ri, a_j = sj*a_rj  =>  a_ri = (relative_sign*sj/si) a_rj
        self.parent[ri] = rj
        self.sign[ri] = relative_sign * sj / si
        return True


# --------------------------------------------------------------------------- #
# domain
# --------------------------------------------------------------------------- #
@dataclass
class FemDomain:
    """Everything the assembler needs, in SI units, with node coords in metres."""

    x: np.ndarray                  # (N,)
    y: np.ndarray                  # (N,)
    tri: np.ndarray                # (E,3), counter-clockwise
    area: np.ndarray               # (E,) m^2, positive
    grad_b: np.ndarray             # (E,3)  dphi_i/dx = b_i / (2A)
    grad_c: np.ndarray             # (E,3)  dphi_i/dy = c_i / (2A)
    material: np.ndarray           # (E,) AIR / STEEL / MAGNET
    nu_linear: np.ndarray          # (E,) reluctivity for the non-steel elements
    j_z: np.ndarray                # (E,) A/m^2
    br: np.ndarray                 # (E,2) T
    steel: Reluctivity             # laminate curve (stator/rotor share NO18-1160)
    curves: Tuple[Reluctivity, ...]   # every BH curve in play
    curve_index: np.ndarray        # (E,) index into `curves`; -1 for non-steel
    reg_code: np.ndarray           # (E,)
    name_of_code: Mapping[int, str]
    source_element: np.ndarray     # (E,) index in the exported mesh, -1 if newly created
    n_export_elements: int
    n_original_nodes: int
    image_constraints: Tuple[Tuple[int, int, float], ...]
    # constraint bookkeeping
    dof_of_node: np.ndarray        # (N,) reduced dof index, -1 when fixed
    dof_sign: np.ndarray           # (N,) a_node = dof_sign * a_dof
    n_dof: int
    dirichlet_nodes: np.ndarray
    constraint_report: dict

    @property
    def n_nodes(self) -> int:
        return int(self.x.size)

    @property
    def n_elements(self) -> int:
        return int(self.tri.shape[0])

    def element_b(self, a_nodal: np.ndarray) -> np.ndarray:
        """(E,2) flux density from nodal A: Bx = dA/dy, By = -dA/dx."""
        a = np.asarray(a_nodal, dtype=np.float64).reshape(-1)
        ga = a[self.tri]
        bx = np.einsum("ej,ej->e", self.grad_c, ga)
        by = -np.einsum("ej,ej->e", self.grad_b, ga)
        return np.stack([bx, by], axis=1)

    def lift_guess(self, a_original: np.ndarray) -> np.ndarray:
        """Turn a guess defined on the exported nodes into a full nodal vector.

        The band remesh appends image nodes whose index depends on the rotor
        position, so a solution carried over from another step only covers the
        first `n_original_nodes` entries. Image nodes get their value from the
        node they mirror, with the anti-periodic sign -- not zero, which would
        put a fake discontinuity right across the air gap and cost the warm start
        exactly the iterations it is supposed to save.
        """
        a = np.zeros(self.n_nodes, dtype=np.float64)
        src = np.asarray(a_original, dtype=np.float64).reshape(-1)
        n = min(src.size, self.n_original_nodes)
        a[:n] = src[:n]
        for image, original, sign in self.image_constraints:
            a[image] = sign * a[original]
        return a

    def to_export_elements(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Re-index a per-element quantity onto the exported mesh's element order.

        The band remesh replaces `a2`, so a solved field lives on a different
        element list than the H5 export. Everything that compares against
        Motor-CAD -- the benchmark's element subset, the Arkkio band -- indexes
        the export, so scatter back before comparing. Elements that no longer
        exist keep `fill`.
        """
        values = np.asarray(values)
        out = np.full((self.n_export_elements,) + values.shape[1:], fill, dtype=np.float64)
        keep = self.source_element >= 0
        out[self.source_element[keep]] = values[keep]
        return out

    def expand(self, a_dof: np.ndarray) -> np.ndarray:
        """Reduced dof vector -> full nodal A (Dirichlet nodes become 0)."""
        a = np.zeros(self.n_nodes, dtype=np.float64)
        free = self.dof_of_node >= 0
        a[free] = self.dof_sign[free] * np.asarray(a_dof)[self.dof_of_node[free]]
        return a

    def restrict(self, a_nodal: np.ndarray) -> np.ndarray:
        """Full nodal A -> reduced dof vector (averaging over each merged class)."""
        a = np.asarray(a_nodal, dtype=np.float64).reshape(-1)
        out = np.zeros(self.n_dof, dtype=np.float64)
        cnt = np.zeros(self.n_dof, dtype=np.float64)
        free = np.flatnonzero(self.dof_of_node >= 0)
        np.add.at(out, self.dof_of_node[free], a[free] * self.dof_sign[free])
        np.add.at(cnt, self.dof_of_node[free], 1.0)
        return out / np.maximum(cnt, 1.0)


def _classify(name: str) -> int:
    for pattern, mat in _MATERIAL_PATTERNS:
        if pattern.search(name):
            return mat
    return AIR


def _ccw(x: np.ndarray, y: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Reorder each triangle counter-clockwise so the signed area is positive."""
    tri = np.array(tri, dtype=np.int64, copy=True)
    x1, y1 = x[tri[:, 0]], y[tri[:, 0]]
    x2, y2 = x[tri[:, 1]], y[tri[:, 1]]
    x3, y3 = x[tri[:, 2]], y[tri[:, 2]]
    two_a = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    flip = two_a < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def _magnet_axes(
    cx: np.ndarray, cy: np.ndarray, area: np.ndarray, mask: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Easy axis (unit vector), polarity, and the block's elongation ratio.

    The magnets are elongated rectangles, so the PCA minor axis of the element
    centroids is the thickness direction, which for a block magnet is the
    magnetisation direction. Polarity comes from the exported field: inside a
    magnet the flux density runs along its own remanence, so the sign of
    ``<B> . minor`` is the sign of Br along that axis.
    """
    p = np.stack([cx[mask], cy[mask]], axis=1)
    w = area[mask]
    mean = (p * w[:, None]).sum(0) / w.sum()
    centred = (p - mean) * np.sqrt(w)[:, None]
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    minor = vt[1]
    b_mean = np.stack([bx[mask], by[mask]], axis=1)
    b_mean = (b_mean * w[:, None]).sum(0) / w.sum()
    polarity = float(np.sign(b_mean @ minor)) or 1.0
    ratio = float(sv[0] / sv[1]) if sv[1] > 0 else np.inf
    return minor * polarity, polarity, ratio


def build_domain(
    record,
    step: int,
    bh_path: str | Path,
    sector_deg: float = SECTOR_DEG,
    angle_tol_deg: float = 2e-3,
    radius_tol_mm: float = 2e-3,
    outer_radius_tol_mm: float = 0.2,
    remesh_band: bool = True,
) -> FemDomain:
    """Assemble the geometry, materials, sources and constraints for one step.

    ``remesh_band`` replaces the exported ``a2`` shear layer with a freshly
    triangulated strip (see `band_remesh.py`). It is on by default because the
    exported layer is only valid for steps 0-2.
    """
    mesh = record.mesh
    sample = record.samples[step]
    x_mm = np.asarray(sample.node_x_mm, dtype=np.float64)
    y_mm = np.asarray(sample.node_y_mm, dtype=np.float64)
    tri_raw = np.stack(mesh.tri, axis=1)
    reg = np.asarray(mesh.reg_code, dtype=np.int64)

    extra_constraints: Sequence[Tuple[int, int, float]] = ()
    remesh_report: dict = {}
    source_element = np.arange(tri_raw.shape[0], dtype=np.int64)
    n_original_nodes = int(x_mm.size)
    if remesh_band:
        from fem_warmstart.band_remesh import rebuild_band

        rb = rebuild_band(x_mm, y_mm, tri_raw, reg, mesh.name_of_code, sector_deg)
        x_mm, y_mm = rb.x_mm, rb.y_mm
        tri_raw = rb.tri
        source_element = rb.source_element
        extra_constraints = rb.constraints
        remesh_report = rb.report
        # per-element data follows the source map; new band elements are plain air
        new = source_element < 0
        reg = np.where(new, -1, reg[np.clip(source_element, 0, None)])

    x, y = x_mm * 1e-3, y_mm * 1e-3
    tri = _ccw(x, y, tri_raw)

    x1, y1 = x[tri[:, 0]], y[tri[:, 0]]
    x2, y2 = x[tri[:, 1]], y[tri[:, 1]]
    x3, y3 = x[tri[:, 2]], y[tri[:, 2]]
    two_a = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * two_a
    if np.any(area <= 0):
        raise DomainError(f"{int(np.sum(area <= 0))} degenerate elements after CCW reorder")

    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)
    grad_b = b / two_a[:, None]
    grad_c = c / two_a[:, None]

    material = np.array([_classify(str(mesh.name_of_code.get(int(r), ""))) for r in reg],
                        dtype=np.int64)

    # ---- sources -----------------------------------------------------------
    # Element fields follow the source map; freshly created band elements are air.
    keep_src = np.clip(source_element, 0, None)
    fresh = source_element < 0
    j_z = np.where(fresh, 0.0,
                   np.asarray(sample.fields["j"], dtype=np.float64)[keep_src]) * 1e6
    bx_fem = np.where(fresh, 0.0, np.asarray(sample.fields["bx"], dtype=np.float64)[keep_src])
    by_fem = np.where(fresh, 0.0, np.asarray(sample.fields["by"], dtype=np.float64)[keep_src])

    br_by_region = parse_magnet_remanence(bh_path)
    br = np.zeros((tri.shape[0], 2), dtype=np.float64)
    cx, cy = (x[tri].mean(axis=1), y[tri].mean(axis=1))
    magnet_report = {}
    for code, name in mesh.name_of_code.items():
        if _classify(str(name)) != MAGNET:
            continue
        mask = reg == int(code)
        if not np.any(mask):
            continue
        br_mag = br_by_region.get(str(name))
        if br_mag is None:
            raise DomainError(f"no remanence for magnet region {name!r} in {bh_path}")
        axis, polarity, ratio = _magnet_axes(cx, cy, area, mask, bx_fem, by_fem)
        br[mask] = br_mag * axis
        magnet_report[str(name)] = {
            "Br_T": br_mag,
            "axis_deg": float(np.degrees(np.arctan2(axis[1], axis[0]))),
            "polarity": polarity,
            "elongation": ratio,
        }

    # ---- reluctivity -------------------------------------------------------
    # Two distinct steels in this model: the laminate (NO18-1160, .bh codes 1 and
    # 17, identical tables for stator and rotor) and the shaft (Stahl 37, code 26,
    # a 32-point curve saturating at 3.41 T). Using the laminate curve for the
    # shaft would over-estimate its permeability near the bore.
    laminate = load_steel_reluctivity(bh_path, 1)
    curves = [laminate]
    shaft_curve_idx = 0
    try:
        curves.append(load_steel_reluctivity(bh_path, 26))
        shaft_curve_idx = 1
    except KeyError:
        pass
    steel = laminate

    is_shaft = np.array(
        [str(mesh.name_of_code.get(int(r), "")).strip().lower() == "shaft" for r in reg]
    )
    curve_index = np.full(tri.shape[0], -1, dtype=np.int64)
    curve_index[material == STEEL] = 0
    curve_index[(material == STEEL) & is_shaft] = shaft_curve_idx

    nu_linear = np.full(tri.shape[0], NU0, dtype=np.float64)
    nu_linear[material == MAGNET] = NU0 / 1.05                       # recoil mu_r
    magnet_mu_r = 1.05

    # ---- constraints -------------------------------------------------------
    theta = np.degrees(np.arctan2(y_mm, x_mm))
    radius = np.hypot(x_mm, y_mm)
    uf = SignedUnionFind(x.size)
    report: dict = {"magnets": magnet_report, "magnet_mu_r": magnet_mu_r}

    # Which nodes belong to the rotor side? Use the radial split at the sliding
    # band: everything strictly inside the band radius is rotor-borne.
    band_r = _band_radius(radius, theta, reg, mesh, tri)
    report["band_radius_mm"] = band_r

    # Only nodes that some element actually uses may take part in a constraint.
    # Rebuilding the band orphans the `a2`-exclusive arc nodes; letting one of
    # those win a cut-plane match would tie a live node to a dead one -- and since
    # dead nodes must end up pinned, that silently grounds a whole live class.
    # (Symptom when this was wrong: every region's |B| collapsed by ~10-25%.)
    referenced = np.zeros(x.size, dtype=bool)
    referenced[tri.ravel()] = True
    original = np.zeros(x.size, dtype=bool)
    original[:n_original_nodes] = True

    def plane_nodes(angle: float, rmin: float, rmax: float) -> np.ndarray:
        return np.flatnonzero(
            (np.abs(theta - angle) < angle_tol_deg)
            & (radius > rmin) & (radius < rmax) & referenced
        )

    # (C1) stator cut planes, outside the band
    n_c1 = _pair_by_radius(
        uf, plane_nodes(0.0, band_r - 1e-3, np.inf),
        plane_nodes(-sector_deg, band_r - 1e-3, np.inf), radius, radius_tol_mm, -1.0)

    # (C2) rotor cut planes -- their angles are read off the mesh, not assumed.
    # Nodes *strictly* inside the band only: the band arc itself is shared with the
    # stator and reaches theta = 0, which would otherwise be read as the rotor's
    # leading cut plane.
    #
    # The two faces are found as angular clusters at the rotor's extremes rather
    # than at "lead" and "lead - 45": the a2 shear layer is redistributed by the
    # solver rather than rotated, so its trailing face sits 0.049 deg past the
    # rigid rotor's (-65.546 vs -65.497 at step 0) and a fixed 45 deg offset
    # would miss it. A 0.06 deg window (41 um at the band radius, versus a
    # ~750 um mesh) catches both without reaching any interior node.
    # Image nodes created by the band remesh are folded copies spread over the whole
    # sector, so they must not take part in locating the rotor's physical cut planes.
    rotor_nodes = np.flatnonzero(
        (radius < band_r - 5e-3) & (radius > 1.0) & referenced & original)
    lead = float(np.max(theta[rotor_nodes]))
    trail = float(np.min(theta[rotor_nodes]))
    report["rotor_cut_deg"] = (lead, trail)
    report["rotor_span_deg"] = lead - trail
    face_window = 0.06
    lead_face = rotor_nodes[theta[rotor_nodes] > lead - face_window]
    trail_face = rotor_nodes[theta[rotor_nodes] < trail + face_window]
    n_c2 = _pair_by_radius(uf, lead_face, trail_face, radius, radius_tol_mm, -1.0)

    # (C3) sliding-band wrap: rotor arc beyond -45 deg <-> stator arc one sector on.
    # Only needed for the *exported* band: once the strip is re-triangulated the
    # coupling is carried by real elements, and the anti-periodic sign travels with
    # the image nodes the remesh created.
    if remesh_band:
        n_c3 = 0
        for image, original, sign in extra_constraints:
            uf.union(int(image), int(original), float(sign))
        report["remesh"] = remesh_report
        report["n_band_image_constraints"] = len(extra_constraints)
    else:
        on_band = np.flatnonzero(np.abs(radius - band_r) < 0.02)
        rotor_arc = on_band[theta[on_band] < -sector_deg + angle_tol_deg]
        stator_arc = on_band[theta[on_band] > -sector_deg - angle_tol_deg]
        n_c3 = _pair_by_angle(uf, rotor_arc, stator_arc, theta, sector_deg, -1.0)

    # (D) Dirichlet on the outer arc. Orphans are not pinned -- they are simply
    # dropped from the dof map after the reduction, which keeps them out of the
    # matrix without grounding whatever class they happen to belong to.
    r_out = float(radius.max())
    dirichlet = np.flatnonzero((radius > r_out - outer_radius_tol_mm) & referenced)

    dof_of_node, dof_sign, n_dof = _reduce(uf, x.size, dirichlet)
    dof_of_node = np.where(referenced, dof_of_node, -1)
    # Compact: a class made up only of orphans would otherwise leave an empty row
    # and column in the reduced matrix, which is singular rather than merely wasteful.
    used = np.unique(dof_of_node[dof_of_node >= 0])
    renumber = np.full(n_dof, -1, dtype=np.int64)
    renumber[used] = np.arange(used.size, dtype=np.int64)
    dof_of_node = np.where(dof_of_node >= 0, renumber[np.clip(dof_of_node, 0, None)], -1)
    n_dof = int(used.size)
    report.update({
        "n_stator_cut_pairs": n_c1,
        "n_rotor_cut_pairs": n_c2,
        "n_band_wrap_pairs": n_c3,
        "n_dirichlet": int(dirichlet.size),
        "n_orphan_nodes": int(np.sum(~referenced)),
        "n_nodes": int(x.size),
        "n_elements": int(tri.shape[0]),
        "n_dof": int(n_dof),
        "sign_conflicts": int(uf.conflicts),
        "outer_radius_mm": r_out,
    })

    return FemDomain(
        x=x, y=y, tri=tri, area=area, grad_b=grad_b, grad_c=grad_c,
        material=material, nu_linear=nu_linear, j_z=j_z, br=br, steel=steel,
        curves=tuple(curves), curve_index=curve_index,
        reg_code=reg, name_of_code=mesh.name_of_code,
        source_element=source_element,
        n_export_elements=int(np.asarray(mesh.reg_code).size),
        n_original_nodes=n_original_nodes,
        image_constraints=tuple(extra_constraints),
        dof_of_node=dof_of_node, dof_sign=dof_sign, n_dof=n_dof,
        dirichlet_nodes=dirichlet, constraint_report=report,
    )


def _band_radius(radius, theta, reg, mesh, tri) -> float:
    """Radius of the stator/rotor sliding interface, from the airgap layer names.

    a1 is the stator-side layer and a2 the rotor-side one (they are the only pair
    that share an arc), so the interface is a1's inner radius.
    """
    code_of_name = {str(v).lower(): int(k) for k, v in mesh.name_of_code.items()}
    if "a1" not in code_of_name:
        raise DomainError("no 'a1' airgap layer; cannot locate the sliding band")
    nodes = np.unique(tri[reg == code_of_name["a1"]].ravel())
    return float(radius[nodes].min())


def _pair_by_radius(uf, side_a, side_b, radius, tol, sign) -> int:
    if side_a.size == 0 or side_b.size == 0:
        return 0
    order = np.argsort(radius[side_b])
    sorted_b = radius[side_b][order]
    paired = 0
    for node in side_a:
        j = int(np.searchsorted(sorted_b, radius[node]))
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < sorted_b.size and abs(sorted_b[cand] - radius[node]) <= tol:
                if uf.union(int(node), int(side_b[order[cand]]), sign):
                    paired += 1
                break
    return paired


def _pair_by_angle(uf, rotor_arc, stator_arc, theta, sector_deg, sign,
                   tol_deg: float = 0.02) -> int:
    """Match each wrapped rotor-arc node to the stator-arc node one sector away."""
    if rotor_arc.size == 0 or stator_arc.size == 0:
        return 0
    order = np.argsort(theta[stator_arc])
    sorted_t = theta[stator_arc][order]
    paired = 0
    for node in rotor_arc:
        target = theta[node] + sector_deg
        j = int(np.searchsorted(sorted_t, target))
        best, best_d = None, np.inf
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < sorted_t.size:
                d = abs(sorted_t[cand] - target)
                if d < best_d:
                    best, best_d = cand, d
        if best is not None and best_d <= tol_deg:
            if uf.union(int(node), int(stator_arc[order[best]]), sign):
                paired += 1
    return paired


def _reduce(uf: SignedUnionFind, n_nodes: int, dirichlet: np.ndarray):
    """Turn the union-find classes into reduced dof indices."""
    roots = np.empty(n_nodes, dtype=np.int64)
    signs = np.empty(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        r, s = uf.find(i)
        roots[i], signs[i] = r, s

    fixed_roots = set(int(roots[i]) for i in dirichlet)
    dof_of_root: Dict[int, int] = {}
    for r in roots:
        r = int(r)
        if r in fixed_roots or r in dof_of_root:
            continue
        dof_of_root[r] = len(dof_of_root)

    dof_of_node = np.full(n_nodes, -1, dtype=np.int64)
    dof_sign = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        r = int(roots[i])
        if r in dof_of_root:
            dof_of_node[i] = dof_of_root[r]
            dof_sign[i] = signs[i]
    return dof_of_node, dof_sign, len(dof_of_root)
