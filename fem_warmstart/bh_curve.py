"""BH-curve -> reluctivity nu(B) for the FEM warm-start PoC.

Parses a Motor-CAD ".bh" autofile (region-blocked H[A/m], B[Tesla] tables) and builds
nu(B) = H/B with a saturated linear tail, plus dnu/dB for the Newton Jacobian. numpy-only
(runs on host or in the container). No prior nu(B) code existed in eMach -- MATLAB paths
returned raw (H,B) only.

.bh format (per region block):
    Material BH characteristics <date>
    Code:1 (Stator) Material: NO18-1160
    26
    1  0  0
    2  20  0.06305075398
    ...
    26  1000000  3.160027239
    Code:17 (Rotor) Material: NO18-1160
    ...
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MU0 = 4e-7 * np.pi
NU0 = 1.0 / MU0

_BLOCK_RE = re.compile(r"Code:\s*(\d+)\s*\(([^)]*)\)\s*Material:\s*(.*)")


def parse_bh_file(path: str | Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return {region_code: (H[A/m], B[T])} from a .bh autofile.

    Robust to the non-UTF8 locale header (latin-1 + skip non-numeric lines).
    """
    text = Path(path).read_bytes().decode("latin-1", errors="replace")
    lines = text.splitlines()
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    i = 0
    while i < len(lines):
        m = _BLOCK_RE.search(lines[i])
        if not m:
            i += 1
            continue
        code = int(m.group(1))
        # next line = point count
        try:
            n = int(lines[i + 1].strip())
        except (ValueError, IndexError):
            i += 1
            continue
        H, B = [], []
        for k in range(n):
            parts = lines[i + 2 + k].split()
            if len(parts) < 3:
                break
            H.append(float(parts[1]))
            B.append(float(parts[2]))
        Ha, Ba = np.asarray(H), np.asarray(B)
        # ensure strictly increasing B (drop duplicates), keep origin
        keep = np.concatenate([[True], np.diff(Ba) > 0])
        out[code] = (Ha[keep], Ba[keep])
        i += 2 + n
    if not out:
        raise ValueError(f"no BH blocks parsed from {path}")
    return out


class Reluctivity:
    """nu(B) = H/B for one steel curve, with a saturated linear H(B) tail.

    Linear interpolation of H(B) between tabulated points (numpy-only). Above B_max the
    steel is fully saturated: dH/dB -> 1/mu0, so H(B) = H_max + (B-B_max)*NU0. At B->0,
    nu -> initial slope H1/B1 (finite). Provides nu(B) and dnu/dB (for the Newton Jacobian).
    """

    def __init__(self, H: np.ndarray, B: np.ndarray):
        # drop the (0,0) origin for the H/B table but keep initial slope
        assert B[0] == 0.0 and H[0] == 0.0, "expected (H,B) to start at origin"
        self.B = B
        self.H = H
        self.Bmax = float(B[-1])
        self.Hmax = float(H[-1])
        self.nu_init = float(H[1] / B[1])          # nu at B->0 (initial reluctivity)
        self.nu_sat_ref = float(H[-1] / B[-1])

    def _H_of_B(self, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return H(b) and dH/db via piecewise-linear interp + saturated tail."""
        b = np.asarray(b, dtype=np.float64)
        H = np.interp(b, self.B, self.H)                    # clamps at ends
        # slope per interval for dH/db
        dHdb = np.gradient(self.H, self.B)                  # nodal slopes
        slope = np.interp(b, self.B, dHdb)
        # saturated tail above Bmax: H = Hmax + (b-Bmax)*NU0
        tail = b > self.Bmax
        H = np.where(tail, self.Hmax + (b - self.Bmax) * NU0, H)
        slope = np.where(tail, NU0, slope)
        return H, slope

    def nu(self, B: np.ndarray) -> np.ndarray:
        """Reluctivity nu(B) = H(B)/B, finite at B=0 (initial slope)."""
        B = np.asarray(B, dtype=np.float64)
        small = B < 1e-9
        Bsafe = np.where(small, 1.0, B)
        H, _ = self._H_of_B(Bsafe)
        val = H / Bsafe
        return np.where(small, self.nu_init, val)

    def nu_and_dnu(self, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (nu(B), dnu/dB). dnu/dB = (H'(B)*B - H(B)) / B^2."""
        B = np.asarray(B, dtype=np.float64)
        small = B < 1e-9
        Bsafe = np.where(small, 1e-9, B)
        H, Hp = self._H_of_B(Bsafe)
        nu = H / Bsafe
        dnu = (Hp * Bsafe - H) / (Bsafe * Bsafe)
        nu = np.where(small, self.nu_init, nu)
        dnu = np.where(small, 0.0, dnu)
        return nu, dnu


def load_steel_reluctivity(bh_path: str | Path, region_code: int | None = None) -> Reluctivity:
    """Load one steel curve. If region_code is None, use the first block (all steel blocks
    are identical when one steel is used, e.g. NO18-1160 for both stator and rotor)."""
    blocks = parse_bh_file(bh_path)
    code = region_code if region_code is not None else sorted(blocks)[0]
    H, B = blocks[code]
    return Reluctivity(H, B)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:/KDH/Sim_4SolverX/DOE_Ext/case_0000/TestCAD1/FEResultsData/Steel_Material_BH_Magnetic_Properties_Autofile.bh"
    blocks = parse_bh_file(p)
    print("blocks:", {k: len(v[0]) for k, v in blocks.items()})
    r = load_steel_reluctivity(p)
    print(f"nu_init {r.nu_init:.3e}  NU0(air) {NU0:.3e}  Bmax {r.Bmax:.3f} Hmax {r.Hmax:.3e}")
    for b in [0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.16, 4.0]:
        nu, dnu = r.nu_and_dnu(np.array([b]))
        print(f"  B={b:5.2f}  nu={nu[0]:.4e}  mu_r={NU0/nu[0]:8.1f}  dnu/dB={dnu[0]:.3e}")
    # Physical checks for electrical steel (nu is NON-monotonic: mu_r peaks at the knee,
    # so nu dips to a minimum near the peak-permeability B then rises with saturation).
    bs = np.linspace(0.05, 3.0, 300)
    mur = NU0 / r.nu(bs)
    b_peak = bs[int(np.argmax(mur))]
    tail = NU0 / r.nu(np.array([6.0]))[0]      # deep saturation -> approaches air (mu_r->1)
    checks = {
        "mu_r peaks in knee (0.3-1.2 T)": 0.3 <= b_peak <= 1.2,
        "saturates (mu_r<10 by 2.5 T)": NU0 / r.nu(np.array([2.5]))[0] < 10,
        "tail approaches air (mu_r<2 at 6 T)": tail < 2.0,
        "nu positive everywhere": bool(np.all(r.nu(bs) > 0)),
    }
    print(f"mu_r peak at B~{b_peak:.2f} T (max mu_r {mur.max():.0f}); tail mu_r@6T {tail:.2f}")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print("BH_CURVE_OK" if all(checks.values()) else "BH_CURVE_FAIL")
