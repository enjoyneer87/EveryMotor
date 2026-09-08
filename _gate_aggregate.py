#!/usr/bin/env python3
"""Pool infer_phase1_pbc.py outputs over held-out cases and compare arms.

Reads results/gate_<arm>_case<k>.npz (written by infer_phase1_pbc.py, which
compares raw model output with raw targets -- physical units regardless of the
--target-normalization the arm trained with) and reports per-channel RMSE and
rel = RMSE / std(gt), per case and pooled over all nodes of all cases.
"""
import argparse
import json
from pathlib import Path

import numpy as np

CH = ["bx", "by", "a", "je"]


def rmse_rel(gt, pr):
    if gt.size == 0:
        return {"rmse": float("nan"), "rel": float("nan")}
    rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))
    sd = float(np.std(gt))
    # An all-zero target (Je in doe_data_120: no fields/je in the exports) has no
    # meaningful relative error -- report NaN rather than RMSE/1e-8.
    return {"rmse": rmse, "rel": float(rmse / sd) if sd > 1e-12 else float("nan")}


def load(arm, cases):
    pooled = {f"{p}_{c}": [] for p in ("gt", "pred") for c in CH}
    per_case = {}
    for k in cases:
        f = Path(f"results/gate_{arm}_case{k}.npz")
        if not f.exists():
            per_case[str(k)] = None
            continue
        z = np.load(f, allow_pickle=True)
        m = {}
        for c in CH:
            gt, pr = z[f"gt_{c}"], z[f"pred_{c}"]
            pooled[f"gt_{c}"].append(gt)
            pooled[f"pred_{c}"].append(pr)
            m[c] = rmse_rel(gt, pr)
        m["bmag"] = rmse_rel(np.hypot(z["gt_bx"], z["gt_by"]), np.hypot(z["pred_bx"], z["pred_by"]))
        per_case[str(k)] = m
    pooled = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in pooled.items()}
    agg = {c: rmse_rel(pooled[f"gt_{c}"], pooled[f"pred_{c}"]) for c in CH}
    agg["bmag"] = rmse_rel(np.hypot(pooled["gt_bx"], pooled["gt_by"]),
                           np.hypot(pooled["pred_bx"], pooled["pred_by"]))
    return {"per_case": per_case, "pooled": agg, "n_nodes": int(pooled["gt_bx"].size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--cases", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {arm: load(arm, a.cases) for arm in a.arms}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    keys = CH + ["bmag"]
    print(f"pooled over held-out cases {a.cases}  (physical units; rel = RMSE/std(gt))")
    print("arm  " + "".join(f"{k:>18}" for k in keys) + "   n_nodes")
    for arm, r in out.items():
        def fmt(m):
            rel = "n/a" if np.isnan(m["rel"]) else f"{m['rel']:.3f}"
            return f"{m['rmse']:.4g}/{rel}".rjust(18)
        cells = "".join(fmt(r["pooled"][k]) for k in keys)
        print(f"{arm:<4} {cells}   {r['n_nodes']}")
    if len(a.arms) == 2 and all(out[x]["n_nodes"] for x in a.arms):
        A, B = (out[x]["pooled"] for x in a.arms)
        print("rmse ratio " + a.arms[1] + "/" + a.arms[0] + ": "
              + "  ".join(f"{k}={B[k]['rmse'] / A[k]['rmse']:.3f}" for k in keys))


if __name__ == "__main__":
    main()
