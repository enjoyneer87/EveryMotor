"""em_scores.py -- compact scorecard table for the EveryMotor curl-MGN runs.

Run on the isolated PC from the repo root (host python is enough, no torch needed):
    python tools/ops/em_scores.py                    # baselines (v2, R9) + E1 family
    python tools/ops/em_scores.py e1airgap5 slotharm  # any tag substrings
    python tools/ops/em_scores.py --all              # every benchmark_v2_nodeB_* run
    python tools/ops/em_scores.py e1airgap5 --cases  # + per-case torque on legacy6

Gate rules (methodology_review sections 25, 29, 36):
  G1' = legacy6 pooled: airgap |B| < 5%  AND  torque nRMSE < 3%.
  new12 / all18 are reported only -- never used for a gate verdict (section 29).
  Improvement vs the common control R9 ep50 (legacy6 11.302 / 6.925 / 3.799):
  airgap < 6.675 success, 6.675..7.175 in-band, > 7.175 harmful;
  harm guard rejects if |B| > 11.552 or torque > 4.049.
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
R9 = (11.302, 6.925, 3.799)
BASELINES = ("doe_v2_bw30_prior", "doe_v3_bw30_prior")


def g(o, *ks):
    for k in ks:
        if not isinstance(o, dict):
            return None
        o = o.get(k)
    return o


def summary(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    mk = [k for k in d.get("models", {}) if k.startswith("curl:")]
    return d["models"][mk[0]]["summary"] if mk else None


def metrics(s):
    return (g(s, "overall", "Bnorm", "nrmse_pct"),
            g(s, "by_region", "airgap", "Bnorm", "nrmse_pct"),
            g(s, "torque", "nrmse_torque_pct"))


def run_names():
    out = []
    for f in glob.glob("results/benchmark_v2_nodeB_*.json"):
        name = os.path.basename(f)[len("benchmark_v2_nodeB_"):-len(".json")]
        if not name.endswith(("_legacy6", "_new12")):
            out.append((os.path.getmtime(f), name))
    return [n for _, n in sorted(out)]


def fmt(v):
    return f"{v:7.3f}" if isinstance(v, (int, float)) else "      -"


def verdict(m):
    b, air, tq = m
    if None in m:
        return ""
    g1 = "G1' PASS" if (air < 5 and tq < 3) else \
        f"G1' air {'ok' if air < 5 else 'x'} tq {'ok' if tq < 3 else 'x'}"
    if b > 11.552 or tq > 4.049:
        imp = "REJECT(harm guard)"
    elif air < 6.675:
        imp = "success"
    elif air <= 7.175:
        imp = "in-band"
    else:
        imp = "harmful"
    return f"{g1} | vs R9: {imp}"


def main(argv):
    tags = [a for a in argv if not a.startswith("--")]
    names = run_names()
    if "--all" in argv:
        sel = names
    elif tags:
        sel = [n for n in names if any(t in n for t in tags)]
    else:
        sel = [n for n in names if n in BASELINES or "e1airgap5" in n]
    print(f"{'run':40s} {'split':7s} {'|B|':>7s} {'airgap':>7s} {'torque':>7s}  verdict (legacy6 only)")
    for n in sel:
        for split, suffix in (("legacy6", "_legacy6"), ("new12", "_new12"), ("all18", "")):
            s = summary(f"results/benchmark_v2_nodeB_{n}{suffix}.json")
            if s is None:
                continue
            m = metrics(s)
            print(f"{n[:40]:40s} {split:7s} {fmt(m[0])} {fmt(m[1])} {fmt(m[2])}  "
                  f"{verdict(m) if split == 'legacy6' else ''}")
        if "--cases" in argv:
            s = summary(f"results/benchmark_v2_nodeB_{n}_legacy6.json") or {}
            for cid, cv in (s.get("by_case") or {}).items():
                t = cv.get("torque") or {}
                print(f"    case {cid:>4s}  torque {fmt(t.get('nrmse_torque_pct'))}  "
                      f"mean_err% {fmt(t.get('mean_torque_error_pct'))}  rmsT {fmt(t.get('rms_torque_true'))}")
    print(f"control R9 ep50 legacy6: |B| {R9[0]} / airgap {R9[1]} / torque {R9[2]}")


if __name__ == "__main__":
    main(sys.argv[1:])
