"""em_export.py -- dump the EveryMotor results as one compact JSON document on stdout.

Consumed by scripts/em_report.py on the relay PC (bash scripts/em.sh report). Host python is
enough (no torch). Everything the HTML report shows comes from here, so the report never
re-derives numbers by hand.

Contents:
  runs     every benchmark_v2_nodeB_doe_v[23]_bw30_prior* run: legacy6 / new12 / all18 pooled
           |B|, airgap, torque, plus the legacy6 per-case torque breakdown
  curves   per-epoch train/val nRMSE parsed from the training logs listed in CURVE_LOGS
  rules    the gate and improvement thresholds (methodology_review 25, 29, 36)
  repo     branch, head, baton
To add a new run's learning curve, append its log paths to CURVE_LOGS.
"""
import datetime
import glob
import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
KDH = r"D:\KDH\NvidiaNemo\results\logs"
CURVE_LOGS = {
    "doe_v2_bw30_prior": [os.path.join(KDH, "v2_320.log"), os.path.join(KDH, "v2_320_resume.log")],
    "doe_v3_bw30_prior": [os.path.join(KDH, "v3_480.log")],
    "doe_v3_bw30_prior_e1airgap5": ["results/logs/e1_airgap_part1.log",
                                    "results/logs/e1_airgap_resume_20260910.log"],
    "doe_v3_bw30_prior_slotharm": ["results/logs/spec_slotharm_part1_FAILED.log"],
}
EP_RE = re.compile(r"ep (\d{4})/(\d+) \| train B \S+ nRMSE\s+([\d.]+)% \| val B \S+ nRMSE\s+([\d.]+)%")
RULES = {"gate": {"airgap_lt": 5.0, "torque_lt": 3.0, "group": "legacy6"},
         "control": {"name": "doe_v3_bw30_prior", "B": 11.302, "airgap": 6.925, "torque": 3.799},
         "improve": {"airgap_success_lt": 6.675, "airgap_harmful_gt": 7.175,
                     "harm_B_gt": 11.552, "harm_torque_gt": 4.049, "band_pp": 0.25}}


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


def pooled(s):
    return {"B": g(s, "overall", "Bnorm", "nrmse_pct"),
            "airgap": g(s, "by_region", "airgap", "Bnorm", "nrmse_pct"),
            "torque": g(s, "torque", "nrmse_torque_pct"),
            "n_cases": s.get("n_cases"), "n_samples": s.get("n_samples")}


def read_text(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def curve(paths):
    pts = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for m in EP_RE.finditer(read_text(p)):
            pts[int(m.group(1))] = (float(m.group(3)), float(m.group(4)))  # later logs win (resumes)
    return [[ep, tr, va] for ep, (tr, va) in sorted(pts.items())]


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8").stdout.strip()


def main():
    runs = []
    for f in sorted(glob.glob("results/benchmark_v2_nodeB_doe_v*_bw30_prior*.json"), key=os.path.getmtime):
        name = os.path.basename(f)[len("benchmark_v2_nodeB_"):-len(".json")]
        if name.endswith(("_legacy6", "_new12", "_groups")):
            continue
        run = {"name": name, "scored": datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d"),
               "splits": {}, "cases": {}}
        for split, suffix in (("legacy6", "_legacy6"), ("new12", "_new12"), ("all18", "")):
            s = summary(f"results/benchmark_v2_nodeB_{name}{suffix}.json")
            if s is not None:
                run["splits"][split] = pooled(s)
                if split == "legacy6":
                    for cid, cv in (s.get("by_case") or {}).items():
                        t = cv.get("torque") or {}
                        run["cases"][cid] = {k: t.get(k) for k in (
                            "nrmse_torque_pct", "mean_torque_error_pct", "mean_torque_true",
                            "mean_torque_pred", "rms_torque_true", "ripple_error_pct")}
        runs.append(run)
    try:
        baton = json.load(open("results/logs/hb_control.json", encoding="utf-8"))
        baton = {k: baton.get(k) for k in ("status_hint", "owner", "experiment", "updated")}
    except (OSError, ValueError):
        baton = {}
    case_meta = {}
    try:
        man = json.load(open(r"D:\KDH\NvidiaNemo\backup\doe_data_v3\doe_manifest.json", encoding="utf-8"))
        ids = {cid for r in runs for cid in r["cases"]}
        for c in man["cases"]:
            if str(c["index"]) in ids:
                e = c.get("electrical") or {}
                case_meta[str(c["index"])] = {"phase_advance_deg": e.get("PhaseAdvance"),
                                              "peak_current_a": e.get("PeakCurrent"),
                                              **{k: v for k, v in (c.get("geometry") or {}).items()
                                                 if isinstance(v, (int, float))}}
    except (OSError, ValueError, KeyError):
        pass
    out = {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="minutes"),
        "case_meta": case_meta,
        "machine": "DESKTOP-513NQ4P / 192.168.0.234 (RTX 3090 24GB, physicsnemo:26.03)",
        "repo": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                 "head": git("log", "-1", "--format=%h %ad %s", "--date=short"), "baton": baton},
        "rules": RULES,
        "runs": runs,
        "curves": {k: curve(v) for k, v in CURVE_LOGS.items()},
    }
    json.dump(out, sys.stdout, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
