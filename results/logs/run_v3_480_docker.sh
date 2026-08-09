#!/bin/bash
# R9: 480-case training (v2's 320 verbatim + the 160-case Ext3 expansion) on the v2
# recipe with NOTHING else changed -- data axis only (single-axis discipline, sections
# 13-21). Design and decision rules are pre-registered in methodology section 30.
#
# PRE-REGISTERED VERDICT (vs the v2 champion's legacy-6 pooled airgap 6.618%):
#   airgap < 6.368%   -> data axis still paying, R9 is the new champion
#   6.368 .. 6.868%   -> inside the +/-0.25 pp run-to-run band; slope flattening
#   > 6.868%          -> regression; suspect the epoch budget (section 21) before the data
# new_current and the 487.9 A interpolation probe are REPORTED, not gated.
#
# PRE-FLIGHT (aborts on mismatch -- each one has burned a run before):
#   1. split digest + sizes -- doe_v3_case_split.json must match the assembled manifest.
#   2. path gate -- every h5 of all 480 cases must resolve FROM INSIDE THE CONTAINER.
#      v2 shipped host-absolute paths once and would have trained silently short
#      (sections 27/29). doe_build_v3.py stores repo-relative POSIX; this proves it.
#   3. prior cache -- complete and readable for the 160 new cases. prior.py keys the
#      cache on the h5 path, so 0..319 reuse doe_data_240/doe_data_v2 and only
#      320..479 live under doe_data_v3; all four dirs are checked.
#
# Reboot recovery (Docker Desktop does NOT auto-start on Windows reboot -- start it
# first, then relaunch with a single line):
#   RESUME=results/mgn_nodeB_doe_v3_bw30_prior.pt.last bash results/logs/run_v3_480_docker.sh
set -u
cd /workspace/app
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-50}"
TAG="${TAG:-doe_v3_bw30_prior}"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"
DATA="backup/doe_data_v3"
SPLIT="eval/splits/doe_v3_case_split.json"
RESUME="${RESUME:-}"

echo "=== R9 pre-flight 1: split digest + sizes vs assembled manifest ==="
python - <<'EOF'
import json, sys
sys.path.insert(0, "/workspace/app")
from eval.case_split import manifest_digest, load_case_split
man = json.load(open("backup/doe_data_v3/doe_manifest.json", encoding="utf-8"))
digest = manifest_digest(man)
split = load_case_split("eval/splits/doe_v3_case_split.json", expected_digest=digest)
print(f"digest {digest} OK; train/val/test = "
      f"{len(split.train)}/{len(split.val)}/{len(split.test)}")
assert man["n_cases"] == 480, man["n_cases"]
assert (len(split.train), len(split.val), len(split.test)) == (450, 12, 18)
# val/test must be byte-identical to v2's -- the whole point of holding them fixed
# across scales (section 20) is that R9 vs v2 is a like-for-like comparison.
v2 = json.load(open("eval/splits/doe_v2_case_split.json", encoding="utf-8"))
assert sorted(v2["val"]) == sorted(split.val), "val drifted from v2"
assert sorted(v2["test"]) == sorted(split.test), "test drifted from v2"
print("val/test identical to v2 -- comparison is like-for-like")
EOF
[ $? -eq 0 ] || { echo "R9 PREFLIGHT FAILED -- split digest/size/holdout mismatch"; exit 1; }

echo "=== R9 pre-flight 2: container path gate, all 480 cases ==="
python - <<'EOF'
import json, sys
sys.path.insert(0, "/workspace/app")
from pathlib import Path
from eval.doe_dataset import resolve_h5_path
data_dir = Path("backup/doe_data_v3")
man = json.loads((data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
bad = []
for c in man["cases"]:
    idx = int(c["index"])
    if any(resolve_h5_path(p, data_dir, idx) is None for p in c["h5_paths"]):
        bad.append(idx)
print(f"resolved {len(man['cases']) - len(bad)}/{len(man['cases'])} cases")
if bad:
    print(f"  unresolved: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    print(f"  example recorded path: {man['cases'][bad[0]]['h5_paths'][0]}")
sys.exit(1 if bad else 0)
EOF
[ $? -eq 0 ] || { echo "R9 PREFLIGHT FAILED -- h5 paths do not resolve in the container"; exit 1; }

echo "=== R9 pre-flight 3: prior cache complete for all 480 ==="
python - <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
man = json.loads(Path("backup/doe_data_v3/doe_manifest.json").read_text(encoding="utf-8"))
root = Path("results/prior_cache")
leftover = sorted(str(p) for d in root.iterdir() if d.is_dir() for p in d.glob("*.tmp.npz"))
if leftover:
    sys.exit(f"tmp files present (a cache writer is active or died mid-write): {leftover[:6]}")
bad = []
for c in man["cases"]:
    idx = int(c["index"])
    # prior.py: dataset = <h5>/../../.. name, i.e. the dir the h5 physically lives in.
    dataset = Path(c["h5_paths"][0]).parent.parent.parent.name
    f = root / dataset / f"case_{idx:04d}.npz"
    if not f.exists():
        bad.append(f"{dataset}/case_{idx:04d} missing"); continue
    try:
        with np.load(f) as z:
            steps = {int(k[5:]) for k in z.files if k.startswith("step_")}
            if steps != set(range(45)):
                bad.append(f"{dataset}/case_{idx:04d} incomplete ({len(steps)} steps)")
    except Exception as exc:                                    # noqa: BLE001
        bad.append(f"{dataset}/case_{idx:04d} unreadable ({exc!r})")
if bad:
    sys.exit(f"prior cache not ready ({len(bad)} bad): " + "; ".join(bad[:6]))
print(f"prior cache: {len(man['cases'])}/480 complete, 45 steps each")
EOF
[ $? -eq 0 ] || { echo "R9 PREFLIGHT FAILED -- prior cache not ready"; exit 1; }
echo "PREFLIGHT OK"

echo "=== R9 train  BAND_W=${BAND_W}  epochs=${EPOCHS}  h256/p15  batch=1  +prior ==="
python -u train_doe_curl_mgn.py \
  --data-dir "${DATA}" \
  --split "${SPLIT}" \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --prior-features \
  --model mgn --hidden-dim 256 --processor-size 15 \
  --band-spectral-weight "${BAND_W}" \
  --epochs "${EPOCHS}" --batch-size 1 --step-stride 1 \
  --ckpt "${CKPT}" --ckpt-every 1 ${RESUME:+--resume "${RESUME}"}
TRAIN_EXIT=$?
echo "TRAIN exit=${TRAIN_EXIT}"
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "R9 TRAIN FAILED -- skipping scoring. Resume with:"
  echo "  RESUME=${CKPT}.last bash results/logs/run_v3_480_docker.sh"
  exit 1
fi

echo "=== R9 scoring: all 18 test cases (legacy 6 + new-current 12) ==="
rm -f "${SCORE}"     # never let a retry summarize a stale artifact
python -u -m eval.benchmark \
  --data-dir "${DATA}" --split "${SPLIT}" --subset test \
  --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out "${SCORE}"
SCORE_EXIT=$?
echo "SCORE exit=${SCORE_EXIT}"
if [ "${SCORE_EXIT}" -ne 0 ]; then
  echo "R9 SCORING FAILED -- no DONE banner, the section-30 verdict must not use ${SCORE}"
  exit 1
fi

echo "=== grouped summary: legacy_650A vs new_current (pooled AND per-case mean) ==="
python - <<EOF
import json, math
split = json.load(open("${SPLIT}", encoding="utf-8"))
groups = {k: set(v) for k, v in split["test_groups"].items()}
d = json.load(open("${SCORE}", encoding="utf-8"))
model_key = "curl:mgn_nodeB_${TAG}"
s = d["models"][model_key]["summary"]
by_case = s["by_case"]
out = {"model": model_key, "n_test_cases": len(by_case), "groups": {}}

def pooled(rows):
    # Pooled nRMSE = RMSE over all samples / RMS of truth over all samples -- the SAME
    # statistic as every campaign scorecard. The mean of per-case percentages is a
    # DIFFERENT statistic that near-zero-torque cases (geometry 32) blow up; section 29
    # documents the trap. Emit both, clearly labelled; gates read pooled only.
    nB = eB = rB = nT = eT = rT = 0.0
    for v in rows.values():
        b = v["channels"]["Bnorm"]
        ref = b["rmse"] / (b["nrmse_pct"] / 100.0)
        nB += b["n"]; eB += b["n"] * b["rmse"] ** 2; rB += b["n"] * ref ** 2
        if "torque" in v:
            t = v["torque"]
            nT += t["n"]; eT += t["n"] * t["rmse_torque"] ** 2; rT += t["n"] * t["rms_torque_true"] ** 2
    pb = 100 * math.sqrt(eB / nB) / math.sqrt(rB / nB) if nB else None
    pt = 100 * math.sqrt(eT / nT) / math.sqrt(rT / nT) if nT else None
    return pb, pt

for gname, gcases in groups.items():
    rows = {int(ci): v for ci, v in by_case.items() if int(ci) in gcases}
    bs = [v["channels"]["Bnorm"]["nrmse_pct"] for v in rows.values()]
    tq = [v["torque"]["nrmse_torque_pct"] for v in rows.values() if "torque" in v]
    pb, pt = pooled(rows)
    out["groups"][gname] = {
        "cases": sorted(rows),
        "n_cases_missing_torque": sum(1 for v in rows.values() if "torque" not in v),
        "pooled": {"bnorm_nrmse_pct": round(pb, 3) if pb else None,
                   "torque_nrmse_pct": round(pt, 3) if pt else None},
        "per_case_mean": {"bnorm_nrmse_pct": round(sum(bs) / len(bs), 3) if bs else None,
                          "torque_nrmse_pct": round(sum(tq) / len(tq), 3) if tq else None},
        "bnorm_nrmse_pct_by_case": {str(k): round(rows[k]["channels"]["Bnorm"]["nrmse_pct"], 3)
                                    for k in sorted(rows)},
    }
out["overall"] = {"bnorm_nrmse_pct": s["overall"]["Bnorm"]["nrmse_pct"],
                  "airgap_bnorm_nrmse_pct": s["by_region"]["airgap"]["Bnorm"]["nrmse_pct"],
                  "torque_nrmse_pct": s["torque"]["nrmse_torque_pct"]}
path = "results/benchmark_v2_nodeB_${TAG}_groups.json"
json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
print("grouped summary ->", path)
print(json.dumps(out["overall"], indent=2))
EOF
GROUPS_EXIT=$?
echo "GROUPS exit=${GROUPS_EXIT}"
if [ "${GROUPS_EXIT}" -ne 0 ]; then
  echo "R9 GROUPED SUMMARY FAILED -- raw score at ${SCORE} is valid, regroup by hand"
  exit 1
fi
# The legacy-6 subset is what section 30's decision rule reads. Score it directly rather
# than trusting a hand-recomputed number -- section 29 cross-validated the two to 3 dp.
echo "=== R9 legacy-6 direct re-score (the G1' gate subset) ==="
python -u -m eval.benchmark \
  --data-dir "${DATA}" --split eval/splits/doe_v3_case_split_legacy6.json --subset test \
  --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out "results/benchmark_v2_nodeB_${TAG}_legacy6.json"
LEG_EXIT=$?
echo "LEGACY6 exit=${LEG_EXIT}"
if [ "${LEG_EXIT}" -ne 0 ]; then
  echo "R9 LEGACY6 SCORING FAILED -- the section-30 airgap decision rule has no input"
  exit 1
fi
python - <<EOF
import json
d = json.load(open("results/benchmark_v2_nodeB_${TAG}_legacy6.json", encoding="utf-8"))
s = d["models"]["curl:mgn_nodeB_${TAG}"]["summary"]
b = s["overall"]["Bnorm"]["nrmse_pct"]
a = s["by_region"]["airgap"]["Bnorm"]["nrmse_pct"]
t = s["torque"]["nrmse_torque_pct"]
print(f"R9 legacy6 pooled: |B| {b:.3f} / airgap {a:.3f} / torque {t:.3f}")
print("           v2 ref:  |B| 11.107 / airgap 6.618 / torque 3.270")
verdict = ("IMPROVED -- new champion" if a < 6.368 else
           "WITHIN BAND -- slope flattening" if a <= 6.868 else
           "REGRESSED -- suspect epoch budget")
print(f"section-30 airgap rule: {verdict}")
print(f"G1' (airgap<5 AND torque<3): {'PASS' if (a < 5 and t < 3) else 'not yet'}")
EOF
echo "V3 480 DONE"
