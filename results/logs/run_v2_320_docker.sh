#!/bin/bash
# R8 v2: 320-case training (240 fixed-excitation re-labelled + 80 current-axis pilot)
# on the champion recipe + prior features with the newly wired current coupling.
#
# DECISION RECORD (2026-08-06, section 27): prior features are INCLUDED although the
# R7-A pre-registered PRIMARY verdict was negative (airgap 7.138% inside the 7.05-7.55
# dead band). The inclusion rests on (i) the pre-registered SECONDARY metric being
# positive (overall 11.096% < 11.39%), (ii) torque 5.400 -> 3.354% -- the only lever
# that has ever materially moved the torque leg of the adopted G1' gate, and (iii) the
# prior being the only spatially-resolved carrier of the new current axis
# (PRIOR_CURRENT_GATE_PASS: current reaches the prior field, exact superposition).
#
# PRE-FLIGHT (aborts on mismatch):
#   1. split digest -- doe_v2_case_split.json must match the REBUILT v2 manifest
#      (the path fix touched h5_paths only, so digest 8d671fe7 must survive).
#   2. V3 byte-identity -- the ep47 champion-recipe checkpoint must re-score
#      airgap-region |B| 11.096 exactly on the legacy 6-test after the prior.py
#      current-coupling edit (proves the scale gate leaves pre-R8 priors untouched).
#
# SCORING (both gates, per the approved plan):
#   A. legacy fixed-excitation 6-test = v2 test_groups.legacy_650A (same geometries,
#      same h5 as the doe240 gate; labels corrected to the true 650.538 A).
#   B. current-aware v2 split -- all 18 test cases, then grouped into legacy_650A vs
#      new_current in a summary JSON so the section-27 verdict has one artifact.
set -u
cd /workspace/app
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-50}"
TAG="${TAG:-doe_v2_bw30_prior}"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"
DATA="backup/doe_data_v2"
SPLIT="eval/splits/doe_v2_case_split.json"
# Reboot recovery (audit finding): re-running this script bare would overwrite BOTH
# recovery files at epoch 1. After a host restart, relaunch with
#   RESUME=results/mgn_nodeB_doe_v2_bw30_prior.pt.last bash results/logs/run_v2_320_docker.sh
# (start Docker Desktop first -- it does not auto-start on Windows reboot).
RESUME="${RESUME:-}"

echo "=== v2 pre-flight 1: split digest vs rebuilt manifest ==="
python - <<'EOF'
import json, sys
sys.path.insert(0, "/workspace/app")
from eval.case_split import manifest_digest, load_case_split
man = json.load(open("backup/doe_data_v2/doe_manifest.json", encoding="utf-8"))
digest = manifest_digest(man)
split = load_case_split("eval/splits/doe_v2_case_split.json", expected_digest=digest)
print(f"digest {digest} OK; split sizes train/val/test = "
      f"{len(split.train)}/{len(split.val)}/{len(split.test)}")
assert (len(split.train), len(split.val), len(split.test)) == (290, 12, 18)
EOF
[ $? -eq 0 ] || { echo "V2 PREFLIGHT FAILED -- split digest/size mismatch"; exit 1; }

echo "=== v2 pre-flight 2: V3 byte-identity after the current-coupling edit ==="
python -u -m eval.benchmark \
  --data-dir backup/doe_data_240 --split eval/splits/doe240_case_split.json \
  --subset test --skip-curl-floor --skip-grid-floor \
  --curl-ckpt results/mgn_nodeB_doe240_bw30_prior_ep47.pt \
  --out results/_v2_preflight_v3_identity.json
python - <<'EOF'
import json, sys
d = json.load(open("results/_v2_preflight_v3_identity.json", encoding="utf-8"))
s = d["models"]["curl:mgn_nodeB_doe240_bw30_prior_ep47"]["summary"]
b = s["overall"]["Bnorm"]["nrmse_pct"]
t = s["torque"]["nrmse_torque_pct"]
print(f"V3 re-score: |B| {b:.3f} / torque {t:.3f} (expect 11.096 / 3.354)")
sys.exit(0 if abs(b - 11.096) < 5e-3 and abs(t - 3.354) < 5e-3 else 1)
EOF
[ $? -eq 0 ] || { echo "V2 PREFLIGHT FAILED -- V3 path no longer byte-stable"; exit 1; }

echo "=== v2 pre-flight 3: pilot prior cache complete and readable ==="
# Launching while build_prior_cache is still writing these files is the audited
# tmp-collision race; a partial cache also silently costs ~20 min of duplicate solves.
python - <<'EOF'
import sys
from pathlib import Path
import numpy as np
d = Path("results/prior_cache/doe_data_v2")
leftover = sorted(p.name for p in d.glob("*.tmp.npz"))
if leftover:
    sys.exit(f"tmp files present (a cache writer is active or died mid-write): {leftover}")
bad = []
for ci in range(240, 320):
    f = d / f"case_{ci:04d}.npz"
    if not f.exists():
        bad.append(f"{f.name} missing"); continue
    try:
        with np.load(f) as z:
            steps = {int(k[5:]) for k in z.files if k.startswith("step_")}
            scales = {int(k[3:]) for k in z.files if k.startswith("cs_")}
            if steps != set(range(45)) or scales != set(range(45)):
                bad.append(f"{f.name} incomplete ({len(steps)} steps, {len(scales)} scales)")
    except Exception as exc:                                    # noqa: BLE001
        bad.append(f"{f.name} unreadable ({exc!r})")
if bad:
    sys.exit("pilot prior cache not ready: " + "; ".join(bad[:6]))
print("pilot prior cache: 80/80 complete, 45 steps + scales each")
EOF
[ $? -eq 0 ] || { echo "V2 PREFLIGHT FAILED -- pilot prior cache not ready"; exit 1; }
echo "PREFLIGHT OK"

echo "=== v2 train  BAND_W=${BAND_W}  epochs=${EPOCHS}  h256/p15  batch=1  +prior(current-coupled) ==="
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
# Do not score a stale or absent checkpoint: a benchmark JSON produced after a failed
# train looks exactly like a real scorecard and would poison the section-27 verdict.
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "V2 TRAIN FAILED -- skipping scoring. Resume with:"
  echo "  RESUME=${CKPT}.last bash results/logs/run_v2_320_docker.sh"
  exit 1
fi

echo "=== v2 scoring: all 18 test cases (legacy 6 + new-current 12) ==="
rm -f "${SCORE}"     # never let a retry summarize a stale artifact
python -u -m eval.benchmark \
  --data-dir "${DATA}" --split "${SPLIT}" --subset test \
  --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out "${SCORE}"
SCORE_EXIT=$?
echo "SCORE exit=${SCORE_EXIT}"
if [ "${SCORE_EXIT}" -ne 0 ]; then
  echo "V2 SCORING FAILED -- no DONE banner, section-27 verdict must not use ${SCORE}"
  exit 1
fi

echo "=== grouped summary: legacy_650A vs new_current ==="
python - <<EOF
import json
split = json.load(open("${SPLIT}", encoding="utf-8"))
groups = {k: set(v) for k, v in split["test_groups"].items()}
d = json.load(open("${SCORE}", encoding="utf-8"))
model_key = "curl:mgn_nodeB_${TAG}"
s = d["models"][model_key]["summary"]
by_case = s["by_case"]
out = {"model": model_key, "n_test_cases": len(by_case), "groups": {}}
for gname, gcases in groups.items():
    rows = {int(ci): v for ci, v in by_case.items() if int(ci) in gcases}
    bs = [v["channels"]["Bnorm"]["nrmse_pct"] for v in rows.values()]
    # 'torque' is only present when a case produced torque samples -- guard, and
    # surface the anomaly instead of KeyError-ing the verdict artifact away.
    tq = [v["torque"]["nrmse_torque_pct"] for v in rows.values() if "torque" in v]
    out["groups"][gname] = {
        "cases": sorted(rows),
        "n_cases_missing_torque": sum(1 for v in rows.values() if "torque" not in v),
        "bnorm_nrmse_pct_mean": sum(bs) / len(bs) if bs else None,
        "torque_nrmse_pct_mean": sum(tq) / len(tq) if tq else None,
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
  echo "V2 GROUPED SUMMARY FAILED -- raw score at ${SCORE} is valid, regroup by hand"
  exit 1
fi
# The DONE banner is what heartbeat's DONE_RE keys on, and it checks DONE before
# FAIL -- so this line must be unreachable unless train, score and groups ALL passed.
echo "V2 320 DONE"
