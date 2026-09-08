#!/bin/bash
# E1: airgap-weighted loss, pre-registered in methodology_review section 30 ("E1 -- 공극 가중 손실 재방문").
#
# SINGLE-AXIS DISCIPLINE: identical to run_e1_airgap_docker.sh (the R9 champion recipe) except ONE
# added training flag, --airgap-weight 5 (default 1.0). Same data, same split, same model, same
# other losses, same epochs. Any other edit invalidates the comparison against the R9 baseline
# (legacy6 pooled, ep50: |B| 11.302 / airgap 6.925 / torque 3.799).
#
# Pre-registered decision rule (section 30 "E1"), read on the LEGACY 6-test group, pooled,
# vs the R9 ep50 baseline; run-to-run band is +-0.25pp:
#   airgap < 6.675%   -> SUCCESS: airgap weighting is a live lever; adopt and re-baseline
#   6.675 .. 7.175%   -> inside the band: no effect. Record and move on.
#   > 7.175%          -> harmful on the very metric the gate reads.
#   HARM GUARD (reject even if airgap improves): |B| > 11.552% OR torque > 4.049%
# Rationale (section 30): the prior+spectral+current stack changed since section 16, so loss
# re-balancing is open again, and G1' reads exactly the airgap.
#
# Scoring: identical to R9 -- all 18 test cases, then legacy6 / new12 with their v3-digest splits.
set -u
cd /workspace/app
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-50}"
TAG="${TAG:-doe_v3_bw30_prior_e1airgap5}"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"
DATA="backup/doe_data_v3"
SPLIT="eval/splits/doe_v3_case_split.json"
RESUME="${RESUME:-}"        # reboot recovery: RESUME=<ckpt>.last bash <this script>

echo "=== E1 pre-flight 1: split digest + sizes ==="
python - <<'EOF'
import json, sys
sys.path.insert(0, "/workspace/app")
from eval.case_split import manifest_digest, load_case_split
man = json.load(open("backup/doe_data_v3/doe_manifest.json", encoding="utf-8"))
d = manifest_digest(man)
s = load_case_split("eval/splits/doe_v3_case_split.json", expected_digest=d)
print(f"digest {d}; train/val/test = {len(s.train)}/{len(s.val)}/{len(s.test)}")
assert (len(s.train), len(s.val), len(s.test)) == (450, 12, 18)
# val/test must be IDENTICAL to v2 -- the fixed-holdout principle the scaling curve rests on
v2 = json.load(open("eval/splits/doe_v2_case_split.json", encoding="utf-8"))
assert sorted(s.val) == sorted(int(c) for c in v2["val"]), "val drifted from v2"
assert sorted(s.test) == sorted(int(c) for c in v2["test"]), "test drifted from v2"
print("val/test identical to v2: OK")
EOF
[ $? -eq 0 ] || { echo "E1 PREFLIGHT FAILED -- split contract"; exit 1; }

echo "=== E1 pre-flight 2: prior cache complete for every case ==="
python - <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
man = json.load(open("backup/doe_data_v3/doe_manifest.json", encoding="utf-8"))
# cache dir is keyed by the resolved h5 path's grandparent (see fem_warmstart/prior.py)
need = {}
for c in man["cases"]:
    ds = Path(c["h5_paths"][0]).parent.parent.parent.name
    need.setdefault(ds, []).append(int(c["index"]))
bad, leftover = [], []
for ds, idxs in need.items():
    d = Path("results/prior_cache") / ds
    leftover += [p.name for p in d.glob("*.tmp.npz")] if d.exists() else []
    for ci in idxs:
        f = d / f"case_{ci:04d}.npz"
        if not f.exists():
            bad.append(f"{ds}/{f.name} missing"); continue
        try:
            with np.load(f) as z:
                steps = {int(k[5:]) for k in z.files if k.startswith("step_")}
                if steps != set(range(45)):
                    bad.append(f"{ds}/{f.name} has {len(steps)} steps")
        except Exception as exc:                                # noqa: BLE001
            bad.append(f"{ds}/{f.name} unreadable ({exc!r})")
if leftover:
    sys.exit(f"tmp files present (a cache writer is active): {leftover[:5]}")
if bad:
    sys.exit("prior cache not ready: " + "; ".join(bad[:6]) + f"  (+{max(0,len(bad)-6)} more)")
print(f"prior cache: complete for all {sum(len(v) for v in need.values())} cases "
      f"across {sorted(need)}")
EOF
[ $? -eq 0 ] || { echo "E1 PREFLIGHT FAILED -- prior cache"; exit 1; }
echo "PREFLIGHT OK"

echo "=== E1 train (airgap-weight 5)  480 cases  BAND_W=${BAND_W}  epochs=${EPOCHS} ==="
python -u train_doe_curl_mgn.py \
  --data-dir "${DATA}" \
  --split "${SPLIT}" \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --prior-features \
  --model mgn --hidden-dim 256 --processor-size 15 \
  --band-spectral-weight "${BAND_W}" \
  --airgap-weight 5 \
  --epochs "${EPOCHS}" --batch-size 1 --step-stride 1 \
  --ckpt "${CKPT}" --ckpt-every 1 ${RESUME:+--resume "${RESUME}"} \
  ${EXTEND_TO:+--extend-to "${EXTEND_TO}"}
TRAIN_EXIT=$?
echo "TRAIN exit=${TRAIN_EXIT}"
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "E1 TRAIN FAILED -- skipping scoring. Resume with:"
  echo "  RESUME=${CKPT}.last bash results/logs/run_e1_airgap_docker.sh"
  exit 1
fi

echo "=== E1 scoring: all 18 test cases, then each group with its own v3-digest split ==="
for pair in "${SPLIT}:${SCORE}" \
            "eval/splits/doe_v3_case_split_legacy6.json:results/benchmark_v2_nodeB_${TAG}_legacy6.json" \
            "eval/splits/doe_v3_case_split_new12.json:results/benchmark_v2_nodeB_${TAG}_new12.json"; do
  sp="${pair%%:*}"; out="${pair##*:}"
  rm -f "${out}"
  python -u -m eval.benchmark --data-dir "${DATA}" --split "${sp}" --subset test \
    --skip-curl-floor --skip-grid-floor --curl-ckpt "${CKPT}" --out "${out}"
  rc=$?
  echo "SCORE ${out} exit=${rc}"
  [ "${rc}" -eq 0 ] || { echo "E1 SCORING FAILED on ${out}"; exit 1; }
done

echo "E1 AIRGAP5 DONE"
