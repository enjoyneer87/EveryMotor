#!/bin/bash
# R7-A: physics-prior input features on the champion recipe (doe240, 50 ep, bw30).
# Single-axis: ONLY the node-feature layout changes (V2 -> V3 = +prior_a/bx/by from
# fem_warmstart/prior.py, linear mu_r=20 solve of the case geometry with the
# section-24 synthetic winding current). Everything else identical to the
# doe240_bw30_ep50 champion run.
#
# PRE-REGISTERED ACCEPTANCE (recorded before launch, 2026-08-04):
#   compare to champion doe240_bw30_ep50 = TEST |B| 11.638% / airgap 7.30% / tq 5.400%
#   - TEST airgap |B| < 7.05% (>run-band below 7.30)   -> POSITIVE (prior injects information)
#   - TEST overall |B| < 11.39%                        -> POSITIVE on the secondary metric
#   - within band (airgap 7.05-7.55, overall 11.39-11.89) -> NEGATIVE (prior uninformative)
#   - materially worse -> investigate feature scaling before concluding
#
# PRE-FLIGHT (runs first, aborts on mismatch): the builder edit must leave old
# checkpoints byte-compatible -- gate mgn_nodeB_long must re-score 13.324/9.207
# exactly (the section-18 failure shape is a train==eval-consistent corruption
# that gate checks CAN catch here because the V2 path must be untouched).
#
# Prior cache: results/prior_cache/ must be prebuilt (build_prior_cache) or the
# graph build pays ~0.3 s/graph extra. Usage: bash results/logs/run_r7a_docker.sh
set -u
cd /workspace/app
BAND_W="${BAND_W:-30}"
EPOCHS="${EPOCHS:-50}"
TAG="${TAG:-doe240_bw30_prior}"
CKPT="results/mgn_nodeB_${TAG}.pt"
SCORE="results/benchmark_v2_nodeB_${TAG}.json"
DATA="backup/doe_data_240"
SPLIT="eval/splits/doe240_case_split.json"

echo "=== R7-A pre-flight: gate byte-identity after the builder edit ==="
python -u -m eval.benchmark \
  --data-dir backup/doe_data --skip-curl-floor --skip-grid-floor \
  --curl-ckpt results/mgn_nodeB_long.pt \
  --out results/gate_recheck9.json
python - <<'EOF'
import json, sys
d = json.load(open("results/gate_recheck9.json", encoding="utf-8"))
s = d["models"]["curl:mgn_nodeB_long"]["summary"]
b = s["overall"]["Bnorm"]["nrmse_pct"]; t = s["torque"]["nrmse_torque_pct"]
print(f"gate re-score: |B| {b:.3f} / torque {t:.3f}")
ok = abs(b - 13.324) < 5e-4 and abs(t - 9.207) < 5e-4
sys.exit(0 if ok else 1)
EOF
[ $? -eq 0 ] || { echo "R7A PREFLIGHT FAILED -- builder edit broke the V2 path"; exit 1; }
echo "PREFLIGHT OK"

echo "=== R7-A train  BAND_W=${BAND_W}  epochs=${EPOCHS}  h256/p15  batch=1  +prior features ==="
python -u train_doe_curl_mgn.py \
  --data-dir "${DATA}" \
  --split "${SPLIT}" \
  --target B \
  --no-wrap-rotor --no-anti-periodic-edges \
  --prior-features \
  --model mgn --hidden-dim 256 --processor-size 15 \
  --band-spectral-weight "${BAND_W}" \
  --epochs "${EPOCHS}" --batch-size 1 --step-stride 1 \
  --ckpt "${CKPT}"
echo "TRAIN exit=$?"

python -u -m eval.benchmark \
  --data-dir "${DATA}" --split "${SPLIT}" --subset test \
  --skip-curl-floor --skip-grid-floor \
  --curl-ckpt "${CKPT}" \
  --out "${SCORE}"
echo "SCORE exit=$?"
echo "R7A DONE"
