#!/bin/bash
# Score both post-fix checkpoints in one pass. Passing both to a single
# invocation shares the DOE load and the region grouping, which is most of the
# CPU cost, so this is far cheaper than two separate runs.
set -u
cd /workspace/app

python -u -m eval.benchmark \
  --data-dir backup/doe_data \
  --skip-curl-floor \
  --curl-ckpt results/mgn_curl_v2.pt results/mgn_nodeB_v2.pt \
  --out results/benchmark_v2_postfix.json
echo "SCORING DONE exit=$?"
