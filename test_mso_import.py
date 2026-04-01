#!/usr/bin/env python
"""Quick import test for multiscale_operator inside Docker."""
import sys

tests = []

# 1) MeshGraphNet
try:
    from multiscale_operator.operators.mesh_graph_net import EncoderProcessorDecoder
    tests.append(("MeshGraphNet", "OK", str(EncoderProcessorDecoder)))
except Exception as e:
    tests.append(("MeshGraphNet", "FAIL", str(e)))

# 2) BSMS
try:
    import multiscale_operator.operators.bsms as bsms_mod
    names = [x for x in dir(bsms_mod) if not x.startswith("_")]
    tests.append(("BSMS", "OK", str(names)))
except Exception as e:
    tests.append(("BSMS", "FAIL", str(e)))

# 3) Perceiver
try:
    import multiscale_operator.operators.perceiver as perc_mod
    names = [x for x in dir(perc_mod) if not x.startswith("_")]
    tests.append(("Perceiver", "OK", str(names)))
except Exception as e:
    tests.append(("Perceiver", "FAIL", str(e)))

# 4) GNN+Perceiver
try:
    import multiscale_operator.operators.gnn_perceiver as gp_mod
    names = [x for x in dir(gp_mod) if not x.startswith("_")]
    tests.append(("GNN_Perceiver", "OK", str(names)))
except Exception as e:
    tests.append(("GNN_Perceiver", "FAIL", str(e)))

# 5) Trainer
try:
    import multiscale_operator.model.trainer as tr_mod
    names = [x for x in dir(tr_mod) if not x.startswith("_")]
    tests.append(("Trainer", "OK", str(names)))
except Exception as e:
    tests.append(("Trainer", "FAIL", str(e)))

# 6) Data
try:
    import multiscale_operator.data as data_mod
    names = [x for x in dir(data_mod) if not x.startswith("_")]
    tests.append(("Data", "OK", str(names)))
except Exception as e:
    tests.append(("Data", "FAIL", str(e)))

# 7) Transforms
try:
    import multiscale_operator.transforms as tf_mod
    names = [x for x in dir(tf_mod) if not x.startswith("_")]
    tests.append(("Transforms", "OK", str(names)))
except Exception as e:
    tests.append(("Transforms", "FAIL", str(e)))

for name, status, detail in tests:
    print(f"  [{status:4s}] {name}: {detail}", flush=True)

ok_count = sum(1 for _, s, _ in tests if s == "OK")
print(f"\n{ok_count}/{len(tests)} modules imported successfully", flush=True)
