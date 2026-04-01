#!/usr/bin/env python
# coding: utf-8

# # PhysicsNeMo Training Notebook (from pyMCAD outputs)
# 
# ? ???? `pyMCAD4SolverX.ipynb`? ???? ??? ??? ????,
# `Mag_*.h5` ?? `MagTransient*.txt`? ?? PhysicsNeMo `MeshGraphNet` ???? ?????.
# 
# - ?? ??: pyMCAD magnetic timeseries (`steps`, `mesh/*`, `fields/*`, `meta/*`) ?? txt table blocks
# - ??? ??: ???? -> edge, ?? ?? ?? ??, ?? ??(Bx, By) ??
# - ??: `physicsnemo.models.meshgraphnet.MeshGraphNet`
# 

# In[1]:


import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from physicsnemo.models.meshgraphnet import MeshGraphNet

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())


# In[ ]:


# Optional: repo tools import (??? ??, ??? fallback parser ??)
repo_root = Path.cwd().resolve()
while not ((repo_root / "tools").exists() or (repo_root / "tool").exists()) and repo_root != repo_root.parent:
    repo_root = repo_root.parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

PYMCAD_AVAILABLE = False
get_magnetic_timeseries_from_file = None
inspect_magnetic_timeseries_h5 = None

try:
    from tools.motorCAD.pyMCAD import get_magnetic_timeseries_from_file, inspect_magnetic_timeseries_h5
    PYMCAD_AVAILABLE = True
    print("pyMCAD tools available from:", repo_root)
except Exception as e:
    print("pyMCAD tools import failed -> fallback parser mode")
    print("reason:", repr(e))


# In[ ]:


# =========================
# Config
# =========================
# 1) ?? ??? ?? ?? (??)
H5_PATH = None
TXT_PATH = '/workspace/host_data/data/MagTransient_20260130_152245.txt'

# ??:
# H5_PATH = r"/workspace/data/Mag_OnLoadTorque_result_xxx.h5"
# TXT_PATH = r"/workspace/data/MagTransient_20260130_152245.txt"

# 2) ?? ?? ??
SEARCH_ROOT = Path.cwd().resolve()

# 3) parser ??
STEP_KEY = "time_index"   # txt??? ??: "time_index" or "solution"
MAX_STEPS = None          # ?? ????: ?) 12

# 4) training ??
TRAIN_RATIO = 0.8
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEED = 42

# 5) ?? ??
CKPT_PATH = SEARCH_ROOT / "physicsnemo_meshgraphnet_ckpt.pt"


# In[ ]:


def _find_latest_file(root: Path, pattern: str):
    cands = list(root.rglob(pattern))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def resolve_input_path():
    if H5_PATH:
        p = Path(H5_PATH)
        if not p.exists():
            raise FileNotFoundError(f"H5_PATH not found: {p}")
        return p
    if TXT_PATH:
        p = Path(TXT_PATH)
        if not p.exists():
            raise FileNotFoundError(f"TXT_PATH not found: {p}")
        return p

    p = _find_latest_file(SEARCH_ROOT, "Mag_*.h5")
    if p is not None:
        return p

    p = _find_latest_file(SEARCH_ROOT, "MagTransient*.txt")
    if p is not None:
        return p

    raise FileNotFoundError(
        f"No Mag_*.h5 or MagTransient*.txt found under {SEARCH_ROOT}.\n"
        "Set H5_PATH or TXT_PATH explicitly."
    )


in_path = resolve_input_path()
print("input:", in_path)


# In[ ]:


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def parse_txt_timeseries(path: Path, key="time_index", max_steps=None):
    # returns list[record], record schema:
    # {
    #   step_key, solution, time_s, rotate_step,
    #   node_xy: dict[node_id] -> (x,y),
    #   node_1,node_2,node_3,reg_code,bx,by,a,j (np arrays, len=n_elem)
    # }
    step_re = re.compile(
        r"^10\s+Solution\s+(?P<solution>\d+)(?:\s+Time\s+index\s+(?P<time_index>\d+)\s+Time\s+(?P<time_s>[-+0-9.Ee]+)\s+\[s\])?\s+Rotate\s+Step\s+(?P<rotate_step>[-+0-9.Ee]+)",
        re.IGNORECASE,
    )

    def _read_until(in_file, marker):
        while True:
            line = in_file.readline()
            if not line:
                return None
            if marker in line:
                return line

    def _skip_header(in_file, n=4):
        for _ in range(n):
            in_file.readline()

    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            m = step_re.match(line.strip())
            if not m:
                continue

            solution = int(m.group("solution"))
            time_index = int(m.group("time_index")) if m.group("time_index") else None
            time_s = _safe_float(m.group("time_s"), 0.0) if m.group("time_s") else 0.0
            rotate_step = _safe_float(m.group("rotate_step"), 0.0)

            if key == "time_index":
                step_key = int(time_index) if time_index is not None else int(solution)
            else:
                step_key = int(solution)

            h_el = _read_until(f, "ElementsTable")
            if h_el is None:
                continue
            try:
                n_el = int(h_el.strip().split()[1])
            except Exception:
                continue
            _skip_header(f, 4)

            node_1 = np.zeros((n_el,), dtype=np.int32)
            node_2 = np.zeros((n_el,), dtype=np.int32)
            node_3 = np.zeros((n_el,), dtype=np.int32)
            reg_code = np.zeros((n_el,), dtype=np.int32)
            bx = np.zeros((n_el,), dtype=np.float32)
            by = np.zeros((n_el,), dtype=np.float32)
            a = np.zeros((n_el,), dtype=np.float32)
            j = np.zeros((n_el,), dtype=np.float32)

            for i in range(n_el):
                row = f.readline().split(",")
                if len(row) < 9:
                    continue
                try:
                    node_1[i] = int(row[1])
                    node_2[i] = int(row[2])
                    node_3[i] = int(row[3])
                    reg_code[i] = int(row[4])
                    bx[i] = _safe_float(row[5])
                    by[i] = _safe_float(row[6])
                    a[i] = _safe_float(row[7])
                    j[i] = _safe_float(row[8])
                except Exception:
                    continue

            node_xy = {}
            h_nodes = _read_until(f, "NodesTable")
            if h_nodes is not None:
                try:
                    n_nodes = int(h_nodes.strip().split()[1])
                except Exception:
                    n_nodes = 0
                _skip_header(f, 4)
                for _ in range(n_nodes):
                    row = f.readline().split(",")
                    if len(row) < 3:
                        continue
                    try:
                        nid = int(row[0])
                        x = _safe_float(row[1])
                        y = _safe_float(row[2])
                        node_xy[nid] = (x, y)
                    except Exception:
                        continue

            rec = {
                "step_key": int(step_key),
                "solution": int(solution),
                "time_s": float(time_s),
                "rotate_step": float(rotate_step),
                "node_xy": node_xy,
                "node_1": node_1,
                "node_2": node_2,
                "node_3": node_3,
                "reg_code": reg_code,
                "bx": bx,
                "by": by,
                "a": a,
                "j": j,
            }
            records.append(rec)

            if max_steps is not None and len(records) >= int(max_steps):
                break

    return records


def parse_h5_timeseries_fallback(path: Path, max_steps=None):
    # pyMCAD magnetic timeseries h5 parser (fallback mode)
    import h5py

    records = []
    with h5py.File(path, "r") as f:
        if "steps" not in f:
            raise ValueError(f"Invalid magnetic h5 (missing 'steps'): {path}")

        steps = np.asarray(f["steps"][:], dtype=np.int32)

        node_id = np.asarray(f["mesh/node_id"][:], dtype=np.int32)
        node_x0 = np.asarray(f["mesh/node_x_mm"][:], dtype=np.float64)
        node_y0 = np.asarray(f["mesh/node_y_mm"][:], dtype=np.float64)

        node_1 = np.asarray(f["mesh/node_1"][:], dtype=np.int32)
        node_2 = np.asarray(f["mesh/node_2"][:], dtype=np.int32)
        node_3 = np.asarray(f["mesh/node_3"][:], dtype=np.int32)
        reg_code = np.asarray(f["mesh/reg_code"][:], dtype=np.int32)

        bx_mat = np.asarray(f["fields/bx"][:], dtype=np.float32)
        by_mat = np.asarray(f["fields/by"][:], dtype=np.float32)
        a_mat = np.asarray(f["fields/a"][:], dtype=np.float32) if "fields/a" in f else np.zeros_like(bx_mat)
        j_mat = np.asarray(f["fields/j"][:], dtype=np.float32) if "fields/j" in f else np.zeros_like(bx_mat)

        meta_time = np.asarray(f["meta/time_s"][:], dtype=np.float64) if "meta/time_s" in f else None
        meta_rot = np.asarray(f["meta/rotate_step"][:], dtype=np.float64) if "meta/rotate_step" in f else None
        meta_sol = np.asarray(f["meta/solution"][:], dtype=np.int32) if "meta/solution" in f else None

        x_by_step = np.asarray(f["mesh/node_x_mm_by_step"][:], dtype=np.float64) if "mesh/node_x_mm_by_step" in f else None
        y_by_step = np.asarray(f["mesh/node_y_mm_by_step"][:], dtype=np.float64) if "mesh/node_y_mm_by_step" in f else None

        x_by_step_moving = np.asarray(f["mesh/node_x_mm_by_step_moving"][:], dtype=np.float64) if "mesh/node_x_mm_by_step_moving" in f else None
        y_by_step_moving = np.asarray(f["mesh/node_y_mm_by_step_moving"][:], dtype=np.float64) if "mesh/node_y_mm_by_step_moving" in f else None
        moving_indices = np.asarray(f["mesh/moving_node_indices"][:], dtype=np.int32) if "mesh/moving_node_indices" in f else None

        n_steps = int(steps.shape[0])
        for si in range(n_steps):
            x = node_x0.copy()
            y = node_y0.copy()

            if x_by_step is not None and y_by_step is not None:
                xs = np.asarray(x_by_step[si], dtype=np.float64)
                ys = np.asarray(y_by_step[si], dtype=np.float64)
                mask = np.isfinite(xs) & np.isfinite(ys)
                x[mask] = xs[mask]
                y[mask] = ys[mask]

            if (
                x_by_step_moving is not None
                and y_by_step_moving is not None
                and moving_indices is not None
                and moving_indices.size > 0
            ):
                xs = np.asarray(x_by_step_moving[si], dtype=np.float64)
                ys = np.asarray(y_by_step_moving[si], dtype=np.float64)
                for li, ni in enumerate(moving_indices.tolist()):
                    if 0 <= int(ni) < int(node_id.size):
                        xv = float(xs[li])
                        yv = float(ys[li])
                        if np.isfinite(xv) and np.isfinite(yv):
                            x[int(ni)] = xv
                            y[int(ni)] = yv

            node_xy = {int(node_id[i]): (float(x[i]), float(y[i])) for i in range(node_id.size)}

            rec = {
                "step_key": int(steps[si]),
                "solution": int(meta_sol[si]) if meta_sol is not None else int(si + 1),
                "time_s": float(meta_time[si]) if meta_time is not None else 0.0,
                "rotate_step": float(meta_rot[si]) if meta_rot is not None else 0.0,
                "node_xy": node_xy,
                "node_1": node_1,
                "node_2": node_2,
                "node_3": node_3,
                "reg_code": reg_code,
                "bx": bx_mat[si],
                "by": by_mat[si],
                "a": a_mat[si],
                "j": j_mat[si],
            }
            records.append(rec)

            if max_steps is not None and len(records) >= int(max_steps):
                break

    return records


# In[ ]:


def load_records_from_input(path: Path):
    suf = path.suffix.lower()

    if PYMCAD_AVAILABLE:
        # pyMCAD ?? ?? ?? (?? ??? ?? ??)
        ts = get_magnetic_timeseries_from_file(path, key=STEP_KEY)
        steps = ts.steps
        if MAX_STEPS is not None:
            steps = steps[: int(MAX_STEPS)]

        records = []
        for st in steps:
            mr = ts.by_step[int(st)]
            meta = ts.meta.get(int(st), {}) if getattr(ts, "meta", None) else {}

            node_xy = dict(getattr(mr, "node_xy", {}) or {})
            if not node_xy:
                continue

            elems = []
            for region in getattr(mr, "_regions", []):
                elems.extend(getattr(region, "elements", []) or [])
            if not elems:
                continue

            node_1 = np.asarray([int(e.node_1) for e in elems], dtype=np.int32)
            node_2 = np.asarray([int(e.node_2) for e in elems], dtype=np.int32)
            node_3 = np.asarray([int(e.node_3) for e in elems], dtype=np.int32)
            reg_code = np.asarray([int(e.reg_code) for e in elems], dtype=np.int32)
            bx = np.asarray([_safe_float(getattr(e, "bx", 0.0)) for e in elems], dtype=np.float32)
            by = np.asarray([_safe_float(getattr(e, "by", 0.0)) for e in elems], dtype=np.float32)
            a = np.asarray([_safe_float(getattr(e, "a", 0.0)) for e in elems], dtype=np.float32)
            j = np.asarray([_safe_float(getattr(e, "j", 0.0)) for e in elems], dtype=np.float32)

            records.append(
                {
                    "step_key": int(st),
                    "solution": int(meta.get("solution") or 0),
                    "time_s": float(meta.get("time_s") or 0.0),
                    "rotate_step": float(meta.get("rotate_step") or 0.0),
                    "node_xy": node_xy,
                    "node_1": node_1,
                    "node_2": node_2,
                    "node_3": node_3,
                    "reg_code": reg_code,
                    "bx": bx,
                    "by": by,
                    "a": a,
                    "j": j,
                }
            )

        return records

    # fallback mode
    if suf in {".h5", ".hdf5"}:
        return parse_h5_timeseries_fallback(path, max_steps=MAX_STEPS)

    return parse_txt_timeseries(path, key=STEP_KEY, max_steps=MAX_STEPS)


records = load_records_from_input(in_path)
print("records:", len(records))
if not records:
    raise RuntimeError("No records loaded")

r0 = records[0]
print("sample step:", r0["step_key"], "nodes:", len(r0["node_xy"]), "elements:", len(r0["node_1"]))


# In[ ]:


def build_graph_from_record(rec):
    node_xy = rec["node_xy"]
    if not node_xy:
        return None

    node_ids = sorted(int(k) for k in node_xy.keys())
    n = len(node_ids)
    if n == 0:
        return None

    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    pos = np.asarray([node_xy[nid] for nid in node_ids], dtype=np.float32)

    n1 = np.asarray(rec["node_1"], dtype=np.int64)
    n2 = np.asarray(rec["node_2"], dtype=np.int64)
    n3 = np.asarray(rec["node_3"], dtype=np.int64)
    reg = np.asarray(rec["reg_code"], dtype=np.int32)
    bx = np.asarray(rec["bx"], dtype=np.float32)
    by = np.asarray(rec["by"], dtype=np.float32)
    aa = np.asarray(rec["a"], dtype=np.float32)
    jj = np.asarray(rec["j"], dtype=np.float32)

    m = int(min(len(n1), len(n2), len(n3), len(reg), len(bx), len(by), len(aa), len(jj)))
    if m <= 0:
        return None

    sum_bx = np.zeros((n,), dtype=np.float32)
    sum_by = np.zeros((n,), dtype=np.float32)
    sum_a = np.zeros((n,), dtype=np.float32)
    sum_j = np.zeros((n,), dtype=np.float32)
    cnt = np.zeros((n,), dtype=np.float32)
    reg_votes = defaultdict(list)
    edges = set()

    for i in range(m):
        try:
            i1 = id2idx[int(n1[i])]
            i2 = id2idx[int(n2[i])]
            i3 = id2idx[int(n3[i])]
        except KeyError:
            continue

        tri = [i1, i2, i3]
        for u in tri:
            sum_bx[u] += float(bx[i])
            sum_by[u] += float(by[i])
            sum_a[u] += float(aa[i])
            sum_j[u] += float(jj[i])
            cnt[u] += 1.0
            reg_votes[u].append(int(reg[i]))

        for a, b in [(i1, i2), (i2, i3), (i3, i1)]:
            edges.add((a, b))
            edges.add((b, a))

    if not edges:
        return None

    cnt = np.clip(cnt, 1.0, None)
    node_bx = sum_bx / cnt
    node_by = sum_by / cnt
    node_a = sum_a / cnt
    node_j = sum_j / cnt

    node_reg = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        votes = reg_votes.get(i, [])
        if votes:
            vals, counts = np.unique(np.asarray(votes, dtype=np.int32), return_counts=True)
            node_reg[i] = float(vals[np.argmax(counts)])

    t_s = float(rec.get("time_s", 0.0) or 0.0)
    rot = float(rec.get("rotate_step", 0.0) or 0.0)

    x = np.concatenate(
        [
            pos,
            node_a[:, None],
            node_j[:, None],
            node_reg[:, None],
            np.full((n, 1), t_s, dtype=np.float32),
            np.full((n, 1), rot, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    # ?€ê²?y)??Bx, By ë¿ë§Œ ?„ë‹ˆ??A(Magnetic Vector Potential), J(Current Density) ì¶”ê?
    y = np.stack([node_bx, node_by, node_a, node_j], axis=1).astype(np.float32)

    edge_list = sorted(edges)
    edge_index = np.asarray(edge_list, dtype=np.int64).T
    src = edge_index[0]
    dst = edge_index[1]
    dxy = pos[dst] - pos[src]
    dist = np.linalg.norm(dxy, axis=1, keepdims=True)
    edge_attr = np.concatenate([dxy, dist], axis=1).astype(np.float32)

    g = Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        pos=torch.from_numpy(pos),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
    )
    return g


graphs = []
used_steps = []
for rec in records:
    g = build_graph_from_record(rec)
    if g is not None:
        graphs.append(g)
        used_steps.append(int(rec.get("step_key", len(used_steps))))

print("graphs:", len(graphs))
if len(graphs) < 2:
    raise RuntimeError("Need at least 2 graph snapshots for training")

print("graph[0] x/y/edge_attr:", graphs[0].x.shape, graphs[0].y.shape, graphs[0].edge_attr.shape)


# In[ ]:


# time-order split
n_total = len(graphs)
n_train = max(1, int(n_total * TRAIN_RATIO))
train_graphs = graphs[:n_train]
val_graphs = graphs[n_train:] if n_total > n_train else graphs[-1:]

# normalize with train stats only
x_train = torch.cat([g.x for g in train_graphs], dim=0)
y_train = torch.cat([g.y for g in train_graphs], dim=0)

x_mean = x_train.mean(dim=0, keepdim=True)
x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
y_mean = y_train.mean(dim=0, keepdim=True)
y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)

for g in train_graphs + val_graphs:
    g.x = (g.x - x_mean) / x_std
    g.y = (g.y - y_mean) / y_std

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

print("train/val:", len(train_graphs), len(val_graphs))
print("node feature dim:", train_graphs[0].x.shape[1], "target dim:", train_graphs[0].y.shape[1])


# In[ ]:


torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MeshGraphNet(
    input_dim_nodes=train_graphs[0].x.shape[1],
    input_dim_edges=train_graphs[0].edge_attr.shape[1],
    output_dim=train_graphs[0].y.shape[1],
    processor_size=10,
    hidden_dim_processor=128,
    hidden_dim_node_encoder=128,
    hidden_dim_edge_encoder=128,
    hidden_dim_node_decoder=128,
    aggregation="sum",
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total = 0.0
    n_batch = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(training):
            pred = model(batch.x, batch.edge_attr, batch)
            loss = F.mse_loss(pred, batch.y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.item())
        n_batch += 1
    return total / max(n_batch, 1)

train_hist = []
val_hist = []
for ep in range(1, EPOCHS + 1):
    tr = run_epoch(train_loader, training=True)
    va = run_epoch(val_loader, training=False)
    train_hist.append(tr)
    val_hist.append(va)
    if ep == 1 or ep % 5 == 0:
        print(f"epoch {ep:03d} | train {tr:.6f} | val {va:.6f}")


# In[ ]:


plt.figure(figsize=(6, 4))
plt.plot(train_hist, label="train")
plt.plot(val_hist, label="val")
plt.xlabel("epoch")
plt.ylabel("MSE (normalized)")
plt.grid(alpha=0.3)
plt.legend()
plt.title("MeshGraphNet training curve")
plt.savefig('training_result.png')


# In[ ]:


# validation snapshot visualization (denormalized)
model.eval()
g = val_graphs[0].to(device)
with torch.no_grad():
    pred_n = model(g.x, g.edge_attr, g).cpu()

pred = pred_n * y_std.cpu() + y_mean.cpu()
true = g.y.cpu() * y_std.cpu() + y_mean.cpu()
pos = g.pos.cpu().numpy()

b_true = torch.linalg.norm(true, dim=1).numpy()
b_pred = torch.linalg.norm(pred, dim=1).numpy()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sc0 = ax[0].scatter(pos[:, 0], pos[:, 1], c=b_true, s=4, cmap="turbo")
ax[0].set_title("|B| true")
ax[0].set_aspect("equal")
plt.colorbar(sc0, ax=ax[0])

sc1 = ax[1].scatter(pos[:, 0], pos[:, 1], c=b_pred, s=4, cmap="turbo")
ax[1].set_title("|B| pred")
ax[1].set_aspect("equal")
plt.colorbar(sc1, ax=ax[1])

plt.tight_layout()
plt.savefig('training_result.png')

mae = torch.mean(torch.abs(pred - true), dim=0)
print("MAE Bx, By:", mae.tolist())


# In[ ]:


CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "used_steps": used_steps,
        "input_path": str(in_path),
        "train_hist": train_hist,
        "val_hist": val_hist,
    },
    CKPT_PATH,
)

print("saved:", CKPT_PATH)

