$ErrorActionPreference = "Stop"
$path = "d:\KDH\NvidiaNemo\phase1_tutorial.ipynb"
$nb = Get-Content -Raw $path | ConvertFrom-Json

function Set-CellSource([string]$cellId, [string]$text) {
  $cell = $nb.cells | Where-Object { $_.id -eq $cellId } | Select-Object -First 1
  if (-not $cell) { throw "Cell not found: $cellId" }
  $cell.source = ($text -split "`r?`n")
}

Set-CellSource "#VSC-45a180a0" @'
## 12) Stepwise 추론 (기본: step 1 전용)

학습된 step별 ckpt를 사용해 case별 추론을 수행합니다.
기본 계획은 step 1만 실행(INFER_STEP_PLAN=[1])이며, 이후 [1, 2, 3]으로 확장 가능합니다.
'@

Set-CellSource "#VSC-9de49b42" @'
## 13) Stepwise GT vs Pred 시각화

대표 case를 골라 step별 GT vs Pred scatter를 비교합니다.
'@

Set-CellSource "#VSC-89783a84" @'
## 13-B) Stepwise |GT - Pred| Error Map

대표 case에 대해 특정 step의 채널별 absolute error를 시각화합니다.
'@

Set-CellSource "#VSC-62b9c687" @'
## 13-C) Anti-Periodic 대칭을 이용한 Full Motor 복원 (Stepwise)

1/8 섹터 stepwise 추론 결과를 8회 회전 + anti-periodic sign flip으로 full 360° 모터 단면을 복원합니다.

**변환 규칙 (섹터 k = 0…7, alpha = 45°):**
- 위치: (x, y) -> 회전 kalpha
- 스칼라 (A, Je): (-1)^k 부호 반전
- 벡터 (Bx, By): (-1)^k x 회전 행렬 적용
'@

Set-CellSource "#VSC-ea47af37" @'
from datetime import datetime
from pathlib import Path
import json
import shutil

ROOT = globals().get("ROOT", Path.cwd())
LOG_DIR = globals().get("LOG_DIR", ROOT / "logs" / "tutorial")
LOG_DIR.mkdir(parents=True, exist_ok=True)

STEPWISE_CASE_INDICES = list(range(40))
STEPWISE_SOURCE_FILE_TYPES = ["OnLoadTorque"]
STEPWISE_MAX_STEPS_PER_CASE = 10
STEPWISE_EPOCHS = 120
STEPWISE_LR = 5e-4
STEPWISE_BATCH_SIZE = 4
STEPWISE_HIDDEN_DIM = 128
STEPWISE_SEED = 42
STEPWISE_STEPS = [1, 2, 3]
STEPWISE_CKPT_DIR = ROOT / "results" / "stepwise_ckpts"
STEPWISE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

STEPWISE_TRAIN_SUMMARY = []

for step_idx in STEPWISE_STEPS:
    ckpt_path = STEPWISE_CKPT_DIR / f"symm_mgn_step{step_idx}.pt"
    train_log = f"11_step{step_idx}_train.log"

    if ckpt_path.exists():
        ckpt_backup_dir = ROOT / "results" / "backups" / "ckpt"
        ckpt_backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = ckpt_backup_dir / f"{ckpt_path.stem}_{ts}{ckpt_path.suffix}"
        shutil.copy2(ckpt_path, dst)
        print(f"[backup] step {step_idx} ckpt -> {dst}")

    train_cmd = " ".join(
        [
            "python -m phase1_static.train",
            "--input-format doe",
            "--data-dir doe_data",
            "--case-indices " + " ".join(str(i) for i in STEPWISE_CASE_INDICES),
            "--source-file-types " + " ".join(STEPWISE_SOURCE_FILE_TYPES),
            f"--max-steps-per-case {STEPWISE_MAX_STEPS_PER_CASE}",
            f"--step-index {step_idx}",
            f"--epochs {STEPWISE_EPOCHS}",
            f"--batch-size {STEPWISE_BATCH_SIZE}",
            f"--hidden-dim {STEPWISE_HIDDEN_DIM}",
            f"--seed {STEPWISE_SEED}",
            f"--lr {STEPWISE_LR}",
            "--lr-decay-factor 0.5",
            "--lr-decay-patience 12",
            "--lr-decay-min-lr 1e-6",
            f"--ckpt-out results/stepwise_ckpts/{ckpt_path.name}",
        ]
    )

    cp_step_train = run_docker(train_cmd, check=False)
    save_log(train_log, cp_step_train)

    combined = (cp_step_train.stdout or "") + "\n" + (cp_step_train.stderr or "")
    epoch_lines = [line for line in combined.splitlines() if "epoch=" in line]

    item = {
        "step_idx": step_idx,
        "returncode": cp_step_train.returncode,
        "epochs": STEPWISE_EPOCHS,
        "lr": STEPWISE_LR,
        "last_epoch": epoch_lines[-1] if epoch_lines else "",
        "ckpt_saved": ckpt_path.exists(),
        "ckpt_size_kb": round(ckpt_path.stat().st_size / 1024, 1) if ckpt_path.exists() else 0,
        "log_file": str(LOG_DIR / train_log),
    }
    STEPWISE_TRAIN_SUMMARY.append(item)
    print(json.dumps(item, ensure_ascii=False))

    if cp_step_train.returncode != 0:
        print("=== train stderr tail ===")
        print("\n".join(combined.splitlines()[-60:]))
        raise RuntimeError(f"Step {step_idx} 학습 실패. {train_log} 를 확인하세요.")

print("=== Stepwise train summary ===")
print(json.dumps(STEPWISE_TRAIN_SUMMARY, indent=2, ensure_ascii=False))
'@

Set-CellSource "#VSC-9fde9543" @'
import json
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np

ROOT = globals().get("ROOT", Path.cwd())
CONTAINER_NAME = globals().get("CONTAINER_NAME", "motor_compare")
LOG_DIR = globals().get("LOG_DIR", ROOT / "logs" / "tutorial")
LOG_DIR.mkdir(parents=True, exist_ok=True)

STEPWISE_CASE_INDICES = globals().get("STEPWISE_CASE_INDICES", list(range(40)))
STEPWISE_SOURCE_FILE_TYPES = globals().get("STEPWISE_SOURCE_FILE_TYPES", ["OnLoadTorque"])
STEPWISE_STEPS_AVAILABLE = globals().get("STEPWISE_STEPS", [1, 2, 3])
INFER_STEP_PLAN = globals().get("INFER_STEP_PLAN", [1])

for step_idx in INFER_STEP_PLAN:
    if step_idx not in STEPWISE_STEPS_AVAILABLE:
        raise ValueError(f"INFER_STEP_PLAN contains unsupported step: {step_idx}")

STEPWISE_CKPT_DIR = globals().get("STEPWISE_CKPT_DIR", ROOT / "results" / "stepwise_ckpts")
STEPWISE_CKPT_MAP = {
    int(step_idx): STEPWISE_CKPT_DIR / f"symm_mgn_step{int(step_idx)}.pt"
    for step_idx in INFER_STEP_PLAN
}
missing_ckpt = [str(path) for path in STEPWISE_CKPT_MAP.values() if not path.exists()]
if missing_ckpt:
    raise FileNotFoundError(
        "Stepwise ckpt가 없습니다. 먼저 11) 셀을 실행하세요:\n" + "\n".join(missing_ckpt)
    )

STEPWISE_INFER_DIR = ROOT / "results" / "stepwise_infer"
STEPWISE_INFER_DIR.mkdir(parents=True, exist_ok=True)
STEPWISE_INFER_LOG = "12_stepwise_infer.log"

existing_npz = list(STEPWISE_INFER_DIR.glob("*.npz"))
if existing_npz:
    backup_dir = ROOT / "results" / "backups" / f"stepwise_infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for src in existing_npz:
        shutil.move(str(src), str(backup_dir / src.name))
    print(f"[backup] moved {len(existing_npz)} npz files -> {backup_dir}")

STEPWISE_INFER_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch

    os.chdir("/workspace/app")
    sys.path.insert(0, "/workspace/app")

    from infer_phase1_pbc import (
        CHANNEL_ORDER,
        compute_per_channel_metrics,
        load_symm_mgn,
        run_inference,
        save_infer_npz,
    )
    from phase1_static.motor_dataset import build_samples_from_doe_manifest

    CASE_INDICES = __CASE_INDICES__
    STEP_PLAN = __STEP_PLAN__
    SOURCE_FILE_TYPES = __SOURCE_FILE_TYPES__
    CKPT_MAP = {int(k): Path(v) for k, v in __CKPT_MAP__.items()}
    OUT_DIR = Path("/workspace/app/results/stepwise_infer")

    script_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    failed = []
    max_step = max(STEP_PLAN) if STEP_PLAN else 1

    for step_idx in STEP_PLAN:
        model, ckpt = load_symm_mgn(CKPT_MAP[step_idx], device)
        channel_order = ckpt.get("channel_order", CHANNEL_ORDER)
        channel_keys = [name.lower() for name in channel_order]

        for loop_idx, case_idx in enumerate(CASE_INDICES, start=1):
            samples = build_samples_from_doe_manifest(
                "/workspace/app/doe_data",
                max_steps_per_case=max_step,
                case_indices=[case_idx],
                source_file_types=SOURCE_FILE_TYPES,
            )
            matches = [
                sample
                for sample in samples
                if int(sample.get("step_index", -1)) == int(step_idx)
            ]
            if not matches:
                failed.append(
                    {
                        "case_idx": case_idx,
                        "step_idx": step_idx,
                        "reason": "missing_sample",
                    }
                )
                continue

            actual_sample = matches[0]
            t0 = time.time()
            result = run_inference(model, [actual_sample], device, batch_size=1)
            elapsed = time.time() - t0
            metrics = compute_per_channel_metrics(result)

            out_path = OUT_DIR / f"stepwise_case{case_idx:04d}_step{step_idx}.npz"
            meta = {
                "model_name": "SymMGN",
                "ckpt": str(CKPT_MAP[step_idx]),
                "case_idx": case_idx,
                "step_idx": int(step_idx),
                "n_samples": 1,
                "elapsed_s": round(elapsed, 3),
                "elapsed_ms": round(elapsed * 1000.0, 1),
                "channel_order": channel_order,
                "pbc_enabled": True,
                "pbc_rotation_deg": -45.0,
                "source_file_types": SOURCE_FILE_TYPES,
                "source_file_name": actual_sample.get("source_file_name", ""),
                "step_semantics": actual_sample.get("step_semantics", ""),
                "time_s": float(actual_sample.get("time_s", 0.0)),
                "rotate_step": float(actual_sample.get("rotate_step", 0.0)),
            }
            save_infer_npz(out_path, result, metrics, meta)

            summary = {
                "case_idx": case_idx,
                "step_idx": int(step_idx),
                "elapsed_ms": round(elapsed * 1000.0, 1),
                "size_kb": round(out_path.stat().st_size / 1024.0, 1),
            }
            for channel_name in channel_keys:
                summary[f"rmse_{channel_name}"] = round(
                    float(metrics[f"rmse_{channel_name}"]),
                    6,
                )
            summaries.append(summary)

            if loop_idx % 10 == 0 or loop_idx == len(CASE_INDICES):
                print(
                    json.dumps(
                        {
                            "step": int(step_idx),
                            "progress": f"{loop_idx}/{len(CASE_INDICES)}",
                        },
                        ensure_ascii=False,
                    )
                )

    aggregate_rmse = {}
    for step_idx in STEP_PLAN:
        step_rows = [row for row in summaries if int(row.get("step_idx", -1)) == int(step_idx)]
        if not step_rows:
            continue
        channel_names = [key.replace("rmse_", "") for key in step_rows[0].keys() if key.startswith("rmse_")]
        aggregate_rmse[str(step_idx)] = {}
        for channel_name in channel_names:
            values = [float(row[f"rmse_{channel_name}"]) for row in step_rows]
            values_arr = np.asarray(values, dtype=float)
            aggregate_rmse[str(step_idx)][channel_name] = {
                "mean": round(float(values_arr.mean()), 6),
                "std": round(float(values_arr.std()), 6),
                "max": round(float(values_arr.max()), 6),
            }

    peak_gpu_mb = None
    if torch.cuda.is_available():
        peak_gpu_mb = round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 1)

    avg_elapsed_ms = None
    if summaries:
        avg_elapsed_ms = round(
            float(np.mean([row["elapsed_ms"] for row in summaries])),
            1,
        )

    print(
        json.dumps(
            {
                "requested_cases": len(CASE_INDICES),
                "requested_steps": STEP_PLAN,
                "saved_count": len(summaries),
                "failed": failed,
                "wall_time_s": round(time.time() - script_t0, 3),
                "avg_elapsed_ms": avg_elapsed_ms,
                "peak_gpu_mb": peak_gpu_mb,
                "aggregate_rmse": aggregate_rmse,
            },
            ensure_ascii=False,
        )
    )
    """
).replace("__CASE_INDICES__", json.dumps(STEPWISE_CASE_INDICES)).replace(
    "__STEP_PLAN__", json.dumps(INFER_STEP_PLAN)
).replace(
    "__SOURCE_FILE_TYPES__", json.dumps(STEPWISE_SOURCE_FILE_TYPES)
).replace(
    "__CKPT_MAP__",
    json.dumps(
        {
            str(k): f"/workspace/app/results/stepwise_ckpts/{v.name}"
            for k, v in STEPWISE_CKPT_MAP.items()
        }
    ),
)

cp_stepwise_infer = subprocess.run(
    ["docker", "exec", CONTAINER_NAME, "python", "-c", STEPWISE_INFER_SCRIPT],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
)

save_log_fn = globals().get("save_log")
if callable(save_log_fn):
    save_log_fn(STEPWISE_INFER_LOG, cp_stepwise_infer)
else:
    (LOG_DIR / STEPWISE_INFER_LOG).write_text(
        (cp_stepwise_infer.stdout or "") + "\n" + (cp_stepwise_infer.stderr or ""),
        encoding="utf-8",
    )

stdout_lines = [
    line.strip()
    for line in (cp_stepwise_infer.stdout or "").splitlines()
    if line.strip()
]
parsed_lines = []
for line in stdout_lines:
    try:
        parsed_lines.append(json.loads(line))
    except json.JSONDecodeError:
        print(line)

summary = next((item for item in reversed(parsed_lines) if "saved_count" in item), {})
progress_updates = [
    f"step {item['step']}: {item['progress']}"
    for item in parsed_lines
    if "progress" in item
]

print("=== Stepwise Inference Summary ===")
print(
    json.dumps(
        {
            "returncode": cp_stepwise_infer.returncode,
            "infer_step_plan": INFER_STEP_PLAN,
            "progress_updates": progress_updates,
            **summary,
            "log_file": str(LOG_DIR / STEPWISE_INFER_LOG),
        },
        indent=2,
        ensure_ascii=False,
    )
)

expected_count = len(STEPWISE_CASE_INDICES) * len(INFER_STEP_PLAN)
if cp_stepwise_infer.returncode != 0:
    print("=== infer stderr tail ===")
    print("\n".join((cp_stepwise_infer.stderr or "").splitlines()[-60:]))
    raise RuntimeError("Stepwise 추론 실패. 12_stepwise_infer.log를 확인하세요.")

if summary.get("saved_count", 0) != expected_count:
    raise RuntimeError(
        f"Stepwise NPZ 저장 수가 예상과 다릅니다: "
        f"{summary.get('saved_count', 0)} / {expected_count}"
    )

if summary.get("failed"):
    raise RuntimeError(
        "일부 case/step 저장이 실패했습니다. "
        "12_stepwise_infer.log를 확인하세요."
    )
'@

Set-CellSource "#VSC-3e2d260d" @'
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = globals().get("ROOT", Path.cwd())
STEPWISE_INFER_DIR = globals().get("STEPWISE_INFER_DIR", ROOT / "results" / "stepwise_infer")
VIS_CASE_INDICES = globals().get("VIS_CASE_INDICES", [0, 1, 2, 3])
VIS_STEP_INDICES = globals().get("INFER_STEP_PLAN", [1])

channels = ["bx", "by", "a", "je"]
labels = ["Bx", "By", "A", "Je"]

for case_idx in VIS_CASE_INDICES:
    for step_idx in VIS_STEP_INDICES:
        npz_path = STEPWISE_INFER_DIR / f"stepwise_case{case_idx:04d}_step{step_idx}.npz"
        if not npz_path.exists():
            print(f"case {case_idx} step {step_idx}: NPZ 없음, skip")
            continue

        arr = np.load(npz_path, allow_pickle=True)
        meta = json.loads(arr["meta"].item()) if "meta" in arr.files else {}
        actual_step_idx = int(meta.get("step_idx", step_idx))
        pos_x = arr["pos_x"]
        pos_y = arr["pos_y"]

        fig, axes = plt.subplots(2, len(channels), figsize=(22, 7))
        fig.suptitle(
            f"stepwise case {case_idx:04d} - step {actual_step_idx} GT (top) vs Pred (bottom)",
            fontsize=13,
        )

        for col_idx, (ch, label) in enumerate(zip(channels, labels)):
            gt = arr[f"gt_{ch}"]
            pred = arr[f"pred_{ch}"]
            vmin = float(min(gt.min(), pred.min()))
            vmax = float(max(gt.max(), pred.max()))

            sc_gt = axes[0, col_idx].scatter(
                pos_x,
                pos_y,
                c=gt,
                cmap="RdBu_r",
                s=6,
                vmin=vmin,
                vmax=vmax,
            )
            axes[0, col_idx].set_title(f"GT {label}")
            axes[0, col_idx].set_aspect("equal")
            axes[0, col_idx].set_xticks([])
            axes[0, col_idx].set_yticks([])
            plt.colorbar(sc_gt, ax=axes[0, col_idx], fraction=0.04)

            sc_pred = axes[1, col_idx].scatter(
                pos_x,
                pos_y,
                c=pred,
                cmap="RdBu_r",
                s=6,
                vmin=vmin,
                vmax=vmax,
            )
            axes[1, col_idx].set_title(f"Pred {label}")
            axes[1, col_idx].set_aspect("equal")
            axes[1, col_idx].set_xticks([])
            axes[1, col_idx].set_yticks([])
            plt.colorbar(sc_pred, ax=axes[1, col_idx], fraction=0.04)

        plt.tight_layout()
        save_path = ROOT / "logs" / f"stepwise_case{case_idx:04d}_step{actual_step_idx}_gt_vs_pred.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.show()
        plt.close()
        print(
            json.dumps(
                {
                    "case_idx": case_idx,
                    "step_idx": actual_step_idx,
                    "saved_png": str(save_path),
                },
                ensure_ascii=False,
            )
        )
'@

Set-CellSource "#VSC-469aa40c" @'
import numpy as np
import matplotlib.pyplot as plt

STEPWISE_INFER_DIR = globals().get("STEPWISE_INFER_DIR", ROOT / "results" / "stepwise_infer")
VIS_CASE_INDICES = [0, 10, 20, 30]
ERROR_MAP_STEP = int(globals().get("ERROR_MAP_STEP", globals().get("INFER_STEP_PLAN", [1])[0]))
channels = ["bx", "by", "a", "je"]
labels = ["Bx", "By", "A", "Je"]

fig, axes = plt.subplots(
    len(VIS_CASE_INDICES),
    len(channels),
    figsize=(20, 4 * len(VIS_CASE_INDICES)),
)
axes = np.atleast_2d(axes)
fig.suptitle(f"Stepwise SymMGN - step {ERROR_MAP_STEP} - |GT - Pred| error map", fontsize=14)

for row_idx, case_idx in enumerate(VIS_CASE_INDICES):
    npz_path = STEPWISE_INFER_DIR / f"stepwise_case{case_idx:04d}_step{ERROR_MAP_STEP}.npz"
    if not npz_path.exists():
        for col_idx in range(len(channels)):
            axes[row_idx, col_idx].text(
                0.5,
                0.5,
                f"case {case_idx}\nNPZ 없음",
                ha="center",
                va="center",
            )
            axes[row_idx, col_idx].set_axis_off()
        continue

    arr = np.load(npz_path, allow_pickle=True)
    pos_x, pos_y = arr["pos_x"], arr["pos_y"]

    for col_idx, (ch, label) in enumerate(zip(channels, labels)):
        gt = arr[f"gt_{ch}"]
        pred = arr[f"pred_{ch}"]
        error = np.abs(gt - pred)
        vmax_err = float(np.percentile(error, 95))

        sc = axes[row_idx, col_idx].scatter(
            pos_x,
            pos_y,
            c=error,
            cmap="hot_r",
            s=6,
            vmin=0,
            vmax=vmax_err,
        )
        axes[row_idx, col_idx].set_aspect("equal")
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        if row_idx == 0:
            axes[row_idx, col_idx].set_title(f"|GT-Pred| {label}")
        if col_idx == 0:
            axes[row_idx, col_idx].set_ylabel(f"case {case_idx:04d}")
        plt.colorbar(sc, ax=axes[row_idx, col_idx], fraction=0.04)

plt.tight_layout()
vis_path = ROOT / "logs" / f"stepwise_gt_vs_pred_error_step{ERROR_MAP_STEP}.png"
vis_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(vis_path, dpi=110, bbox_inches="tight")
plt.show()
plt.close()
print(f"saved: {vis_path}")
'@

Set-CellSource "#VSC-b53b0959" @'
import numpy as np
import matplotlib.pyplot as plt

STEPWISE_INFER_DIR = globals().get("STEPWISE_INFER_DIR", ROOT / "results" / "stepwise_infer")
STEPWISE_VIS_CASE = int(globals().get("FULL_VIS_CASE", 0))
STEPWISE_VIS_STEP = int(globals().get("FULL_VIS_STEP", globals().get("INFER_STEP_PLAN", [1])[0]))

SECTOR_COUNT = 8
SECTOR_ANGLE_DEG = 45.0


def reconstruct_full_motor(pos_x, pos_y, fields_dict, n_sectors=8, angle_deg=45.0):
    all_x, all_y = [], []
    all_fields = {key: [] for key in fields_dict}

    for sector_idx in range(n_sectors):
        rad = np.radians(sector_idx * angle_deg)
        cos_k = np.cos(rad)
        sin_k = np.sin(rad)
        sign = (-1.0) ** sector_idx

        rot_x = pos_x * cos_k - pos_y * sin_k
        rot_y = pos_x * sin_k + pos_y * cos_k
        all_x.append(rot_x)
        all_y.append(rot_y)

        bx_orig = fields_dict["bx"]
        by_orig = fields_dict["by"]
        all_fields["bx"].append(sign * (bx_orig * cos_k - by_orig * sin_k))
        all_fields["by"].append(sign * (bx_orig * sin_k + by_orig * cos_k))
        all_fields["a"].append(sign * fields_dict["a"])
        all_fields["je"].append(sign * fields_dict["je"])

    full_x = np.concatenate(all_x)
    full_y = np.concatenate(all_y)
    full_fields = {key: np.concatenate(values) for key, values in all_fields.items()}
    return full_x, full_y, full_fields


npz_path = STEPWISE_INFER_DIR / f"stepwise_case{STEPWISE_VIS_CASE:04d}_step{STEPWISE_VIS_STEP}.npz"
if not npz_path.exists():
    raise FileNotFoundError(f"NPZ not found: {npz_path}")

arr = np.load(npz_path, allow_pickle=True)
sector_x = arr["pos_x"]
sector_y = arr["pos_y"]

channels = ["bx", "by", "a", "je"]
labels = ["Bx", "By", "A", "Je"]
gt_fields = {channel: arr[f"gt_{channel}"] for channel in channels}
pred_fields = {channel: arr[f"pred_{channel}"] for channel in channels}

gt_full_x, gt_full_y, gt_full = reconstruct_full_motor(sector_x, sector_y, gt_fields)
pred_full_x, pred_full_y, pred_full = reconstruct_full_motor(sector_x, sector_y, pred_fields)

fig, axes = plt.subplots(2, len(channels), figsize=(24, 10))
fig.suptitle(
    f"Full Motor (8 sectors) - stepwise case {STEPWISE_VIS_CASE:04d} step {STEPWISE_VIS_STEP} GT (top) vs Pred (bottom)",
    fontsize=14,
)

for col_idx, (channel, label) in enumerate(zip(channels, labels)):
    gt_vals = gt_full[channel]
    pred_vals = pred_full[channel]
    vmin = float(min(gt_vals.min(), pred_vals.min()))
    vmax = float(max(gt_vals.max(), pred_vals.max()))

    sc_gt = axes[0, col_idx].scatter(
        gt_full_x,
        gt_full_y,
        c=gt_vals,
        cmap="RdBu_r",
        s=1.5,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, col_idx].set_title(f"GT {label}")
    axes[0, col_idx].set_aspect("equal")
    axes[0, col_idx].set_xticks([])
    axes[0, col_idx].set_yticks([])
    plt.colorbar(sc_gt, ax=axes[0, col_idx], fraction=0.046)

    sc_pred = axes[1, col_idx].scatter(
        pred_full_x,
        pred_full_y,
        c=pred_vals,
        cmap="RdBu_r",
        s=1.5,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, col_idx].set_title(f"Pred {label}")
    axes[1, col_idx].set_aspect("equal")
    axes[1, col_idx].set_xticks([])
    axes[1, col_idx].set_yticks([])
    plt.colorbar(sc_pred, ax=axes[1, col_idx], fraction=0.046)

plt.tight_layout()
save_path = ROOT / "logs" / f"full_motor_stepwise_case{STEPWISE_VIS_CASE:04d}_step{STEPWISE_VIS_STEP}_gt_vs_pred.png"
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=130, bbox_inches="tight")
plt.show()
plt.close()
print(f"saved: {save_path}")
print(f"total nodes: {len(gt_full_x)} ({len(sector_x)} x {SECTOR_COUNT} sectors)")
'@

$json = $nb | ConvertTo-Json -Depth 100
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($path, $json, $utf8NoBom)
Write-Host "UPDATED_NOTEBOOK_JSON"
