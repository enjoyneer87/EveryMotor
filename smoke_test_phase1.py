"""Phase 1 Smoke Test — 독립 실행 스크립트

phase1_tutorial.ipynb 의 검증 게이트(5절)와 3-case DOE smoke test(9-10절)를
노트북 없이 단독으로 실행합니다.

실행 방법:
    python smoke_test_phase1.py

전제 조건:
    - `motor_compare` Docker 컨테이너가 실행 중이어야 합니다.
    - 호스트 Python: PyMotorEnv_310 venv (pytest, h5py, numpy, matplotlib)
    - Docker 내부: PhysicsNeMo + phase1_static 패키지

출력:
    - logs/ 디렉토리에 각 단계별 로그 파일
    - results/ 에 smoke test 체크포인트 + NPZ
    - logs/ 에 GT vs Pred 시각화 PNG
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
CONTAINER_NAME = "motor_compare"
LOG_DIR = ROOT / "logs" / "smoke_test"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PY_EXEC_HOST = r"c:/Users/moa/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe"

SMOKE_CASE_INDICES = [0, 1, 2]
SMOKE_SOURCE_FILE_TYPES = ["OnLoadTorque"]
SMOKE_MAX_STEPS_PER_CASE = 2
SMOKE_EPOCHS = 2
SMOKE_BATCH_SIZE = 2
SMOKE_HIDDEN_DIM = 64
SMOKE_SEED = 42
SMOKE_CKPT_NAME = "symm_mgn_smoke_test.pt"
SMOKE_CKPT_PATH = ROOT / "results" / SMOKE_CKPT_NAME
SMOKE_INFER_CASE_IDX = 0
SMOKE_INFER_NPZ_PATH = ROOT / "results" / f"smoke_case{SMOKE_INFER_CASE_IDX:04d}_step1.npz"
SMOKE_VIS_PNG_PATH = ROOT / "logs" / f"smoke_case{SMOKE_INFER_CASE_IDX:04d}_gt_vs_pred.png"


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def run_local(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[local] {cmd}")
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, check=check)


def run_docker(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    payload = f"cd /workspace/app && {cmd}"
    payload = payload.replace("\\", "\\\\").replace('"', '\\"')
    full_cmd = f'docker exec {CONTAINER_NAME} bash -lc "{payload}"'
    print(f"[docker] {cmd}")
    return run_local(full_cmd, check=check)


def save_log(name: str, cp: subprocess.CompletedProcess) -> Path:
    path = LOG_DIR / name
    text = [f"$ returncode={cp.returncode}\n"]
    if cp.stdout:
        text += ["\n[stdout]\n", cp.stdout]
    if cp.stderr:
        text += ["\n[stderr]\n", cp.stderr]
    path.write_text("".join(text), encoding="utf-8")
    print(f"  → log: {path}")
    return path


def _assert_ok(label: str, cp: subprocess.CompletedProcess) -> None:
    if cp.returncode != 0:
        print(f"\n[FAIL] {label} (exit {cp.returncode})")
        print(cp.stdout[-800:] if cp.stdout else "")
        print(cp.stderr[-800:] if cp.stderr else "")
        sys.exit(1)
    print(f"[OK]   {label}")


# ---------------------------------------------------------------------------
# Gate 1 — Contract 경계 테스트 (Docker pytest)
# ---------------------------------------------------------------------------

def gate_contract_tests() -> None:
    print("\n=== Gate 1: Contract 경계 테스트 (Docker) ===")
    cp = run_docker(
        "PYTHONPATH=/workspace/app pytest -q tests/test_phase1_contract_boundaries.py",
        check=False,
    )
    save_log("01_contract_tests.log", cp)
    print(cp.stdout)
    _assert_ok("contract boundary tests", cp)


# ---------------------------------------------------------------------------
# Gate 2 — PBC 호스트 테스트 (host Python, torch 불필요)
# ---------------------------------------------------------------------------

def gate_pbc_host_tests() -> None:
    print("\n=== Gate 2: PBC 호스트 테스트 ===")
    cp = subprocess.run(
        [
            PY_EXEC_HOST, "-m", "pytest", "-q",
            "tests/test_phase1_pbc_boundary.py",
            "tests/test_phase1_pbc_contracts.py",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    save_log("02_pbc_host_tests.log", cp)
    print(cp.stdout)
    if cp.returncode != 0:
        print("[stderr]", cp.stderr[:600])
    _assert_ok("pbc host tests", cp)


# ---------------------------------------------------------------------------
# Gate 3 — Overfit-Single 게이트 (Docker)
# ---------------------------------------------------------------------------

OVERFIT_SCRIPT = textwrap.dedent("""
import sys, json, tempfile, subprocess
import numpy as np
from pathlib import Path

try:
    angles = np.deg2rad(np.array([0.0, 22.5, 45.0], dtype=np.float64))
    inner = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    pos = inner[None]
    node_type = np.ones((1, 3, 1), dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]], dtype=np.int64)
    ie = edges.T[None]
    pe = np.zeros((1, 2, 0), dtype=np.int64)
    pa = np.zeros((1, 0, 1), dtype=np.float32)
    y = np.random.RandomState(42).randn(1, 3, 5).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "overfit.npz"
        np.savez(
            npz_path,
            pos=pos,
            node_type_onehot=node_type,
            interior_edge_index=ie,
            pbc_edge_index=pe,
            pbc_edge_attr=pa,
            y=y,
        )

        r = subprocess.run(
            [sys.executable, "-m", "phase1_static.train",
             "--input-format", "npz", "--data", str(npz_path),
             "--overfit-single", "--epochs", "150",
             "--lr", "1e-2", "--hidden-dim", "32",
             "--seed", "42", "--overfit-target", "1e-2"],
            capture_output=True, text=True, cwd="/workspace/app",
        )
        combined = r.stdout + r.stderr
        epoch_lines = [l for l in combined.splitlines() if "epoch=" in l]
        overfit_lines = [l for l in combined.splitlines() if "Overfit" in l or "overfit" in l]
        print(json.dumps({
            "returncode": r.returncode,
            "last_epoch": epoch_lines[-1] if epoch_lines else "",
            "overfit_gate": overfit_lines[-1] if overfit_lines else "",
        }))
except Exception as exc:
    import traceback
    print(json.dumps({"error": str(exc), "tb": traceback.format_exc()[-400:]}))
""")


def gate_overfit_single() -> None:
    print("\n=== Gate 3: Overfit-Single 게이트 (Docker) ===")
    cp = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "python", "-c", OVERFIT_SCRIPT],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    save_log("03_overfit_single.log", cp)
    for line in cp.stdout.splitlines():
        try:
            d = json.loads(line)
            print(json.dumps(d, indent=2, ensure_ascii=False))
            if d.get("returncode", 1) != 0 or d.get("error"):
                print("[FAIL] Overfit gate failed")
                sys.exit(1)
        except Exception:
            print(line)
    _assert_ok("overfit-single gate", cp)


# ---------------------------------------------------------------------------
# Gate 4 — PBC bundle 전체 테스트 (host Python)
# ---------------------------------------------------------------------------

def gate_pbc_bundle_tests() -> None:
    print("\n=== Gate 4: PBC Bundle 전체 테스트 ===")
    cp = subprocess.run(
        [
            PY_EXEC_HOST, "-m", "pytest", "-q",
            "tests/test_phase1_pbc_bundle.py",
            "tests/test_phase1_pbc_boundary.py",
            "tests/test_phase1_pbc_contracts.py",
            "tests/test_phase1_pbc_pairing.py",
            "tests/test_phase1_pbc_prior.py",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    save_log("04_pbc_bundle_tests.log", cp)
    print(cp.stdout)
    if cp.returncode != 0:
        print("[stderr]", cp.stderr[:600])
    _assert_ok("pbc bundle tests", cp)


# ---------------------------------------------------------------------------
# Gate 5 — 3-case DOE Smoke 학습 (Docker)
# ---------------------------------------------------------------------------

def gate_smoke_train() -> None:
    print("\n=== Gate 5: 3-case DOE Smoke 학습 (Docker) ===")
    train_cmd = " ".join([
        "python -m phase1_static.train",
        "--input-format doe",
        "--data-dir doe_data",
        "--case-indices " + " ".join(str(i) for i in SMOKE_CASE_INDICES),
        "--source-file-types " + " ".join(SMOKE_SOURCE_FILE_TYPES),
        f"--max-steps-per-case {SMOKE_MAX_STEPS_PER_CASE}",
        f"--epochs {SMOKE_EPOCHS}",
        f"--batch-size {SMOKE_BATCH_SIZE}",
        f"--hidden-dim {SMOKE_HIDDEN_DIM}",
        f"--seed {SMOKE_SEED}",
        f"--ckpt-out results/{SMOKE_CKPT_NAME}",
    ])
    cp = run_docker(train_cmd, check=False)
    save_log("05_smoke_train.log", cp)

    combined = (cp.stdout or "") + "\n" + (cp.stderr or "")
    epoch_lines = [l for l in combined.splitlines() if "epoch=" in l]
    print(json.dumps({
        "returncode": cp.returncode,
        "last_epoch": epoch_lines[-1] if epoch_lines else "",
        "ckpt_saved": SMOKE_CKPT_PATH.exists(),
        "ckpt_size_kb": round(SMOKE_CKPT_PATH.stat().st_size / 1024, 1) if SMOKE_CKPT_PATH.exists() else 0,
    }, indent=2, ensure_ascii=False))
    _assert_ok("smoke train", cp)


# ---------------------------------------------------------------------------
# Gate 6 — Smoke 추론 (Docker)
# ---------------------------------------------------------------------------

def gate_smoke_infer() -> None:
    print("\n=== Gate 6: Smoke 추론 (Docker) ===")
    infer_cmd = " ".join([
        "python infer_phase1_pbc.py",
        f"--ckpt /workspace/app/results/{SMOKE_CKPT_NAME}",
        "--data-dir /workspace/app/doe_data",
        f"--case-idx {SMOKE_INFER_CASE_IDX}",
        "--source-file-types " + " ".join(SMOKE_SOURCE_FILE_TYPES),
        "--max-steps 1",
        "--batch-size 1",
        f"--out /workspace/app/results/{SMOKE_INFER_NPZ_PATH.name}",
    ])
    cp = run_docker(infer_cmd, check=False)
    save_log("06_smoke_infer.log", cp)

    combined = (cp.stdout or "") + "\n" + (cp.stderr or "")
    metric_lines = [l for l in combined.splitlines() if "RMSE=" in l]
    print(json.dumps({
        "returncode": cp.returncode,
        "npz_exists": SMOKE_INFER_NPZ_PATH.exists(),
        "metric_lines": metric_lines[-3:],
    }, indent=2, ensure_ascii=False))
    _assert_ok("smoke infer", cp)


# ---------------------------------------------------------------------------
# Gate 7 — GT vs Pred 시각화 (host matplotlib)
# ---------------------------------------------------------------------------

def gate_smoke_visualize() -> None:
    print("\n=== Gate 7: Smoke GT vs Pred 시각화 ===")
    import matplotlib.pyplot as plt
    import numpy as np

    if not SMOKE_INFER_NPZ_PATH.exists():
        print(f"[FAIL] NPZ 없음: {SMOKE_INFER_NPZ_PATH}")
        sys.exit(1)

    arr = np.load(SMOKE_INFER_NPZ_PATH, allow_pickle=True)
    pos_x = arr["pos_x"]
    pos_y = arr["pos_y"]
    channels = ["bx", "by", "a", "je"]
    labels = ["Bx", "By", "A", "Je"]

    fig, axes = plt.subplots(2, len(channels), figsize=(18, 7))
    fig.suptitle(
        f"Smoke case {SMOKE_INFER_CASE_IDX:04d} — GT (top) vs Pred (bottom)",
        fontsize=12,
    )
    for col_idx, (ch, lbl) in enumerate(zip(channels, labels)):
        gt = arr[f"gt_{ch}"]
        pred = arr[f"pred_{ch}"]
        vmin_v = float(min(gt.min(), pred.min()))
        vmax_v = float(max(gt.max(), pred.max()))
        for row, vals in enumerate([gt, pred]):
            sc = axes[row, col_idx].scatter(pos_x, pos_y, c=vals, cmap="RdBu_r",
                                            s=8, vmin=vmin_v, vmax=vmax_v)
            axes[row, col_idx].set_title(f"{'GT' if row == 0 else 'Pred'} {lbl}")
            axes[row, col_idx].set_aspect("equal")
            axes[row, col_idx].set_xticks([])
            axes[row, col_idx].set_yticks([])
            plt.colorbar(sc, ax=axes[row, col_idx], fraction=0.04)

    plt.tight_layout()
    SMOKE_VIS_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SMOKE_VIS_PNG_PATH, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[OK]   시각화 저장: {SMOKE_VIS_PNG_PATH}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 1 Smoke Test")
    print(f"  ROOT          : {ROOT}")
    print(f"  CONTAINER     : {CONTAINER_NAME}")
    print(f"  CASE_INDICES  : {SMOKE_CASE_INDICES}")
    print(f"  SOURCE_TYPES  : {SMOKE_SOURCE_FILE_TYPES}")
    print("=" * 60)

    gate_contract_tests()
    gate_pbc_host_tests()
    gate_overfit_single()
    gate_pbc_bundle_tests()
    gate_smoke_train()
    gate_smoke_infer()
    gate_smoke_visualize()

    print("\n" + "=" * 60)
    print("Phase 1 Smoke Test — ALL GATES PASSED")
    print(f"  로그 디렉토리 : {LOG_DIR}")
    print(f"  체크포인트   : {SMOKE_CKPT_PATH}")
    print(f"  시각화 PNG   : {SMOKE_VIS_PNG_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
