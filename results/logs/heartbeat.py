#!/usr/bin/env python3
"""Heartbeat for the EveryMotor spectral sweep + R3.

Writes results/viz/train_status.json and appends results/viz/train_status_history.log
every INTERVAL seconds, so the sandboxed twice-daily brief (which can only see
results/viz) has real training state.

Current-run state is read from results/logs/hb_control.json, which the operator
overwrites as each run launches. Set "status_hint": "stop" there to end the loop.

Run:
  python results/logs/heartbeat.py --once      # single write, for testing
  python results/logs/heartbeat.py             # loop forever (INTERVAL, default 600s)
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

ROOT = "D:/KDH/NvidiaNemo"
VIZ = os.path.join(ROOT, "results", "viz")
CONTROL = os.path.join(ROOT, "results", "logs", "hb_control.json")
STATUS_JSON = os.path.join(VIZ, "train_status.json")
HISTORY_LOG = os.path.join(VIZ, "train_status_history.log")
INTERVAL = int(os.environ.get("INTERVAL", "600"))

EP_RE = re.compile(
    r"ep\s+(\d+)/(\d+).*?val B [\d.]+ nRMSE\s+([\d.]+)%.*?spec\s+([\d.eE+\-]+).*?\|\s*(\d+)s\s*$"
)
# Runner scripts end with a banner line whose last word is DONE ("SPECTRAL bw=30 DONE",
# "R3 DONE", "DOE240 DONE"). Match the shape, not each campaign's name -- the old
# hard-coded alternation silently reported a finished DOE run as still "running".
DONE_RE = re.compile(r"^.*\bDONE\s*$", re.M)
FAIL_RE = re.compile(r"CUDA out of memory|Traceback \(most recent call last\)|RuntimeError|Killed")


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_control():
    try:
        with open(CONTROL, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"experiment": "unknown", "band_w": None, "total_epochs": 100,
                "logfile": "", "status_hint": "running", "next_in_queue": "",
                "gpu_note_extra": "", "_control_error": str(e)}


ANSYS_RE = re.compile(r"ansys|mapdl|fluent|ansysedt|icepak|maxwell|cfx|mechanical",
                      re.IGNORECASE)


def gpu_note(extra):
    parts = []
    used_mib = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        util, used, total = [x.strip() for x in out.split(",")]
        used_mib = int(used)
        parts.append(f"util {util}% mem {used}/{total}MiB")
    except Exception as e:
        parts.append(f"smi-unavailable({e})")
    # Contention: on WDDM every desktop app shows as a compute-app, so counting is
    # meaningless. Flag only an ANSYS-family process name, or an unexpected memory jump.
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
        names = [r.strip() for r in apps.splitlines() if r.strip()]
        ansys = sorted({os.path.basename(n) for n in names if ANSYS_RE.search(n)})
        if ansys:
            parts.append("CONTENTION-ANSYS(" + ",".join(ansys) + ")")
    except Exception:
        pass
    if used_mib is not None and used_mib > 18000:
        parts.append(f"HIGH-MEM({used_mib}MiB — check for other GPU jobs)")
    if extra:
        parts.append(str(extra))
    return "; ".join(parts)


def parse_log(logfile):
    """Return dict with epoch, total, latest_val, best_val, sec_per_epoch, last_write, log_status."""
    res = {"epoch": 0, "total": None, "latest_val_nrmse": None,
           "best_val_nrmse": None, "sec_per_epoch": None,
           "last_log_write_iso": None, "log_status": None}
    if not logfile or not os.path.isfile(logfile):
        return res
    try:
        mt = os.path.getmtime(logfile)
        res["last_log_write_iso"] = datetime.fromtimestamp(mt, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass
    best = None
    last = None
    first_ep = None
    try:
        with open(logfile, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return res
    for line in text.splitlines():
        m = EP_RE.search(line)
        if not m:
            continue
        # group(5) is CUMULATIVE elapsed seconds (log prints every ~5 epochs), not per-epoch.
        ep, tot, val, cum = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(5))
        if first_ep is None:
            first_ep = ep
        last = (ep, tot, val, cum)
        if best is None or val < best:
            best = val
    if last:
        ep, tot, val, cum = last
        res["epoch"], res["total"], res["latest_val_nrmse"] = ep, tot, val
        # The elapsed seconds are counted from THIS process's start, so on a resumed
        # run (--resume, which begins at e.g. epoch 48) dividing by the absolute epoch
        # number understates the pace by ~50x and the ETA lands in the past. Divide by
        # the epochs this process has actually run.
        done_here = ep - (first_ep - 1) if first_ep else ep
        res["sec_per_epoch"] = round(cum / done_here, 1) if done_here > 0 else None
        res["best_val_nrmse"] = best
    if DONE_RE.search(text):
        res["log_status"] = "completed"
    elif FAIL_RE.search(text):
        res["log_status"] = "failed"
    elif last:
        res["log_status"] = "running"
    return res


def build_status(ctrl):
    p = parse_log(ctrl.get("logfile", ""))
    total = p["total"] or ctrl.get("total_epochs", 100)
    status = p["log_status"] or ctrl.get("status_hint", "running")
    eta = None
    if status == "running" and p["sec_per_epoch"] and p["epoch"] and total:
        remaining = max(0, total - p["epoch"]) * p["sec_per_epoch"]
        eta = (datetime.now(timezone.utc) + timedelta(seconds=remaining)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "experiment": ctrl.get("experiment"),
        "band_w": ctrl.get("band_w"),
        "epoch": p["epoch"],
        "total_epochs": total,
        "latest_val_nrmse": p["latest_val_nrmse"],
        "best_val_nrmse": p["best_val_nrmse"],
        "sec_per_epoch": p["sec_per_epoch"],
        "eta_iso": eta,
        "last_log_write_iso": p["last_log_write_iso"],
        "gpu_note": gpu_note(ctrl.get("gpu_note_extra", "")),
        "status": status,
        "next_in_queue": ctrl.get("next_in_queue"),
        "heartbeat_iso": now_iso(),
    }


def write_status(st):
    os.makedirs(VIZ, exist_ok=True)
    tmp = STATUS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATUS_JSON)
    line = (f'{st["heartbeat_iso"]} exp={st["experiment"]} bw={st["band_w"]} '
            f'ep={st["epoch"]}/{st["total_epochs"]} val={st["latest_val_nrmse"]} '
            f'best={st["best_val_nrmse"]} spe={st["sec_per_epoch"]}s '
            f'status={st["status"]} eta={st["eta_iso"]} next={st["next_in_queue"]} '
            f'[{st["gpu_note"]}]\n')
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def main():
    once = "--once" in sys.argv
    while True:
        ctrl = read_control()
        if ctrl.get("status_hint") == "stop":
            st = build_status(ctrl)
            st["status"] = "stopped"
            write_status(st)
            print("heartbeat: stop requested, exiting")
            return
        st = build_status(ctrl)
        write_status(st)
        print(f"heartbeat {st['heartbeat_iso']}: bw={st['band_w']} ep={st['epoch']}/{st['total_epochs']} "
              f"status={st['status']} best={st['best_val_nrmse']}")
        if once:
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
