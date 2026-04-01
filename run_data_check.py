#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def run_docker(container: str, container_host_data: str, timeout: int) -> int:
    cmd = [
        "docker",
        "exec",
        container,
        "bash",
        "-c",
        (
            "python -u {p}/check_data_quality.py > {p}/data_quality.log 2>&1; "
            "echo EXIT_CODE=$?"
        ).format(p=container_host_data),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    print("Done:", r.stdout.strip())
    if r.stderr:
        print("stderr:", r.stderr.strip())
    return r.returncode


def run_local(timeout: int) -> int:
    cmd = [sys.executable, "-u", "check_data_quality.py"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    print(r.stdout.strip())
    if r.stderr:
        print("stderr:", r.stderr.strip())
    return r.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "docker", "local"], default="auto")
    parser.add_argument("--container", default="motor_compare")
    parser.add_argument("--container-host-data", default="/workspace/host_data")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    docker_available = shutil.which("docker") is not None
    mode = args.mode
    if mode == "auto":
        mode = "docker" if docker_available else "local"

    if mode == "docker" and not docker_available:
        print("docker is not available; falling back to local mode")
        mode = "local"

    if mode == "docker":
        return run_docker(args.container, args.container_host_data, args.timeout)
    return run_local(args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
