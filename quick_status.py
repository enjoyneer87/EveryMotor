import subprocess, sys

cmds = [
    "tail -3 /workspace/host_data/train_gino.log",
    "ls -la /workspace/host_data/doe_gino_ckpt.pt 2>/dev/null || echo NO_CKPT",
    "ps aux | grep train_doe | grep -v grep | wc -l",
]

for cmd in cmds:
    r = subprocess.run(
        ["docker", "exec", "motor_compare", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=10
    )
    print(f"CMD: {cmd}")
    print(f"OUT: {r.stdout.strip()}")
    if r.stderr.strip():
        print(f"ERR: {r.stderr.strip()[:200]}")
    print()
