import subprocess

def docker_exec(cmd):
    r = subprocess.run(['docker', 'exec', 'motor_compare', 'bash', '-c', cmd],
                       capture_output=True, text=True, timeout=30)
    return r.stdout.strip(), r.stderr.strip()

print("="*60)

# Checkpoints
print("[CHECKPOINTS]")
for m in ['fno', 'gino', 'rnn', 'meshgraphnet']:
    path = f'/workspace/host_data/doe_{m}_ckpt.pt'
    out, _ = docker_exec(f'ls -lh {path} 2>/dev/null || echo NOT_FOUND')
    print(f"  {m}: {out}")

# Processes
print("\n[PROCESSES]")
out, _ = docker_exec('ps aux | grep python | grep -v grep')
print(out if out else "  No python processes running")

# GINO log
print("\n[GINO LOG - last 15 lines]")
out, _ = docker_exec('tail -15 /workspace/host_data/train_gino.log 2>/dev/null')
print(out if out else "  No GINO log")

# RNN log
print("\n[RNN LOG - last 5 lines]")
out, _ = docker_exec('tail -5 /workspace/host_data/train_rnn.log 2>/dev/null')
print(out if out else "  No RNN log")

print("\n" + "="*60)
