import subprocess, sys

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return r.stdout.strip() + ("\nERR: " + r.stderr.strip() if r.stderr.strip() else "")

print("="*60)
print("MOTOR COMPARE STATUS CHECK")
print("="*60)

# 1. Container status
print("\n[1] Container Status:")
print(run('docker ps -a --filter "name=motor_compare" --format "ID={{.ID}} Status={{.Status}}"'))

# 2. Check if container is running
ps_out = run('docker ps --filter "name=motor_compare" -q')
if not ps_out:
    print("\nContainer is NOT running. Trying to start...")
    print(run('docker start motor_compare'))
    import time; time.sleep(3)

# 3. Check training processes
print("\n[2] Running Training Processes:")
print(run('docker exec motor_compare bash -c "ps aux | grep python | grep -v grep"'))

# 4. Check checkpoints
print("\n[3] Checkpoint Files:")
for model in ['fno', 'gino', 'rnn', 'meshgraphnet']:
    ckpt = f'/workspace/host_data/doe_{model}_ckpt.pt'
    result = run(f'docker exec motor_compare bash -c "ls -lh {ckpt} 2>/dev/null || echo NOT_FOUND"')
    print(f"  {model}: {result}")

# 5. Check GINO log
print("\n[4] GINO Training Log (last 10 lines):")
print(run('docker exec motor_compare bash -c "tail -10 /workspace/host_data/train_gino.log 2>/dev/null || echo NO_LOG"'))

# 6. Check RNN log
print("\n[5] RNN Training Log (last 5 lines):")
print(run('docker exec motor_compare bash -c "tail -5 /workspace/host_data/train_rnn.log 2>/dev/null || echo NO_LOG"'))

print("\n" + "="*60)
print("CHECK COMPLETE")
print("="*60)
