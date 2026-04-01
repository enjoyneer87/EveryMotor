import subprocess

r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'tail -10 /workspace/host_data/train_rnn.log; echo "---"; ps aux | grep train_doe_rnn | grep -v grep'],
    capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace'
)
print(r.stdout)
if r.stderr:
    print("ERR:", r.stderr[:300])
