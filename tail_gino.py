import subprocess

r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'tail -10 /workspace/host_data/train_gino.log; echo "---"; ps aux | grep train_doe_gino | grep -v grep'],
    capture_output=True, text=True, timeout=30
)
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr)
