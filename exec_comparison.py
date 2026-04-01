import subprocess

r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'python', '-u',
     '/workspace/host_data/run_comparison.py'],
    capture_output=True, text=True, timeout=120,
    encoding='utf-8', errors='replace'
)
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr[-500:])
