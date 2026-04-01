import subprocess

r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'python -u /workspace/host_data/run_comparison.py > /workspace/host_data/comparison_output.log 2>&1; echo EXIT=$?'],
    capture_output=True, text=True, timeout=120,
    encoding='utf-8', errors='replace'
)
print("Result:", r.stdout.strip())
