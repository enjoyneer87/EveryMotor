import subprocess

r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'cat /workspace/host_data/comparison_output.log'],
    capture_output=True, text=True, timeout=30,
    encoding='utf-8', errors='replace'
)
print(r.stdout[-3000:] if r.stdout else "(empty)")
