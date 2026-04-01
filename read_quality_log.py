import subprocess
r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'cat /workspace/host_data/data_quality.log'],
    capture_output=True, text=True, timeout=30
)
print(r.stdout[-4000:] if r.stdout else "(empty)")
if r.stderr:
    print("ERR:", r.stderr[:500])
