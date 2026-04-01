import subprocess
r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'grep -A 100 "DATA SUMMARY" /workspace/host_data/data_quality.log'],
    capture_output=True, text=True, timeout=30
)
print(r.stdout if r.stdout else "(no match)")
