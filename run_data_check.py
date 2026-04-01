import subprocess
r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'python -u /workspace/host_data/check_data_quality.py > /workspace/host_data/data_quality.log 2>&1; echo EXIT_CODE=$?'],
    capture_output=True, text=True, timeout=180
)
print("Done:", r.stdout.strip())
