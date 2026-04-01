import subprocess

# Start Jupyter server inside Docker container
r = subprocess.run(
    ['docker', 'exec', '-d', 'motor_compare', 'bash', '-c',
     'jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root '
     '--NotebookApp.token=motor2026 --NotebookApp.notebook_dir=/workspace/host_data '
     '> /workspace/host_data/jupyter.log 2>&1'],
    capture_output=True, text=True, timeout=30
)
print("Start exit code:", r.returncode)

import time; time.sleep(5)

# Check if running
r2 = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'ps aux | grep jupyter | grep -v grep; echo "---"; tail -5 /workspace/host_data/jupyter.log'],
    capture_output=True, text=True, timeout=15, encoding='utf-8', errors='replace'
)
print(r2.stdout)
