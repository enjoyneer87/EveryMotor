import subprocess

# Check if jupyter is installed in the container
r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'which jupyter 2>/dev/null || echo NOT_FOUND; pip list 2>/dev/null | grep -i jupyter | head -5'],
    capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace'
)
print("Jupyter check:", r.stdout.strip())
