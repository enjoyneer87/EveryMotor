import subprocess

r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'grep -n -E "Building|Total|Epoch|Done|ERROR|built|train_loss|val_loss" /workspace/host_data/train_gino.log; echo "---LINE_COUNT---"; wc -l /workspace/host_data/train_gino.log'],
    capture_output=True, text=True, timeout=30
)
print("STDOUT:")
print(r.stdout if r.stdout else "(empty)")
print("STDERR:")
print(r.stderr if r.stderr else "(empty)")
