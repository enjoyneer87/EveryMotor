import subprocess, time

# Kill existing GINO
r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'pkill -f train_doe_gino; sleep 2; ps aux | grep train_doe_gino | grep -v grep | wc -l'],
    capture_output=True, text=True, timeout=30
)
print("Kill:", r.stdout.strip())

time.sleep(2)

# Restart
r2 = subprocess.run(
    ['docker', 'exec', '-d', 'motor_compare', 'bash', '-c',
     'python -u /workspace/host_data/train_doe_gino.py '
     '--data-dir /workspace/host_data/doe_data '
     '--latent-res 32 --epochs 60 --lr 1e-3 '
     '--fno-modes 16 --fno-layers 4 --fno-hidden 64 '
     '--gno-radius 0.1 '
     '--ckpt /workspace/host_data/doe_gino_ckpt.pt '
     '> /workspace/host_data/train_gino.log 2>&1'],
    capture_output=True, text=True, timeout=30
)
print("Restart exit:", r2.returncode)
