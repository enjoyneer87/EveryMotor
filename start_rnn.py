import subprocess

r = subprocess.run(
    ['docker', 'exec', '-d', 'motor_compare', 'bash', '-c',
     'python -u /workspace/host_data/train_doe_rnn.py '
     '--data-dir /workspace/host_data/doe_data '
     '--grid-res 64 --seq-in 4 --seq-out 4 '
     '--epochs 60 --batch-size 4 --lr 1e-3 '
     '--hidden-channels 64 '
     '--ckpt /workspace/host_data/doe_rnn_ckpt.pt '
     '> /workspace/host_data/train_rnn.log 2>&1'],
    capture_output=True, text=True, timeout=30
)
print("Exit code:", r.returncode)
if r.stderr:
    print("ERR:", r.stderr[:300])
