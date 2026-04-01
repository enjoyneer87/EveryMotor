import subprocess
r = subprocess.run(["docker", "exec", "motor_compare", "bash", "-c", 
    "tail -5 /workspace/host_data/train_gino.log; echo '==='; "
    "ls -la /workspace/host_data/doe_gino_ckpt.pt 2>/dev/null || echo 'no ckpt'; "
    "ps aux | grep train_doe | grep -v grep | wc -l"], 
    capture_output=True, text=True, timeout=15)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr[:200] if r.stderr else "")
