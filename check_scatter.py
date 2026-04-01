import subprocess

r = subprocess.run(
    ['docker', 'exec', 'motor_compare', 'bash', '-c',
     'python -c "import torch_scatter; print(torch_scatter.__version__)" 2>&1; '
     'python -c "import torch_cluster; print(torch_cluster.__version__)" 2>&1; '
     'python -c "from torch_geometric.nn import radius; print(radius)" 2>&1'],
    capture_output=True, text=True, timeout=30
)
print(r.stdout)
