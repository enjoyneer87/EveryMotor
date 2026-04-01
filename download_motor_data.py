#!/usr/bin/env python
"""Download the FSG motor dataset."""
import sys, os
os.chdir("/workspace/multiscale-pde-operators")
sys.path.insert(0, "/workspace/multiscale-pde-operators")

from multiscale_operator.data.download_data import download_motordata
print("Downloading motor data...", flush=True)
folder = download_motordata()
print(f"Done: {folder}", flush=True)

# Verify
from pathlib import Path
txt_files = list(Path(folder).rglob("*Bnorm_matinfo.txt"))
print(f"Found {len(txt_files)} Bnorm_matinfo.txt files", flush=True)
if txt_files:
    print(f"  Example: {txt_files[0]}", flush=True)
