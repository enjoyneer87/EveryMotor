# Moving this work to another machine

Code and results are in git. The DOE fields and the model checkpoints are not —
they are gitignored on purpose (348 MB + 28 MB each), and the repo's `.git` is
already 1.1 GB, so git-lfs would burn the GitHub free quota immediately.

**OneDrive is the transport** — the client is already running on the source
machine, so the copy is done and syncing. ~401 MB total. Google Drive and
HuggingFace instructions are kept at the end as alternatives.

Transfer as a **plain folder, not a zip**: `backup/doe_data` is 197 files
(156 H5 + 41 small), largest 8.7 MB, and the H5 datasets are already gzip-
compressed internally — zipping buys almost nothing and costs a pack/unpack
round trip. That file count is far below the level where Drive's per-file sync
overhead matters.

## What is where

| Piece | Size | In git? |
|---|---|---|
| Code, tests, `eval/`, `phase1_static/`, `tools/` | — | yes |
| `eval/splits/doe40_case_split.json` (the case holdout) | 1 KB | **yes** — the split is reproducible from git alone |
| `results/benchmark_v2_*.json` (all scorecards) | ~1 MB | yes |
| `results/viz/*.png` | 200 KB | yes |
| `backup/doe_data/` (40 DOE cases, H5) | **348 MB** | no — `.gitignore: doe_data/` |
| `results/mgn_nodeB_long.pt` (current best, 13.32% / 9.20%) | **28 MB** | no — `.gitignore: *.pt` |
| `results/mgn_nodeB_notime.pt` (no-time_s run, stopped at epoch 20) | **28 MB** | no |
| `results/viz/*.gif` | 3.2 MB | no — regenerate with `tools/make_field_gif.py` |
| Motor-CAD original solve (`D:\KDH\Sim_4SolverX\DOE4TrainingData`) | 141 MB/case | no — only needed to redo the Motor-CAD torque validation |

Older checkpoints (`doe_*.pt`, `mgn_curl_*.pt`, `mgn_nodeB_v2.pt`, …) are
deliberately **not** in the migration set: two of them take 11 node features
(they were fed A and J) and the harness refuses to score them, and the rest are
superseded and cheaper to retrain than to carry.

## Already staged on OneDrive

`C:\Users\moa\OneDrive\EveryMotor_migration\` on the source machine:

```
EveryMotor_migration/          199 files, 401.2 MB
├── doe_data/                  197 files, 347.2 MB  <- copy of backup/doe_data
└── checkpoints/
    ├── mgn_nodeB_long.pt      27.0 MB  current best: 13.32% |B| / 9.20% torque
    └── mgn_nodeB_notime.pt    27.0 MB  no-time_s run, stopped at epoch 20
```

Verified byte-for-byte after the copy: 197/197 files and 364,026,511 bytes on
both sides, and both checkpoints match their source length exactly.

It went into `C:\Users\moa\OneDrive`, which is the **personal** account — the
machine has both a Personal and a Business1 account registered, and a business
account would have mounted as `OneDrive - <Company>`. If the intent was the
business tenant, move the folder there and adjust the path below.

## On the new machine — setup

```bash
git clone --recurse-submodules https://github.com/enjoyneer87/EveryMotor.git
cd EveryMotor
git checkout codex/phase-1-static-1-8-model
git submodule update --init --recursive        # if clone missed it
```

Then bring the data across. Sign into the **same OneDrive account** and let it
sync, or download the folder from onedrive.live.com:

```powershell
$src = "$env:USERPROFILE\OneDrive\EveryMotor_migration"
New-Item -ItemType Directory -Force backup | Out-Null
Copy-Item -Recurse "$src\doe_data" backup\doe_data
Copy-Item "$src\checkpoints\*.pt" results\
```

If OneDrive has Files On-Demand on, the files may be cloud-only placeholders
(`Attributes` shows `Offline`) and the copy will pull them down on demand —
slower but correct. To force them local first, right-click the folder →
*Always keep on this device*, or:

```powershell
attrib -U +P "$src\*" /S      # unpin -> pin, i.e. make locally available
```

Verify you got everything before trusting it:

```powershell
$d = Get-ChildItem backup\doe_data -Recurse -File
"{0} files, {1} bytes" -f $d.Count, ($d | Measure-Object Length -Sum).Sum
# expect: 197 files, 364026511 bytes
```

The loader expects the data at **`backup/doe_data`** — that exact path is the
default for `--data-dir` everywhere. Then:

```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:26.03
```

Verify before trusting anything:

```bash
python -m pytest tests/ -q --ignore=tests/test_phase1_contract_boundaries.py \
                           --ignore=tests/test_curl_trainer_batching.py   # 220 pass, host, no torch

docker run --rm -v "$PWD:/workspace/app" -w /workspace/app \
    nvcr.io/nvidia/physicsnemo/physicsnemo:26.03 \
    python -m eval.benchmark --data-dir backup/doe_data --skip-curl-floor \
        --curl-ckpt results/mgn_nodeB_long.pt --out /tmp/check.json
# expect  |B| nRMSE 13.32%   torque nRMSE 9.20%
```

If those two match, the migration is faithful.

## Resuming the no-time_s run

The run was stopped at **epoch 20 of 60** — not because it failed, but because
other ANSYS processes on this workstation had taken the GPU to 98% memory and an
epoch had gone from 85 s to 508 s.

**The trainer has no resume path**: it always starts from a fresh model, and
`mgn_nodeB_notime.pt` holds only the best weights, no optimizer state. So the new
machine restarts the 60-epoch run from scratch. The epoch-20 checkpoint is worth
carrying anyway — it is enough for a preliminary read on whether removing
`time_s` killed the early-cycle error concentration, without waiting ~75 min.

```bash
docker run --rm --gpus all --ipc=host -v "$PWD:/workspace/app" -w /workspace/app \
    nvcr.io/nvidia/physicsnemo/physicsnemo:26.03 \
    bash results/logs/run_no_timefeat.sh
```

Two things that cost time here, worth not rediscovering:

* **Keep `--batch-size 4`.** Batch 8 fills a 24 GB card and allocator thrashing
  made an epoch **12x** slower (50 s → 10 min+). Check `nvidia-smi` before
  blaming the model — the same thing happened at batch 4 once other GPU work
  showed up.
* **PowerShell here-strings break `bash -lc`** (CRLF). Run the `.sh` files in
  `results/logs/` directly, as above.

## Alternative: Google Drive

Drive for Desktop is not installed on the source machine (no rclone either), so
this needs one of:

```powershell
# a. Drive for Desktop (google.com/drive/download) mounts as a letter, often G:
$dst = "G:\My Drive\EveryMotor_migration"
New-Item -ItemType Directory -Force "$dst\checkpoints" | Out-Null
Copy-Item -Recurse backup\doe_data "$dst\doe_data"
Copy-Item results\mgn_nodeB_long.pt, results\mgn_nodeB_notime.pt "$dst\checkpoints\"
```

```bash
# b. rclone — scriptable, no client install
rclone config                                  # new remote "gdrive", type: drive
rclone copy backup/doe_data gdrive:EveryMotor_migration/doe_data -P
rclone copy results/mgn_nodeB_long.pt   gdrive:EveryMotor_migration/checkpoints/ -P
rclone copy results/mgn_nodeB_notime.pt gdrive:EveryMotor_migration/checkpoints/ -P
```

Or drag the folders in at drive.google.com — fine for a one-off.

## Alternative: HuggingFace private dataset

Worth it only if this becomes a recurring sync and you want versioning plus a
one-line scripted fetch. A repo created with `private=True` is visible to you
and anyone you explicitly add — not public, not indexed.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                     # WRITE token from hf.co/settings/tokens
huggingface-cli repo create everymotor-doe40 --type dataset --private

huggingface-cli upload enjoyneer87/everymotor-doe40 backup/doe_data doe_data --repo-type dataset
huggingface-cli upload enjoyneer87/everymotor-doe40 \
    results/mgn_nodeB_long.pt checkpoints/mgn_nodeB_long.pt --repo-type dataset
huggingface-cli upload enjoyneer87/everymotor-doe40 \
    results/mgn_nodeB_notime.pt checkpoints/mgn_nodeB_notime.pt --repo-type dataset

# on the new machine (READ token is enough)
huggingface-cli download enjoyneer87/everymotor-doe40 --repo-type dataset \
    --local-dir . --include "doe_data/*"
mkdir -p backup && mv doe_data backup/
```

## Where the work stands

State, open question and next steps are in
`.github/plans/methodology_review_20260720.md` — sections 6–9 are this session.
Short version: the torque metric is validated against Motor-CAD's virtual work to
0.35% on the mean, the current best surrogate is 13.32% |B| / 9.20% torque
against a 1.58% representation floor, and the open item is whether removing the
`time_s` shortcut feature (section 9) clears the early-cycle error.
