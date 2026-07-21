# Moving this work to another machine

Code and results are in git. The DOE fields and the model checkpoints are not —
they are gitignored on purpose (348 MB + 28 MB each), and the repo's `.git` is
already 1.1 GB, so git-lfs would burn the GitHub free quota immediately.

**The department network share is the transport** -- both machines already
reach it, so there is no account, no OAuth and no upload wait. ~401 MB.

    \\192.168.0.165\디지털융합사업본부\01_EM사업부\강도현\EveryMotor_migration\ 

**Privacy: this is a shared departmental folder.** Anyone with permissions on
that share -- colleagues, IT -- can read the DOE fields and the checkpoints.
That is a real difference from the personal-cloud options below, which are
private to one account. Nothing here is secret, but decide deliberately rather
than by default.

rclone (OneDrive or Google Drive) is kept below for the case where the two
machines are not on the same network, or where the data should not sit on a
share.

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
| `results/viz/*.png` (torque three-way, Motor-CAD validation, GIF previews) | 500 KB | yes |
| `backup/doe_data/` (40 DOE cases, H5) | **348 MB** | no — `.gitignore: doe_data/` |
| `results/mgn_nodeB_long.pt` (current best, 13.32% / 9.20%) | **28 MB** | no — `.gitignore: *.pt` |
| `results/mgn_nodeB_notime.pt` (no-time_s run, stopped at epoch 20) | **28 MB** | no |
| `results/viz/*.gif` | 3.2 MB | no — `.gitignore: *.gif`; **on the share**, and regenerable via `results/logs/run_viz.sh` |
| `results/logs/*.log`, `*.err` (training curves, scoring runs) | 66 KB | no — `.gitignore: *.log`; on the share |
| `results/diag_infer/*.npz` (early-cycle diagnosis dumps) | 4.2 MB | no — on the share; regenerable |
| Motor-CAD original solve (`D:\KDH\Sim_4SolverX\DOE4TrainingData`) | 141 MB/case | no — only needed to redo the Motor-CAD torque validation |

Older checkpoints (`doe_*.pt`, `mgn_curl_*.pt`, `mgn_nodeB_v2.pt`, …) are
deliberately **not** in the migration set: two of them take 11 node features
(they were fed A and J) and the harness refuses to score them, and the rest are
superseded and cheaper to retrain than to carry.

## Already copied to the share

    \\192.168.0.165\디지털융합사업본부\01_EM사업부\강도현\EveryMotor_migration

```
EveryMotor_migration/          256 files, 428,723,034 B
├── doe_data/                  197 files, 364,026,511 B   40 DOE cases
├── checkpoints/                 2 files,  56,693,298 B
│   ├── mgn_nodeB_long.pt      28,347,381 B  current best: 13.32% |B| / 9.20% torque
│   └── mgn_nodeB_notime.pt    28,345,917 B  no-time_s run, stopped at epoch 20
├── viz/                         6 files,   3,697,890 B   GIFs + the figures (see below)
├── diag_infer/                 15 files,   4,238,751 B   early-cycle diagnosis dumps (§9 evidence)
└── logs/                       36 files,      66,584 B   training/scoring logs + .sh drivers
```

Verified after each copy: **file counts and byte totals match on both sides.**

`viz/` is the figures themselves. Only the GIFs actually need carrying — the PNGs
are in git (`.gitignore: *.gif` but not `*.png`) — but they are 100 KB and keeping
the folder whole is simpler than explaining the split:

| file | bytes | in git? |
|---|---|---|
| `case0004_representative.gif` | 1,730,107 | no |
| `case0032_worst.gif` | 1,460,233 | no |
| `case0004_representative_preview.png` | 97,113 | yes |
| `case0032_worst_preview.png` | 87,974 | yes |
| `torque_three_way_case0004.png` | 179,238 | yes |
| `torque_validation_motorcad.png` | 143,225 | yes |

**Not carried, deliberately:** `results/backups/` (95 MB) is April-era checkpoints and
inference dumps from the superseded `symm_mgn_doe_full_onloadtorque` pipeline.

## Reproducing the figures on the new machine

Everything needed is either in git or in the list above:

| piece | where |
|---|---|
| `tools/make_field_gif.py`, `eval/`, `phase1_static/`, both trainers | git |
| `results/logs/run_viz.sh` (the exact commands used) | git |
| `backup/doe_data`, `results/mgn_nodeB_long.pt` | share |
| Python environment | the NGC container — **nothing custom** |

```bash
docker run --rm --gpus all --ipc=host -v "$PWD:/workspace/app" -w /workspace/app \
    nvcr.io/nvidia/physicsnemo/physicsnemo:26.03 \
    bash results/logs/run_viz.sh
```

**The container needs no `pip install`.** A fresh `docker pull` of
`physicsnemo:26.03` already carries everything both the harness and the GIF tool
import — verified on the image itself:

```
torch 2.10.0a0+a36e1d39eb.nv26.01.42222806   torch_geometric 2.7.0   physicsnemo 2.0.0
numpy 1.26.4   h5py 3.15.1   scipy 1.17.1   matplotlib 3.10.8   pillow 12.1.1   pytest 8.1.1
```

Nothing was installed ad hoc inside a running container during this work, so there
is no hidden state to reproduce. (The one `pip install --user pytest` in the session
history was on the **host**, not in the container — see below.)

### Host environment (only for the torch-free test run)

The `python -m pytest tests/` check below runs on the host, not in the container,
and the host therefore needs a few packages. This machine used Python 3.13 with:

```
pytest  numpy  h5py  scipy  matplotlib
```

No torch on the host — the tests that need it are the two that get `--ignore`d.
If installing on the host is inconvenient, run the same suite in the container
instead (it has pytest and all of the above).

A local staging copy also sits at `C:\Users\moa\EveryMotor_migration\` (same
content). It is only a source for re-uploads and can be deleted once the other
machine has pulled from the share.

## On the new machine — setup

```bash
git clone --recurse-submodules https://github.com/enjoyneer87/EveryMotor.git
cd EveryMotor
git checkout codex/phase-1-static-1-8-model
git submodule update --init --recursive        # if clone missed it
```

Then pull the data from the share:

```powershell
$share = "\\192.168.0.165\디지털융합사업본부\01_EM사업부\강도현\EveryMotor_migration"
robocopy "$share\doe_data" backup\doe_data /E /NFL /NDL /NJH /NP /MT:8
Copy-Item "$share\checkpoints\*.pt" results\ -Force
```

robocopy exits **1** on a successful copy (0 means *nothing needed copying*), so
do not read that as an error.

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

## Alternative: rclone -> OneDrive

For when the machines are not on the same network, or the data should not live
on a departmental share. `tools/bin/rclone.exe` (v1.74.4, portable, gitignored)
is already downloaded.

### Authorize (browser, one time)

OAuth needs a browser, and doing it yourself is what guarantees only *your*
Microsoft account is ever linked:

```powershell
D:\KDH\NvidiaNemo\tools\bin\rclone.exe config
```

Answer: `n` (new remote) -> name **`od`** -> storage **`onedrive`** ->
client_id/client_secret **blank** (Enter) -> region **`1`** (Microsoft Cloud
Global) -> advanced config **`n`** -> use web browser **`y`**. Your browser
opens; sign in and approve.

rclone then asks which kind of drive to connect. Pick **OneDrive Personal or
Business** (option `1`), and if it lists more than one drive, choose the one you
want. Confirm the drive it found, then `y` to save and `q` to quit.

This machine has both a Personal and a Business1 account registered with the
desktop client, so the sign-in page may offer a choice -- pick whichever account
the other machine will also use. The desktop client's failed sign-in does not
constrain this; rclone authorizes independently.

Verify:

```powershell
D:\KDH\NvidiaNemo\tools\bin\rclone.exe listremotes   # expect: od:
D:\KDH\NvidiaNemo\tools\bin\rclone.exe about od:     # expect: your quota
```

### Upload

```powershell
$rc = "D:\KDH\NvidiaNemo\tools\bin\rclone.exe"
& $rc copy "C:\Users\moa\EveryMotor_migration" od:EveryMotor_migration --progress
```

Or straight from the originals, skipping the staging folder entirely:

```powershell
& $rc copy D:\KDH\NvidiaNemo\backup\doe_data od:EveryMotor_migration/doe_data --progress
& $rc copy D:\KDH\NvidiaNemo\results\mgn_nodeB_long.pt   od:EveryMotor_migration/checkpoints/ --progress
& $rc copy D:\KDH\NvidiaNemo\results\mgn_nodeB_notime.pt od:EveryMotor_migration/checkpoints/ --progress
```

Confirm what landed:

```powershell
& $rc size od:EveryMotor_migration     # expect 199 files, ~401 MB
& $rc check "C:\Users\moa\EveryMotor_migration" od:EveryMotor_migration --size-only
```

### Who can reach it

- **OneDrive files are private to your Microsoft account by default.** rclone
  creates no sharing links and changes no permissions; nothing is shared unless
  you later share it yourself in the OneDrive web UI.
- **The OAuth token is stored in `C:\Users\moa\AppData\Roaming\rclone\rclone.conf`.**
  It is plain text. Anyone who can log into *this Windows account* can use it --
  so the data is exactly as private as your Windows login, no more.
- Want more than that? Encrypt the config:
  ```powershell
  D:\KDH\NvidiaNemo\tools\bin\rclone.exe config encryption set
  ```
  Every later rclone call then prompts for that password. Lose it and the remote
  must be re-authorized (the data in the cloud is unaffected).
- If this is the **Business** tenant, your organisation's admin can in principle
  reach the data and retention policies apply. Use the Personal account if that
  matters.
- `tools/bin/` is gitignored, so the binary cannot be pushed by accident, and
  `rclone.conf` lives outside the repo regardless.

## Alternative: Google Drive

Same rclone, different backend -- worth it if the Microsoft account is the
constrained one (business retention, admin visibility) and the 2 TB Google quota
is not.

```powershell
D:\KDH\NvidiaNemo\tools\bin\rclone.exe config
# n -> name 'gdrive' -> storage 'drive' -> client_id/secret blank -> scope 1
# -> root_folder_id/service_account blank -> advanced n -> browser y

$rc = "D:\KDH\NvidiaNemo\tools\bin\rclone.exe"
& $rc copy "C:\Users\moa\EveryMotor_migration" gdrive:EveryMotor_migration --progress
```

Google Drive files are likewise private to the account by default, and the token
lands in the same `rclone.conf` with the same exposure.

Note the OneDrive **desktop client** is not a route at all until it is signed in
(see the top of this file); that is independent of the rclone onedrive backend.

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
