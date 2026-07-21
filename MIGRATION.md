# Moving this work to another machine

Code and results are in git. The DOE fields and the model checkpoints are not —
they are gitignored on purpose (348 MB + 28 MB each), and the repo's `.git` is
already 1.1 GB, so git-lfs would burn the GitHub free quota immediately.

**rclone → Google Drive is the transport.** ~401 MB total.

**OneDrive on the source machine is installed but not signed in, and syncs
nothing.** Verified, not assumed:

| check | result |
|---|---|
| registered sync roots (`SyncRootManager`) | **0** |
| `Accounts\Personal` → `UserEmail` / `UserFolder` / `cid` | **all absent** |
| `LastSignInResult` | **`0x8004E4C8`** — an error HRESULT |

A running `OneDrive.exe` proves nothing; sign-in is what matters and it failed.
The earlier reading of "files show `Archive`, so they're uploading" was wrong —
`Archive` only means *not a cloud placeholder*, which is exactly what you get
when nothing is syncing at all.

So `C:\Users\moa\OneDrive\EveryMotor_migration\` is **a local folder on C:**,
nothing more. It is still a fine staging directory to upload *from*, and the
byte-verified copy there is real — it just has not left the machine.

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

## Staged locally, ready to upload

`C:\Users\moa\OneDrive\EveryMotor_migration\` (a plain local folder — see above):

```
EveryMotor_migration/          199 files, 401.2 MB
├── doe_data/                  197 files, 347.2 MB  <- copy of backup/doe_data
└── checkpoints/
    ├── mgn_nodeB_long.pt      27.0 MB  current best: 13.32% |B| / 9.20% torque
    └── mgn_nodeB_notime.pt    27.0 MB  no-time_s run, stopped at epoch 20
```

Verified byte-for-byte: 197/197 files and 364,026,511 bytes on both sides, and
both checkpoints match their source length exactly.

## rclone → Google Drive

`tools/bin/rclone.exe` (v1.74.4, portable, gitignored) is already downloaded.

### 1. Authorize — you must do this step

OAuth needs a browser, and doing it yourself is what guarantees only *your*
Google account is ever linked:

```powershell
D:\KDH\NvidiaNemo\tools\bin\rclone.exe config
```

Answer: `n` (new remote) → name **`gdrive`** → storage **`drive`** →
client_id/secret **blank** (Enter) → scope **`1`** (full access) →
root_folder_id and service_account_file **blank** → advanced config **`n`** →
use web browser **`y`**. Your browser opens, you approve, rclone writes the
token. Then `q` to quit.

Verify:

```powershell
D:\KDH\NvidiaNemo\tools\bin\rclone.exe listremotes      # expect: gdrive:
D:\KDH\NvidiaNemo\tools\bin\rclone.exe about gdrive:    # expect: your 2 TB quota
```

### 2. Upload

```powershell
$rc = "D:\KDH\NvidiaNemo\tools\bin\rclone.exe"
& $rc copy "C:\Users\moa\OneDrive\EveryMotor_migration" gdrive:EveryMotor_migration --progress
```

Or straight from the originals, skipping the staging folder entirely:

```powershell
& $rc copy D:\KDH\NvidiaNemo\backup\doe_data gdrive:EveryMotor_migration/doe_data --progress
& $rc copy D:\KDH\NvidiaNemo\results\mgn_nodeB_long.pt   gdrive:EveryMotor_migration/checkpoints/ --progress
& $rc copy D:\KDH\NvidiaNemo\results\mgn_nodeB_notime.pt gdrive:EveryMotor_migration/checkpoints/ --progress
```

Confirm what landed:

```powershell
& $rc size gdrive:EveryMotor_migration     # expect 199 files, ~401 MB
& $rc check "C:\Users\moa\OneDrive\EveryMotor_migration" gdrive:EveryMotor_migration --size-only
```

### Who can reach it

- **Google Drive files are private to your account by default.** rclone creates
  no sharing links and sets no permissions; nothing is public unless you later
  share it in the Drive UI.
- **The OAuth token is stored in `C:\Users\moa\AppData\Roaming\rclone\rclone.conf`.**
  It is plain text. Anyone who can log into *this Windows account* can use it —
  so the data is exactly as private as your Windows login, no more.
- Want more than that? Encrypt the config:
  ```powershell
  D:\KDH\NvidiaNemo\tools\bin\rclone.exe config encryption set
  ```
  Every later rclone call then prompts for that password. Lose it and the remote
  must be re-authorized (the data on Drive is unaffected).
- `tools/bin/` is gitignored, so neither the binary nor anything beside it can be
  pushed by accident. `rclone.conf` lives outside the repo regardless.

## On the new machine — setup

```bash
git clone --recurse-submodules https://github.com/enjoyneer87/EveryMotor.git
cd EveryMotor
git checkout codex/phase-1-static-1-8-model
git submodule update --init --recursive        # if clone missed it
```

Then pull the data with rclone, authorizing against the **same Google account**:

```powershell
# fetch rclone once (portable, ~28 MB, gitignored)
Invoke-WebRequest https://downloads.rclone.org/rclone-current-windows-amd64.zip -OutFile $env:TEMP\rclone.zip
Expand-Archive $env:TEMP\rclone.zip $env:TEMP\rclone_x -Force
New-Item -ItemType Directory -Force tools\bin | Out-Null
Copy-Item (Get-ChildItem $env:TEMP\rclone_x -Recurse -Filter rclone.exe)[0].FullName tools\bin\rclone.exe

.\tools\bin\rclone.exe config      # n -> gdrive -> drive -> blanks -> scope 1 -> browser auth

.\tools\bin\rclone.exe copy gdrive:EveryMotor_migration/doe_data     backup\doe_data --progress
.\tools\bin\rclone.exe copy gdrive:EveryMotor_migration/checkpoints  results         --progress
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

## Alternative: OneDrive

Only if OneDrive gets signed in first — right now it is not (see the top of this
file). Once `SyncRootManager` shows a registered root and `Accounts\Personal`
carries a `UserEmail`, the staged folder at
`C:\Users\moa\OneDrive\EveryMotor_migration\` would sync on its own and the new
machine could just sign into the same account.

Note the machine has both a Personal and a Business1 account registered; a
business tenant would mount as `OneDrive - <Company>`, which does not exist here,
so the staging folder sits under the personal profile.

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
