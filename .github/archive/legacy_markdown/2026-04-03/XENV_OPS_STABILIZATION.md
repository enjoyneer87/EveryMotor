# XENV-6 Ops Stabilization

## Added
- ops_status_snapshot.py: unified ops health snapshot in JSON.
- Streamlit quick action wired to run ops status snapshot.

## Command

```powershell
python ops_status_snapshot.py --repo . --out ops_status_snapshot.json
```

## Output sections
- git branch/head/submodule status
- runtime paths and manifest existence
- checkpoint availability
- command availability (python/docker/pwsh)
- overnight automation script existence

## Usage in Notion evidence
- Attach summary path in 비고 or 완료근거.
- Use snapshot to justify 완료/홀드 transitions.
