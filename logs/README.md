# Log Tracking Policy

This repository tracks only small, operational logs needed for cross-server visibility.

Tracked files:
- `logs/policy-sync/policy_sync.jsonl`
- `logs/policy-sync/policy_sync.log`

All other logs remain ignored to avoid repository bloat.

Guidelines:
- Keep tracked logs append-only and machine-readable where possible.
- If a tracked log grows too large, rotate or truncate it and keep only recent history.
- Do not store secrets or tokens in tracked logs.
