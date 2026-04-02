---
name: xenv-subprocess-bridge
description: "Use when: validating subprocess JSON payload/result contract and executing xenv_subprocess_bridge safely."
---

# XENV Subprocess Bridge

## Purpose
Validate and run subprocess tasks through the JSON contract used by xenv_subprocess_bridge.py.

## Input contract
Required:
- command: non-empty string array

Optional:
- cwd
- timeout_sec (default 300)

Example payload:
{
  "command": ["python", "check_env.py"],
  "cwd": ".",
  "timeout_sec": 60
}

## Output contract
Must include:
- command, ok, cwd, timeout_sec, returncode
- stdout, stderr
- start_utc, end_utc, duration_sec

Schemas:
- schemas/xenv_payload.schema.json
- schemas/xenv_result.schema.json

## Execution
- python xenv_subprocess_bridge.py --payload xenv_payload_example.json
- python xenv_subprocess_bridge.py --payload xenv_payload_example.json --out out/bridge_result.json

## Safety rules
- Keep shell disabled by default.
- Use explicit command arrays, not shell command strings.
- Reject payloads with empty command.

## Streamlit linkage
streamlit_bridge_app.py should call run_from_payload in xenv_subprocess_bridge.py.
