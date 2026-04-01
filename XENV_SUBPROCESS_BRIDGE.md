# XENV Subprocess Bridge

This document defines the JSON contract used by xenv_subprocess_bridge.py.

## Input Payload

Required fields:
- command: string list, non-empty

Optional fields:
- cwd: working directory
- timeout_sec: integer, default 300

Example:

```json
{
  "command": ["python", "check_env.py"],
  "cwd": ".",
  "timeout_sec": 60
}
```

## Output JSON

Fields:
- command: executed command list
- cwd: resolved working directory
- timeout_sec: timeout used
- returncode: process return code
- stdout: captured stdout
- stderr: captured stderr
- start_utc: ISO timestamp
- end_utc: ISO timestamp
- duration_sec: elapsed seconds

## CLI Usage

```powershell
python xenv_subprocess_bridge.py --payload xenv_payload_example.json
python xenv_subprocess_bridge.py --payload xenv_payload_example.json --out out/bridge_result.json
```

## Streamlit Wiring

streamlit_bridge_app.py calls run_from_payload from xenv_subprocess_bridge.py.
Buttons are preconfigured for:
- Submodule branch check
- Runtime contract check
- Notion field sync
- Custom payload execution

## Notes

- This bridge is non-shell by default (shell=False) for safer execution.
- Use explicit command arrays instead of shell strings.
