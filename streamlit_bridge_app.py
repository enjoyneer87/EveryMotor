#!/usr/bin/env python3
"""Streamlit control panel for overnight development actions.

This app wires quick buttons to subprocess actions via xenv_subprocess_bridge.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from xenv_subprocess_bridge import run_from_payload

ROOT = Path(__file__).resolve().parent


def detect_powershell() -> str | None:
    for cmd in ("pwsh", "powershell"):
        if shutil.which(cmd):
            return cmd
    return None


def python_cmd() -> str:
    return "python"


def make_actions() -> dict[str, dict]:
    ps = detect_powershell()

    actions: dict[str, dict] = {
        "Submodule Branch Check": {
            "command": [
                python_cmd(),
                str(ROOT / "check_submodule_branch.py"),
                "--repo",
                str(ROOT),
                "--submodule",
                "eMach",
                "--expected-branch",
                "devVeriACLoss",
            ],
            "timeout_sec": 60,
            "cwd": str(ROOT),
        },
        "Runtime Contract Check": {
            "command": [
                python_cmd(),
                str(ROOT / "check_runtime_contract.py"),
                "--host-data",
                str(ROOT),
                "--doe-data",
                str(ROOT),
                "--mso-root",
                str(ROOT),
            ],
            "timeout_sec": 60,
            "cwd": str(ROOT),
        },
        "Ops Status Snapshot": {
            "command": [
                python_cmd(),
                str(ROOT / "ops_status_snapshot.py"),
                "--repo",
                str(ROOT),
                "--out",
                str(ROOT / "ops_status_snapshot.json"),
            ],
            "timeout_sec": 60,
            "cwd": str(ROOT),
        },
    }

    if ps:
        token = os.getenv("NOTION_TOKEN", "")
        dbid = os.getenv("NOTION_DATABASE_ID", "33507031978c81e5a8eaeaf27372f37d")
        actions["Notion Field Sync"] = {
            "command": [
                ps,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "sync_notion_fields.ps1"),
                "-Token",
                token,
                "-DatabaseId",
                dbid,
            ],
            "timeout_sec": 120,
            "cwd": str(ROOT),
        }

    return actions


def run_payload(payload: dict) -> dict:
    result = run_from_payload(payload)
    return asdict(result)


def main() -> None:
    st.set_page_config(page_title="EveryMotor XENV Bridge", layout="wide")
    st.title("EveryMotor XENV Bridge")
    st.caption("Wire Streamlit buttons to subprocess actions")

    actions = make_actions()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quick Actions")
        selected = st.selectbox("Action", list(actions.keys()))
        if st.button("Run Action", type="primary"):
            payload = actions[selected]
            with st.spinner("Running command..."):
                try:
                    result = run_payload(payload)
                    st.session_state["last_result"] = result
                    if result.get("returncode", 1) == 0:
                        st.success("Action completed successfully")
                    else:
                        st.error(f"Action failed with return code {result.get('returncode')}")
                except Exception as exc:
                    st.error(f"Action error: {exc}")

    with col2:
        st.subheader("Custom JSON Payload")
        default_payload = {
            "command": [python_cmd(), str(ROOT / "check_env.py")],
            "cwd": str(ROOT),
            "timeout_sec": 60,
        }
        text = st.text_area("Payload", value=json.dumps(default_payload, indent=2), height=220)
        if st.button("Run Custom Payload"):
            try:
                payload = json.loads(text)
                result = run_payload(payload)
                st.session_state["last_result"] = result
                if result.get("returncode", 1) == 0:
                    st.success("Custom payload completed successfully")
                else:
                    st.error(f"Custom payload failed with return code {result.get('returncode')}")
            except Exception as exc:
                st.error(f"Invalid payload or execution error: {exc}")

    st.subheader("Last Result")
    result = st.session_state.get("last_result")
    if result:
        st.json(result)
        st.code(result.get("stdout", ""), language="text")
        if result.get("stderr"):
            st.code(result.get("stderr", ""), language="text")
    else:
        st.info("Run an action to see output.")

    st.subheader("CLI Launcher")
    st.code("streamlit run streamlit_bridge_app.py")


if __name__ == "__main__":
    main()
