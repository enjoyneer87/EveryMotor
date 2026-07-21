"""Tests for the graph-op preflight and the empty-scorecard backstop.

The failure these guard against is not a crash but a *quiet success*: with
`torch_scatter` missing, physicsnemo raises only inside the forward pass, the
harness folds that into per-sample failures, and the run reports `no samples`
and exits 0. Both halves are tested here — the preflight that stops the run
before it starts, and the exit-code backstop that catches any other cause of an
empty scorecard.

These run on the host: the guard only imports packages by name, so it is
exercised with those lookups patched out rather than by uninstalling anything.
"""

import pytest

from eval import runtime_guard
from eval.benchmark import models_with_no_samples
from eval.runtime_guard import (
    GRAPH_OP_PACKAGES,
    GraphOpsUnavailable,
    probe_graph_ops,
    require_graph_ops,
)


def _stub_stack(monkeypatch, *, versions, accepts):
    """Pretend the machine has exactly `versions` installed."""
    monkeypatch.setattr(runtime_guard, "_version_of", lambda name: versions.get(name))
    monkeypatch.setattr(runtime_guard, "_torch_build", lambda: "2.11.0+cu128")
    monkeypatch.setattr(runtime_guard, "_physicsnemo_accepts_stack", lambda: accepts)


HEALTHY = {
    "torch_geometric": "2.8.0",
    "torch_scatter": "2.1.2+pt211cu128",
    "physicsnemo": "2.1.1",
}


# --- preflight ---------------------------------------------------------------


def test_a_healthy_stack_passes_and_reports_what_it_found(monkeypatch):
    _stub_stack(monkeypatch, versions=HEALTHY, accepts=True)

    probe = require_graph_ops()

    assert probe["usable"] is True
    assert probe["packages"]["torch_scatter"] == "2.1.2+pt211cu128"
    assert probe["torch"] == "2.11.0+cu128"


@pytest.mark.parametrize("absent", GRAPH_OP_PACKAGES)
def test_a_missing_graph_package_fails_before_any_scoring(monkeypatch, absent):
    _stub_stack(
        monkeypatch,
        versions={k: v for k, v in HEALTHY.items() if k != absent},
        accepts=True,
    )

    with pytest.raises(GraphOpsUnavailable) as excinfo:
        require_graph_ops(context="mgn_nodeB_long.pt")

    message = str(excinfo.value)
    assert absent in message
    assert "mgn_nodeB_long.pt" in message
    # The message has to say why a clean-looking run is not evidence of health,
    # because that is the mistake it exists to prevent.
    assert "no samples" in message
    assert "exit 0" in message


def test_the_failure_names_the_pinned_wheel(monkeypatch):
    """A torch upgrade is the usual cause, so the fix must be in the message."""
    _stub_stack(
        monkeypatch,
        versions={**HEALTHY, "torch_scatter": None},
        accepts=True,
    )

    with pytest.raises(GraphOpsUnavailable, match=r"2\.1\.2\+pt211cu128"):
        require_graph_ops()


def test_an_abi_mismatch_is_reported_as_a_mismatch_not_a_missing_install(monkeypatch):
    """Both packages import, but physicsnemo's version spec rejects them.

    This is what a torch upgrade looks like: `import torch_scatter` still
    succeeds while physicsnemo quietly routes around the whole MeshGraphNet
    path, so the diagnosis has to differ from the missing-package one.
    """
    _stub_stack(monkeypatch, versions=HEALTHY, accepts=False)

    with pytest.raises(GraphOpsUnavailable) as excinfo:
        require_graph_ops()

    message = str(excinfo.value)
    assert "version-spec mismatch" in message
    assert "not importable" not in message


def test_an_unreadable_physicsnemo_flag_does_not_block_a_usable_stack(monkeypatch):
    """physicsnemo may move the flag; that is not evidence of a broken stack."""
    _stub_stack(monkeypatch, versions=HEALTHY, accepts=None)

    probe = require_graph_ops()

    assert probe["physicsnemo_accepts_stack"] is None
    assert probe["usable"] is True


@pytest.mark.parametrize("accepts,versions", [(True, {}), (False, HEALTHY)])
def test_the_message_survives_a_cp949_console(monkeypatch, accepts, versions):
    """A guard nobody can read is not a guard.

    The default console on the Windows machine that runs this is cp949. An
    em-dash in the message turns the intended failure into a
    UnicodeEncodeError when anything print()s it, which buries the diagnosis
    under an unrelated traceback.
    """
    _stub_stack(monkeypatch, versions=versions, accepts=accepts)

    with pytest.raises(GraphOpsUnavailable) as excinfo:
        require_graph_ops()

    str(excinfo.value).encode("cp949")


def test_probe_never_raises_so_it_can_be_recorded(monkeypatch):
    _stub_stack(monkeypatch, versions={}, accepts=False)

    probe = probe_graph_ops()

    assert probe["usable"] is False
    assert all(probe["packages"][name] is None for name in GRAPH_OP_PACKAGES)


def test_probe_on_the_real_interpreter_returns_a_json_safe_shape():
    """No patching: whatever this machine has, the shape must be recordable."""
    probe = probe_graph_ops()

    assert set(probe["packages"]) == set(GRAPH_OP_PACKAGES)
    assert isinstance(probe["usable"], bool)


# --- empty-scorecard backstop ------------------------------------------------


def test_a_model_that_scored_nothing_is_singled_out():
    artifact = {
        "models": {
            "fem_identity": {"summary": {"n_samples": 270}},
            "curl:mgn_nodeB_long": {"summary": {"n_samples": 0, "note": "no samples scored"}},
        }
    }

    assert models_with_no_samples(artifact) == ["curl:mgn_nodeB_long"]


def test_a_fully_scored_run_reports_nothing():
    artifact = {"models": {"fem_identity": {"summary": {"n_samples": 270}}}}

    assert models_with_no_samples(artifact) == []


def test_a_missing_summary_counts_as_empty():
    """A malformed card must not read as success by omission."""
    artifact = {"models": {"broken": {}}}

    assert models_with_no_samples(artifact) == ["broken"]
