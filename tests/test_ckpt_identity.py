"""Checkpoint identity: hashing, registry lookup, and refusal on contradiction.

The bug this guards against is silent: two different runs were stored under the
name `mgn_nodeB_notime.pt` at the same time (epoch 34 on the share, epoch 55 in
the repo). Nothing raised. The scorecard recorded a path, so the number could not
be traced back to the weights that produced it.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from eval.ckpt_identity import (
    CheckpointIdentityError,
    ambiguous_names,
    find_by_sha,
    identify,
    load_registry,
    sha256_file,
)

PAYLOAD = b"not a real checkpoint, but hashing does not care" * 64


@pytest.fixture()
def fake_ckpt(tmp_path):
    p = tmp_path / "mgn_nodeB_notime.pt"
    p.write_bytes(PAYLOAD)
    return p


@pytest.fixture()
def registry(fake_ckpt):
    sha = hashlib.sha256(PAYLOAD).hexdigest()
    return {
        "version": "checkpoints/v1",
        "checkpoints": [
            {
                "sha256": sha,
                "bytes": len(PAYLOAD),
                "epoch": 55,
                "machine": "HPC134",
                "canonical_name": "mgn_nodeB_notime_ep55_HPC134.pt",
                "seen_as": ["mgn_nodeB_notime.pt"],
            }
        ],
        "ambiguous_names": {
            "mgn_nodeB_notime.pt": {
                "reason": "two runs shared this name",
                "seen_as": ["4ef2c475... (epoch 34, WSL2-container)",
                            "6937a2e2... (epoch 55, HPC134)"],
            }
        },
    }


def test_sha256_file_matches_hashlib(fake_ckpt):
    assert sha256_file(fake_ckpt) == hashlib.sha256(PAYLOAD).hexdigest()


def test_streaming_hash_handles_multi_chunk(tmp_path):
    """The real checkpoints are ~28 MB, so the reader must not stop at one chunk."""
    big = tmp_path / "big.pt"
    blob = b"\xa5" * (3 * (1 << 20) + 7)          # 3 MB + change, spans chunks
    big.write_bytes(blob)
    assert sha256_file(big) == hashlib.sha256(blob).hexdigest()


def test_identify_resolves_known_hash_to_canonical_name(fake_ckpt, registry):
    ident = identify(fake_ckpt, epoch=55, registry=registry)
    assert ident["registered"] is True
    assert ident["canonical_name"] == "mgn_nodeB_notime_ep55_HPC134.pt"
    assert ident["epoch"] == 55
    assert ident["bytes"] == len(PAYLOAD)


def test_identify_flags_the_ambiguous_filename(fake_ckpt, registry):
    """Resolving by hash is not enough on its own -- the name must be called out,
    because the *other* machine's copy of that name is different weights."""
    ident = identify(fake_ckpt, epoch=55, registry=registry)
    assert ident["name_is_ambiguous"] is True
    assert "two runs shared this name" in ident["ambiguity_note"]


def test_identify_refuses_when_epoch_contradicts_registry(fake_ckpt, registry):
    """Same bytes, different epoch claim: one of the two is wrong, so scoring it
    would produce a number nobody can trace back."""
    with pytest.raises(CheckpointIdentityError, match="epoch 55.*reports epoch 34"):
        identify(fake_ckpt, epoch=34, registry=registry)


def test_non_strict_downgrades_refusal_to_a_warning(fake_ckpt, registry):
    ident = identify(fake_ckpt, epoch=34, registry=registry, strict=False)
    assert "refusing" in ident["warning"]
    assert ident["registered"] is True


def test_unknown_hash_is_allowed_but_marked(tmp_path, registry):
    """A new training run is normal. It must score, and it must say it is new."""
    fresh = tmp_path / "mgn_nodeB_notime_ep60_HPC134.pt"
    fresh.write_bytes(b"a different run entirely")
    ident = identify(fresh, epoch=60, registry=registry)
    assert ident["registered"] is False
    assert "canonical_name" not in ident
    assert ident.get("name_is_ambiguous") is None


def test_missing_registry_is_empty_not_fatal(tmp_path):
    """A fresh clone that has not synced the share still has to run."""
    reg = load_registry(tmp_path / "nope.json")
    assert reg["checkpoints"] == []
    assert find_by_sha("deadbeef", reg) is None
    assert ambiguous_names(reg) == {}


def test_shipped_registry_is_self_consistent():
    """The checked-in registry must not contradict itself: one hash, one entry,
    and every ambiguous name must actually map to more than one checkpoint."""
    reg = load_registry()
    shas = [e["sha256"] for e in reg["checkpoints"]]
    assert len(shas) == len(set(shas)), "duplicate sha256 in registry"

    for entry in reg["checkpoints"]:
        assert entry.get("canonical_name"), f"{entry['sha256'][:12]} has no canonical name"
        assert len(entry["sha256"]) == 64

    for name, info in reg.get("ambiguous_names", {}).items():
        holders = [e for e in reg["checkpoints"] if name in e.get("seen_as", [])]
        assert len(holders) > 1, f"{name} is marked ambiguous but only {len(holders)} checkpoint claims it"
        assert info.get("reason")


def test_eval_identity_module_does_not_import_torch():
    """`eval/` runs on hosts without torch. Identity checking must too --
    otherwise a machine cannot even verify what it just downloaded."""
    import subprocess
    import sys

    code = (
        "import sys; import eval.ckpt_identity as m; "
        "assert 'torch' not in sys.modules, 'ckpt_identity pulled in torch'; "
        "print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout
