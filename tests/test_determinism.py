"""Tests for the pinned float32 settings.

These assert the *settings*, not the numbers. The numbers are not bitwise
reproducible on GPU and this module is explicit that no flag here makes them so
— three benchmark runs of the same checkpoint on the same machine moved in the
8th significant digit while the numpy floors stayed bit-identical.

Runs on the host: everything torch-specific is skipped when torch is absent,
because `eval.benchmark` is meant to stay importable without it.
"""

import pytest

from eval.determinism import (
    CUBLAS_WORKSPACE_ENV,
    DEFAULT_SEED,
    determinism_state,
    pin_determinism,
)

torch = pytest.importorskip("torch", reason="settings under test are torch's")


def test_tf32_is_off_after_pinning():
    """TF32 truncates the fp32 mantissa to 10 bits inside every matmul.

    torch 2.11 already defaults matmul TF32 off, which is exactly why this is
    pinned: the default has moved between releases, and torch here resolves
    from the global interpreter, so it can move without the venv changing.
    """
    pin_determinism()

    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"


def test_cudnn_autotuning_is_off_after_pinning():
    """Autotuning picks a kernel by timing it, so machine load leaks into the result."""
    pin_determinism()

    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_pinning_reports_what_it_pinned():
    state = pin_determinism()

    assert state["seed"] == DEFAULT_SEED
    assert state["matmul_allow_tf32"] is False
    assert state["cudnn_benchmark"] is False
    assert state["float32_matmul_precision"] == "highest"
    assert state["torch"] == str(torch.__version__)


def test_the_record_bounds_its_own_reproducibility_claim():
    """Bitwise equality here is measured on one machine, and not enforced.

    Repeated runs were byte-identical after pinning, so the flag is worth
    claiming — but `warn_only=True` means an op with no deterministic variant
    would warn rather than fail, and nothing here speaks to a *different*
    machine. A scorecard that said only "reproducible" would overstate both.
    """
    state = pin_determinism()

    assert state["bitwise_reproducible"] is True
    assert state["reproducibility_scope"] == "same machine, same stack, same settings"
    assert "warn_only" in state["note"]
    assert "13.324" in state["note"], "the cross-machine contract stays 3 decimals"


def test_deterministic_algorithms_stay_warn_only():
    """This is the flag that removed the drift, and it must not abort a run.

    torch_scatter delegates its sum to torch's `scatter_add_`, so torch's
    deterministic implementation does reach it. warn_only keeps some *other*
    op without a deterministic variant from turning a scoring run into a crash.
    """
    state = pin_determinism()

    assert state["deterministic_algorithms"] == "warn_only"
    assert torch.are_deterministic_algorithms_enabled() is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="drift is GPU-only")
def test_the_scatter_reduction_is_repeatable_after_pinning():
    """Guards the actual mechanism, not just the flag that enables it.

    Scatter-add over a shuffled index is the operation that moved the gate's
    last digits: float addition is not associative, so an atomicAdd kernel's
    scheduling order changes the result. Under pinning it must not.
    """
    pin_determinism()
    src = torch.randn(200_000, 8, device="cuda")
    index = torch.randint(0, 512, (200_000,), device="cuda")

    def reduce_once():
        out = torch.zeros(512, 8, device="cuda")
        return out.index_add_(0, index, src).clone()

    assert torch.equal(reduce_once(), reduce_once())


def test_pinning_is_idempotent():
    first = pin_determinism()
    second = pin_determinism()

    assert first == second


def test_seed_is_applied_to_every_generator():
    import random

    import numpy as np

    pin_determinism(seed=1234)
    drawn = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    pin_determinism(seed=1234)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == drawn


def test_cublas_workspace_is_reported_not_faked():
    """Setting it mid-process is too late for cuBLAS, so it is observed only."""
    state = pin_determinism()
    import os

    assert state[CUBLAS_WORKSPACE_ENV.lower()] == os.environ.get(CUBLAS_WORKSPACE_ENV)


def test_state_reader_does_not_mutate_settings():
    pin_determinism()
    torch.backends.cudnn.benchmark = True

    observed = determinism_state()

    assert observed["cudnn_benchmark"] is True, "reader must report, not impose"
    torch.backends.cudnn.benchmark = False


def test_the_recorded_state_is_json_safe():
    import json

    json.dumps(pin_determinism())
