"""Tests for ane.kernel — high-level ANE kernel wrapper."""

from __future__ import annotations

import numpy as np
import pytest

# ANEKernel imports _bridge at module level
_ane_bridge = pytest.importorskip("ane._bridge", reason="Cython bridge not compiled")
_kernel_mod = pytest.importorskip("ane.kernel", reason="ane.kernel import failed")

ANEKernel = _kernel_mod.ANEKernel


class _DummyKernel:
    """Minimal stand-in for ane._bridge.Kernel to test ANEKernel logic."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def eval(self):
        return None

    def write_input(self, idx: int, arr: np.ndarray):
        return None

    def read_output(self, idx: int, arr: np.ndarray):
        return None


def test_ane_kernel_closed_property():
    """ANEKernel.closed reflects whether the underlying kernel is None."""
    dummy = _DummyKernel()
    k = ANEKernel(dummy, [(1, 4, 1, 2)], [(1, 4, 1, 2)], np.float16)

    assert not k.closed
    k.close()
    assert k.closed
    assert dummy.closed


def test_ane_kernel_context_manager():
    """ANEKernel supports context manager — close on exit."""
    dummy = _DummyKernel()
    k = ANEKernel(dummy, [(1, 4, 1, 2)], [(1, 4, 1, 2)], np.float16)

    with k:
        assert not k.closed
    assert k.closed
    assert dummy.closed
