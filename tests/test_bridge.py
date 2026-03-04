"""Tests for ane._bridge — Cython bindings to libane_bridge."""

from __future__ import annotations

import numpy as np
import pytest

_bridge_mod = pytest.importorskip("ane._bridge", reason="Cython bridge not compiled")

Bridge = _bridge_mod.Bridge
Kernel = _bridge_mod.Kernel
build_weight_blob = _bridge_mod.build_weight_blob


def test_init_idempotent():
    """Calling Bridge.init() twice must not raise."""
    b = Bridge()
    b.init()
    b.init()  # second call should be a no-op


def test_compile_count():
    """compile_count should be a non-negative integer."""
    b = Bridge()
    b.init()
    count = b.compile_count
    assert isinstance(count, int)
    assert count >= 0


def test_build_weight_blob_shape():
    """3x4 float32 -> blob = 128-byte header + 3*4*2 bytes fp16 data = 152 bytes."""
    src = np.zeros((3, 4), dtype=np.float32)
    blob = build_weight_blob(src)
    expected_size = 128 + 3 * 4 * 2  # header + fp16 data
    assert blob.dtype == np.uint8
    assert len(blob) == expected_size


def test_build_weight_blob_transposed():
    """Transposed blob should differ from non-transposed for non-square matrix."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    blob_normal = build_weight_blob(src, transposed=False)
    blob_transposed = build_weight_blob(src, transposed=True)
    # Transposed version has different dimensions: 4x3 vs 3x4 in fp16
    assert not np.array_equal(blob_normal, blob_transposed)


def test_kernel_context_manager():
    """Kernel supports context manager protocol for resource cleanup."""
    # We can't compile a real kernel without valid MIL, but we can verify
    # that Kernel has __enter__/__exit__ attributes.
    assert hasattr(Kernel, "__enter__")
    assert hasattr(Kernel, "__exit__")
