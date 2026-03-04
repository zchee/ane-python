"""Shared fixtures for ANE test suite."""

from __future__ import annotations

import contextlib

import pytest

# Detect whether the Cython bridge is importable (requires compilation + macOS)
try:
    import ane

    HAS_BRIDGE = True
except ImportError:
    HAS_BRIDGE = False


@pytest.fixture(scope="session", autouse=True)
def init_ane():
    """Initialize ANE runtime once per test session.

    Silently skips if the bridge extension is not available.
    """
    if HAS_BRIDGE:
        with contextlib.suppress(RuntimeError):
            ane.init()
