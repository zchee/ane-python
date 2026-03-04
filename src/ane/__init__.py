"""ane — Python bindings for Apple Neural Engine."""

from ane._bridge import Bridge, Kernel, build_weight_blob, compile, compile_multi
from ane.kernel import ANEKernel

__version__ = "0.1.0"
__all__ = [
    "ANEKernel",
    "Bridge",
    "Kernel",
    "build_weight_blob",
    "compile",
    "compile_multi",
    "get_compile_count",
    "init",
    "reset_compile_count",
]

_bridge: Bridge = Bridge()


def init() -> None:
    """Initialize ANE runtime. Call once before any ANE operation."""
    _bridge.init()


def get_compile_count() -> int:
    """Current ANE kernel compilation count."""
    return _bridge.compile_count


def reset_compile_count() -> None:
    """Reset the ANE compilation counter."""
    _bridge.reset_compile_count()
