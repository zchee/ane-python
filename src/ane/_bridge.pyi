"""Type stubs for ane._bridge Cython extension."""

from types import TracebackType

import numpy as np

class Bridge:
    """Singleton interface to ANE runtime initialization."""

    def init(self) -> None:
        """Initialize the ANE runtime. Raises RuntimeError on failure."""
        ...

    @property
    def compile_count(self) -> int:
        """Current ANE compile count (for exec() restart budgeting)."""
        ...

    def reset_compile_count(self) -> None:
        """Reset the compile counter to zero."""
        ...

class Kernel:
    """Wrapper around a compiled ANE kernel handle.

    Supports context-manager protocol for automatic cleanup.
    """

    n_inputs: int
    n_outputs: int
    input_sizes: tuple[int, ...]
    output_sizes: tuple[int, ...]

    def eval(self) -> None:
        """Run the compiled kernel on ANE. Raises RuntimeError on failure."""
        ...

    def write_input(self, idx: int, arr: np.ndarray) -> None:
        """Write numpy array data to kernel input tensor at index ``idx``.

        The array must be C-contiguous and its byte size must match
        the expected input size.
        """
        ...

    def read_output(self, idx: int, arr: np.ndarray) -> None:
        """Read kernel output tensor at index ``idx`` into pre-allocated numpy array.

        The array must be C-contiguous and its byte size must match
        the expected output size.
        """
        ...

    def close(self) -> None:
        """Free the kernel handle. Idempotent."""
        ...

    def __enter__(self) -> Kernel: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool: ...

def compile(
    mil_text: bytes,
    *,
    weight_data: bytes | np.ndarray | None = None,
    input_sizes: list[int],
    output_sizes: list[int],
) -> Kernel:
    """Compile a MIL program into an ANE kernel.

    Args:
        mil_text: UTF-8 MIL program text as bytes.
        weight_data: Optional raw weight blob (bytes or uint8 ndarray).
        input_sizes: List of byte sizes for each input tensor.
        output_sizes: List of byte sizes for each output tensor.

    Returns:
        Kernel handle ready for eval().

    Raises:
        RuntimeError: If compilation fails.
    """
    ...

def compile_multi(
    mil_text: bytes,
    *,
    weights: dict[str, np.ndarray],
    input_sizes: list[int],
    output_sizes: list[int],
) -> Kernel:
    """Compile a MIL program with multiple named weight blobs.

    Args:
        mil_text: UTF-8 MIL program text as bytes.
        weights: Dict mapping weight names to weight data (uint8 ndarray).
        input_sizes: List of byte sizes for each input tensor.
        output_sizes: List of byte sizes for each output tensor.

    Returns:
        Kernel handle ready for eval().

    Raises:
        RuntimeError: If compilation fails.
    """
    ...

def build_weight_blob(
    src: np.ndarray,
    transposed: bool = False,
) -> np.ndarray:
    """Build an ANE-format weight blob from a float32 2D array.

    Args:
        src: 2D float32 array [rows x cols].
        transposed: If True, build transposed weight blob.

    Returns:
        1D uint8 numpy array containing the weight blob
        (128-byte header + fp16 data).

    Raises:
        ValueError: If src is not 2D float32 C-contiguous.
        RuntimeError: If blob construction fails.
    """
    ...
