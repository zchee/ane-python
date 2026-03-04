"""High-level ANE kernel wrapper with shape-aware I/O."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Protocol

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ane._bridge import compile, compile_multi


class _KernelLike(Protocol):
    """Kernel protocol used by ANEKernel for static typing."""

    def write_input(self, idx: int, arr: NDArray) -> None: ...
    def read_output(self, idx: int, arr: NDArray) -> None: ...
    def eval(self) -> None: ...
    def close(self) -> None: ...


class ANEKernel:
    """Shape-aware wrapper around a compiled ANE kernel.

    Usage::

        with ANEKernel.from_mil(mil_text, weights={...},
                                input_shapes=[(1, 768, 1, 256)],
                                output_shapes=[(1, 4608, 1, 256)]) as k:
            k.set_input(0, data)
            k.run()
            out = k.get_output(0)
    """

    __slots__ = ("_dtype", "_input_shapes", "_kernel", "_output_shapes")
    _dtype: np.dtype
    _input_shapes: list[tuple[int, ...]]
    _kernel: _KernelLike | None
    _output_shapes: list[tuple[int, ...]]

    def __init__(
        self,
        kernel: _KernelLike,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        dtype: DTypeLike,
    ) -> None:
        self._kernel = kernel
        self._input_shapes = input_shapes
        self._output_shapes = output_shapes
        self._dtype = np.dtype(dtype)

    @classmethod
    def from_mil(
        cls,
        mil_text: str | bytes,
        *,
        weights: dict[str, NDArray[np.uint8]] | None = None,
        input_shapes: Sequence[tuple[int, ...]],
        output_shapes: Sequence[tuple[int, ...]],
        dtype: DTypeLike = np.float16,
    ) -> ANEKernel:
        """Compile a MIL program and return a shape-aware kernel.

        Args:
            mil_text: MIL program (str or UTF-8 bytes).
            weights: Optional dict of weight name → blob (uint8 ndarray).
            input_shapes: Shape tuples for each input tensor.
            output_shapes: Shape tuples for each output tensor.
            dtype: Data type for I/O (default fp16, matching ANE).
        """
        if isinstance(mil_text, str):
            mil_text = mil_text.encode("utf-8")

        dt = np.dtype(dtype)
        input_shapes = [tuple(s) for s in input_shapes]
        output_shapes = [tuple(s) for s in output_shapes]
        input_sizes = [math.prod(s) * dt.itemsize for s in input_shapes]
        output_sizes = [math.prod(s) * dt.itemsize for s in output_shapes]

        if weights:
            kernel = compile_multi(
                mil_text,
                weights=weights,
                input_sizes=input_sizes,
                output_sizes=output_sizes,
            )
        else:
            kernel = compile(
                mil_text,
                input_sizes=input_sizes,
                output_sizes=output_sizes,
            )

        return cls(kernel, input_shapes, output_shapes, dt)

    @property
    def input_shapes(self) -> list[tuple[int, ...]]:
        return list(self._input_shapes)

    @property
    def output_shapes(self) -> list[tuple[int, ...]]:
        return list(self._output_shapes)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def closed(self) -> bool:
        return self._kernel is None

    def set_input(self, idx: int, data: NDArray) -> None:
        """Write data to input tensor at *idx*, casting to kernel dtype."""
        if self._kernel is None:
            raise RuntimeError("Kernel is closed")
        arr = np.ascontiguousarray(data, dtype=self._dtype)
        self._kernel.write_input(idx, arr)

    def run(self) -> None:
        """Execute the kernel on ANE."""
        if self._kernel is None:
            raise RuntimeError("Kernel is closed")
        self._kernel.eval()

    def get_output(self, idx: int) -> NDArray:
        """Read output tensor at *idx* into a new array."""
        if self._kernel is None:
            raise RuntimeError("Kernel is closed")
        out = np.empty(self._output_shapes[idx], dtype=self._dtype)
        self._kernel.read_output(idx, out)
        return out

    def close(self) -> None:
        """Release the kernel handle."""
        if self._kernel is not None:
            self._kernel.close()
            self._kernel = None

    def __enter__(self) -> ANEKernel:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False
