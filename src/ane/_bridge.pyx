# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t

cimport numpy as cnp
import numpy as np

cnp.import_array()

from ane._bridge cimport (
    ANEKernelHandle,
    ane_bridge_init,
    ane_bridge_compile,
    ane_bridge_compile_multi_weights,
    ane_bridge_eval,
    ane_bridge_write_input,
    ane_bridge_read_output,
    ane_bridge_free,
    ane_bridge_get_compile_count,
    ane_bridge_reset_compile_count,
    ane_bridge_build_weight_blob,
    ane_bridge_build_weight_blob_transposed,
    ane_bridge_free_blob,
)


cdef class Bridge:
    """Singleton interface to ANE runtime initialization."""

    cdef bint _initialized

    def __cinit__(self):
        self._initialized = False

    def init(self):
        """Initialize the ANE runtime. Raises RuntimeError on failure."""
        if self._initialized:
            return
        cdef int rc = ane_bridge_init()
        if rc == -1:
            raise RuntimeError("ane_bridge_init failed: could not load ANE runtime")
        self._initialized = True

    @property
    def compile_count(self) -> int:
        """Current ANE compile count (for exec() restart budgeting)."""
        return ane_bridge_get_compile_count()

    def reset_compile_count(self):
        """Reset the compile counter to zero."""
        ane_bridge_reset_compile_count()


cdef class Kernel:
    """Wrapper around a compiled ANE kernel handle.

    Supports context-manager protocol for automatic cleanup.
    """

    cdef ANEKernelHandle *_handle
    cdef readonly int n_inputs
    cdef readonly int n_outputs
    cdef readonly tuple input_sizes
    cdef readonly tuple output_sizes

    def __cinit__(self):
        self._handle = NULL
        self.n_inputs = 0
        self.n_outputs = 0
        self.input_sizes = ()
        self.output_sizes = ()

    @staticmethod
    cdef Kernel _wrap(ANEKernelHandle *handle, int n_inputs, tuple input_sizes,
                      int n_outputs, tuple output_sizes):
        cdef Kernel k = Kernel.__new__(Kernel)
        k._handle = handle
        k.n_inputs = n_inputs
        k.n_outputs = n_outputs
        k.input_sizes = input_sizes
        k.output_sizes = output_sizes
        return k

    def eval(self):
        """Run the compiled kernel on ANE. Raises RuntimeError on failure."""
        if self._handle == NULL:
            raise RuntimeError("Kernel already closed")
        cdef bint ok = ane_bridge_eval(self._handle)
        if not ok:
            raise RuntimeError("ane_bridge_eval failed")

    def write_input(self, int idx, cnp.ndarray arr):
        """Write numpy array data to kernel input tensor at index `idx`.

        Performs zero-copy pointer access via cnp.PyArray_DATA.
        Validates index bounds and byte-size match.
        """
        if self._handle == NULL:
            raise RuntimeError("Kernel already closed")
        if idx < 0 or idx >= self.n_inputs:
            raise IndexError(
                f"Input index {idx} out of range [0, {self.n_inputs})"
            )
        cdef size_t expected = <size_t>self.input_sizes[idx]
        cdef size_t actual = <size_t>arr.nbytes
        if actual != expected:
            raise ValueError(
                f"Input {idx}: expected {expected} bytes, got {actual}"
            )
        if not arr.flags['C_CONTIGUOUS']:
            raise ValueError("Input array must be C-contiguous")
        ane_bridge_write_input(self._handle, idx, cnp.PyArray_DATA(arr), expected)

    def read_output(self, int idx, cnp.ndarray arr):
        """Read kernel output tensor at index `idx` into pre-allocated numpy array.

        Performs zero-copy pointer access via cnp.PyArray_DATA.
        Validates index bounds and byte-size match.
        """
        if self._handle == NULL:
            raise RuntimeError("Kernel already closed")
        if idx < 0 or idx >= self.n_outputs:
            raise IndexError(
                f"Output index {idx} out of range [0, {self.n_outputs})"
            )
        cdef size_t expected = <size_t>self.output_sizes[idx]
        cdef size_t actual = <size_t>arr.nbytes
        if actual != expected:
            raise ValueError(
                f"Output {idx}: expected {expected} bytes, got {actual}"
            )
        if not arr.flags['C_CONTIGUOUS']:
            raise ValueError("Output array must be C-contiguous")
        ane_bridge_read_output(self._handle, idx, cnp.PyArray_DATA(arr), expected)

    def close(self):
        """Free the kernel handle. Idempotent."""
        if self._handle != NULL:
            ane_bridge_free(self._handle)
            self._handle = NULL

    def __dealloc__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def compile(bytes mil_text not None, *, object weight_data=None,
            list input_sizes not None, list output_sizes not None) -> Kernel:
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
    cdef const char *mil_ptr = mil_text
    cdef size_t mil_len = <size_t>len(mil_text)

    # Weight data
    cdef const uint8_t *w_ptr = NULL
    cdef size_t w_len = 0
    cdef cnp.ndarray w_arr
    if weight_data is not None:
        if isinstance(weight_data, bytes):
            w_ptr = <const uint8_t *><const char *>weight_data
            w_len = <size_t>len(weight_data)
        elif isinstance(weight_data, np.ndarray):
            w_arr = weight_data
            if not w_arr.flags['C_CONTIGUOUS']:
                raise ValueError("weight_data array must be C-contiguous")
            w_ptr = <const uint8_t *>cnp.PyArray_DATA(w_arr)
            w_len = <size_t>w_arr.nbytes
        else:
            raise TypeError("weight_data must be bytes or ndarray")

    cdef int n_in = <int>len(input_sizes)
    cdef int n_out = <int>len(output_sizes)

    # Allocate size arrays
    cdef size_t *in_sizes = <size_t *>malloc(n_in * sizeof(size_t))
    cdef size_t *out_sizes = <size_t *>malloc(n_out * sizeof(size_t))
    if in_sizes == NULL or out_sizes == NULL:
        free(in_sizes)
        free(out_sizes)
        raise MemoryError("Failed to allocate size arrays")

    cdef int i
    for i in range(n_in):
        in_sizes[i] = <size_t>input_sizes[i]
    for i in range(n_out):
        out_sizes[i] = <size_t>output_sizes[i]

    cdef ANEKernelHandle *handle = ane_bridge_compile(
        mil_ptr, mil_len, w_ptr, w_len,
        n_in, in_sizes, n_out, out_sizes,
    )

    free(in_sizes)
    free(out_sizes)

    if handle == NULL:
        raise RuntimeError("ane_bridge_compile failed: compilation returned NULL")

    return Kernel._wrap(handle, n_in, tuple(input_sizes), n_out, tuple(output_sizes))


def compile_multi(bytes mil_text not None, *, dict weights not None,
                  list input_sizes not None, list output_sizes not None) -> Kernel:
    """Compile a MIL program with multiple named weight blobs.

    Args:
        mil_text: UTF-8 MIL program text as bytes.
        weights: Dict mapping weight names (str) to weight data (uint8 ndarray).
        input_sizes: List of byte sizes for each input tensor.
        output_sizes: List of byte sizes for each output tensor.

    Returns:
        Kernel handle ready for eval().

    Raises:
        RuntimeError: If compilation fails.
    """
    cdef const char *mil_ptr = mil_text
    cdef size_t mil_len = <size_t>len(mil_text)
    cdef int n_weights = <int>len(weights)
    cdef int n_in = <int>len(input_sizes)
    cdef int n_out = <int>len(output_sizes)

    # Allocate arrays for weight names, data pointers, and lengths
    cdef const char **w_names = <const char **>malloc(n_weights * sizeof(char *))
    cdef const uint8_t **w_datas = <const uint8_t **>malloc(n_weights * sizeof(uint8_t *))
    cdef size_t *w_lens = <size_t *>malloc(n_weights * sizeof(size_t))
    cdef size_t *in_sizes = <size_t *>malloc(n_in * sizeof(size_t))
    cdef size_t *out_sizes = <size_t *>malloc(n_out * sizeof(size_t))

    if (w_names == NULL or w_datas == NULL or w_lens == NULL
            or in_sizes == NULL or out_sizes == NULL):
        free(w_names)
        free(w_datas)
        free(w_lens)
        free(in_sizes)
        free(out_sizes)
        raise MemoryError("Failed to allocate arrays for compile_multi")

    # Keep Python references alive to prevent GC during the C call
    cdef list name_refs = []
    cdef list data_refs = []

    cdef int i
    cdef bytes name_bytes
    cdef cnp.ndarray arr
    cdef ANEKernelHandle *handle = NULL

    try:
        i = 0
        for key, val in weights.items():
            if isinstance(key, str):
                name_bytes = key.encode('utf-8')
            elif isinstance(key, bytes):
                name_bytes = key
            else:
                raise TypeError(f"Weight name must be str or bytes, got {type(key)}")
            name_refs.append(name_bytes)
            w_names[i] = name_bytes

            if not isinstance(val, np.ndarray):
                raise TypeError(f"Weight data must be ndarray, got {type(val)}")
            arr = val
            if not arr.flags['C_CONTIGUOUS']:
                raise ValueError(f"Weight data for '{key}' must be C-contiguous")
            data_refs.append(arr)
            w_datas[i] = <const uint8_t *>cnp.PyArray_DATA(arr)
            w_lens[i] = <size_t>arr.nbytes
            i += 1

        for i in range(n_in):
            in_sizes[i] = <size_t>input_sizes[i]
        for i in range(n_out):
            out_sizes[i] = <size_t>output_sizes[i]

        handle = ane_bridge_compile_multi_weights(
            mil_ptr, mil_len,
            w_names, w_datas, w_lens, n_weights,
            n_in, in_sizes, n_out, out_sizes,
        )
    finally:
        free(w_names)
        free(w_datas)
        free(w_lens)
        free(in_sizes)
        free(out_sizes)

    if handle == NULL:
        raise RuntimeError("ane_bridge_compile_multi_weights failed: compilation returned NULL")

    return Kernel._wrap(handle, n_in, tuple(input_sizes), n_out, tuple(output_sizes))


def build_weight_blob(cnp.ndarray src not None, bint transposed=False) -> cnp.ndarray:
    """Build an ANE-format weight blob from a float32 2D array.

    Args:
        src: 2D float32 array [rows x cols].
        transposed: If True, build transposed weight blob.

    Returns:
        1D uint8 numpy array containing the weight blob (128-byte header + fp16 data).

    Raises:
        ValueError: If src is not 2D float32 C-contiguous.
        RuntimeError: If blob construction fails.
    """
    if src.ndim != 2:
        raise ValueError(f"src must be 2D, got {src.ndim}D")
    if src.dtype != np.float32:
        raise ValueError(f"src must be float32, got {src.dtype}")
    if not src.flags['C_CONTIGUOUS']:
        raise ValueError("src must be C-contiguous")

    cdef int rows = <int>src.shape[0]
    cdef int cols = <int>src.shape[1]
    cdef const float *data_ptr = <const float *>cnp.PyArray_DATA(src)
    cdef size_t out_len = 0
    cdef uint8_t *blob

    if transposed:
        blob = ane_bridge_build_weight_blob_transposed(data_ptr, rows, cols, &out_len)
    else:
        blob = ane_bridge_build_weight_blob(data_ptr, rows, cols, &out_len)

    if blob == NULL:
        raise RuntimeError("ane_bridge_build_weight_blob failed: returned NULL")

    # Copy into numpy array then free the C buffer
    cdef cnp.ndarray result = np.empty(out_len, dtype=np.uint8)
    memcpy(cnp.PyArray_DATA(result), blob, out_len)
    ane_bridge_free_blob(blob)

    return result
