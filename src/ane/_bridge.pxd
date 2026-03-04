# cython: language_level=3

from libc.stddef cimport size_t
from libc.stdint cimport uint8_t

cdef extern from "ane_bridge.h":
    # Opaque kernel handle
    ctypedef struct ANEKernelHandle:
        pass

    int ane_bridge_init()

    ANEKernelHandle *ane_bridge_compile(
        const char *mil_text, size_t mil_len,
        const uint8_t *weight_data, size_t weight_len,
        int n_inputs, const size_t *input_sizes,
        int n_outputs, const size_t *output_sizes,
    )

    ANEKernelHandle *ane_bridge_compile_multi_weights(
        const char *mil_text, size_t mil_len,
        const char **weight_names, const uint8_t **weight_datas,
        const size_t *weight_lens, int n_weights,
        int n_inputs, const size_t *input_sizes,
        int n_outputs, const size_t *output_sizes,
    )

    bint ane_bridge_eval(ANEKernelHandle *kernel)

    void ane_bridge_write_input(
        ANEKernelHandle *kernel, int idx,
        const void *data, size_t bytes,
    )

    void ane_bridge_read_output(
        ANEKernelHandle *kernel, int idx,
        void *data, size_t bytes,
    )

    void ane_bridge_free(ANEKernelHandle *kernel)

    int ane_bridge_get_compile_count()

    void ane_bridge_reset_compile_count()

    uint8_t *ane_bridge_build_weight_blob(
        const float *src, int rows, int cols,
        size_t *out_len,
    )

    uint8_t *ane_bridge_build_weight_blob_transposed(
        const float *src, int rows, int cols,
        size_t *out_len,
    )

    void ane_bridge_free_blob(void *ptr)
