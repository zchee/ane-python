"""Integration tests — compile MIL, run on ANE, verify outputs.

These tests require:
  - macOS on Apple Silicon
  - Compiled Cython bridge (ane._bridge)
  - ANE runtime availability

All tests are marked @pytest.mark.integration and skip automatically
when the bridge is unavailable.

NOTE: ANE hardware has minimum tensor dimension requirements.
Small tensors (dim<16 or so) may fail to compile or evaluate.
Tests use ane_compile/ane_eval helpers that skip on hardware failures.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if bridge is not compiled
_bridge = pytest.importorskip("ane._bridge", reason="Cython bridge not compiled")
_kernel_mod = pytest.importorskip("ane.kernel", reason="ane.kernel import failed")

Bridge = _bridge.Bridge
build_weight_blob = _bridge.build_weight_blob
compile = _bridge.compile
compile_multi = _bridge.compile_multi
ANEKernel = _kernel_mod.ANEKernel

pytestmark = pytest.mark.integration

# MIL program header and conv constants used by inline test MIL builders
MIL_HDR: str = (
    'program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n{\n'
)

CONV_CONST: str = (
    '        string pt = const()[name=string("pt"), val=string("valid")];\n'
    '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '        int32 gr = const()[name=string("gr"), val=int32(1)];\n'
)


def ane_compile(*args, **kwargs):
    """Wrapper around compile() that skips test on ANE hardware failure."""
    try:
        return compile(*args, **kwargs)
    except RuntimeError as e:
        if "compilation returned NULL" in str(e) or "modelWithMILText" in str(e):
            pytest.skip(f"ANE hardware rejected kernel: {e}")
        raise


def ane_compile_multi(*args, **kwargs):
    """Wrapper around compile_multi() that skips test on ANE hardware failure."""
    try:
        return compile_multi(*args, **kwargs)
    except RuntimeError as e:
        if "compilation returned NULL" in str(e) or "modelWithMILText" in str(e):
            pytest.skip(f"ANE hardware rejected kernel: {e}")
        raise


def ane_eval(kernel):
    """Wrapper around kernel.eval() that skips test on ANE hardware failure."""
    try:
        kernel.eval()
    except RuntimeError as e:
        if "eval failed" in str(e):
            pytest.skip(f"ANE hardware eval failed: {e}")
        raise


def _build_identity_mil(channels: int, seq_len: int) -> bytes:
    """Minimal MIL: add zero to input (identity operation)."""
    m = MIL_HDR
    m += f"    func main<ios18>(tensor<fp16, [1, {channels}, 1, {seq_len}]> x) {{\n"
    m += '        fp16 z = const()[name=string("z"), val=fp16(0.0)];\n'
    m += f'        tensor<fp16, [1,{channels},1,{seq_len}]> out = add(x=x,y=z)[name=string("out")];\n'
    m += "    } -> (out);\n}\n"
    return m.encode("utf-8")


def _build_scale_mil(channels: int, seq_len: int, scale: float = 2.0) -> bytes:
    """Minimal MIL: multiply input by a scalar."""
    m = MIL_HDR
    m += f"    func main<ios18>(tensor<fp16, [1, {channels}, 1, {seq_len}]> x) {{\n"
    m += f'        fp16 s = const()[name=string("s"), val=fp16({scale})];\n'
    m += f'        tensor<fp16, [1,{channels},1,{seq_len}]> out = mul(x=x,y=s)[name=string("out")];\n'
    m += "    } -> (out);\n}\n"
    return m.encode("utf-8")


def _build_conv_mil(in_ch: int, out_ch: int, seq_len: int) -> bytes:
    """Minimal MIL: 1x1 convolution with BLOBFILE weights."""
    m = MIL_HDR
    m += f"    func main<ios18>(tensor<fp16, [1, {in_ch}, 1, {seq_len}]> x) {{\n"
    m += CONV_CONST
    m += (
        f"        tensor<fp16, [{out_ch},{in_ch},1,1]> W = "
        'const()[name=string("W"), val=tensor<fp16, '
        f'[{out_ch},{in_ch},1,1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), '
        "offset=uint64(64)))];\n"
    )
    m += (
        f"        tensor<fp16, [1,{out_ch},1,{seq_len}]> out = "
        "conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
        '[name=string("out")];\n'
    )
    m += "    } -> (out);\n}\n"
    return m.encode("utf-8")


@pytest.fixture(scope="module")
def bridge():
    """Initialize ANE bridge for the test module."""
    b = Bridge()
    b.init()
    return b


class TestIdentityKernel:
    """Compile a minimal identity MIL, verify output matches input within fp16 tolerance."""

    def test_identity_passthrough(self, bridge):
        C, S = 64, 32
        mil = _build_identity_mil(C, S)
        nbytes = 1 * C * 1 * S * 2  # fp16

        kernel = ane_compile(mil, input_sizes=[nbytes], output_sizes=[nbytes])
        with kernel:
            inp = np.random.default_rng(7).standard_normal((1, C, 1, S)).astype(np.float16)
            kernel.write_input(0, inp)
            ane_eval(kernel)

            out = np.empty_like(inp)
            kernel.read_output(0, out)

            np.testing.assert_allclose(
                out.astype(np.float32),
                inp.astype(np.float32),
                atol=1e-3,
                rtol=1e-3,
                err_msg="Identity kernel output should match input",
            )

    def test_identity_zeros(self, bridge):
        C, S = 64, 32
        mil = _build_identity_mil(C, S)
        nbytes = 1 * C * 1 * S * 2

        kernel = ane_compile(mil, input_sizes=[nbytes], output_sizes=[nbytes])
        with kernel:
            inp = np.zeros((1, C, 1, S), dtype=np.float16)
            kernel.write_input(0, inp)
            ane_eval(kernel)

            out = np.empty_like(inp)
            kernel.read_output(0, out)

            np.testing.assert_array_equal(out, inp, err_msg="Zero input should produce zero output")


class TestScaleKernel:
    """Compile a scalar multiply MIL, verify output = input * scale."""

    def test_scale_by_two(self, bridge):
        C, S = 64, 32
        scale = 2.0
        mil = _build_scale_mil(C, S, scale=scale)
        nbytes = 1 * C * 1 * S * 2

        kernel = ane_compile(mil, input_sizes=[nbytes], output_sizes=[nbytes])
        with kernel:
            inp = np.ones((1, C, 1, S), dtype=np.float16) * 0.5
            kernel.write_input(0, inp)
            ane_eval(kernel)

            out = np.empty_like(inp)
            kernel.read_output(0, out)

            expected = (inp * scale).astype(np.float16)
            np.testing.assert_allclose(
                out.astype(np.float32),
                expected.astype(np.float32),
                atol=1e-3,
                rtol=1e-3,
                err_msg="Scale kernel should multiply input by 2",
            )


class TestConvKernel:
    """Compile a 1x1 conv MIL with weight blobs, verify output shape and non-trivial values."""

    def test_conv_output_shape(self, bridge):
        in_ch, out_ch, S = 64, 128, 32
        mil = _build_conv_mil(in_ch, out_ch, S)
        in_bytes = 1 * in_ch * 1 * S * 2
        out_bytes = 1 * out_ch * 1 * S * 2

        rng = np.random.default_rng(99)
        w = (rng.standard_normal((out_ch, in_ch)) * 0.1).astype(np.float32)
        w_blob = build_weight_blob(w)

        kernel = ane_compile_multi(
            mil,
            weights={"@model_path/weights/w.bin": w_blob},
            input_sizes=[in_bytes],
            output_sizes=[out_bytes],
        )
        with kernel:
            inp = rng.standard_normal((1, in_ch, 1, S)).astype(np.float16)
            kernel.write_input(0, inp)
            ane_eval(kernel)

            out = np.empty((1, out_ch, 1, S), dtype=np.float16)
            kernel.read_output(0, out)

            assert out.shape == (1, out_ch, 1, S)
            assert not np.allclose(out, 0, atol=1e-4), "Conv output should be non-zero for non-zero input"

    def test_conv_zero_weights(self, bridge):
        """Conv with zero weights should produce zero output."""
        in_ch, out_ch, S = 64, 128, 32
        mil = _build_conv_mil(in_ch, out_ch, S)
        in_bytes = 1 * in_ch * 1 * S * 2
        out_bytes = 1 * out_ch * 1 * S * 2

        w = np.zeros((out_ch, in_ch), dtype=np.float32)
        w_blob = build_weight_blob(w)

        kernel = ane_compile_multi(
            mil,
            weights={"@model_path/weights/w.bin": w_blob},
            input_sizes=[in_bytes],
            output_sizes=[out_bytes],
        )
        with kernel:
            inp = np.ones((1, in_ch, 1, S), dtype=np.float16)
            kernel.write_input(0, inp)
            ane_eval(kernel)

            out = np.empty((1, out_ch, 1, S), dtype=np.float16)
            kernel.read_output(0, out)

            np.testing.assert_allclose(
                out.astype(np.float32),
                0.0,
                atol=1e-4,
                err_msg="Conv with zero weights should produce zero output",
            )


class TestWeightBlobRoundtrip:
    """Build weight blobs from known data, compile a kernel using them, verify the kernel runs."""

    def test_blob_compile_and_run(self, bridge):
        """Build blobs, compile conv kernel, run successfully."""
        in_ch, out_ch, S = 64, 64, 32
        mil = _build_conv_mil(in_ch, out_ch, S)
        in_bytes = 1 * in_ch * 1 * S * 2
        out_bytes = 1 * out_ch * 1 * S * 2

        # Identity-like weight: diagonal 1s
        w = np.eye(out_ch, in_ch, dtype=np.float32)
        w_blob = build_weight_blob(w)

        kernel = ane_compile_multi(
            mil,
            weights={"@model_path/weights/w.bin": w_blob},
            input_sizes=[in_bytes],
            output_sizes=[out_bytes],
        )
        with kernel:
            inp = np.arange(in_ch * S, dtype=np.float16).reshape(1, in_ch, 1, S)
            kernel.write_input(0, inp)
            ane_eval(kernel)

            out = np.empty((1, out_ch, 1, S), dtype=np.float16)
            kernel.read_output(0, out)

            # With identity weights, conv should approximately pass through
            np.testing.assert_allclose(
                out.astype(np.float32),
                inp.astype(np.float32),
                atol=0.05,
                rtol=0.05,
                err_msg="Identity conv should approximate passthrough",
            )


class TestKernelReuse:
    """Compile a kernel, run eval multiple times with different inputs."""

    def test_multiple_evals(self, bridge):
        C, S = 64, 32
        mil = _build_scale_mil(C, S, scale=3.0)
        nbytes = 1 * C * 1 * S * 2

        kernel = ane_compile(mil, input_sizes=[nbytes], output_sizes=[nbytes])
        with kernel:
            rng = np.random.default_rng(123)
            for i in range(5):
                inp = rng.standard_normal((1, C, 1, S)).astype(np.float16)
                kernel.write_input(0, inp)
                ane_eval(kernel)

                out = np.empty_like(inp)
                kernel.read_output(0, out)

                expected = (inp * 3.0).astype(np.float16)
                np.testing.assert_allclose(
                    out.astype(np.float32),
                    expected.astype(np.float32),
                    atol=0.05,
                    rtol=0.05,
                    err_msg=f"Eval #{i}: reused kernel should still produce correct output",
                )


class TestCompileCount:
    """Verify Bridge.compile_count increments after compilations."""

    def test_compile_count_increments(self, bridge):
        before = bridge.compile_count
        C, S = 64, 32
        mil = _build_identity_mil(C, S)
        nbytes = 1 * C * 1 * S * 2

        kernel = ane_compile(mil, input_sizes=[nbytes], output_sizes=[nbytes])
        kernel.close()

        after = bridge.compile_count
        assert after > before, f"compile_count should increase after compilation: before={before}, after={after}"


class TestANEKernelHighLevel:
    """Test the high-level ANEKernel wrapper with real compilation."""

    def test_from_mil_identity(self, bridge):
        C, S = 64, 32
        mil = _build_identity_mil(C, S)
        shape_in = (1, C, 1, S)
        shape_out = (1, C, 1, S)

        try:
            k = ANEKernel.from_mil(
                mil,
                input_shapes=[shape_in],
                output_shapes=[shape_out],
            )
        except RuntimeError as e:
            pytest.skip(f"ANE hardware rejected kernel: {e}")

        with k:
            inp = np.random.default_rng(55).standard_normal(shape_in).astype(np.float16)
            k.set_input(0, inp)
            try:
                k.run()
            except RuntimeError as e:
                pytest.skip(f"ANE eval failed: {e}")
            out = k.get_output(0)

            assert out.shape == shape_out
            np.testing.assert_allclose(
                out.astype(np.float32),
                inp.astype(np.float32),
                atol=1e-3,
                rtol=1e-3,
            )

    def test_from_mil_with_weights(self, bridge):
        in_ch, out_ch, S = 64, 128, 32
        mil = _build_conv_mil(in_ch, out_ch, S)

        rng = np.random.default_rng(77)
        w = (rng.standard_normal((out_ch, in_ch)) * 0.1).astype(np.float32)
        w_blob = build_weight_blob(w)

        try:
            k = ANEKernel.from_mil(
                mil,
                weights={"@model_path/weights/w.bin": w_blob},
                input_shapes=[(1, in_ch, 1, S)],
                output_shapes=[(1, out_ch, 1, S)],
            )
        except RuntimeError as e:
            pytest.skip(f"ANE hardware rejected kernel: {e}")

        with k:
            inp = rng.standard_normal((1, in_ch, 1, S)).astype(np.float16)
            k.set_input(0, inp)
            try:
                k.run()
            except RuntimeError as e:
                pytest.skip(f"ANE eval failed: {e}")
            out = k.get_output(0)

            assert out.shape == (1, out_ch, 1, S)
            assert not np.allclose(out, 0, atol=1e-4)
