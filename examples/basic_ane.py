#!/usr/bin/env python3
"""Basic ANE example: compile and run a kernel on Apple Neural Engine.

Demonstrates the pure ANE workflow:
  1. Initialize the ANE runtime
  2. Build a MIL program (inline, no training dependencies)
  3. Compile the kernel
  4. Write input, run eval, read output
  5. Verify correctness

Requirements:
  - macOS 15+ on Apple Silicon (M-series)
  - Built and installed `ane` package (`pip install -e .`)

Usage:
  python examples/basic_ane.py
"""

import sys

import numpy as np

import ane
from ane._bridge import build_weight_blob
from ane.kernel import ANEKernel

# MIL program header matching ANE compiler expectations
MIL_HDR = (
    'program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n{\n'
)

CONV_CONST = (
    '        string pt = const()[name=string("pt"), val=string("valid")];\n'
    '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '        int32 gr = const()[name=string("gr"), val=int32(1)];\n'
)


def build_scale_mil(channels: int, seq_len: int, scale: float) -> bytes:
    """Minimal MIL program: multiply input by a scalar constant."""
    m = MIL_HDR
    m += f"    func main<ios18>(tensor<fp16, [1, {channels}, 1, {seq_len}]> x) {{\n"
    m += f'        fp16 s = const()[name=string("s"), val=fp16({scale})];\n'
    m += f'        tensor<fp16, [1,{channels},1,{seq_len}]> out = mul(x=x,y=s)[name=string("out")];\n'
    m += "    } -> (out);\n}\n"
    return m.encode("utf-8")


def build_conv_mil(in_ch: int, out_ch: int, seq_len: int) -> bytes:
    """MIL program: 1x1 convolution with BLOBFILE weights."""
    m = MIL_HDR
    m += f"    func main<ios18>(tensor<fp16, [1, {in_ch}, 1, {seq_len}]> x) {{\n"
    m += CONV_CONST
    m += f'        tensor<fp16, [{out_ch},{in_ch},1,1]> W = const()[name=string("W"), val=tensor<fp16, [{out_ch},{in_ch},1,1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];\n'
    m += f'        tensor<fp16, [1,{out_ch},1,{seq_len}]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("out")];\n'
    m += "    } -> (out);\n}\n"
    return m.encode("utf-8")


def main() -> None:
    # ANE requires seq_len >= 32 for eval to succeed.
    C, S = 64, 32

    # -- Step 1: Initialize ANE runtime --
    print("[1] Initializing ANE runtime...")
    try:
        ane.init()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Requires Apple Silicon with macOS 15+.", file=sys.stderr)
        sys.exit(1)
    print("    OK")

    # -- Step 2: Scalar multiply kernel --
    print("\n[2] Compiling scalar multiply kernel (x * 3.0)...")
    mil = build_scale_mil(C, S, scale=3.0)
    shape = (1, C, 1, S)

    with ANEKernel.from_mil(mil, input_shapes=[shape], output_shapes=[shape]) as k:
        inp = np.ones(shape, dtype=np.float16) * 2.0
        k.set_input(0, inp)
        k.run()
        out = k.get_output(0)

        expected = np.float16(6.0)
        assert np.allclose(out, expected, atol=1e-2), f"Expected {expected}, got {out.flat[0]}"
        print("    Input:  2.0 (all elements)")
        print(f"    Output: {float(out.flat[0]):.1f} (expected 6.0)")
        print(f"    Compile count: {ane.get_compile_count()}")

    # -- Step 3: 1x1 convolution kernel with weight blobs --
    in_ch, out_ch = 64, 128
    print(f"\n[3] Compiling 1x1 conv kernel ({in_ch} -> {out_ch} channels)...")

    mil = build_conv_mil(in_ch, out_ch, S)
    rng = np.random.default_rng(42)
    w = (rng.standard_normal((out_ch, in_ch)) * 0.1).astype(np.float32)
    w_blob = build_weight_blob(w)

    print(f"    Weight blob: {w_blob.nbytes:,} bytes (128-byte header + fp16 data)")

    with ANEKernel.from_mil(
        mil,
        weights={"@model_path/weights/w.bin": w_blob},
        input_shapes=[(1, in_ch, 1, S)],
        output_shapes=[(1, out_ch, 1, S)],
    ) as k:
        inp = rng.standard_normal((1, in_ch, 1, S)).astype(np.float16)
        k.set_input(0, inp)
        k.run()
        out = k.get_output(0)

        print(f"    Input shape:  {inp.shape}")
        print(f"    Output shape: {out.shape}")
        print(f"    Output mean:  {float(out.mean()):.6f}")
        print(f"    Output std:   {float(out.std()):.6f}")
        assert not np.allclose(out, 0, atol=1e-4), "Conv output should be non-zero"
        print(f"    Compile count: {ane.get_compile_count()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
