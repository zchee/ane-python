# ane-python

Python bindings for [Apple Neural Engine (ANE)](https://machinelearning.apple.com/research/neural-engine-transformers) via private `AppleNeuralEngine.framework` APIs using Cython.

Compile and run [MIL (ML Intermediate Language)](https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html) programs directly on the Neural Engine — bypassing CoreML — for low-level kernel control on Apple Silicon.

## Requirements

- macOS 15+ on Apple Silicon (M-series)
- Python 3.11–3.14
- Xcode Command Line Tools (`xcode-select --install`)

## Installation

```bash
uv pip install -e ".[dev]"
```

This compiles the Objective-C bridge (`ane_bridge.m` → `libane.so`) and the Cython extension (`_bridge.pyx` → `_bridge.cpython-*.so`) in one step.

## Quick Start

```python
import numpy as np
import ane
from ane.kernel import ANEKernel

# Initialize ANE runtime (call once)
ane.init()

# MIL program: multiply input by 3.0
mil = b"""program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"},
 {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""},
 {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 32]> x) {
        fp16 s = const()[name=string("s"), val=fp16(3.0)];
        tensor<fp16, [1,64,1,32]> out = mul(x=x,y=s)[name=string("out")];
    } -> (out);
}
"""

shape = (1, 64, 1, 32)
with ANEKernel.from_mil(mil, input_shapes=[shape], output_shapes=[shape]) as k:
    k.set_input(0, np.ones(shape, dtype=np.float16) * 2.0)
    k.run()
    out = k.get_output(0)  # all 6.0
```

## Examples

```bash
# Basic: compile and run scalar multiply + 1x1 convolution kernels
uv run examples/basic_ane.py
```

## Architecture

```
src/ane/
├── bridge/
│   ├── ane_bridge.h      # C API declarations
│   └── ane_bridge.m      # Obj-C: dlopen ANE framework, compile/eval/IO via objc_msgSend
├── _bridge.pxd           # Cython extern declarations for ane_bridge.h
├── _bridge.pyx           # Cython extension: Bridge, Kernel, compile(), build_weight_blob()
├── _bridge.pyi           # Type stubs for IDE support
├── kernel.py             # ANEKernel: high-level shape-aware wrapper with context manager
└── __init__.py           # Public API: init(), get_compile_count(), reset_compile_count()
```

**C Bridge** — Loads `AppleNeuralEngine.framework` via `dlopen`, resolves private classes (`_ANEInMemoryModel`, `_ANERequest`, `_ANEIOSurfaceObject`) through `objc_msgSend`. Manages IOSurface buffers for zero-copy I/O and serializes weight blobs in ANE format (128-byte header + fp16 data).

**Cython Bridge** — Wraps the C API into Python classes. `Kernel` holds a compiled kernel handle with `write_input`/`read_output`/`eval`/`close`. `compile()` and `compile_multi()` take MIL text + optional named weight blobs. `build_weight_blob()` converts float32 numpy arrays to ANE weight format.

**Python API** — `ANEKernel.from_mil()` is the main entry point: takes MIL text, I/O shapes, and optional weights, returns a context-managed kernel that auto-casts inputs to fp16.

## ANE Constraints

- All computation is fp16 on the Neural Engine
- Tensor dimensions must be >= ~32 elements per axis for `eval` to succeed
- ANE has limited compilation slots (~119); track with `ane.get_compile_count()`
- Compilation failures may need a 100ms retry for ANE slot reclamation
- MIL programs require version 1.3 with `buildInfo` dict

## Disclaimer

The files in `src/ane/bridge/` (`ane_bridge.h` and `ane_bridge.m`) are derived from [maderix/ANE](https://github.com/maderix/ANE), which is licensed under the MIT License. The original work by maderix reverse-engineered the private `AppleNeuralEngine.framework` APIs to enable direct ANE access. These files have been adapted for use as a Python/Cython bridge in this project.

This project uses **undocumented private Apple APIs** that are not part of any public SDK. These APIs may change or break at any time with macOS updates. This software is provided for **research and educational purposes only** — use at your own risk.

## License

[Apache License 2.0](LICENSE)
