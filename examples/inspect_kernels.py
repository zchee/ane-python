#!/usr/bin/env python3
"""Inspect all 6 ANE kernel types: MIL programs, weight requirements, I/O shapes.

Demonstrates:
  1. Generate MIL programs for all 6 kernel types used in transformer training
  2. Show the kernel flow: fwd_attn → fwd_ffn → ffn_bwd → sdpa_bwd1 → sdpa_bwd2 → qkv_bwd
  3. Display I/O tensor shapes and byte sizes
  4. Show weight blob sizes and the causal mask
  5. Print MIL text for a selected kernel (optional)

This is a pure-Python exploration tool — no ANE hardware needed.

Usage:
  uv run python examples/inspect_kernels.py
  uv run python examples/inspect_kernels.py --show-mil sdpa_fwd
  uv run python examples/inspect_kernels.py --config tiny
"""

import argparse

import numpy as np
from ane.config import STORIES_110M, ModelConfig
from ane.mil import MILGenerator

CONFIGS = {
    "stories110m": STORIES_110M,
    "tiny": ModelConfig(dim=64, hidden_dim=128, n_heads=4, seq_len=16, n_layers=2, vocab_size=256),
    "small": ModelConfig(dim=256, hidden_dim=512, n_heads=8, seq_len=64, n_layers=4, vocab_size=4096),
}

KERNEL_NAMES = ["sdpa_fwd", "ffn_fwd", "ffn_bwd", "sdpa_bwd1", "sdpa_bwd2", "qkv_bwd"]


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def kernel_io_shapes(cfg: ModelConfig) -> dict[str, dict]:
    """Compute input/output shapes and byte sizes for each kernel type."""
    D, H, S = cfg.dim, cfg.hidden_dim, cfg.seq_len
    SC = cfg.score_ch  # n_heads * seq_len
    bwd2_in = 2 * SC + 2 * D

    return {
        "sdpa_fwd": {
            "desc": "RMSNorm + QKV projections + SDPA + Wo (attention forward + taps)",
            "input": [(1, D, 1, S)],
            "output": [(1, 6 * D, 1, S)],
            "weights": ["rms1", "Wq", "Wk", "Wv", "Wo", "mask"],
        },
        "ffn_fwd": {
            "desc": "RMSNorm + SwiGLU FFN (W1/W3 gate + W2 down-project + taps)",
            "input": [(1, D, 1, S)],
            "output": [(1, 2 * D + 3 * H, 1, S)],
            "weights": ["rms2", "W1", "W2", "W3"],
        },
        "ffn_bwd": {
            "desc": "FFN backward: W2^T + SiLU_bwd + W1^T + W3^T",
            "input": [(1, D + 2 * H, 1, S)],
            "output": [(1, D + 2 * H, 1, S)],
            "weights": ["W2^T", "W1^T", "W3^T"],
        },
        "sdpa_bwd1": {
            "desc": "SDPA backward part 1: Wo^T + dV + softmax probs + dP",
            "input": [(1, 4 * D, 1, S)],
            "output": [(1, D + 2 * SC, 1, S)],
            "weights": ["Wo^T", "mask"],
        },
        "sdpa_bwd2": {
            "desc": "SDPA backward part 2: softmax grad + dQ + dK (no weights)",
            "input": [(1, bwd2_in, 1, S)],
            "output": [(1, 2 * D, 1, S)],
            "weights": [],
        },
        "qkv_bwd": {
            "desc": "QKV backward: Wq^T + Wk^T + Wv^T -> dx",
            "input": [(1, 3 * D, 1, S)],
            "output": [(1, D, 1, S)],
            "weights": ["Wq^T", "Wk^T", "Wv^T"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ANE kernel types")
    parser.add_argument(
        "--config",
        choices=list(CONFIGS),
        default="stories110m",
        help="Model config to use",
    )
    parser.add_argument(
        "--show-mil",
        choices=KERNEL_NAMES,
        default=None,
        help="Print full MIL text for a specific kernel",
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    gen = MILGenerator(cfg)

    print(f"Model: {args.config}")
    print(
        f"  dim={cfg.dim}, hidden={cfg.hidden_dim}, heads={cfg.n_heads}, "
        f"seq={cfg.seq_len}, layers={cfg.n_layers}, vocab={cfg.vocab_size}"
    )
    print(f"  head_dim={cfg.head_dim}, score_ch={cfg.score_ch}")
    print(f"  total_params={cfg.total_params:,}")

    # Kernel flow
    print("\n" + "=" * 70)
    print("Per-Step Kernel Flow (6 ANE dispatches per layer)")
    print("=" * 70)
    print("  fwd_attn -> fwd_ffn -> ffn_bwd -> sdpa_bwd1 -> sdpa_bwd2 -> qkv_bwd")

    # Kernel details
    shapes = kernel_io_shapes(cfg)
    mil_generators = {
        "sdpa_fwd": gen.sdpa_fwd_taps,
        "ffn_fwd": gen.ffn_fwd_taps,
        "ffn_bwd": gen.ffn_bwd,
        "sdpa_bwd1": gen.sdpa_bwd1,
        "sdpa_bwd2": gen.sdpa_bwd2,
        "qkv_bwd": gen.qkv_bwd,
    }

    total_mil_bytes = 0
    print(f"\n{'Kernel':<12s}  {'Input Shape':<24s}  {'Output Shape':<24s}  {'MIL Size':>10s}  Weights")
    print("-" * 100)

    for name in KERNEL_NAMES:
        info = shapes[name]
        mil = mil_generators[name]()
        mil_size = len(mil)
        total_mil_bytes += mil_size

        in_shape = str(info["input"][0])
        out_shape = str(info["output"][0])
        weights_str = ", ".join(info["weights"]) if info["weights"] else "(none)"

        print(f"{name:<12s}  {in_shape:<24s}  {out_shape:<24s}  {fmt_bytes(mil_size):>10s}  {weights_str}")

    # I/O byte sizes
    print(f"\n{'Kernel':<12s}  {'In bytes':>12s}  {'Out bytes':>12s}  {'Total I/O':>12s}")
    print("-" * 55)
    total_io = 0
    for name in KERNEL_NAMES:
        info = shapes[name]
        in_bytes = sum(np.prod(s) * 2 for s in info["input"])  # fp16
        out_bytes = sum(np.prod(s) * 2 for s in info["output"])
        total_io += in_bytes + out_bytes
        print(
            f"{name:<12s}  {fmt_bytes(in_bytes):>12s}  {fmt_bytes(out_bytes):>12s}  {fmt_bytes(in_bytes + out_bytes):>12s}"
        )

    # Compilation budget
    n_kernels = 5 * cfg.n_layers + 1  # 5 weight-bearing per layer + 1 shared sdpa_bwd2
    print("\nCompilation summary:")
    print("  Kernels per layer:  5 (sdpa_fwd, ffn_fwd, ffn_bwd, sdpa_bwd1, qkv_bwd)")
    print("  Shared kernels:     1 (sdpa_bwd2 — no weights, reused across layers)")
    print(f"  Total kernels:      {n_kernels}")
    print(f"  Total MIL text:     {fmt_bytes(total_mil_bytes)}")
    print("  ANE compile limit:  ~119 (resource leak workaround: exec() restart)")
    print(f"  Steps per compile:  {119 // n_kernels} (before exec restart needed)")

    # Causal mask blob
    mask = gen.build_causal_mask_blob()
    print(f"\nCausal mask blob: {fmt_bytes(len(mask))} (128-byte header + {cfg.seq_len}x{cfg.seq_len} fp16)")

    # Show MIL text if requested
    if args.show_mil:
        mil = mil_generators[args.show_mil]()
        print(f"\n{'=' * 70}")
        print(f"MIL program: {args.show_mil} ({fmt_bytes(len(mil))})")
        print("=" * 70)
        print(mil.decode("utf-8"))


if __name__ == "__main__":
    main()
