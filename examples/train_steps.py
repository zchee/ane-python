#!/usr/bin/env python3
"""Training loop example: run forward+backward steps on ANE and watch loss decrease.

Demonstrates:
  1. Create or load model weights
  2. Initialize the Trainer (compiles all 6 kernel types per layer)
  3. Run training steps with random token batches
  4. Display loss progression and compile budget

This uses the full static pipeline: 6 ANE kernel dispatches per layer per step,
CPU RMSNorm backward, gradient accumulation via numpy/Accelerate, and Adam updates.

Requirements:
  - macOS 15+ on Apple Silicon (M-series)
  - Built and installed `ane` package (`uv pip install -e .`)

Usage:
  uv run python examples/train_steps.py
  uv run python examples/train_steps.py --steps 50 --lr 3e-4
"""

import argparse
import sys
import time

import numpy as np
from ane.config import ModelConfig
from ane.trainer import Trainer
from ane.weights import LayerWeights

import ane


def make_tiny_model(cfg: ModelConfig, rng: np.random.Generator) -> tuple[list[LayerWeights], np.ndarray, np.ndarray]:
    """Create random weights for a tiny transformer model."""
    scale = 0.02
    layers = []
    for _ in range(cfg.n_layers):
        layers.append(
            LayerWeights(
                Wq=rng.standard_normal((cfg.dim, cfg.dim), dtype=np.float32) * scale,
                Wk=rng.standard_normal((cfg.dim, cfg.dim), dtype=np.float32) * scale,
                Wv=rng.standard_normal((cfg.dim, cfg.dim), dtype=np.float32) * scale,
                Wo=rng.standard_normal((cfg.dim, cfg.dim), dtype=np.float32) * scale,
                W1=rng.standard_normal((cfg.hidden_dim, cfg.dim), dtype=np.float32) * scale,
                W2=rng.standard_normal((cfg.dim, cfg.hidden_dim), dtype=np.float32) * scale,
                W3=rng.standard_normal((cfg.hidden_dim, cfg.dim), dtype=np.float32) * scale,
                rms_att=np.ones(cfg.dim, dtype=np.float32),
                rms_ffn=np.ones(cfg.dim, dtype=np.float32),
            )
        )
    rms_final = np.ones(cfg.dim, dtype=np.float32)
    embed = rng.standard_normal((cfg.vocab_size, cfg.dim), dtype=np.float32) * scale
    return layers, rms_final, embed


def main() -> None:
    parser = argparse.ArgumentParser(description="ANE training step demo")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--accum", type=int, default=5, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Use a small config that still runs on ANE.
    # ANE requires seq_len >= 32 for eval to succeed.
    cfg = ModelConfig(
        dim=128,
        hidden_dim=256,
        n_heads=4,
        seq_len=32,
        n_layers=2,
        vocab_size=256,
    )

    print(
        f"Config: dim={cfg.dim}, hidden={cfg.hidden_dim}, heads={cfg.n_heads}, "
        f"seq={cfg.seq_len}, layers={cfg.n_layers}, vocab={cfg.vocab_size}"
    )
    print(f"Total params: {cfg.total_params:,}")
    print(f"Training: {args.steps} steps, lr={args.lr}, accum={args.accum}")

    # Step 1: Initialize ANE
    print("\n[1] Initializing ANE runtime...")
    try:
        ane.init()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Create model
    print("[2] Creating random model weights...")
    rng = np.random.default_rng(args.seed)
    layers, rms_final, embed = make_tiny_model(cfg, rng)

    # Step 3: Initialize trainer
    print("[3] Initializing trainer...")
    trainer = Trainer(
        layer_weights=layers,
        embed=embed,
        rms_final=rms_final,
        cfg=cfg,
        lr=args.lr,
        accum_steps=args.accum,
    )
    print(f"    Compile budget: {trainer.compile_budget_remaining} kernels")

    # Step 4: Compile all kernels
    needed = 5 * cfg.n_layers + 1
    print(f"[4] Compiling {needed} ANE kernels ({5} per layer + 1 shared)...")
    t0 = time.perf_counter()
    try:
        trainer.compile_kernels()
    except RuntimeError as e:
        print(f"ERROR: Kernel compilation failed: {e}", file=sys.stderr)
        sys.exit(1)
    compile_time = time.perf_counter() - t0
    print(f"    Compiled in {compile_time:.2f}s")
    print(f"    Budget remaining: {trainer.compile_budget_remaining} kernels")

    # Step 5: Training loop
    print(f"\n[5] Running {args.steps} training steps...\n")
    print(f"{'Step':>6s}  {'Loss':>10s}  {'Time (ms)':>10s}")
    print("-" * 32)

    losses = []
    with trainer:
        for step in range(1, args.steps + 1):
            # Random token batch (in real training, these come from dataset)
            tokens = rng.integers(0, cfg.vocab_size, size=cfg.seq_len, dtype=np.uint16)
            targets = rng.integers(0, cfg.vocab_size, size=cfg.seq_len, dtype=np.uint16)

            t_step = time.perf_counter()
            loss = trainer.train_step(tokens, targets)
            dt = (time.perf_counter() - t_step) * 1000

            losses.append(loss)

            if step <= 5 or step % 5 == 0 or step == args.steps:
                print(f"{step:6d}  {loss:10.4f}  {dt:10.1f}")

    # Summary
    print("\nTraining complete.")
    if len(losses) >= 5:
        first5 = np.mean(losses[:5])
        last5 = np.mean(losses[-5:])
        print(f"  Mean loss (first 5): {first5:.4f}")
        print(f"  Mean loss (last 5):  {last5:.4f}")
        if last5 < first5:
            print(f"  Loss decreased by {first5 - last5:.4f}")
        else:
            print("  Loss did not decrease (random tokens — expected with short runs)")


if __name__ == "__main__":
    main()
