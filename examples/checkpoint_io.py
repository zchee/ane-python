#!/usr/bin/env python3
"""Checkpoint save/load example: persist and restore training state.

Demonstrates:
  1. Create model weights and optimizer state
  2. Save a training checkpoint in binary format (matching the C implementation)
  3. Load the checkpoint back and verify integrity
  4. Inspect checkpoint header fields
  5. Show how to resume training from a checkpoint

The checkpoint format is binary-compatible with the C trainer (train_large.m),
so checkpoints saved by Python can be loaded in C and vice versa.

No ANE hardware needed — this is pure numpy I/O.

Usage:
  uv run python examples/checkpoint_io.py
  uv run python examples/checkpoint_io.py --output /tmp/my_checkpoint.ckpt
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from ane.config import ModelConfig
from ane.model import CheckpointHeader, load_checkpoint, save_checkpoint
from ane.weights import AdamState, LayerAdam, LayerWeights


def make_dummy_state(cfg: ModelConfig, rng: np.random.Generator) -> dict:
    """Create random model weights and optimizer state for demonstration."""
    scale = 0.02

    layers = []
    layer_adam = []
    for _ in range(cfg.n_layers):
        lw = LayerWeights(
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
        layers.append(lw)
        layer_adam.append(LayerAdam.zeros(cfg))

    rms_final = np.ones(cfg.dim, dtype=np.float32)
    adam_rms_final = AdamState.zeros_like((cfg.dim,))
    embed = rng.standard_normal((cfg.vocab_size, cfg.dim), dtype=np.float32) * scale
    adam_embed = AdamState.zeros_like((cfg.vocab_size, cfg.dim))

    return {
        "layer_weights": layers,
        "layer_adam": layer_adam,
        "rms_final": rms_final,
        "adam_rms_final": adam_rms_final,
        "embed": embed,
        "adam_embed": adam_embed,
    }


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint save/load demo")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: temp file)")
    args = parser.parse_args()

    cfg = ModelConfig(
        dim=64,
        hidden_dim=128,
        n_heads=4,
        seq_len=16,
        n_layers=2,
        vocab_size=256,
    )

    rng = np.random.default_rng(42)
    state = make_dummy_state(cfg, rng)

    print(f"Model: dim={cfg.dim}, hidden={cfg.hidden_dim}, layers={cfg.n_layers}, vocab={cfg.vocab_size}")
    print(f"Total params: {cfg.total_params:,}")

    # -- Save checkpoint --
    if args.output:
        ckpt_path = Path(args.output)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False)
        ckpt_path = Path(tmp.name)
        tmp.close()

    print(f"\n[1] Saving checkpoint to {ckpt_path}...")
    save_checkpoint(
        ckpt_path,
        step=42,
        total_steps=1000,
        lr=1e-4,
        loss=3.14,
        adam_t=42,
        cum_compile=1.5,
        cum_train=120.0,
        cum_wall=180.0,
        cum_steps=42,
        cum_batches=420,
        cfg=cfg,
        **state,
    )
    file_size = ckpt_path.stat().st_size
    print(f"    Saved: {fmt_bytes(file_size)}")

    # -- Checkpoint file layout --
    header_size = CheckpointHeader._STRUCT_SIZE
    weight_params = sum(
        w.size
        for lw in state["layer_weights"]
        for w in [lw.Wq, lw.Wk, lw.Wv, lw.Wo, lw.W1, lw.W2, lw.W3, lw.rms_att, lw.rms_ffn]
    )
    adam_params = weight_params * 2  # m + v per weight
    print("\n    File layout:")
    print(f"      Header:         {header_size} bytes")
    print(f"      Per-layer data: {cfg.n_layers} layers x (weights + adam state)")
    print(f"      Weight params:  {weight_params:,} floats ({fmt_bytes(weight_params * 4)})")
    print(f"      Adam params:    {adam_params:,} floats ({fmt_bytes(adam_params * 4)})")

    # -- Load checkpoint --
    print(f"\n[2] Loading checkpoint from {ckpt_path}...")
    loaded = load_checkpoint(ckpt_path, cfg=cfg)
    hdr = loaded["header"]

    print("\n    Header fields:")
    print(f"      magic:       0x{hdr.magic:08X} ({'OK' if hdr.magic == 0x424C5A54 else 'BAD'})")
    print(f"      version:     {hdr.version}")
    print(f"      step:        {hdr.step} / {hdr.total_steps}")
    print(f"      lr:          {hdr.lr}")
    print(f"      loss:        {hdr.loss:.4f}")
    print(f"      adam_t:      {hdr.adam_t}")
    print(f"      cum_compile: {hdr.cum_compile:.1f}s")
    print(f"      cum_train:   {hdr.cum_train:.1f}s")
    print(f"      cum_wall:    {hdr.cum_wall:.1f}s")
    print(f"      cum_steps:   {hdr.cum_steps}")
    print(f"      cum_batches: {hdr.cum_batches}")
    print(
        f"      model:       dim={hdr.dim} hidden={hdr.hidden_dim} heads={hdr.n_heads} "
        f"seq={hdr.seq_len} layers={hdr.n_layers} vocab={hdr.vocab_size}"
    )

    # -- Verify roundtrip --
    print("\n[3] Verifying roundtrip integrity...")
    ok = True
    for i in range(cfg.n_layers):
        for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "W3", "rms_att", "rms_ffn"):
            orig = getattr(state["layer_weights"][i], name)
            restored = getattr(loaded["layer_weights"][i], name)
            if not np.array_equal(orig, restored.reshape(orig.shape)):
                print(f"    MISMATCH: layer {i} {name}")
                ok = False

    if np.array_equal(state["rms_final"], loaded["rms_final"]):
        pass
    else:
        print("    MISMATCH: rms_final")
        ok = False

    orig_embed = state["embed"]
    loaded_embed = loaded["embed"]
    if np.array_equal(orig_embed, loaded_embed):
        pass
    else:
        print("    MISMATCH: embed")
        ok = False

    if ok:
        print("    All weights match — roundtrip verified.")
    else:
        print("    WARNING: Some weights did not match!")

    # -- Weight statistics --
    print("\n[4] Weight statistics (layer 0):")
    lw = loaded["layer_weights"][0]
    for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "W3", "rms_att", "rms_ffn"):
        w = getattr(lw, name)
        print(
            f"    {name:8s}  shape={w.shape!s:>16s}  "
            f"mean={w.mean():+.6f}  std={w.std():.6f}  "
            f"min={w.min():+.4f}  max={w.max():+.4f}"
        )

    # Cleanup temp file if we created one
    if not args.output:
        ckpt_path.unlink()
        print("\n    Temp file cleaned up.")

    print("\nDone.")


if __name__ == "__main__":
    main()
