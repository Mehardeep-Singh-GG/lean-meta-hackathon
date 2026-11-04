"""
Unit Tests for Milestone M18:
LEAN 8 GB Edge Runtime Engine, Tiled Zero-Spike Bit-Unpacking & Memory-Mapped `.lean30b` Streaming.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.lean.infer.edge_8gb_runtime import (
    Lean30BPackageManager,
    QuantizedKVCache,
    StaticTileWorkspace,
    TiledHybridQuantizedLinear,
)


def test_static_tile_workspace_and_tiled_hybrid_linear() -> None:
    dev = torch.device("cpu")
    ws = StaticTileWorkspace.get_workspace(device=dev, dtype=torch.float16)
    assert ws.buffer_bytes == 20 * 1024 * 1024  # 20.00 MiB exact

    for bits in [1.58, 2.0, 3.0, 4.0]:
        w_ref = torch.randn((256, 256), dtype=torch.float32) * 0.05
        lin = TiledHybridQuantizedLinear(
            d_in=256, d_out=256, bitwidth=bits, group_size=128, wrmf_rank=8, tile_out=128, device=dev
        )
        lin.pack_from_float_weights(w_ref)
        x = torch.randn((4, 256), dtype=torch.float16)
        y = lin(x)
        assert y.shape == (4, 256)
        assert not torch.isnan(y).any()


def test_quantized_kv_cache_and_lean30b_package(tmp_path: Path) -> None:
    dev = torch.device("cpu")
    kv = QuantizedKVCache(num_layers=4, num_kv_heads=4, head_dim=64, max_seq_len=128, group_size=64, device=dev)
    q_in = torch.randn((1, 64, 4, 64), dtype=torch.float16)
    k_in = torch.randn((1, 64, 4, 64), dtype=torch.float16)
    v_in = torch.randn((1, 64, 4, 64), dtype=torch.float16)
    for l_idx in range(4):
        kv.update(l_idx, k_in, v_in)

    k_out, v_out = kv.dequantize_layer(0, active_len=64)
    assert k_out.shape == (1, 64, 4, 64)
    attn_ref = torch.nn.functional.scaled_dot_product_attention(
        q_in.transpose(1, 2), k_in.transpose(1, 2), v_in.transpose(1, 2)
    )
    attn_hat = torch.nn.functional.scaled_dot_product_attention(
        q_in.transpose(1, 2), k_out.transpose(1, 2), v_out.transpose(1, 2)
    )
    rel_err = float(torch.norm(attn_hat.float() - attn_ref.float()) / torch.norm(attn_ref.float()))
    assert rel_err < 0.018

    pkg_dir = tmp_path / "test.lean30b"
    meta = Lean30BPackageManager.export_package(
        pkg_dir=pkg_dir,
        state_tensors={"w": torch.ones((16, 16), dtype=torch.uint8)},
        config={"model_id": "meta-models/Muse-Glimmer-30B", "layers": 52},
    )
    assert meta["pocket_index_bytes"] == 18432
    tensors, cfg, load_sec = Lean30BPackageManager.load_package_mmap(pkg_dir)
    assert "w" in tensors
    assert cfg["layers"] == 52
    assert load_sec < 2.0
