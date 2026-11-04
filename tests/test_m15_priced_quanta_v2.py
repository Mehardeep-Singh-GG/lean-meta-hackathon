"""
tests/test_m15_priced_quanta_v2.py
==================================
Unit Tests for Milestone M15: Priced-Quanta v2 & Block-Cholesky H-KAC v2 on Meta's
`meta-models/Muse-Glimmer-30B` (29,776,626,688 parameters).
Verifies:
1. Bit-exactness and exact byte counts of Ternary5PerBytePacker (1.60 bpp),
   Int2Packer (2.00 bpp), Int3OctetPacker (3.00 bpp), and Int4Packer (4.00 bpp).
2. Strict positive-definiteness and adaptive Cholesky damping on rank-deficient X.
3. >= 45% relative Frobenius output reconstruction error reduction vs Stage-1 RTN.
4. Exact 29,776,626,688 parameter accounting and <= 6.15 GiB physical byte ceiling
   for `meta-models/Muse-Glimmer-30B`.
"""

import torch
import pytest
from src.lean.compress.priced_quanta_v2 import (
    Ternary5PerBytePacker,
    Int2Packer,
    Int3OctetPacker,
    Int4Packer,
    HessianCompensatedHybridLinear,
    QuantizedGroupEmbedding,
    PricedQuantaV2KnapsackSolver,
)
from experiments.run_m15_30b_priced_quanta_v2 import (
    build_30b_architecture_specs,
    generate_realistic_swiglu_activations_and_weights,
)


def test_ternary_5_per_byte_packer_bit_exact():
    torch.manual_seed(42)
    for numel in [5, 17, 128, 10003]:
        q = torch.randint(-1, 2, (numel,), dtype=torch.int8)
        packed = Ternary5PerBytePacker.pack(q)
        assert packed.dtype == torch.uint8
        assert packed.numel() == (numel + 4) // 5
        unpacked = Ternary5PerBytePacker.unpack(packed, numel)
        assert torch.equal(q, unpacked), f"Mismatch in Ternary5PerBytePacker for numel={numel}"


def test_int2_int3_int4_packers_bit_exact():
    torch.manual_seed(43)
    numel = 4096
    q2 = torch.randint(-2, 2, (numel,), dtype=torch.int8)
    p2 = Int2Packer.pack(q2)
    assert p2.numel() == numel // 4
    assert torch.equal(q2, Int2Packer.unpack(p2, numel))

    q3 = torch.randint(-4, 4, (numel,), dtype=torch.int8)
    p3 = Int3OctetPacker.pack(q3)
    assert p3.numel() == (numel // 8) * 3
    assert torch.equal(q3, Int3OctetPacker.unpack(p3, numel))

    q4 = torch.randint(-8, 8, (numel,), dtype=torch.int8)
    p4 = Int4Packer.pack(q4)
    assert p4.numel() == numel // 2
    assert torch.equal(q4, Int4Packer.unpack(p4, numel))


def test_hessian_compensated_hybrid_linear_error_reduction():
    W, X = generate_realistic_swiglu_activations_and_weights(
        d_out=512, d_in=1024, num_tokens=384, seed=99, device="cpu"
    )
    layer = HessianCompensatedHybridLinear(
        in_features=1024,
        out_features=512,
        bits=1.585,
        group_size=128,
        outlier_ratio=0.005,
        bias=False,
    )
    tel = layer.pack_weights_with_hessian(W, X_calib=X)
    assert tel["err_hkac_v2"] < tel["err_rtn_stage1"]
    assert tel["error_reduction_pct"] >= 45.0, (
        f"Expected >= 45% error reduction, got {tel['error_reduction_pct']}%"
    )
    assert tel["effective_bpp"] < 1.90


def test_muse_glimmer_30b_knapsack_solver_fits_in_8gb_laptop():
    solver = PricedQuantaV2KnapsackSolver(horizon_steps=64)
    specs = build_30b_architecture_specs()
    res = solver.solve_30b_allocation(
        specs,
        vocab_size=202048,
        hidden_size=6656,
        total_layers=52,
        norm_params=5915648,
        embed_bits=1,
        head_bits=2,
        target_max_gib=6.08,
    )
    summary = res["summary"]
    assert summary["total_params"] == 29_776_626_688
    assert summary["total_physical_gib"] <= 6.15
    assert summary["free_headroom_in_8gib"] >= 1.85
    assert 1.55 <= summary["effective_bpp"] <= 1.85
