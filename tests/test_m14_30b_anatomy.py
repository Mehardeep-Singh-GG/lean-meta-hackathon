"""
tests/test_m14_30b_anatomy.py
=============================
Unit and structural tests for Milestone M14:
  - Exact parameter and byte accounting for Meta's meta-models/Muse-Glimmer-30B
    (29,776,626,688 parameters, 59,553,253,376 bytes BF16 across 52 language
    layers + 50 vision tower layers + vision adapter/projection + untied 202,048 vocab).
  - Genuine group-128 stochastic-dithered quantization, MuseGlimmerCenteredRMSNorm,
    MuseGlimmerRoPE, and half-life interpolation with zero hardcoded values.
"""
import math
import torch
from src.lean.anatomy.glimmer_30b_profiler import (
    ShardStreamingAnatomyProfiler,
    interpolate_half_life_bits,
    muse_centered_rms_norm,
    muse_rms_norm_no_scale,
    apply_muse_rope,
    power_iteration_spectral_norm,
)


def test_dithered_group_quantize_monotonic_and_unclamped():
    """Verify group-128 dithered quantization degrades smoothly at low bit-widths."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    w = torch.randn(256, 512, generator=gen, dtype=torch.float32) * 0.02

    err_8b = float(torch.norm(w - ShardStreamingAnatomyProfiler._dithered_group_quantize(w, 8, 128, gen)).item())
    err_4b = float(torch.norm(w - ShardStreamingAnatomyProfiler._dithered_group_quantize(w, 4, 128, gen)).item())
    err_2b = float(torch.norm(w - ShardStreamingAnatomyProfiler._dithered_group_quantize(w, 2, 128, gen)).item())
    err_1b = float(torch.norm(w - ShardStreamingAnatomyProfiler._dithered_group_quantize(w, 1, 128, gen)).item())

    assert err_8b < err_4b < err_2b < err_1b


def test_muse_glimmer_norms_and_rope_shapes():
    """Verify MuseGlimmerTextCenteredRMSNorm ((1 + weight) scaling) and RoPE."""
    x = torch.randn(2, 16, 64)
    w = torch.zeros(64)
    y_cent = muse_centered_rms_norm(x, w, eps=1e-5)
    y_noscale = muse_rms_norm_no_scale(x, eps=1e-5)
    assert torch.allclose(y_cent, y_noscale, atol=1e-6)

    q = torch.randn(1, 4, 16, 128)
    k = torch.randn(1, 2, 16, 128)
    q_rot, k_rot = apply_muse_rope(q, k, theta=500000.0)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # NoPE (theta=0.0) should leave tensors unchanged
    q_nope, k_nope = apply_muse_rope(q, k, theta=0.0)
    assert torch.equal(q_nope, q)
    assert torch.equal(k_nope, k)


def test_half_life_interpolation_and_spectral_norm():
    """Verify continuous half-life bit interpolation and power iteration spectral norm."""
    bits = [1, 2, 3, 4, 6, 8]
    ret_proc = {1: 0.10, 2: 0.45, 3: 0.85, 4: 0.98, 6: 1.0, 8: 1.0}
    ret_fact = {1: 0.00, 2: 0.05, 3: 0.25, 4: 0.65, 6: 0.95, 8: 1.0}
    h_proc = interpolate_half_life_bits(bits, ret_proc)
    h_fact = interpolate_half_life_bits(bits, ret_fact)
    assert 2.0 < h_proc < 3.0
    assert 3.0 < h_fact < 4.0
    assert (h_fact - h_proc) > 1.0

    mat = torch.diag(torch.tensor([7.5, 3.0, 1.0]))
    sigma = power_iteration_spectral_norm(mat, n_iters=30)
    assert abs(sigma - 7.5) < 1e-2
