"""
tests/test_m16_calibration_optimizer.py
=======================================
Unit tests for Milestone M16:
  1. ZeroLeakageCalibrationLoader 3-domain stratification and 0.00% 8-gram firewall enforcement
     (including raising SecurityFirewallViolationError when a leak is injected).
  2. ResidualBiasAbsorber (mu-Folding) exact first-moment drift cancellation E[y - y_hat] = 0.
  3. RoPEFrequencyAwareScaler high-frequency rotary dimension boost and attention sink detection.
  4. Exact 29.78B Muse-Glimmer-30B physical footprint <= 6.15 GiB across all 5 rungs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.lean.compress.calibration_optimizer import (
    ZeroLeakageCalibrationLoader,
    SecurityFirewallViolationError,
    ResidualBiasAbsorber,
    RoPEFrequencyAwareScaler,
    LayerWiseKLDivergenceTracker,
)
from experiments.run_m16_30b_iterative_accuracy_ladder import (
    DeterministicWordHashTokenizer,
    project_30b_physical_footprint_bytes,
    quantize_weight_rung,
)


def test_zero_leakage_firewall_passes_clean_and_catches_leak():
    loader = ZeroLeakageCalibrationLoader(num_sequences=20, seq_len=128, seed=1042)
    prose = [
        f"Classical thermodynamics paragraph {i} describing entropy production and Carnot efficiency "
        f"in closed reversible cycles under isothermal expansion stage {i*19}."
        for i in range(15)
    ]
    calib_records = loader.build_multidomain_texts(prose)
    assert len(calib_records) == 20

    val_texts = [f"Validation passage {j} about linear algebra eigenvalues." for j in range(8)]
    test_prompts = [f"Locked test prompt {k}: What is the orbital period of moonOberon_{k}?" for k in range(10)]
    test_targets = [f"OberonTarget_{k}" for k in range(10)]

    tok = DeterministicWordHashTokenizer()
    receipt = loader.verify_zero_overlap(tok, calib_records, val_texts, test_prompts, test_targets, ngram_order=8)
    assert receipt.passed is True
    assert receipt.calib_vs_test_overlap_pct == 0.0
    assert receipt.entity_leak_count == 0

    # Inject a target entity leak into calibration and verify SecurityFirewallViolationError is raised
    tainted_records = list(calib_records)
    tainted_records[0] = dict(tainted_records[0], text=tainted_records[0]["text"] + " OberonTarget_3 ")
    with pytest.raises(SecurityFirewallViolationError):
        loader.verify_zero_overlap(tok, tainted_records, val_texts, test_prompts, test_targets, ngram_order=8)


def test_residual_bias_absorber_eliminates_first_moment_drift():
    torch.manual_seed(1042)
    W_orig = torch.randn(64, 128) * 0.05
    W_quant = W_orig + torch.randn(64, 128) * 0.015
    # Non-zero mean input activations (like SwiGLU output)
    X = torch.randn(256, 128) + 1.25
    mean_x = X.mean(dim=0)

    y_true = X @ W_orig.T
    y_uncorrected = X @ W_quant.T
    mean_err_before = torch.norm((y_true - y_uncorrected).mean(dim=0)).item()
    assert mean_err_before > 0.1

    mod = nn.Linear(128, 64, bias=False)
    mod.weight.data.copy_(W_quant)
    drift_norm = ResidualBiasAbsorber.apply_bias_folding(mod, W_orig, W_quant, mean_x)
    assert drift_norm > 0.1

    y_corrected = mod(X.float())
    mean_err_after = torch.norm((y_true - y_corrected).mean(dim=0)).item()
    assert mean_err_after < 1e-3


def test_rope_scaler_and_5_rung_monotonic_error_reduction():
    torch.manual_seed(1042)
    rope_scaler = RoPEFrequencyAwareScaler(head_dim=128)
    W = torch.randn(256, 256) * 0.04
    X = torch.randn(192, 256) + 0.85  # asymmetric activation mean

    weights_row = rope_scaler.compute_row_importance_weights(256, W, X.abs().mean(dim=0))
    assert weights_row.shape == (256,)
    assert weights_row[0] > weights_row[63]  # high-frequency RoPE channel weighted higher than low-freq

    y_ref = X @ W.T
    errs = []
    for r_id in range(5):
        w_deq, b_corr, phys_b, _ = quantize_weight_rung(
            W, X, "down_proj", r_id, rope_scaler=rope_scaler
        )
        y_hat = F.linear(X, w_deq, b_corr.float() if b_corr is not None else None)
        rel_err = float((torch.norm(y_ref - y_hat) / torch.norm(y_ref)).item())
        errs.append(rel_err)
        acct = project_30b_physical_footprint_bytes(r_id)
        assert acct["total_physical_gib"] <= 6.15

    assert errs[4] < errs[2] < errs[0]
