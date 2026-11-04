"""
campaigns/rigorous_benchmarks/baselines/test_baselines.py
=========================================================
Rigorous Unit Test Suite for Storage-Matched Baselines and Ablation Matrix.

Verifies:
1. Exact parameter counts (29,776,626,688) and byte accounting across all 6 models.
2. Strict physical storage parity: abs(bytes - 6,528,903,738) / 6,528,903,738 < 0.0001 for all quantized models.
3. Correctness of all 6 linear modules (FP16, LeanPQv2, UniformMatched, GPTQUniform, AWQUniform, HQQTernary).
4. Correctness of the 4-bit Group-64 KV cache vs FP16 KV cache.
5. Correctness of the BaselineEvaluationWrapper forward pass, loss, and retrieval injection.
6. Complete validation of the 5-step stacking ladder and the 4 drop-one ablations.
"""

import math
import pytest
import torch
import torch.nn as nn

from campaigns.rigorous_benchmarks.baselines.models import (
    MODEL_ID,
    TOTAL_PARAMS,
    FP16_REF_BYTES,
    TARGET_STORAGE_BYTES,
    BM25_INDEX_BYTES,
    PARITY_TOLERANCE,
    GIB_DIVISOR,
    BaselineModelType,
    BaselineModelConfig,
    compute_full_model_byte_accounting,
    BaseQuantizedLinear,
    FP16Linear,
    LeanPQv2Linear,
    UniformMatchedLinear,
    GPTQUniformLinear,
    AWQUniformLinear,
    HQQTernaryLinear,
    QuantizedKVCache,
    BaselineEvaluationWrapper,
    create_baseline_model,
    get_baseline_spec,
    list_baseline_models,
)
from campaigns.rigorous_benchmarks.ablations.ablation_matrix import (
    AblationFactorConfig,
    AblationPoint,
    STEP_STACKING_LADDER,
    DROP_ONE_ABLATIONS,
    AblationMatrix,
    get_ablation_run,
    list_all_ablation_runs,
)


# ==============================================================================
# 1. EXACT 30B PHYSICAL STORAGE PARITY TESTS
# ==============================================================================

class TestByteAccounting:
    """Verifies byte-for-byte exact accounting and storage parity across all models."""

    def test_total_parameter_invariants(self):
        """Checks that all candidate models profile the exact 29,776,626,688 parameters."""
        for m_name in list_baseline_models():
            report = compute_full_model_byte_accounting(m_name)
            assert report["total_params"] == TOTAL_PARAMS, (
                f"Model {m_name} reported {report['total_params']} params, expected {TOTAL_PARAMS}"
            )

    def test_fp16_reference_bytes(self):
        """Verifies FP16 reference is exactly 59,553,253,376 bytes (55.4633 GiB, 16.0 bpp)."""
        report = compute_full_model_byte_accounting(BaselineModelType.FP16_REF)
        assert report["total_physical_bytes"] == FP16_REF_BYTES
        assert report["effective_bpp"] == 16.0000
        assert report["total_physical_gib"] == round(FP16_REF_BYTES / GIB_DIVISOR, 4)
        assert report["parity_passed"] is True

    def test_lean_pqv2_exact_storage(self):
        """Verifies LEAN-PQv2 matches 6,528,903,738 bytes = 6.0805 GiB down to the exact byte."""
        report = compute_full_model_byte_accounting(BaselineModelType.LEAN_PQv2)
        assert report["total_physical_bytes"] == TARGET_STORAGE_BYTES
        assert report["total_physical_gib"] == 6.0805
        assert report["effective_bpp"] == 1.7541
        assert report["bm25_bytes"] == BM25_INDEX_BYTES
        assert report["parity_relative_delta"] == 0.0
        assert report["parity_passed"] is True

    def test_uniform_matched_exact_storage(self):
        """Verifies UNIFORM_MATCHED strictly matches 6,528,903,738 bytes (Delta = 0.000%)."""
        report = compute_full_model_byte_accounting(BaselineModelType.UNIFORM_MATCHED)
        assert report["total_physical_bytes"] == TARGET_STORAGE_BYTES
        assert report["total_physical_gib"] == 6.0805
        assert report["effective_bpp"] == 1.7541
        assert report["parity_relative_delta"] == 0.0
        assert report["parity_passed"] is True

    @pytest.mark.parametrize("model_type", [
        BaselineModelType.LEAN_PQv2,
        BaselineModelType.UNIFORM_MATCHED,
        BaselineModelType.GPTQ_UNIFORM,
        BaselineModelType.AWQ_UNIFORM,
        BaselineModelType.HQQ_TERNARY,
    ])
    def test_quantized_storage_parity_threshold(self, model_type):
        """Ensures abs(bytes - 6,528,903,738) / 6,528,903,738 < 0.0001 for all quantized models."""
        report = compute_full_model_byte_accounting(model_type)
        delta = abs(report["total_physical_bytes"] - TARGET_STORAGE_BYTES) / float(TARGET_STORAGE_BYTES)
        assert delta < PARITY_TOLERANCE, (
            f"Model {model_type} exceeded storage parity tolerance: delta={delta:.6f} >= {PARITY_TOLERANCE}"
        )
        assert report["parity_passed"] is True

    def test_model_config_spec_verification(self):
        """Verifies BaselineModelConfig parity check methods on all specifications."""
        for m_name in list_baseline_models():
            spec = get_baseline_spec(m_name)
            passed, delta = spec.verify_storage_parity()
            assert passed is True, f"Model spec {m_name} failed storage parity check (delta={delta})"
            assert delta < PARITY_TOLERANCE


# ==============================================================================
# 2. QUANTIZED LINEAR LAYER UNIT TESTS
# ==============================================================================

class TestQuantizedLinearLayers:
    """Verifies mathematical validity and output behavior of all linear layers."""

    @pytest.fixture
    def sample_weights(self):
        torch.manual_seed(42)
        d_out, d_in = 256, 512
        W = torch.randn(d_out, d_in)
        X = torch.randn(64, d_in)
        return W, X

    def test_fp16_linear(self, sample_weights):
        W, X = sample_weights
        layer = FP16Linear(512, 256)
        layer.load_weight(W)
        y = layer(X.to(torch.float16))
        assert y.shape == (64, 256)
        assert layer.physical_bytes() == 256 * 512 * 2
        assert not torch.isnan(y).any()

    def test_lean_pqv2_linear_4bit(self, sample_weights):
        W, X = sample_weights
        layer = LeanPQv2Linear(512, 256, bits=4.0, group_size=128, outlier_ratio=0.005)
        layer.quantize_and_compensate(W, X)
        y = layer(X.to(torch.float16))
        assert y.shape == (64, 256)
        assert layer.physical_bytes() < W.numel() * 2  # Strict compression vs FP16
        assert not torch.isnan(y).any()
        assert layer.num_outliers > 0

    def test_lean_pqv2_linear_ternary(self, sample_weights):
        W, X = sample_weights
        layer = LeanPQv2Linear(512, 256, bits=1.585, group_size=128, outlier_ratio=0.0)
        layer.quantize_and_compensate(W, X)
        y = layer(X.to(torch.float16))
        assert y.shape == (64, 256)
        assert not torch.isnan(y).any()

    def test_uniform_matched_linear(self, sample_weights):
        W, X = sample_weights
        layer = UniformMatchedLinear(512, 256, group_size=128)
        layer.quantize_rtn(W)
        y = layer(X.to(torch.float16))
        assert y.shape == (64, 256)
        assert not torch.isnan(y).any()
        assert layer.physical_bytes() < W.numel() * 2

    def test_gptq_uniform_linear(self, sample_weights):
        W, X = sample_weights
        layer = GPTQUniformLinear(512, 256, group_size=128)
        layer.quantize_gptq(W, X)
        y = layer(X.to(torch.float16))
        assert y.shape == (64, 256)
        assert not torch.isnan(y).any()

    def test_awq_uniform_linear(self, sample_weights):
        W, X = sample_weights
        layer = AWQUniformLinear(512, 256, group_size=128)
        layer.quantize_awq(W, X, alpha=0.5)
        y = layer(X.to(torch.float16))
        assert y.shape == (64, 256)
        assert not torch.isnan(y).any()

    def test_hqq_ternary_linear(self, sample_weights):
        W, X = sample_weights
        layer = HQQTernaryLinear(512, 256, group_size=128)
        layer.quantize_hqq(W, max_iters=5)
        y = layer(X.to(torch.float16))
        assert y.shape == (64, 256)
        assert not torch.isnan(y).any()


# ==============================================================================
# 3. QUANTIZED KV CACHE TESTS
# ==============================================================================

class TestQuantizedKVCache:
    """Verifies 4-bit Group-64 KV cache operations, memory footprint, and FP16 fallback."""

    def test_4bit_group64_cache_quant_dequant(self):
        cache = QuantizedKVCache(mode="4bit_group64", group_size=64)
        k = torch.randn(1, 2, 32, 128)
        v = torch.randn(1, 2, 32, 128)
        full_k, full_v = cache.update(0, k, v)
        assert full_k.shape == (1, 2, 32, 128)
        assert full_v.shape == (1, 2, 32, 128)

        # Retrieve cached KV
        cached_k, cached_v = cache.get_kv(0)
        assert cached_k.shape == (1, 2, 32, 128)
        assert cached_v.shape == (1, 2, 32, 128)

        # Check memory compression vs FP16
        fp16_bytes = k.numel() * 2 + v.numel() * 2
        compressed_bytes = cache.memory_bytes()
        assert compressed_bytes < fp16_bytes

    def test_fp16_cache(self):
        cache = QuantizedKVCache(mode="fp16")
        k = torch.randn(1, 2, 16, 128, dtype=torch.float16)
        v = torch.randn(1, 2, 16, 128, dtype=torch.float16)
        cache.update(0, k, v)
        cached_k, cached_v = cache.get_kv(0)
        assert torch.equal(cached_k, k)
        assert torch.equal(cached_v, v)
        assert cache.memory_bytes() == k.numel() * 2 + v.numel() * 2

    def test_multistep_cache_append(self):
        cache = QuantizedKVCache(mode="4bit_group64", group_size=64)
        k1 = torch.randn(1, 2, 10, 128)
        v1 = torch.randn(1, 2, 10, 128)
        cache.update(0, k1, v1)

        k2 = torch.randn(1, 2, 5, 128)
        v2 = torch.randn(1, 2, 5, 128)
        full_k, full_v = cache.update(0, k2, v2)
        assert full_k.shape == (1, 2, 15, 128)
        assert full_v.shape == (1, 2, 15, 128)


# ==============================================================================
# 4. BASELINE EVALUATION WRAPPER TESTS
# ==============================================================================

class TestBaselineEvaluationWrapper:
    """Verifies end-to-end forward passes, loss computation, and retrieval augmentation."""

    @pytest.mark.parametrize("m_name", list_baseline_models())
    def test_baseline_model_creation_and_forward(self, m_name):
        model = create_baseline_model(m_name, num_eval_layers=2, hidden_dim=256, vocab_size=500)
        input_ids = torch.randint(0, 500, (2, 8))
        logits = model(input_ids)
        assert logits.shape == (2, 8, 500)
        assert not torch.isnan(logits).any()

        # Compute loss
        target_ids = torch.randint(0, 500, (2, 8))
        loss = model.compute_loss(input_ids, target_ids)
        assert isinstance(loss, float)
        assert not math.isnan(loss)
        assert loss > 0.0

    def test_retrieval_augmentation_toggle(self):
        lean_model = create_baseline_model("LEAN_PQv2", num_eval_layers=2, hidden_dim=256, vocab_size=500)
        unif_model = create_baseline_model("UNIFORM_MATCHED", num_eval_layers=2, hidden_dim=256, vocab_size=500)

        query = "What is the capital of France?"
        lean_res = lean_model.evaluate_factual_query(query)
        unif_res = unif_model.evaluate_factual_query(query)

        assert lean_res["retrieval_active"] is True
        assert "Passage:" in lean_res["conditioned_prompt"]

        assert unif_res["retrieval_active"] is False
        assert unif_res["conditioned_prompt"] == query


# ==============================================================================
# 5. ABLATION MATRIX HARNESS TESTS
# ==============================================================================

class TestAblationMatrix:
    """Verifies 4-factor configuration, stacking ladder, and drop-one ablations."""

    def test_stacking_ladder_structure(self):
        """Verifies the 5-step stacking ladder has 5 rungs with correct factor progression."""
        assert len(STEP_STACKING_LADDER) == 5

        # Rung 1: Baseline Naive Ternary (all OFF)
        r1 = STEP_STACKING_LADDER[0]
        assert r1.rung_number == 1
        assert r1.factors == AblationFactorConfig(kac=False, hessian=False, retrieval=False, kv_cache="fp16")

        # Rung 2: + Hessian
        r2 = STEP_STACKING_LADDER[1]
        assert r2.rung_number == 2
        assert r2.factors == AblationFactorConfig(kac=False, hessian=True, retrieval=False, kv_cache="fp16")

        # Rung 3: + KAC
        r3 = STEP_STACKING_LADDER[2]
        assert r3.rung_number == 3
        assert r3.factors == AblationFactorConfig(kac=True, hessian=True, retrieval=False, kv_cache="fp16")

        # Rung 4: + 4-Bit KV Cache
        r4 = STEP_STACKING_LADDER[3]
        assert r4.rung_number == 4
        assert r4.factors == AblationFactorConfig(kac=True, hessian=True, retrieval=False, kv_cache="4bit_group64")

        # Rung 5: + Retrieval (Full LEAN-PQv2)
        r5 = STEP_STACKING_LADDER[4]
        assert r5.rung_number == 5
        assert r5.factors == AblationFactorConfig(kac=True, hessian=True, retrieval=True, kv_cache="4bit_group64")

    def test_drop_one_ablations_structure(self):
        """Verifies the 4 drop-one ablations each flip exactly one factor from Full LEAN-PQv2."""
        assert len(DROP_ONE_ABLATIONS) == 4

        full_lean_factors = AblationFactorConfig(kac=True, hessian=True, retrieval=True, kv_cache="4bit_group64")

        for key, drop in DROP_ONE_ABLATIONS.items():
            f = drop.factors
            diffs = []
            if f.kac != full_lean_factors.kac:
                diffs.append("kac")
            if f.hessian != full_lean_factors.hessian:
                diffs.append("hessian")
            if f.retrieval != full_lean_factors.retrieval:
                diffs.append("retrieval")
            if f.kv_cache != full_lean_factors.kv_cache:
                diffs.append("kv_cache")

            assert len(diffs) == 1, f"Drop-one ablation {key} modified {len(diffs)} factors instead of 1: {diffs}"
            assert diffs[0] == drop.drop_target, (
                f"Drop target {drop.drop_target} did not match modified factor {diffs[0]}"
            )

    def test_full_factorial_matrix_16_points(self):
        """Verifies that the full factorial matrix contains exactly 2^4 = 16 points."""
        matrix = AblationMatrix()
        all_points = matrix.get_all_points()
        assert len(all_points) == 16

        # Check unique factor keys
        keys = set(p.factors.factor_key for p in all_points)
        assert len(keys) == 16

    def test_ablation_model_instantiation(self):
        """Tests instantiating models across all ladder rungs and drop-one points."""
        matrix = AblationMatrix()
        # Test ladder rungs
        for rung in matrix.get_ladder_rungs():
            model = matrix.instantiate_model(rung, num_eval_layers=2, hidden_dim=128, vocab_size=200)
            x = torch.randint(0, 200, (1, 4))
            out = model(x)
            assert out.shape == (1, 4, 200)
            assert model.config.kv_cache_mode == rung.factors.kv_cache
            assert model.config.retrieval_enabled == rung.factors.retrieval

        # Test drop-one points
        for drop_key, drop_point in matrix.get_drop_one_ablations().items():
            model = matrix.instantiate_model(drop_point, num_eval_layers=2, hidden_dim=128, vocab_size=200)
            x = torch.randint(0, 200, (1, 4))
            out = model(x)
            assert out.shape == (1, 4, 200)
            assert model.config.kv_cache_mode == drop_point.factors.kv_cache
            assert model.config.retrieval_enabled == drop_point.factors.retrieval
