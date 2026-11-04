"""
campaigns/rigorous_benchmarks/baselines/__init__.py
===================================================
Baseline model definitions and evaluation wrappers for Project LEAN.
"""

from .models import (
    MODEL_ID,
    TOTAL_PARAMS,
    FP16_REF_BYTES,
    TARGET_STORAGE_BYTES,
    BM25_INDEX_BYTES,
    BaselineModelType,
    BaselineModelConfig,
    compute_full_model_byte_accounting,
    QuantizedKVCache,
    BaseQuantizedLinear,
    FP16Linear,
    LeanPQv2Linear,
    UniformMatchedLinear,
    GPTQUniformLinear,
    AWQUniformLinear,
    HQQTernaryLinear,
    BaselineEvaluationWrapper,
    create_baseline_model,
    get_baseline_spec,
    list_baseline_models,
)

__all__ = [
    "MODEL_ID",
    "TOTAL_PARAMS",
    "FP16_REF_BYTES",
    "TARGET_STORAGE_BYTES",
    "BM25_INDEX_BYTES",
    "BaselineModelType",
    "BaselineModelConfig",
    "compute_full_model_byte_accounting",
    "QuantizedKVCache",
    "BaseQuantizedLinear",
    "FP16Linear",
    "LeanPQv2Linear",
    "UniformMatchedLinear",
    "GPTQUniformLinear",
    "AWQUniformLinear",
    "HQQTernaryLinear",
    "BaselineEvaluationWrapper",
    "create_baseline_model",
    "get_baseline_spec",
    "list_baseline_models",
]
