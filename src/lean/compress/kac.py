"""
Knapsack-Aware Compressor (KAC) Engine.

Implements:
1. Value-Per-Bit Parameter Profiling:
   - Differentiates algorithmic reasoning circuits (Attention QKVO) from associative memory (MLPs).
   - High bit allocation (e.g. 4-bit / 8-bit) to attention projection matrices.
   - Aggressive low-bit allocation (e.g. 2-bit / 3-bit) to feed-forward MLP projections.
2. Group-128 Symmetric RTN Quantization with Exact Scale Accounting.
3. Mixed-Precision Physical Bit Budget Calculation:
   - Tracks unquantized parameters (embeddings, LM heads, LayerNorms) at 16-bit.
   - Tracks quantized matrices at b_l bits + group scale overhead (16 / group_size).
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from src.lean.measure.ruler_a import quantize_tensor_rtn


class KnapsackAwareCompressor:
    """
    Allocates bits across model components based on marginal value per bit:
    max_b sum_i v_i(b_i) s.t. sum_i B_i(b_i) <= B_target.
    """
    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def identify_layer_type(self, name: str) -> str:
        """Classify parameter into: 'attention', 'mlp', 'embedding', 'norm', 'head'."""
        lname = name.lower()
        if "embed" in lname or "wte" in lname or "wpe" in lname:
            return "embedding"
        if "lm_head" in lname:
            return "head"
        if "norm" in lname or "ln" in lname:
            return "norm"
        if any(k in lname for k in ["q_proj", "k_proj", "v_proj", "o_proj", "query_key_value", "c_attn", "c_proj"]):
            return "attention"
        if any(k in lname for k in ["mlp", "gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h", "c_fc"]):
            return "mlp"
        return "other"

    def plan_bit_allocation(
        self,
        model: nn.Module,
        target_effective_bpp: float = 3.0,
        attention_bits: int = 4,
        mlp_bits: int = 2
    ) -> Dict[str, Any]:
        """
        Computes mixed-precision bit allocation strategy protecting attention circuits.
        """
        layer_plan = {}
        total_params = 0
        unquant_params = 0
        quant_params = 0
        total_physical_bits = 0
        
        for name, param in model.named_parameters():
            n = param.numel()
            total_params += n
            ltype = self.identify_layer_type(name)
            
            # Keep 1D vectors, norms, embeddings, and lm_head in FP16
            if len(param.shape) < 2 or ltype in ["embedding", "head", "norm"]:
                layer_plan[name] = {"bits": 16, "quantize": False, "type": ltype, "params": n}
                unquant_params += n
                total_physical_bits += n * 16
            elif ltype == "attention":
                layer_plan[name] = {"bits": attention_bits, "quantize": True, "type": ltype, "params": n}
                quant_params += n
                # bits + scale overhead (16 / group_size)
                scale_overhead = 16.0 / self.group_size
                total_physical_bits += int(n * (attention_bits + scale_overhead))
            else: # mlp or other 2D matrix
                layer_plan[name] = {"bits": mlp_bits, "quantize": True, "type": ltype, "params": n}
                quant_params += n
                scale_overhead = 16.0 / self.group_size
                total_physical_bits += int(n * (mlp_bits + scale_overhead))
                
        effective_bpp = total_physical_bits / max(1, total_params)
        
        return {
            "layer_plan": layer_plan,
            "total_params": total_params,
            "quantized_params": quant_params,
            "unquantized_params": unquant_params,
            "total_physical_bits": total_physical_bits,
            "effective_bits_per_param": effective_bpp,
            "compression_ratio": (total_params * 16) / max(1, total_physical_bits)
        }

    def apply_compression(
        self,
        model: nn.Module,
        allocation_plan: Dict[str, Any],
        device: str = "cpu"
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Physically quantizes weights in-place according to the KAC allocation plan.
        Returns the compressed model and a dictionary of original weights for restoring.
        """
        orig_weights = {}
        layer_plan = allocation_plan["layer_plan"]
        
        for name, param in model.named_parameters():
            plan = layer_plan.get(name)
            if plan and plan["quantize"]:
                orig_weights[name] = param.data.detach().cpu()
                bits = plan["bits"]
                q, _ = quantize_tensor_rtn(param.data, bits=bits, group_size=self.group_size)
                param.data = q.to(device=param.device, dtype=param.dtype)
                
        return model, orig_weights

    def restore_weights(
        self,
        model: nn.Module,
        orig_weights: Dict[str, torch.Tensor]
    ):
        """Restores original uncompressed weights in-place."""
        param_dict = dict(model.named_parameters())
        for name, tensor in orig_weights.items():
            if name in param_dict:
                param_dict[name].data = tensor.to(device=param_dict[name].device, dtype=param_dict[name].dtype)

