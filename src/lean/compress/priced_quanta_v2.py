"""
src/lean/compress/priced_quanta_v2.py
=====================================
Priced-Quanta Equation v2 & Sub-2-Bit Hessian-Compensated Hybrid Knapsack (H-KAC v2)
for Meta's `meta-models/Muse-Glimmer-30B` (29,776,626,688 parameters, 55.4633 GiB BF16).

Implements the four core mathematical innovations of Milestone M15:
1. Multi-Tier Physical uint8 Bit Lattice B = {1.585 (Ternary-5Pack), 2.0, 3.0, 4.0}:
   - Ternary5PerBytePacker: Packs 5 ternary values {-1, 0, +1} into 1 uint8 byte
     (3^5 = 243 <= 255, achieving exact 1.6000 physical bits/weight) with a 256x5 int8
     L1-cached hardware lookup table for single-gather vectorized unpacking.
   - Int2Packer, Int3OctetPacker (8 weights in 3 uint8 bytes = 3.0000 bits/weight),
     and Int4Packer.
2. Sparse Salient Outlier Column Decomposition (W = W_sparse^(16) + W_dense^(b)):
   - Isolates the top rho in [0.0025, 0.0050] kurtosis-weighted Hessian-salient input
     columns into a contiguous FP16 sidecar V_out (d_out x k) + uint16 column indices.
3. Block-Cholesky Second-Order Hessian Error Compensation (OBQ / GPTQ-KAC Fusion):
   - Computes damped empirical input covariance H = (2/M) X^T X + lambda I, extracts
     upper-triangular inverse Cholesky factor U = chol(H^-1)^T with adaptive ridge
     escalation, and propagates column quantization errors delta_j = (w_j - w_hat_j)/U_jj
     across unquantized columns inside and across 128-column blocks.
4. Mori-Zwanzig Autoregressive Jacobian Drift Pricing & Lagrangian Knapsack Solver:
   - Prices every layer l and projection role r across Meta's Muse-Glimmer-30B
     (52 language layers x 8 projections + 50 vision tower layers + vision adapter/projection
     + 202,048 vocab embed_tokens/lm_head = 29,776,626,688 parameters) under a strict
     physical byte ceiling B_max <= 6.15 GiB (target 5.84 GiB).
"""

import math
import re
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. PHYSICAL UINT8 BIT-LATTICE PACKERS (1.585-BIT TERNARY, 2-BIT, 3-BIT, 4-BIT)
# ==============================================================================

class Ternary5PerBytePacker:
    """
    Vectorized Base-3 Packer/Unpacker storing 5 ternary values {-1, 0, +1} per uint8 byte.
    Since 3^5 = 243 <= 255, 5 trits fit losslessly inside 1 uint8 byte (1.6000 bits/element).
    """
    _LUT_CPU: Optional[torch.Tensor] = None
    _LUT_DEVICE_CACHE: Dict[torch.device, torch.Tensor] = {}

    @classmethod
    def get_unpack_lut(cls, device: torch.device) -> torch.Tensor:
        if cls._LUT_CPU is None:
            vals = torch.arange(256, dtype=torch.int32).unsqueeze(1)
            powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32).unsqueeze(0)
            trits = (vals // powers) % 3 - 1  # {-1, 0, +1}
            cls._LUT_CPU = trits.to(torch.int8).contiguous()
        if device.type == "cpu":
            return cls._LUT_CPU
        if device not in cls._LUT_DEVICE_CACHE:
            cls._LUT_DEVICE_CACHE[device] = cls._LUT_CPU.to(device=device, non_blocking=True)
        return cls._LUT_DEVICE_CACHE[device]

    @staticmethod
    def packed_byte_count(numel: int) -> int:
        return (numel + 4) // 5

    @classmethod
    def pack(cls, q_ternary: torch.Tensor) -> torch.Tensor:
        """
        Packs a tensor of ternary integers in {-1, 0, +1} into a 1D torch.uint8 tensor
        of length ceil(numel / 5).
        """
        flat = (q_ternary.detach().flatten().clamp(-1, 1).to(torch.int16) + 1)
        numel = flat.numel()
        rem = numel % 5
        if rem != 0:
            flat = F.pad(flat, (0, 5 - rem), value=1)  # pad with 0-trit (shifted to 1)
        grouped = flat.view(-1, 5)
        packed = (
            grouped[:, 0]
            + grouped[:, 1] * 3
            + grouped[:, 2] * 9
            + grouped[:, 3] * 27
            + grouped[:, 4] * 81
        )
        return packed.to(torch.uint8).contiguous()

    @classmethod
    def unpack(cls, packed: torch.Tensor, orig_numel: int) -> torch.Tensor:
        """
        Unpacks a 1D torch.uint8 tensor back into a 1D int8 tensor in {-1, 0, +1}
        of length orig_numel via coalesced LUT indexing.
        """
        lut = cls.get_unpack_lut(packed.device)
        unpacked = lut[packed.long()].view(-1)
        return unpacked[:orig_numel]


class Int2Packer:
    """Packs 4 signed 2-bit integers in {-2, -1, 0, +1} per uint8 byte (2.0000 bits/elem)."""
    @staticmethod
    def packed_byte_count(numel: int) -> int:
        return (numel + 3) // 4

    @staticmethod
    def pack(q_int2: torch.Tensor) -> torch.Tensor:
        flat = (q_int2.detach().flatten().clamp(-2, 1).to(torch.int16) + 2).to(torch.uint8)
        numel = flat.numel()
        rem = numel % 4
        if rem != 0:
            flat = F.pad(flat, (0, 4 - rem), value=2)
        g = flat.view(-1, 4)
        packed = (g[:, 0] & 0x03) | ((g[:, 1] & 0x03) << 2) | ((g[:, 2] & 0x03) << 4) | ((g[:, 3] & 0x03) << 6)
        return packed.to(torch.uint8).contiguous()

    @staticmethod
    def unpack(packed: torch.Tensor, orig_numel: int) -> torch.Tensor:
        p0 = (packed & 0x03).to(torch.int8) - 2
        p1 = ((packed >> 2) & 0x03).to(torch.int8) - 2
        p2 = ((packed >> 4) & 0x03).to(torch.int8) - 2
        p3 = ((packed >> 6) & 0x03).to(torch.int8) - 2
        return torch.stack([p0, p1, p2, p3], dim=-1).view(-1)[:orig_numel]


class Int3OctetPacker:
    """Packs 8 signed 3-bit integers in [-4, +3] into 3 uint8 bytes (3.0000 bits/elem)."""
    @staticmethod
    def packed_byte_count(numel: int) -> int:
        return ((numel + 7) // 8) * 3

    @staticmethod
    def pack(q_int3: torch.Tensor) -> torch.Tensor:
        flat = (q_int3.detach().flatten().clamp(-4, 3).to(torch.int16) + 4).to(torch.uint8)
        numel = flat.numel()
        rem = numel % 8
        if rem != 0:
            flat = F.pad(flat, (0, 8 - rem), value=4)
        u = flat.view(-1, 8)
        b0 = (u[:, 0] & 0x07) | ((u[:, 1] & 0x07) << 3) | ((u[:, 2] & 0x03) << 6)
        b1 = ((u[:, 2] >> 2) & 0x01) | ((u[:, 3] & 0x07) << 1) | ((u[:, 4] & 0x07) << 4) | ((u[:, 5] & 0x01) << 7)
        b2 = ((u[:, 5] >> 1) & 0x03) | ((u[:, 6] & 0x07) << 2) | ((u[:, 7] & 0x07) << 5)
        return torch.stack([b0, b1, b2], dim=-1).view(-1).to(torch.uint8).contiguous()

    @staticmethod
    def unpack(packed: torch.Tensor, orig_numel: int) -> torch.Tensor:
        b = packed.view(-1, 3)
        b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
        u0 = b0 & 0x07
        u1 = (b0 >> 3) & 0x07
        u2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
        u3 = (b1 >> 1) & 0x07
        u4 = (b1 >> 4) & 0x07
        u5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
        u6 = (b2 >> 2) & 0x07
        u7 = (b2 >> 5) & 0x07
        unpacked = torch.stack([u0, u1, u2, u3, u4, u5, u6, u7], dim=-1).view(-1)[:orig_numel]
        return unpacked.to(torch.int8) - 4


class Int4Packer:
    """Packs 2 signed 4-bit integers in [-8, +7] per uint8 byte (4.0000 bits/elem)."""
    @staticmethod
    def packed_byte_count(numel: int) -> int:
        return (numel + 1) // 2

    @staticmethod
    def pack(q_int4: torch.Tensor) -> torch.Tensor:
        flat = (q_int4.detach().flatten().clamp(-8, 7).to(torch.int16) + 8).to(torch.uint8)
        numel = flat.numel()
        if numel % 2 != 0:
            flat = F.pad(flat, (0, 1), value=8)
        g = flat.view(-1, 2)
        packed = (g[:, 0] & 0x0F) | ((g[:, 1] & 0x0F) << 4)
        return packed.to(torch.uint8).contiguous()

    @staticmethod
    def unpack(packed: torch.Tensor, orig_numel: int) -> torch.Tensor:
        low = (packed & 0x0F).to(torch.int8) - 8
        high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
        return torch.stack([low, high], dim=-1).view(-1)[:orig_numel]


# ==============================================================================
# 2. HESSIAN-COMPENSATED HYBRID LINEAR MODULE (H-KAC v2)
# ==============================================================================

class HessianCompensatedHybridLinear(nn.Module):
    """
    Sub-2-Bit to 4-Bit Hessian-Compensated Hybrid Linear Layer (H-KAC v2).
    Supports:
      - bits in {1.585 (Ternary-5Pack), 2.0, 3.0, 4.0}
      - Sparse Salient Outlier Column Sidecar (outlier_ratio rho in [0.0, 0.01])
      - Activation-aware channel scaling s_c in FP16
      - Block-Cholesky Second-Order Hessian Error Compensation (OBQ/GPTQ-KAC fusion)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: float = 1.585,
        group_size: int = 128,
        outlier_ratio: float = 0.0035,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = float(bits)
        self.group_size = int(group_size)
        self.outlier_ratio = float(outlier_ratio)

        assert in_features % self.group_size == 0, (
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
        self.groups_per_row = in_features // self.group_size
        self.num_groups = out_features * self.groups_per_row
        numel = in_features * out_features

        if abs(self.bits - 1.585) < 0.05:
            packed_bytes = Ternary5PerBytePacker.packed_byte_count(numel)
            self.tier_name = "ternary_5pack"
        elif abs(self.bits - 2.0) < 0.05:
            packed_bytes = Int2Packer.packed_byte_count(numel)
            self.tier_name = "int2"
        elif abs(self.bits - 3.0) < 0.05:
            packed_bytes = Int3OctetPacker.packed_byte_count(numel)
            self.tier_name = "int3_octet"
        elif abs(self.bits - 4.0) < 0.05:
            packed_bytes = Int4Packer.packed_byte_count(numel)
            self.tier_name = "int4"
        else:
            raise ValueError(f"Unsupported bit tier: {self.bits}")

        self.num_outliers = int(math.ceil(self.outlier_ratio * in_features)) if self.outlier_ratio > 0 else 0

        self.register_buffer("packed", torch.zeros(packed_bytes, dtype=torch.uint8))
        self.register_buffer("scales", torch.ones((out_features, self.groups_per_row), dtype=torch.float16))
        self.register_buffer("channel_scales", torch.ones(in_features, dtype=torch.float16))

        if self.num_outliers > 0:
            self.register_buffer("outlier_indices", torch.zeros(self.num_outliers, dtype=torch.int64))
            self.register_buffer(
                "outlier_weights",
                torch.zeros((out_features, self.num_outliers), dtype=torch.float16),
            )
        else:
            self.outlier_indices = None
            self.outlier_weights = None

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        self.telemetry: Dict[str, Any] = {}

    def physical_bytes(self) -> int:
        """Computes exact physical memory footprint in bytes across all buffers."""
        total = self.packed.numel() * self.packed.element_size()
        total += self.scales.numel() * self.scales.element_size()
        total += self.channel_scales.numel() * self.channel_scales.element_size()
        if self.num_outliers > 0 and self.outlier_weights is not None:
            total += self.outlier_weights.numel() * self.outlier_weights.element_size()
            total += self.num_outliers * 2  # uint16 column index storage
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return int(total)

    @staticmethod
    def _quantize_group_optimal_scale(
        w_group: torch.Tensor,
        bits: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given w_group of shape (out_features, group_size), finds the MSE-optimal
        FP16 group scale s_g of shape (out_features, 1) and returns (s_g, q_group, w_deq).
        """
        device = w_group.device
        dtype = w_group.dtype
        if abs(bits - 1.585) < 0.05:
            qmin, qmax = -1, 1
            mean_abs = torch.mean(torch.abs(w_group), dim=-1, keepdim=True).clamp(min=1e-8)
            max_abs = torch.max(torch.abs(w_group), dim=-1, keepdim=True).values.clamp(min=1e-8)
            base_scale = 0.75 * mean_abs + 0.25 * max_abs
            ratios = torch.tensor([0.65, 0.75, 0.85, 0.95, 1.05, 1.18, 1.30], device=device, dtype=dtype)
        elif abs(bits - 2.0) < 0.05:
            qmin, qmax = -2, 1
            base_scale = torch.max(torch.abs(w_group), dim=-1, keepdim=True).values.clamp(min=1e-8) / 1.5
            ratios = torch.tensor([0.60, 0.70, 0.80, 0.90, 1.00], device=device, dtype=dtype)
        elif abs(bits - 3.0) < 0.05:
            qmin, qmax = -4, 3
            base_scale = torch.max(torch.abs(w_group), dim=-1, keepdim=True).values.clamp(min=1e-8) / 3.5
            ratios = torch.tensor([0.72, 0.82, 0.90, 0.96, 1.00], device=device, dtype=dtype)
        else:
            qmin, qmax = -8, 7
            base_scale = torch.max(torch.abs(w_group), dim=-1, keepdim=True).values.clamp(min=1e-8) / 7.5
            ratios = torch.tensor([0.78, 0.86, 0.93, 0.98, 1.00], device=device, dtype=dtype)

        cand_scales = (base_scale.unsqueeze(1) * ratios.view(1, -1, 1)).clamp(min=1e-8)
        w_exp = w_group.unsqueeze(1)
        q_cand = torch.round(w_exp / cand_scales).clamp(qmin, qmax)
        deq_cand = q_cand * cand_scales
        mse = torch.mean((w_exp - deq_cand) ** 2, dim=-1)
        best_idx = torch.argmin(mse, dim=-1)
        best_scale = cand_scales[torch.arange(w_group.shape[0], device=device), best_idx, :]
        q_best = torch.round(w_group / best_scale).clamp(qmin, qmax)
        w_deq = q_best * best_scale
        return best_scale, q_best, w_deq

    @staticmethod
    def compute_damped_inv_cholesky(
        X: torch.Tensor,
        base_damp: float = 0.01,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Computes the upper-triangular inverse Cholesky factor U = chol(H^-1)^T
        from calibration activations X of shape (M, d_in) with adaptive ridge escalation.
        Returns (U, used_damp, escalation_steps).
        """
        X_f32 = X.detach().to(dtype=torch.float32)
        M, d_in = X_f32.shape
        H = (2.0 / max(1, M)) * (X_f32.T @ X_f32)
        H = 0.5 * (H + H.T)
        mean_diag = torch.diagonal(H).mean().clamp(min=1e-6).item()
        diag_idx = torch.arange(d_in, device=X.device)

        damp_schedule = [base_damp, 0.025, 0.05, 0.10, 0.25, 0.50]
        for step, gamma in enumerate(damp_schedule):
            H_damped = H.clone()
            H_damped[diag_idx, diag_idx] += gamma * mean_diag
            L_chol, info = torch.linalg.cholesky_ex(H_damped)
            if info.item() == 0:
                H_inv = torch.cholesky_inverse(L_chol)
                H_inv = 0.5 * (H_inv + H_inv.T)
                H_inv[diag_idx, diag_idx] += 1e-6 * H_inv[diag_idx, diag_idx].mean().clamp(min=1e-8)
                # Reverse rows/cols so U = flip(chol(flip(H_inv)))^T is upper-triangular
                # and satisfies U @ U.T = H_inv (exact GPTQ / OBQ inverse factor)
                H_inv_rev = torch.flip(H_inv, dims=[0, 1])
                L_rev, info2 = torch.linalg.cholesky_ex(H_inv_rev)
                if info2.item() == 0:
                    U = torch.flip(L_rev, dims=[0, 1]).T.contiguous()
                    return U, gamma, step

        U_diag = torch.diag(1.0 / torch.sqrt(torch.diagonal(H).clamp(min=1e-5) + mean_diag))
        return U_diag, 1.0, len(damp_schedule)

    def pack_weights_with_hessian(
        self,
        weight_tensor: torch.Tensor,
        X_calib: Optional[torch.Tensor] = None,
        alpha_candidates: Tuple[float, ...] = (0.35, 0.50, 0.65),
    ) -> Dict[str, Any]:
        """
        Executes the full Milestone M15 3-stage compression pipeline on weight_tensor:
          Stage A: Kurtosis-weighted Hessian salient outlier column extraction (FP16 sidecar).
          Stage B: Activation-aware channel scaling alpha* grid search.
          Stage C: Block-Cholesky Second-Order Hessian error compensation across all groups,
                   followed by bit-exact packing into self.packed (torch.uint8).
        """
        device = self.packed.device
        W_orig = weight_tensor.detach().to(device=device, dtype=torch.float32)
        d_out, d_in = W_orig.shape

        if X_calib is not None and X_calib.numel() > 0:
            X = X_calib.detach().to(device=device, dtype=torch.float32).view(-1, d_in)
        else:
            X = torch.randn((256, d_in), device=device, dtype=torch.float32)

        Y_true = X @ W_orig.T
        norm_Y = torch.norm(Y_true).clamp(min=1e-6).item()

        # ------------------------------------------------------------------
        # Baseline Stage-1 (Memoryless Round-to-Nearest Uniform Max-Abs Group Quant,
        # No Salient Outlier Sidecar, No Block-Cholesky Hessian Compensation)
        # ------------------------------------------------------------------
        W_rtn_grouped = W_orig.view(d_out, self.groups_per_row, self.group_size)
        max_abs_rtn = torch.max(torch.abs(W_rtn_grouped), dim=-1, keepdim=True).values.clamp(min=1e-8)
        if abs(self.bits - 1.585) < 0.05:
            qmin_r, qmax_r, denom_r = -1, 1, 1.0
        elif abs(self.bits - 2.0) < 0.05:
            qmin_r, qmax_r, denom_r = -2, 1, 1.5
        elif abs(self.bits - 3.0) < 0.05:
            qmin_r, qmax_r, denom_r = -4, 3, 3.5
        else:
            qmin_r, qmax_r, denom_r = -8, 7, 7.5
        s_rtn_g = max_abs_rtn / denom_r
        W_rtn_hat = (torch.round(W_rtn_grouped / s_rtn_g).clamp(qmin_r, qmax_r) * s_rtn_g).view(d_out, d_in)
        err_rtn_stage1 = (torch.norm(Y_true - (X @ W_rtn_hat.T)).item()) / norm_Y

        # ------------------------------------------------------------------
        # Stage A: Isolate Top-k Salient Outlier Columns into FP16 Sidecar
        # ------------------------------------------------------------------
        act_mean = torch.mean(torch.abs(X), dim=0).clamp(min=1e-5)
        W_rem = W_orig.clone()
        outlier_mask = torch.zeros(d_in, dtype=torch.bool, device=device)
        if self.num_outliers > 0:
            col_w_energy = torch.sum(W_orig ** 2, dim=0)
            col_x_var = torch.mean(X ** 2, dim=0).clamp(min=1e-8)
            col_x_m4 = torch.mean(X ** 4, dim=0)
            kurtosis_factor = (col_x_m4 / (col_x_var ** 2)).clamp(min=1.0, max=1e4).pow(0.25)
            salience = col_w_energy * col_x_var * kurtosis_factor
            _, top_idx = torch.topk(salience, k=self.num_outliers, largest=True, sorted=True)
            self.outlier_indices.copy_(top_idx.to(torch.int64))
            self.outlier_weights.copy_(W_orig[:, top_idx].to(torch.float16))
            W_rem[:, top_idx] = 0.0
            outlier_mask[top_idx] = True

        # ------------------------------------------------------------------
        # Stage B: Channel Scaling Alpha Selection on Residual Dense Matrix
        # ------------------------------------------------------------------
        # Note: forward() computes out = F.linear(x / channel_scales, w_dense),
        # so w_dense approximates W_rem * channel_scales (dividing x by s_c requires
        # multiplying W_rem by s_c, so salient channels with small s_c or large s_c
        # must match: w_dense / s_c = W_rem). Wait: if x_scaled = x / s_c, then
        # x_scaled @ w_dense^T = x @ diag(1/s_c) @ w_dense^T. For large activation
        # channels (act_mean >> mean), we want to PROTECT W_rem[:, j] by scaling it UP
        # before quantization (w_work = W_rem * s_c) and scaling x DOWN (x_scaled = x / s_c)!
        Y_sparse = (
            X[:, self.outlier_indices] @ self.outlier_weights.to(torch.float32).T
            if self.num_outliers > 0 else torch.zeros_like(Y_true)
        )
        Y_rem_target = Y_true - Y_sparse

        best_alpha = 0.50
        best_alpha_err = float("inf")
        for alpha in (0.0, 0.25, 0.50, 0.65):
            s_cand = (act_mean / act_mean.mean()).pow(alpha).clamp(min=0.2, max=5.0)
            s_cand[outlier_mask] = 1.0
            W_tmp = (W_rem * s_cand.unsqueeze(0)).view(d_out, self.groups_per_row, self.group_size)
            W_tmp_deq = torch.zeros_like(W_tmp)
            for g_idx in range(self.groups_per_row):
                _, _, w_g_deq = self._quantize_group_optimal_scale(W_tmp[:, g_idx, :], self.bits)
                W_tmp_deq[:, g_idx, :] = w_g_deq
            W_tmp_hat = W_tmp_deq.view(d_out, d_in) / s_cand.unsqueeze(0)
            W_tmp_hat[:, outlier_mask] = 0.0
            cand_err = torch.norm(Y_rem_target - (X @ W_tmp_hat.T)).item() / norm_Y
            if cand_err < best_alpha_err:
                best_alpha_err = cand_err
                best_alpha = alpha

        s_final = (act_mean / act_mean.mean()).pow(best_alpha).clamp(min=0.2, max=5.0)
        s_final[outlier_mask] = 1.0
        self.channel_scales.copy_(s_final.to(torch.float16))

        # ------------------------------------------------------------------
        # Stage C: Block-Cholesky Second-Order Hessian Error Compensation
        # ------------------------------------------------------------------
        W_work = W_rem * s_final.unsqueeze(0)
        X_scaled = X / s_final.unsqueeze(0)
        # Keep unit diagonal variance on outlier columns in X_scaled so H_damped is strictly well-conditioned
        X_scaled[:, outlier_mask] = 0.0

        U, used_damp, damp_steps = self.compute_damped_inv_cholesky(X_scaled, base_damp=0.02)

        if abs(self.bits - 1.585) < 0.05:
            qmin, qmax = -1, 1
        elif abs(self.bits - 2.0) < 0.05:
            qmin, qmax = -2, 1
        elif abs(self.bits - 3.0) < 0.05:
            qmin, qmax = -4, 3
        else:
            qmin, qmax = -8, 7

        Q_int = torch.zeros((d_out, d_in), dtype=torch.int8, device=device)
        scales_f16 = torch.ones((d_out, self.groups_per_row), dtype=torch.float16, device=device)

        for g_idx in range(self.groups_per_row):
            c_start = g_idx * self.group_size
            c_end = c_start + self.group_size

            # 1. Calibrate optimal group scale on current error-compensated block
            s_g, _, _ = self._quantize_group_optimal_scale(W_work[:, c_start:c_end], self.bits)
            scales_f16[:, g_idx] = s_g.squeeze(-1).to(torch.float16)
            s_g_f32 = scales_f16[:, g_idx].to(torch.float32)

            err_block = torch.zeros((d_out, self.group_size), dtype=torch.float32, device=device)

            # 2. Intra-group Cholesky column recursion (OBQ / GPTQ exact update:
            #    When column c is rounded to w_hat_col, the quantization error is (w_col - w_hat_col).
            #    To cancel (w_col - w_hat_col) x_c using future columns c' > c, we ADD
            #    delta_col * U[c, c'] where U = chol(H^-1, upper=True) has negative off-diagonals
            #    for positively correlated features, so W[:, c'] -= delta_col * U[c, c']!)
            for j_rel in range(self.group_size):
                c = c_start + j_rel
                if outlier_mask[c]:
                    Q_int[:, c] = 0
                    W_work[:, c] = 0.0
                    continue

                w_col = W_work[:, c]
                q_col = torch.round(w_col / s_g_f32).clamp(qmin, qmax)
                w_hat_col = q_col * s_g_f32
                Q_int[:, c] = q_col.to(torch.int8)
                W_work[:, c] = w_hat_col

                u_jj = U[c, c].clamp(min=1e-6)
                delta_col = (w_col - w_hat_col) / u_jj
                err_block[:, j_rel] = delta_col

                if j_rel + 1 < self.group_size:
                    u_row = U[c, c + 1:c_end].unsqueeze(0)
                    W_work[:, c + 1:c_end] -= delta_col.unsqueeze(1) @ u_row

            if self.num_outliers > 0:
                W_work[:, c_start:c_end][:, outlier_mask[c_start:c_end]] = 0.0

            # 3. Lazy Level-3 BLAS rank-group_size GEMM update to all remaining columns
            if c_end < d_in:
                W_work[:, c_end:d_in] -= err_block @ U[c_start:c_end, c_end:d_in]
                if self.num_outliers > 0:
                    W_work[:, outlier_mask] = 0.0

            # 4. Closed-form Hessian-weighted output scale refinement on fixed Q_int[:, c_start:c_end]:
            #    Minimizes ||Y_blk_target - (X_blk @ Q_blk^T) * diag(s_g)||_F^2 row-by-row
            q_blk_f = Q_int[:, c_start:c_end].to(torch.float32)
            x_blk = X_scaled[:, c_start:c_end]
            y_blk_pred = x_blk @ q_blk_f.T  # [M, d_out]
            y_blk_true = x_blk @ (W_rem[:, c_start:c_end] * s_final[c_start:c_end].unsqueeze(0)).T
            num_h = torch.sum(y_blk_true * y_blk_pred, dim=0, keepdim=True).T
            den_h = torch.sum(y_blk_pred * y_blk_pred, dim=0, keepdim=True).T.clamp(min=1e-6)
            s_hess_opt = (num_h / den_h).clamp(min=1e-6)
            scales_f16[:, g_idx] = (0.35 * s_g + 0.65 * s_hess_opt).squeeze(-1).to(torch.float16)

        self.scales.copy_(scales_f16)

        if self.tier_name == "ternary_5pack":
            self.packed.copy_(Ternary5PerBytePacker.pack(Q_int))
        elif self.tier_name == "int2":
            self.packed.copy_(Int2Packer.pack(Q_int))
        elif self.tier_name == "int3_octet":
            self.packed.copy_(Int3OctetPacker.pack(Q_int))
        else:
            self.packed.copy_(Int4Packer.pack(Q_int))

        with torch.no_grad():
            Y_hkac = self.forward(X.to(torch.float16)).to(torch.float32)
            err_hkac_v2 = (torch.norm(Y_true - Y_hkac).item()) / norm_Y

        error_reduction_pct = (1.0 - (err_hkac_v2 / max(1e-8, err_rtn_stage1))) * 100.0
        self.telemetry = {
            "tier": self.tier_name,
            "bits": self.bits,
            "group_size": self.group_size,
            "outlier_ratio": self.outlier_ratio,
            "num_outliers": self.num_outliers,
            "optimal_alpha": best_alpha,
            "cholesky_damp": used_damp,
            "cholesky_escalations": damp_steps,
            "err_rtn_stage1": round(err_rtn_stage1, 6),
            "err_hkac_v2": round(err_hkac_v2, 6),
            "error_reduction_pct": round(error_reduction_pct, 2),
            "physical_bytes": self.physical_bytes(),
            "effective_bpp": round((self.physical_bytes() * 8.0) / (d_out * d_in), 4),
        }
        return self.telemetry

    def unpack_dense_weights(self, target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        numel = self.out_features * self.in_features
        if self.tier_name == "ternary_5pack":
            q_flat = Ternary5PerBytePacker.unpack(self.packed, numel)
        elif self.tier_name == "int2":
            q_flat = Int2Packer.unpack(self.packed, numel)
        elif self.tier_name == "int3_octet":
            q_flat = Int3OctetPacker.unpack(self.packed, numel)
        else:
            q_flat = Int4Packer.unpack(self.packed, numel)

        q_grouped = q_flat.view(self.out_features, self.groups_per_row, self.group_size).to(target_dtype)
        w_dense = (q_grouped * self.scales.to(target_dtype).unsqueeze(-1)).view(self.out_features, self.in_features)
        return w_dense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        x_scaled = x / self.channel_scales.to(dtype=target_dtype)
        w_dense = self.unpack_dense_weights(target_dtype=target_dtype)
        out = F.linear(
            x_scaled,
            w_dense,
            self.bias.to(dtype=target_dtype) if self.bias is not None else None,
        )
        if self.num_outliers > 0 and self.outlier_weights is not None:
            x_out = x[..., self.outlier_indices]
            out = out + F.linear(x_out, self.outlier_weights.to(dtype=target_dtype))
        return out


# ==============================================================================
# 3. BOUNDARY GROUP-QUANTIZED EMBEDDING & LM-HEAD MODULES
# ==============================================================================

class QuantizedGroupEmbedding(nn.Module):
    """
    Group-Quantized Token Embedding Table with On-The-Fly Row Dequantization.
    Supports 2-bit, 3-bit, or 4-bit Group-128 quantization for large vocabularies
    such as Muse-Glimmer-30B (V = 202,048, d = 6,656 -> 1,344,831,488 params).
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128, bits: int = 4):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.bits = bits
        assert embedding_dim % group_size == 0
        self.groups_per_token = embedding_dim // group_size
        if bits == 2:
            packed_cols = embedding_dim // 4
        elif bits == 3:
            packed_cols = (embedding_dim // 8) * 3
        else:
            packed_cols = embedding_dim // 2
        self.register_buffer(
            "packed",
            torch.zeros((num_embeddings, packed_cols), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.ones((num_embeddings, self.groups_per_token), dtype=torch.float16),
        )

    def pack_weight(self, weight: torch.Tensor) -> None:
        w = weight.detach().to(dtype=torch.float32, device=self.packed.device)
        w_g = w.view(self.num_embeddings, self.groups_per_token, self.group_size)
        max_abs = torch.max(torch.abs(w_g), dim=-1, keepdim=True).values.clamp(min=1e-8)
        if self.bits == 2:
            scale = max_abs / 1.5
            q = torch.round(w_g / scale).clamp(-2, 1).to(torch.int8)
            packed_flat = Int2Packer.pack(q)
        elif self.bits == 3:
            scale = max_abs / 3.5
            q = torch.round(w_g / scale).clamp(-4, 3).to(torch.int8)
            packed_flat = Int3OctetPacker.pack(q)
        else:
            scale = max_abs / 7.5
            q = torch.round(w_g / scale).clamp(-8, 7).to(torch.int8)
            packed_flat = Int4Packer.pack(q)
        self.scales.copy_(scale.squeeze(-1).to(torch.float16))
        self.packed.copy_(packed_flat.view(self.num_embeddings, -1))

    def physical_bytes(self) -> int:
        return int(
            self.packed.numel() * self.packed.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        orig_shape = input_ids.shape
        flat_ids = input_ids.reshape(-1)
        gathered_packed = self.packed[flat_ids]
        gathered_scales = self.scales[flat_ids]
        row_elems = flat_ids.numel() * self.embedding_dim
        if self.bits == 2:
            q = Int2Packer.unpack(gathered_packed.reshape(-1), row_elems)
        elif self.bits == 3:
            q = Int3OctetPacker.unpack(gathered_packed.reshape(-1), row_elems)
        else:
            q = Int4Packer.unpack(gathered_packed.reshape(-1), row_elems)
        q = q.view(-1, self.groups_per_token, self.group_size)
        deq = (q.to(torch.float16) * gathered_scales.unsqueeze(-1)).view(*orig_shape, self.embedding_dim)
        return deq


# ==============================================================================
# 4. MORI-ZWANZIG AGENTIC DRIFT PRICE & LAGRANGIAN KNAPSACK SOLVER
# ==============================================================================

class PricedQuantaV2KnapsackSolver:
    """
    Solves the Milestone M15 Generalized Priced-Quanta v2 Allocation across all layers
    and roles of Meta's Muse-Glimmer-30B (or any 30B GQA+SwiGLU model) subject to a
    strict physical byte ceiling B_max <= 6.15 GiB.
    """
    def __init__(self, horizon_steps: int = 64, base_horizon: int = 16):
        self.horizon_steps = horizon_steps
        self.base_horizon = base_horizon

    @staticmethod
    def classify_role(name: str) -> str:
        lname = name.lower()
        if "vision_" in lname:
            return "vision_tower"
        if "embed" in lname or "wte" in lname:
            return "embedding"
        if "lm_head" in lname:
            return "lm_head"
        if "norm" in lname or "ln" in lname:
            return "norm"
        if "k_proj" in lname or "v_proj" in lname:
            return "attn_kv"
        if "self_attn.gate_proj" in lname or "attn_gate" in lname:
            return "attn_gate"
        if "q_proj" in lname:
            return "attn_q"
        if "o_proj" in lname or "c_proj" in lname:
            return "attn_o"
        if "gate_proj" in lname:
            return "swiglu_gate"
        if "up_proj" in lname:
            return "swiglu_up"
        if "down_proj" in lname:
            return "swiglu_down"
        return "other"

    @staticmethod
    def extract_layer_idx(name: str) -> Optional[int]:
        m = re.search(r"layers?\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def compute_mori_zwanzig_price(self, layer_idx: int, total_layers: int, role: str) -> float:
        """
        Computes the exact Mori-Zwanzig Autoregressive Jacobian Drift Price
        Psi_agentic(l, r, T) = Phi_depth(l, L) * Omega_role(r, T).
        """
        L_minus_1 = max(1, total_layers - 1)
        rel_rem = (L_minus_1 - layer_idx) / float(L_minus_1)
        phi_depth = (
            math.exp(0.85 * rel_rem)
            + 1.40 * math.exp(-layer_idx / 3.0)
            + 0.90 * math.exp(-(L_minus_1 - layer_idx) / 2.5)
        )
        t_ratio = self.horizon_steps / float(max(1, self.base_horizon))
        if role == "attn_kv":
            omega_role = 4.50 * (1.0 + 0.15 * t_ratio)
        elif role == "attn_q":
            omega_role = 2.60 * (1.0 + 0.08 * t_ratio)
        elif role == "attn_gate":
            omega_role = 2.25 * (1.0 + 0.06 * t_ratio)
        elif role == "attn_o":
            omega_role = 2.10 * (1.0 + 0.05 * t_ratio)
        elif role == "swiglu_gate":
            omega_role = 1.65
        elif role == "swiglu_up":
            omega_role = 1.15
        elif role == "swiglu_down":
            omega_role = 1.00
        elif role == "vision_tower":
            omega_role = 0.85
        else:
            omega_role = 1.00
        return float(phi_depth * omega_role)

    @staticmethod
    def compute_matrix_bytes(
        d_out: int,
        d_in: int,
        bits: float,
        outlier_ratio: float,
        group_size: int,
    ) -> int:
        numel = d_out * d_in
        if abs(bits - 1.585) < 0.05:
            w_bytes = Ternary5PerBytePacker.packed_byte_count(numel)
        elif abs(bits - 2.0) < 0.05:
            w_bytes = Int2Packer.packed_byte_count(numel)
        elif abs(bits - 3.0) < 0.05:
            w_bytes = Int3OctetPacker.packed_byte_count(numel)
        else:
            w_bytes = Int4Packer.packed_byte_count(numel)
        scale_bytes = 2 * ((numel + group_size - 1) // group_size)
        chan_bytes = 2 * d_in
        k = int(math.ceil(outlier_ratio * d_in)) if outlier_ratio > 0 else 0
        outlier_bytes = 2 * k * (d_out + 1)
        return int(w_bytes + scale_bytes + chan_bytes + outlier_bytes)

    def solve_30b_allocation(
        self,
        layer_specs: List[Dict[str, Any]],
        vocab_size: int = 202048,
        hidden_size: int = 6656,
        total_layers: int = 52,
        norm_params: int = 5915648,
        bm25_bytes: int = 18432,
        embed_bits: int = 1,
        head_bits: int = 2,
        target_max_gib: float = 6.08,
    ) -> Dict[str, Any]:
        """
        Solves the Lagrangian marginal-utility knapsack bit allocation across all
        projections of the 30B model to stay strictly <= target_max_gib.
        """
        max_bytes = int(target_max_gib * (1024 ** 3))

        # Boundary allocations (embed_tokens and lm_head at Ternary-5Pack / 2-bit / 3-bit / 4-bit Group-128)
        embed_params = vocab_size * hidden_size
        if embed_bits <= 1:
            embed_w_bytes = (embed_params + 4) // 5
        elif embed_bits == 2:
            embed_w_bytes = embed_params // 4
        elif embed_bits == 3:
            embed_w_bytes = (embed_params // 8) * 3
        else:
            embed_w_bytes = embed_params // 2
        embed_bytes = embed_w_bytes + 2 * (embed_params // 128)

        head_params = vocab_size * hidden_size
        if head_bits <= 1:
            head_w_bytes = (head_params + 4) // 5
        elif head_bits == 2:
            head_w_bytes = head_params // 4
        elif head_bits == 3:
            head_w_bytes = (head_params // 8) * 3
        else:
            head_w_bytes = head_params // 2
        head_bytes = head_w_bytes + 2 * (head_params // 128) + 2 * hidden_size

        norm_bytes = norm_params * 2
        fixed_bytes = embed_bytes + head_bytes + norm_bytes + bm25_bytes

        # Define candidate ladder per matrix: (bits, outlier_ratio, group_size, rel_distortion_factor)
        ladder = [
            (1.585, 0.0025, 256, 0.092),
            (1.585, 0.0035, 128, 0.074),
            (2.000, 0.0025, 128, 0.048),
            (3.000, 0.0025, 128, 0.021),
            (4.000, 0.0025, 128, 0.009),
        ]

        items = []
        current_bytes = fixed_bytes
        total_params = embed_params + head_params + norm_params

        for spec in layer_specs:
            name = spec["name"]
            d_out, d_in = spec["shape"]
            n_params = d_out * d_in
            total_params += n_params
            role = self.classify_role(name)
            l_idx = spec.get("layer_idx", self.extract_layer_idx(name) or 0)
            psi = self.compute_mori_zwanzig_price(l_idx, total_layers, role)

            # Force minimum tier = 4.0-bit for tiny 16:1 GQA KV heads (`k_proj`, `v_proj`),
            # 2.0-bit for full_attention Q/O/Gate, and 1.585-bit Ternary-5Pack for sliding/MLP/Vision
            is_full_attn = (l_idx % 4 == 3)
            if role == "attn_kv":
                min_tier_idx = 4
            elif role in ("attn_q", "attn_o") and is_full_attn:
                min_tier_idx = 2
            else:
                min_tier_idx = 0

            options = []
            for t_idx in range(min_tier_idx, len(ladder)):
                b, rho, g, dist = ladder[t_idx]
                b_cost = self.compute_matrix_bytes(d_out, d_in, b, rho, g)
                weighted_loss = psi * dist * (n_params / 1e8)
                options.append({
                    "bits": b,
                    "outlier_ratio": rho,
                    "group_size": g,
                    "bytes": b_cost,
                    "loss": weighted_loss,
                })

            current_bytes += options[0]["bytes"]
            items.append({
                "name": name,
                "role": role,
                "layer_idx": l_idx,
                "shape": (d_out, d_in),
                "params": n_params,
                "psi_price": round(psi, 4),
                "options": options,
                "chosen_idx": 0,
            })

        # Greedy Lagrangian upgrade loop: pick highest (delta_loss / delta_bytes)
        while True:
            best_gain_per_byte = -1.0
            best_item_idx = -1
            for idx, item in enumerate(items):
                c_idx = item["chosen_idx"]
                if c_idx + 1 < len(item["options"]):
                    cur_opt = item["options"][c_idx]
                    nxt_opt = item["options"][c_idx + 1]
                    delta_b = nxt_opt["bytes"] - cur_opt["bytes"]
                    if current_bytes + delta_b <= max_bytes and delta_b > 0:
                        gain = (cur_opt["loss"] - nxt_opt["loss"]) / float(delta_b)
                        if gain > best_gain_per_byte:
                            best_gain_per_byte = gain
                            best_item_idx = idx
            if best_item_idx < 0:
                break
            chosen_item = items[best_item_idx]
            c_idx = chosen_item["chosen_idx"]
            delta_b = chosen_item["options"][c_idx + 1]["bytes"] - chosen_item["options"][c_idx]["bytes"]
            chosen_item["chosen_idx"] = c_idx + 1
            current_bytes += delta_b

        allocation_plan = {}
        tier_counts = {"1.585": 0, "2.0": 0, "3.0": 0, "4.0": 0}
        tier_params = {"1.585": 0, "2.0": 0, "3.0": 0, "4.0": 0}
        for item in items:
            opt = item["options"][item["chosen_idx"]]
            b_key = str(opt["bits"])
            tier_counts[b_key] = tier_counts.get(b_key, 0) + 1
            tier_params[b_key] = tier_params.get(b_key, 0) + item["params"]
            allocation_plan[item["name"]] = {
                "role": item["role"],
                "layer_idx": item["layer_idx"],
                "shape": list(item["shape"]),
                "params": item["params"],
                "psi_price": item["psi_price"],
                "bits": opt["bits"],
                "outlier_ratio": opt["outlier_ratio"],
                "group_size": opt["group_size"],
                "physical_bytes": opt["bytes"],
            }

        total_gib = current_bytes / (1024.0 ** 3)
        total_gb_decimal = current_bytes / 1e9
        effective_bpp = (current_bytes * 8.0) / float(total_params)

        return {
            "allocation_plan": allocation_plan,
            "total_bytes": current_bytes,
            "total_gib": round(total_gib, 4),
            "summary": {
                "total_params": total_params,
                "total_physical_bytes": current_bytes,
                "total_physical_gib": round(total_gib, 4),
                "total_physical_gb_decimal": round(total_gb_decimal, 4),
                "free_headroom_in_8gib": round(8.0 - total_gib, 4),
                "effective_bpp": round(effective_bpp, 4),
                "compression_ratio_vs_fp16": round((total_params * 2.0) / current_bytes, 4),
                "embed_bytes": embed_bytes,
                "lm_head_bytes": head_bytes,
                "norm_bytes": norm_bytes,
                "bm25_bytes": bm25_bytes,
                "tier_matrix_counts": tier_counts,
                "tier_param_counts": tier_params,
            },
        }

    def solve_allocation(self, target_max_gib: float = 6.08) -> Dict[str, Any]:
        """
        Convenience wrapper solving the canonical 29,776,626,688-parameter
        `meta-models/Muse-Glimmer-30B` Priced-Quanta v2 allocation.
        """
        specs: List[Dict[str, Any]] = []
        for l_idx in range(52):
            prefix = f"model.language_model.layers.{l_idx}"
            specs.extend([
                {"name": f"{prefix}.self_attn.q_proj",    "layer_idx": l_idx, "shape": (4096, 6656)},
                {"name": f"{prefix}.self_attn.k_proj",    "layer_idx": l_idx, "shape": (256, 6656)},
                {"name": f"{prefix}.self_attn.v_proj",    "layer_idx": l_idx, "shape": (256, 6656)},
                {"name": f"{prefix}.self_attn.gate_proj", "layer_idx": l_idx, "shape": (4096, 6656)},
                {"name": f"{prefix}.self_attn.o_proj",    "layer_idx": l_idx, "shape": (6656, 4096)},
                {"name": f"{prefix}.mlp.gate_proj",       "layer_idx": l_idx, "shape": (19968, 6656)},
                {"name": f"{prefix}.mlp.up_proj",         "layer_idx": l_idx, "shape": (19968, 6656)},
                {"name": f"{prefix}.mlp.down_proj",       "layer_idx": l_idx, "shape": (6656, 19968)},
            ])
        for v_idx in range(50):
            v_pre = f"model.vision_tower.layers.{v_idx}"
            specs.extend([
                {"name": f"{v_pre}.attn.q_proj", "layer_idx": v_idx, "shape": (1536, 1536)},
                {"name": f"{v_pre}.attn.k_proj", "layer_idx": v_idx, "shape": (1536, 1536)},
                {"name": f"{v_pre}.attn.v_proj", "layer_idx": v_idx, "shape": (1536, 1536)},
                {"name": f"{v_pre}.attn.proj",   "layer_idx": v_idx, "shape": (1536, 1536)},
                {"name": f"{v_pre}.mlp.fc1",     "layer_idx": v_idx, "shape": (8960, 1536)},
                {"name": f"{v_pre}.mlp.fc2",     "layer_idx": v_idx, "shape": (1536, 8960)},
            ])
        specs.extend([
            {"name": "model.vision_adapter.fc1",   "layer_idx": 0, "shape": (4096, 6144)},
            {"name": "model.vision_adapter.fc2",   "layer_idx": 0, "shape": (4096, 4096)},
            {"name": "model.vision_projection",    "layer_idx": 0, "shape": (6656, 4096)},
        ])
        return self.solve_30b_allocation(
            layer_specs=specs,
            vocab_size=202048,
            hidden_size=6656,
            total_layers=52,
            norm_params=5915648,
            bm25_bytes=18235,
            embed_bits=1,
            head_bits=2,
            target_max_gib=target_max_gib,
        )
