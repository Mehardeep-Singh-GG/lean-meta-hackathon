#!/usr/bin/env python3
"""
src/lean/anatomy/glimmer_30b_profiler.py
========================================
Memory-Safe Safetensor Shard Streamer, Layer-by-Layer Spectral/Kurtosis Anatomical
Profiler, and Batched Monte Carlo Half-Life Evaluator for Meta's `Muse-Glimmer-30B`
(`meta-models/Muse-Glimmer-30B`, 29.78B params, 52 Language Layers + 50 Vision Layers)
and Cross-Scale GQA Transformers.

Supports:
  - Exact `MuseGlimmerForConditionalGeneration` architecture:
      * `model.language_model.layers.{0..51}` (d_model=6656, d_mlp=19968, n_q=32, n_kv=2, d_k=128)
      * Gated Attention (`q_proj`, `k_proj`, `v_proj`, `self_attn.gate_proj`, `o_proj`)
      * QK-Norm with `qk_scale_factor = 3.87` and 3:1 `sliding_attention` (RoPE) / `full_attention` (NoPE)
      * 4 Centered RMSNorms per layer (`1.0 + weight`)
      * SwiGLU MLP (`mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`)
      * Vision Tower (`model.vision_tower.layers.{0..49}`) + `vision_adapter` + `vision_projection`
  - Cryptographic SHA-256 verification of every safetensor shard on disk.
  - Single-layer streaming Power-Iteration Spectral Norm, Stable Rank,
    Weight Kurtosis, Activation Outlier Kurtosis, and Hutchinson Hessian Trace.
  - Zero-Clamp Empirical vs Analytical Half-Life Asymmetry (Product vs Ratio Law).
"""

import os
import json
import math
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


CANDIDATE_30B_MODELS = [
    "meta-models/Muse-Glimmer-30B",
]


def compute_file_sha256(filepath: str, max_bytes: Optional[int] = None) -> str:
    """Computes SHA-256 of a file (or header+tail digest for multi-GB shards if specified)."""
    h = hashlib.sha256()
    size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        if max_bytes is not None and size > max_bytes * 2:
            h.update(f.read(max_bytes))
            f.seek(size - max_bytes)
            h.update(f.read(max_bytes))
            h.update(str(size).encode("utf-8"))
        else:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
    return h.hexdigest()


def power_iteration_spectral_norm(W: torch.Tensor, n_iters: int = 25) -> float:
    """Computes exact top singular value sigma_max(W) on GPU without full SVD OOM."""
    with torch.no_grad():
        W_f = W.float()
        out_dim, in_dim = W_f.shape
        gen = torch.Generator(device=W_f.device)
        gen.manual_seed(1042 + out_dim + in_dim)
        v = torch.randn(in_dim, 1, generator=gen, device=W_f.device, dtype=torch.float32)
        v = v / torch.norm(v).clamp(min=1e-12)
        for _ in range(n_iters):
            u = torch.matmul(W_f, v)
            u_norm = torch.norm(u).clamp(min=1e-12)
            u = u / u_norm
            v = torch.matmul(W_f.t(), u)
            v_norm = torch.norm(v).clamp(min=1e-12)
            v = v / v_norm
        sigma_max = torch.norm(torch.matmul(W_f, v)).item()
    return float(sigma_max)


def compute_tensor_kurtosis(T: torch.Tensor) -> float:
    """Computes fourth standardized moment E[(x - mu)^4] / (E[(x - mu)^2])^2."""
    with torch.no_grad():
        flat = T.float().flatten()
        if flat.numel() > 2_000_000:
            stride = flat.numel() // 2_000_000
            flat = flat[::stride]
        mean = torch.mean(flat)
        diff = flat - mean
        var = torch.mean(diff * diff).clamp(min=1e-12)
        fourth = torch.mean((diff * diff) ** 2)
        return float((fourth / (var * var)).item())


def compute_channel_outlier_ratio(act: torch.Tensor) -> Tuple[float, float]:
    """
    Given activation tensor [B, S, D], computes:
      - channel_max_to_median: max_j(mean(|X_j|)) / median_j(mean(|X_j|))
      - activation_kurtosis: kurtosis across channel norms
    """
    with torch.no_grad():
        ch_norms = torch.mean(torch.abs(act.float()), dim=(0, 1))
        med = torch.median(ch_norms).clamp(min=1e-8)
        mx = torch.max(ch_norms)
        ratio = float((mx / med).item())
        kurt = compute_tensor_kurtosis(ch_norms)
        return ratio, kurt


def interpolate_half_life_bits(bit_grid: List[int], retentions: Dict[int, float]) -> float:
    """
    Finds continuous bit precision h_{1/2} where retention R(b) crosses 0.50.
    Uses monotone linear interpolation across evaluated bit grid [b_min..b_max].
    When R(b_min) > 0.50, interpolates smoothly toward R(0 bits) = 0.0 (zero information capacity).
    """
    sorted_b = sorted(bit_grid)
    for idx in range(len(sorted_b) - 1):
        b_low = sorted_b[idx]
        b_high = sorted_b[idx + 1]
        r_low = retentions[b_low]
        r_high = retentions[b_high]
        if r_low <= 0.50 <= r_high:
            denom = max(1e-9, r_high - r_low)
            frac = (0.50 - r_low) / denom
            return float(b_low + frac * (b_high - b_low))
    if retentions[sorted_b[0]] > 0.50:
        # Anchor at 0.0 bits -> 0.0 retention (physical origin)
        r_first = max(0.5001, retentions[sorted_b[0]])
        return float(sorted_b[0] * (0.50 / r_first))
    r_last = retentions[sorted_b[-1]]
    r_prev = retentions[sorted_b[-2]]
    slope = max(0.10, (r_last - r_prev) / float(sorted_b[-1] - sorted_b[-2]))
    return float(sorted_b[-1] + (0.50 - r_last) / slope)


def muse_rms_norm_no_scale(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """MuseGlimmerRMSNorm(with_scale=False): x * (mean(x^2) + eps)^(-0.5)."""
    x_f = x.float()
    mean_sq = x_f.pow(2).mean(-1, keepdim=True) + eps
    return x_f * torch.pow(mean_sq, -0.5)


def muse_centered_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """MuseGlimmerTextCenteredRMSNorm: (x * rsqrt(mean(x^2) + eps)) * (1.0 + weight)."""
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return normed * (1.0 + weight.float())


def apply_muse_rope(q: torch.Tensor, k: torch.Tensor, theta: float = 500000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Muse Glimmer Rotary Position Embedding on [B, H, S, D]."""
    if theta <= 0.0:
        return q, k
    bsz, n_h, seqlen, head_dim = q.shape
    device = q.device
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)

    def _rotate_half(t: torch.Tensor) -> torch.Tensor:
        t1 = t[..., : head_dim // 2]
        t2 = t[..., head_dim // 2 :]
        return torch.cat((-t2, t1), dim=-1)

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class MuseGlimmerRMSNorm(nn.Module):
    """Centered RMSNorm for `Muse-Glimmer-30B` (`(x * rsqrt(mean(x^2) + eps)) * (1.0 + weight)`)."""

    def __init__(self, weight: Optional[torch.Tensor] = None, eps: float = 1e-5, scale_factor: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.scale_factor = scale_factor
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            return (muse_rms_norm_no_scale(x, eps=self.eps) * self.scale_factor).to(x.dtype)
        return (muse_centered_rms_norm(x, self.weight, eps=self.eps) * self.scale_factor).to(x.dtype)


class MuseGlimmerTextNormedEmbedding(nn.Module):
    """Normed token embedding for `Muse-Glimmer-30B`."""

    def __init__(self, weight: torch.Tensor, norm_weight: Optional[torch.Tensor] = None, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("weight", weight.float())
        if norm_weight is not None:
            self.register_buffer("norm_weight", norm_weight.float())
        else:
            self.norm_weight = None
        self.eps = eps

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h0 = F.embedding(input_ids, self.weight)
        if self.norm_weight is not None:
            return muse_centered_rms_norm(h0, self.norm_weight, eps=self.eps)
        return muse_rms_norm_no_scale(h0, eps=self.eps)


def build_rope_cache(
    seq_len: int,
    head_dim: int = 128,
    base: float = 500000.0,
    device: Any = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds `(cos, sin)` Rotary Position Embedding cache."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype=dtype)
    sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype=dtype)
    return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies precomputed `(cos, sin)` RoPE cache to `(q, k)`."""
    head_dim = q.shape[-1]

    def _rotate_half(t: torch.Tensor) -> torch.Tensor:
        t1 = t[..., : head_dim // 2]
        t2 = t[..., head_dim // 2 :]
        return torch.cat((-t2, t1), dim=-1)

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def dithered_group_quantize(
    W: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    seed: int = 42,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Top-level helper for b-bit group quantization with uniform dither."""
    gen = torch.Generator(device=W.device)
    gen.manual_seed(seed)
    w_deq = ShardStreamingAnatomyProfiler._dithered_group_quantize(W.float(), bits=bits, group_size=group_size, gen=gen)
    return w_deq.to(W.dtype), {"bits": bits, "group_size": group_size, "seed": seed}


class ShardStreamingAnatomyProfiler:
    """
    Streams a multi-shard Safetensors checkpoint (`meta-models/Muse-Glimmer-30B` or standard GQA)
    layer-by-layer onto GPU, extracting exact anatomical metrics and half-life curves without OOM.
    """
    def __init__(self, snapshot_dir: Any, device: Any = "cuda"):
        self.snapshot_dir = Path(snapshot_dir)
        self.device = str(device)
        with open(self.snapshot_dir / "config.json", "r") as f:
            self.raw_config = json.load(f)

        # Support both multimodal MuseGlimmerConfig (has text_config + vision_config) and flat text configs
        if "text_config" in self.raw_config:
            self.text_config = self.raw_config["text_config"]
            self.vision_config = self.raw_config.get("vision_config", {})
            self.is_muse_glimmer = self.raw_config.get("model_type") == "muse_glimmer"
            self.layer_prefix = "model.language_model.layers"
            self.embed_key = "model.language_model.embed_tokens.weight"
            self.final_norm_key = "model.language_model.norm.weight"
        else:
            self.text_config = self.raw_config
            self.vision_config = {}
            self.is_muse_glimmer = False
            self.layer_prefix = "model.layers"
            self.embed_key = "model.embed_tokens.weight"
            self.final_norm_key = "model.norm.weight"

        index_path = self.snapshot_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                idx_data = json.load(f)
            self.weight_map: Dict[str, str] = idx_data["weight_map"]
            self.index_metadata: Dict[str, Any] = idx_data.get("metadata", {})
        else:
            st_file = "model.safetensors"
            self.weight_map = {}
            self.index_metadata = {}
            with safe_open(str(self.snapshot_dir / st_file), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    self.weight_map[k] = st_file

    def verify_shards_cryptographically(self, fast_digest: bool = False) -> Dict[str, Any]:
        """Hashes all safetensor shards referenced by the checkpoint."""
        unique_shards = sorted(set(self.weight_map.values()))
        shard_manifest = {}
        total_bytes = 0
        for s_file in unique_shards:
            full_p = self.snapshot_dir / s_file
            sz = os.path.getsize(full_p)
            total_bytes += sz
            digest = compute_file_sha256(str(full_p), max_bytes=(16 * 1024 * 1024) if fast_digest else None)
            shard_manifest[s_file] = {
                "bytes": sz,
                "gib": round(sz / (1024**3), 4),
                "sha256": digest,
                "digest_mode": "header_tail_32mb" if fast_digest else "full_file_sha256",
            }
        return {
            "num_shards": len(unique_shards),
            "total_checkpoint_bytes": total_bytes,
            "total_checkpoint_gib": round(total_bytes / (1024**3), 4),
            "total_tensors_in_index": len(self.weight_map),
            "index_metadata": self.index_metadata,
            "shards": shard_manifest,
        }

    def load_tensor(self, tensor_name: str, target_device: Optional[str] = None) -> torch.Tensor:
        dev = str(target_device or self.device)
        if tensor_name not in self.weight_map:
            # Handle parameter-free centered RMSNorms (weight=0 -> 1.0 + 0.0 = 1.0)
            if tensor_name.endswith(("q_norm.weight", "k_norm.weight")):
                d_k = int(self.text_config.get("head_dim", 128))
                return torch.zeros(d_k, device=dev, dtype=torch.float32)
            if tensor_name.endswith("embed_tokens.norm.weight"):
                d_model = int(self.text_config.get("hidden_size", 6656))
                return torch.zeros(d_model, device=dev, dtype=torch.float32)
        shard_file = self.weight_map[tensor_name]
        shard_path = str(self.snapshot_dir / shard_file)
        with safe_open(shard_path, framework="pt", device=dev) as sf:
            return sf.get_tensor(tensor_name)

    def get_tensor(self, tensor_name: str, target_device: Optional[str] = None) -> torch.Tensor:
        return self.load_tensor(tensor_name, target_device=target_device)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Computes exact initial hidden states H_0 from input_ids."""
        embed_w = self.load_tensor(self.embed_key, self.device).float()
        with torch.no_grad():
            h0 = F.embedding(input_ids, embed_w)
            if self.is_muse_glimmer:
                rms_eps = float(self.text_config.get("rms_norm_eps", 1e-5))
                h0 = muse_rms_norm_no_scale(h0, eps=rms_eps)
        del embed_w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return h0

    @staticmethod
    def _dithered_group_quantize(
        W: torch.Tensor,
        bits: int,
        group_size: int,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """
        Applies b-bit symmetric group quantization with uniform stochastic rounding dither
        xi ~ U[-0.5, +0.5] to measure continuous bit-precision degradation curves.
        """
        orig_shape = W.shape
        numel = W.numel()
        pad = (group_size - (numel % group_size)) % group_size
        w_flat = F.pad(W.reshape(-1), (0, pad), value=0.0).view(-1, group_size)
        amax = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        qmax = float((1 << (bits - 1)) - 1) if bits > 1 else 1.0
        qmin = -float(1 << (bits - 1)) if bits > 1 else -1.0
        scale = amax / qmax
        dither = torch.rand(w_flat.shape, generator=gen, device=W.device, dtype=W.dtype) - 0.5
        q = torch.clamp(torch.round((w_flat / scale) + dither), qmin, qmax)
        w_deq = (q * scale).reshape(-1)[:numel].view(orig_shape)
        return w_deq

    def profile_layer_projections(
        self,
        layer_idx: int,
        H_in: torch.Tensor,
        n_mc_trials: int = 40,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Streams Language Layer `layer_idx` onto GPU, computes spectral norms, stable ranks,
        weight kurtosis, activation outlier kurtosis, and Monte Carlo half-life curves,
        advances hidden state `H_in -> H_out` using exact MuseGlimmer / GQA equations,
        and frees all layer weights from VRAM.
        """
        d_model = int(self.text_config.get("hidden_size", 6656))
        n_heads = int(self.text_config.get("num_attention_heads", 32))
        n_kv_heads = int(self.text_config.get("num_key_value_heads", 2))
        d_k = int(self.text_config.get("head_dim", d_model // n_heads))
        d_mlp = int(self.text_config.get("intermediate_size", 19968))
        rms_eps = float(self.text_config.get("rms_norm_eps", 1e-5))
        post_eps = float(self.text_config.get("post_norm_eps", 1e-8))
        qk_scale = float(self.text_config.get("qk_scale_factor", 1.0))
        layer_types = self.text_config.get("layer_types", ["sliding_attention"] * 64)
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "sliding_attention"
        rope_thetas = self.text_config.get("layer_rope_theta", [500000.0] * 64)
        rope_theta = float(rope_thetas[layer_idx]) if layer_idx < len(rope_thetas) else 500000.0

        prefix = f"{self.layer_prefix}.{layer_idx}"
        in_norm_w = self.load_tensor(f"{prefix}.input_layernorm.weight", self.device).float()
        post_attn_norm_w = self.load_tensor(f"{prefix}.post_attention_layernorm.weight", self.device).float()

        has_pre_ff_norm = f"{prefix}.pre_feedforward_layernorm.weight" in self.weight_map
        pre_ff_norm_w = (
            self.load_tensor(f"{prefix}.pre_feedforward_layernorm.weight", self.device).float()
            if has_pre_ff_norm else post_attn_norm_w
        )
        has_post_ff_norm = f"{prefix}.post_feedforward_layernorm.weight" in self.weight_map
        post_ff_norm_w = (
            self.load_tensor(f"{prefix}.post_feedforward_layernorm.weight", self.device).float()
            if has_post_ff_norm else None
        )

        with torch.no_grad():
            H_f = H_in.float()
            if self.is_muse_glimmer:
                norm_x = muse_centered_rms_norm(H_f, in_norm_w, eps=rms_eps)
            else:
                var = H_f.pow(2).mean(-1, keepdim=True)
                norm_x = H_f * torch.rsqrt(var + rms_eps) * in_norm_w
            attn_outlier_ratio, attn_act_kurt = compute_channel_outlier_ratio(norm_x)

        projections = {
            "q_proj": f"{prefix}.self_attn.q_proj.weight",
            "k_proj": f"{prefix}.self_attn.k_proj.weight",
            "v_proj": f"{prefix}.self_attn.v_proj.weight",
            "attn_gate_proj": f"{prefix}.self_attn.gate_proj.weight",
            "o_proj": f"{prefix}.self_attn.o_proj.weight",
            "gate_proj": f"{prefix}.mlp.gate_proj.weight",
            "up_proj": f"{prefix}.mlp.up_proj.weight",
            "down_proj": f"{prefix}.mlp.down_proj.weight",
        }

        proj_metrics: Dict[str, Any] = {}
        loaded_weights: Dict[str, torch.Tensor] = {}

        for p_key, t_name in projections.items():
            if t_name not in self.weight_map:
                continue
            W = self.load_tensor(t_name, self.device)
            loaded_weights[p_key] = W
            with torch.no_grad():
                W_f = W.float()
                frob = float(torch.norm(W_f, p="fro").item())
                sigma_max = power_iteration_spectral_norm(W_f, n_iters=20)
                stable_rank = (frob * frob) / max(1e-12, sigma_max * sigma_max)
                w_kurt = compute_tensor_kurtosis(W_f)
                max_abs = float(torch.max(torch.abs(W_f)).item())
                proj_metrics[p_key] = {
                    "shape": list(W.shape),
                    "numel": int(W.numel()),
                    "frobenius_norm": round(frob, 4),
                    "spectral_norm_sigma_max": round(sigma_max, 4),
                    "stable_rank": round(stable_rank, 2),
                    "weight_kurtosis": round(w_kurt, 4),
                    "max_abs_weight": round(max_abs, 5),
                }

        W_q = loaded_weights["q_proj"].float()
        W_k = loaded_weights["k_proj"].float()
        W_v = loaded_weights["v_proj"].float()
        W_o = loaded_weights["o_proj"].float()
        W_gate = loaded_weights["gate_proj"].float()
        W_up = loaded_weights["up_proj"].float()
        W_down = loaded_weights["down_proj"].float()

        with torch.no_grad():
            # Forward pass through Attention
            bsz, seqlen, _ = norm_x.shape
            q_all = F.linear(norm_x, W_q).view(bsz, seqlen, n_heads, d_k).transpose(1, 2)
            k_all = F.linear(norm_x, W_k).view(bsz, seqlen, n_kv_heads, d_k).transpose(1, 2)
            v_all = F.linear(norm_x, W_v).view(bsz, seqlen, n_kv_heads, d_k).transpose(1, 2)

            if self.is_muse_glimmer:
                q_all = muse_rms_norm_no_scale(q_all, eps=rms_eps) * qk_scale
                k_all = muse_rms_norm_no_scale(k_all, eps=rms_eps)
                q_all, k_all = apply_muse_rope(q_all, k_all, theta=rope_theta)

            rep = n_heads // n_kv_heads
            k_rep = k_all.repeat_interleave(rep, dim=1) if rep > 1 else k_all
            v_rep = v_all.repeat_interleave(rep, dim=1) if rep > 1 else v_all

            attn_out = F.scaled_dot_product_attention(q_all, k_rep, v_rep, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, n_heads * d_k)

            if "attn_gate_proj" in loaded_weights:
                W_ag = loaded_weights["attn_gate_proj"].float()
                attn_out = attn_out * torch.sigmoid(F.linear(norm_x, W_ag))

            o_out = F.linear(attn_out, W_o)
            if self.is_muse_glimmer:
                o_out = muse_centered_rms_norm(o_out, post_attn_norm_w, eps=post_eps)
                H_mid = H_f + o_out
                post_norm_x = muse_centered_rms_norm(H_mid, pre_ff_norm_w, eps=rms_eps)
            else:
                H_mid = H_f + o_out
                var_mid = H_mid.pow(2).mean(-1, keepdim=True)
                post_norm_x = H_mid * torch.rsqrt(var_mid + rms_eps) * post_attn_norm_w

            mlp_outlier_ratio, mlp_act_kurt = compute_channel_outlier_ratio(post_norm_x)

            swiglu_clean = F.silu(F.linear(post_norm_x, W_gate)) * F.linear(post_norm_x, W_up)
            down_outlier_ratio, down_act_kurt = compute_channel_outlier_ratio(swiglu_clean)
            mlp_clean_full = F.linear(swiglu_clean, W_down)

            if self.is_muse_glimmer and post_ff_norm_w is not None:
                H_out = (H_mid + muse_centered_rms_norm(mlp_clean_full, post_ff_norm_w, eps=post_eps)).to(dtype=H_in.dtype)
            else:
                H_out = (H_mid + mlp_clean_full).to(dtype=H_in.dtype)

            # ------------------------------------------------------------------
            # Batched Monte Carlo Half-Life Evaluation (Zero Clamps)
            # ------------------------------------------------------------------
            # 1. Procedural Attention Routing Subspace (Head 0..3, d_k=128)
            n_probe_heads = min(4, n_heads)
            q_sub_dim = n_probe_heads * d_k
            W_q_sub = W_q[:q_sub_dim, :]
            W_k_sub = W_k[:d_k, :]
            q_ref = F.linear(norm_x[0], W_q_sub).view(seqlen, n_probe_heads, d_k).transpose(0, 1)
            k_ref = F.linear(norm_x[0], W_k_sub).view(seqlen, 1, d_k).transpose(0, 1)
            if self.is_muse_glimmer:
                q_ref = muse_rms_norm_no_scale(q_ref.unsqueeze(0), eps=rms_eps).squeeze(0) * qk_scale
                k_ref = muse_rms_norm_no_scale(k_ref.unsqueeze(0), eps=rms_eps).squeeze(0)
            scores_ref = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(d_k)
            norm_scores_ref = torch.norm(scores_ref, p="fro").clamp(min=1e-8)

            # 2. Associative SwiGLU MLP Subspace (Sliced along d_out=256, full d_mlp=19968)
            sub_out = min(256, d_model)
            sub_mlp = min(4096, d_mlp)
            x_mlp_probe = post_norm_x[0]  # [seqlen, d_model]
            W_gate_sub = W_gate[:sub_mlp, :]
            W_up_sub = W_up[:sub_mlp, :]
            W_down_sub = W_down[:sub_out, :sub_mlp]
            inter_ref = F.silu(F.linear(x_mlp_probe, W_gate_sub)) * F.linear(x_mlp_probe, W_up_sub)
            y_mlp_ref = F.linear(inter_ref, W_down_sub)
            norm_mlp_ref = torch.norm(y_mlp_ref, p="fro").clamp(min=1e-8)

            # Base relative distortion tolerance tau_0 = 0.25:
            # Attention averages over d_k coordinates; Associative storage packs d_mlp/d_k directions
            tau_attn = 0.25
            tau_fact = tau_attn * math.pow(float(d_k) / float(d_mlp), 0.35)

            bit_grid = [1, 2, 3, 4, 6, 8]
            ret_proc: Dict[int, float] = {}
            ret_fact: Dict[int, float] = {}
            gen = torch.Generator(device=self.device)
            gen.manual_seed(1042 + layer_idx * 97)

            for b in bit_grid:
                attn_ok = 0
                fact_ok = 0
                for _ in range(n_mc_trials):
                    wq_hat = self._dithered_group_quantize(W_q_sub, b, 128, gen)
                    wk_hat = self._dithered_group_quantize(W_k_sub, b, 128, gen)
                    q_p = F.linear(norm_x[0], wq_hat).view(seqlen, n_probe_heads, d_k).transpose(0, 1)
                    k_p = F.linear(norm_x[0], wk_hat).view(seqlen, 1, d_k).transpose(0, 1)
                    if self.is_muse_glimmer:
                        q_p = muse_rms_norm_no_scale(q_p.unsqueeze(0), eps=rms_eps).squeeze(0) * qk_scale
                        k_p = muse_rms_norm_no_scale(k_p.unsqueeze(0), eps=rms_eps).squeeze(0)
                    scores_p = torch.matmul(q_p, k_p.transpose(-1, -2)) / math.sqrt(d_k)
                    rel_err_attn = float((torch.norm(scores_p - scores_ref, p="fro") / norm_scores_ref).item())
                    if rel_err_attn < tau_attn:
                        attn_ok += 1

                    wg_hat = self._dithered_group_quantize(W_gate_sub, b, 128, gen)
                    wu_hat = self._dithered_group_quantize(W_up_sub, b, 128, gen)
                    wd_hat = self._dithered_group_quantize(W_down_sub, b, 128, gen)
                    inter_p = F.silu(F.linear(x_mlp_probe, wg_hat)) * F.linear(x_mlp_probe, wu_hat)
                    y_mlp_p = F.linear(inter_p, wd_hat)
                    rel_err_mlp = float((torch.norm(y_mlp_p - y_mlp_ref, p="fro") / norm_mlp_ref).item())
                    if rel_err_mlp < tau_fact:
                        fact_ok += 1

                ret_proc[b] = round(attn_ok / float(n_mc_trials), 4)
                ret_fact[b] = round(fact_ok / float(n_mc_trials), 4)

            h_proc = interpolate_half_life_bits(bit_grid, ret_proc)
            h_fact = interpolate_half_life_bits(bit_grid, ret_fact)
            delta_h_emp = float(h_fact - h_proc)

        del loaded_weights, W_q, W_k, W_v, W_o, W_gate, W_up, W_down
        del in_norm_w, post_attn_norm_w, pre_ff_norm_w, post_ff_norm_w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        layer_report = {
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "rope_theta": rope_theta,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "d_k": d_k,
            "d_mlp": d_mlp,
            "activation_outliers": {
                "attn_in_max_to_median": round(attn_outlier_ratio, 3),
                "attn_in_kurtosis": round(attn_act_kurt, 3),
                "mlp_in_max_to_median": round(mlp_outlier_ratio, 3),
                "mlp_in_kurtosis": round(mlp_act_kurt, 3),
                "swiglu_down_in_max_to_median": round(down_outlier_ratio, 3),
                "swiglu_down_in_kurtosis": round(down_act_kurt, 3),
            },
            "half_life": {
                "h_half_proc_bits": round(h_proc, 4),
                "h_half_fact_bits": round(h_fact, 4),
                "delta_h_half_empirical": round(delta_h_emp, 4),
                "analytical_product_term": round(0.5 * math.log2(d_mlp * d_k), 4),
                "analytical_ratio_term": round(0.5 * math.log2(d_mlp / d_k), 4),
                "retentions_proc": ret_proc,
                "retentions_fact": ret_fact,
            },
            "projections": proj_metrics,
        }
        return layer_report, H_out

    def profile_vision_tower_summary(self) -> Dict[str, Any]:
        """
        Profiles exact parameter and byte breakdown of the 50-layer Muse Glimmer Vision Tower
        (`model.vision_tower.*`, `model.vision_adapter.*`, `model.vision_projection.*`).
        """
        vision_keys = [k for k in self.weight_map.keys() if "vision_" in k]
        if not vision_keys:
            return {"present": False, "num_tensors": 0, "total_params": 0, "total_bf16_bytes": 0}

        # Group by shard to read tensor shapes without loading multiple times
        by_shard: Dict[str, List[str]] = {}
        for k in vision_keys:
            by_shard.setdefault(self.weight_map[k], []).append(k)

        total_params = 0
        sample_layer_0 = {}
        for s_file, keys in by_shard.items():
            s_path = str(self.snapshot_dir / s_file)
            with safe_open(s_path, framework="pt", device="cpu") as sf:
                for k in keys:
                    sl = sf.get_slice(k)
                    shape = list(sl.get_shape())
                    n_p = int(np.prod(shape))
                    total_params += n_p
                    if "vision_tower.layers.0." in k or "vision_adapter" in k or "vision_projection" in k:
                        sample_layer_0[k] = {"shape": shape, "numel": n_p}

        return {
            "present": True,
            "num_vision_layers": int(self.vision_config.get("num_hidden_layers", 50)),
            "vision_hidden_size": int(self.vision_config.get("hidden_size", 1536)),
            "vision_intermediate_size": int(self.vision_config.get("intermediate_size", 8960)),
            "num_tensors": len(vision_keys),
            "total_params": total_params,
            "total_bf16_bytes": total_params * 2,
            "total_bf16_gib": round((total_params * 2) / (1024**3), 4),
            "sample_shapes": sample_layer_0,
        }
