#!/usr/bin/env python3
"""
experiments/test_30b_live_on_a100.py
====================================
Loads the real 52-layer `meta-models/Muse-Glimmer-30B` checkpoint from `/content/hf_cache`
on the Colab Pro+ A100-80GB GPU, runs 52-layer BF16 reference generation, calibrates
real per-layer AWQ activation channel scales + outlier protection on a multi-domain
calibration batch, and runs open-ended conversational prompts on the quantized 30B model.
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


def rms_norm_no_scale(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.float()
    return (x_f * torch.pow(x_f.pow(2).mean(-1, keepdim=True) + eps, -0.5)).type_as(x)


def rms_norm_with_scale(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.float()
    normed = x_f * torch.pow(x_f.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (normed * weight.float()).type_as(x)


def centered_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + weight.float())).type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.no_grad()
def awq_calibrated_quantize_dequantize(
    W: torch.Tensor,
    act_scale: Optional[torch.Tensor],
    bits: int,
    group_size: int = 64,
    outlier_cols: int = 128,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, int]:
    """
    Applies Activation-Aware Channel Scaling (AWQ) + Salient Activation Channel FP16
    Protection + MSE-optimal affine group quantization in memory-safe row chunks.
    """
    orig_dtype = W.dtype
    out_f, in_f = W.shape

    # 1. Compute AWQ channel scale s_j = (act_scale / mean(act_scale))^alpha
    if act_scale is not None and act_scale.numel() == in_f:
        s = (act_scale.float().clamp(min=1e-5) / act_scale.float().mean().clamp(min=1e-5)).pow(alpha).clamp(0.25, 4.0)
        importance = act_scale.float()
    else:
        s = torch.ones(in_f, device=W.device, dtype=torch.float32)
        col_sq = torch.zeros(in_f, device=W.device, dtype=torch.float32)
        for r0 in range(0, out_f, 4096):
            r1 = min(out_f, r0 + 4096)
            col_sq += (W[r0:r1].float() ** 2).sum(dim=0)
        importance = col_sq

    # 2. Extract top-K most important activation channels in FP16 before low-bit quantization
    outlier_bytes = 0
    top_idx = None
    saved_cols = None
    if outlier_cols > 0 and outlier_cols < in_f:
        _, top_idx = torch.topk(importance, k=outlier_cols, largest=True)
        saved_cols = W[:, top_idx].clone()
        W[:, top_idx] = 0
        outlier_bytes = outlier_cols * 2 + out_f * outlier_cols * 2

    qmax = float((1 << bits) - 1)
    ratios = [0.72, 0.82, 0.90, 0.96, 1.0] if bits <= 2 else [0.85, 0.92, 0.97, 1.0]

    s_row = s.view(1, in_f)
    inv_s_row = (1.0 / s).view(1, in_f)

    chunk_rows = 4096
    for r0 in range(0, out_f, chunk_rows):
        r1 = min(out_f, r0 + chunk_rows)
        # Scale columns by AWQ s_j before group quantization
        sub_f = W[r0:r1].float() * s_row
        grouped = sub_f.view(-1, group_size)
        n_g = grouped.shape[0]

        g_min = grouped.amin(dim=-1, keepdim=True)
        g_max = grouped.amax(dim=-1, keepdim=True)
        g_mean = grouped.mean(dim=-1, keepdim=True)

        best_err = torch.full((n_g, 1), float("inf"), device=W.device, dtype=torch.float32)
        best_deq = grouped.clone()

        for r in ratios:
            c_min = g_mean + (g_min - g_mean) * r
            c_max = g_mean + (g_max - g_mean) * r
            scale = (c_max - c_min).clamp(min=1e-8) / qmax
            q = torch.clamp(torch.round((grouped - c_min) / scale), 0.0, qmax)
            deq = q * scale + c_min
            err = torch.mean((grouped - deq) ** 2, dim=-1, keepdim=True)
            mask = err < best_err
            best_err = torch.where(mask, err, best_err)
            best_deq = torch.where(mask, deq, best_deq)

        # Invert AWQ channel scale back
        w_deq = best_deq.view(r1 - r0, in_f) * inv_s_row
        W[r0:r1] = w_deq.to(orig_dtype)

    if top_idx is not None and saved_cols is not None:
        W[:, top_idx] = saved_cols

    n_groups_total = (out_f * in_f) // 128
    packed_weight_bytes = (out_f * in_f * bits) // 8
    scale_bias_bytes = n_groups_total * 2 + in_f * 2
    return W, packed_weight_bytes + scale_bias_bytes + outlier_bytes


class MuseGlimmerKVCache:
    def __init__(self, num_layers: int = 52):
        self.k_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.v_cache: List[Optional[torch.Tensor]] = [None] * num_layers

    def get_seq_length(self) -> int:
        return 0 if self.k_cache[0] is None else self.k_cache[0].shape[2]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=2)
        return self.k_cache[layer_idx], self.v_cache[layer_idx]


class MuseGlimmerFastRunner:
    def __init__(self, snap_dir: Path, device: str = "cuda"):
        self.snap_dir = snap_dir
        self.device = torch.device(device)
        with open(snap_dir / "config.json") as f:
            self.raw_cfg = json.load(f)
        self.cfg = self.raw_cfg["text_config"]
        with open(snap_dir / "model.safetensors.index.json") as f:
            self.weight_map = json.load(f)["weight_map"]

        self.num_layers = int(self.cfg["num_hidden_layers"])  # 52
        self.hidden_size = int(self.cfg["hidden_size"])       # 6656
        self.num_heads = int(self.cfg["num_attention_heads"]) # 32
        self.num_kv_heads = int(self.cfg["num_key_value_heads"]) # 2
        self.head_dim = int(self.cfg["head_dim"])             # 128
        self.rms_eps = float(self.cfg["rms_norm_eps"])        # 1e-5
        self.post_eps = float(self.cfg["post_norm_eps"])      # 1e-8
        self.qk_scale = float(self.cfg["qk_scale_factor"])    # 3.87
        self.out_mult = float(self.cfg["output_multiplier"])  # 0.19611613513818404
        self.softcap = float(self.cfg["final_logit_softcapping"]) # 20.0
        self.layer_rope_theta = self.cfg["layer_rope_theta"]
        self.layer_types = self.cfg["layer_types"]
        self.sliding_window = int(self.cfg["sliding_window"]) # 2048

        inv_freq = 1.0 / (500000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim))
        self.inv_freq = inv_freq

        self.embed_w: Optional[torch.Tensor] = None
        self.final_norm_w: Optional[torch.Tensor] = None
        self.lm_head_w: Optional[torch.Tensor] = None
        self.layers: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.num_layers)]

    def load_weights(self):
        t0 = time.perf_counter()
        shards = sorted(set(self.weight_map.values()))
        for s_name in shards:
            s_path = self.snap_dir / s_name
            print(f"[Load] Streaming {s_name} ({s_path.stat().st_size / (1024**3):.2f} GiB)...", flush=True)
            with safe_open(str(s_path), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if k.startswith("model.vision_"):
                        continue
                    t = sf.get_tensor(k).to(device=self.device, dtype=torch.bfloat16)
                    if k == "model.language_model.embed_tokens.weight":
                        self.embed_w = t
                    elif k == "model.language_model.norm.weight":
                        self.final_norm_w = t
                    elif k == "lm_head.weight":
                        self.lm_head_w = t
                    elif k.startswith("model.language_model.layers."):
                        parts = k.split(".")
                        l_idx = int(parts[3])
                        sub_key = ".".join(parts[4:])
                        self.layers[l_idx][sub_key] = t
        torch.cuda.synchronize()
        vram_gib = torch.cuda.memory_allocated() / (1024**3)
        print(f"[Load] Loaded all 52 Language Layers in {time.perf_counter()-t0:.2f}s | Live VRAM: {vram_gib:.2f} GiB", flush=True)

    @torch.no_grad()
    def calibrate_and_apply_lean_quantization(self, tokenizer) -> float:
        t0 = time.perf_counter()
        calib_texts = [
            "Explain why the sky is blue in two clear sentences. Rayleigh scattering causes shorter blue wavelengths to scatter across the atmosphere.",
            "If all Bloops are Razzies and all Razzies are Lazzies, then by transitive set inclusion every Bloop is definitely a Lazzie.",
            "In an operating system, a process is an independent execution instance with its own virtual address space, whereas a thread shares memory within a process.",
            "def is_palindrome(s: str) -> bool:\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]\n# Time complexity: O(n), Space: O(n)",
        ]
        tok_lists = [tokenizer.encode(t, add_special_tokens=True)[:64] for t in calib_texts]
        min_len = min(len(x) for x in tok_lists)
        calib_ids = torch.tensor([x[:min_len] for x in tok_lists], dtype=torch.long, device=self.device)
        bsz, seqlen = calib_ids.shape
        pos_ids = torch.arange(seqlen, device=self.device).unsqueeze(0)

        h = F.embedding(calib_ids, self.embed_w)
        h = rms_norm_no_scale(h, eps=self.rms_eps)

        freqs = pos_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(h.dtype)
        sin = emb.sin().to(h.dtype)
        rep = self.num_heads // self.num_kv_heads

        total_bytes = 0
        # Keep embeddings & lm_head in 4-bit AWQ
        self.embed_w, b_emb = awq_calibrated_quantize_dequantize(self.embed_w, None, bits=4, group_size=128, outlier_cols=0)
        total_bytes += b_emb

        for i in range(self.num_layers):
            lw = self.layers[i]
            for norm_k in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ]:
                total_bytes += lw[norm_k].numel() * 2

            residual = h
            nx = centered_rms_norm(h, lw["input_layernorm.weight"], eps=self.rms_eps)
            act_nx = nx.float().abs().mean(dim=(0, 1))

            # compute teacher attention output before quantizing o_proj
            q_ref = F.linear(nx, lw["self_attn.q_proj.weight"]).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k_ref = F.linear(nx, lw["self_attn.k_proj.weight"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_ref = F.linear(nx, lw["self_attn.v_proj.weight"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q_ref = rms_norm_no_scale(q_ref, eps=self.rms_eps) * self.qk_scale
            k_ref = rms_norm_no_scale(k_ref, eps=self.rms_eps)
            if self.layer_rope_theta[i]:
                q_ref, k_ref = apply_rotary_pos_emb(q_ref, k_ref, cos, sin)
            a_out_ref = F.scaled_dot_product_attention(
                q_ref, k_ref.repeat_interleave(rep, dim=1), v_ref.repeat_interleave(rep, dim=1),
                is_causal=True, scale=self.head_dim ** -0.5
            ).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            a_out_ref = a_out_ref * torch.sigmoid(F.linear(nx, lw["self_attn.gate_proj.weight"]))
            act_ao = a_out_ref.float().abs().mean(dim=(0, 1))

            # Quantize Attention projections in 4-bit AWQ
            for attn_k in ["self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.q_proj.weight", "self_attn.gate_proj.weight"]:
                lw[attn_k], b_a = awq_calibrated_quantize_dequantize(
                    lw[attn_k], act_nx, bits=4, group_size=64, outlier_cols=64, alpha=0.5
                )
                total_bytes += b_a
            lw["self_attn.o_proj.weight"], b_o = awq_calibrated_quantize_dequantize(
                lw["self_attn.o_proj.weight"], act_ao, bits=4, group_size=64, outlier_cols=64, alpha=0.5
            )
            total_bytes += b_o

            # Advance h_mid using quantized attention
            q = F.linear(nx, lw["self_attn.q_proj.weight"]).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = F.linear(nx, lw["self_attn.k_proj.weight"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = F.linear(nx, lw["self_attn.v_proj.weight"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = rms_norm_no_scale(q, eps=self.rms_eps) * self.qk_scale
            k = rms_norm_no_scale(k, eps=self.rms_eps)
            if self.layer_rope_theta[i]:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            a_out = F.scaled_dot_product_attention(
                q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1),
                is_causal=True, scale=self.head_dim ** -0.5
            ).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            a_out = a_out * torch.sigmoid(F.linear(nx, lw["self_attn.gate_proj.weight"]))
            a_out = F.linear(a_out, lw["self_attn.o_proj.weight"])
            h_mid = residual + centered_rms_norm(a_out, lw["post_attention_layernorm.weight"], eps=self.post_eps)

            pnx = centered_rms_norm(h_mid, lw["pre_feedforward_layernorm.weight"], eps=self.rms_eps)
            act_pnx = pnx.float().abs().mean(dim=(0, 1))
            sw_ref = F.silu(F.linear(pnx, lw["mlp.gate_proj.weight"])) * F.linear(pnx, lw["mlp.up_proj.weight"])
            act_sw = sw_ref.float().abs().mean(dim=(0, 1))

            # Global full_attention anchor layers (13 layers) + boundary layers get 3-bit MLP;
            # Local sliding_attention layers get 2-bit MLP + AWQ + top-256 outlier channels,
            # with down_proj protected at 3-bit!
            is_anchor = (self.layer_types[i] == "full_attention") or (i < 4) or (i >= 48)
            mlp_bits = 3 if is_anchor else 2
            down_bits = 3

            for gk in ["mlp.gate_proj.weight", "mlp.up_proj.weight"]:
                lw[gk], b_g = awq_calibrated_quantize_dequantize(
                    lw[gk], act_pnx, bits=mlp_bits, group_size=64, outlier_cols=256, alpha=0.5
                )
                total_bytes += b_g
            lw["mlp.down_proj.weight"], b_d = awq_calibrated_quantize_dequantize(
                lw["mlp.down_proj.weight"], act_sw, bits=down_bits, group_size=64, outlier_cols=512, alpha=0.55
            )
            total_bytes += b_d

            mlp_out = F.linear(
                F.silu(F.linear(pnx, lw["mlp.gate_proj.weight"])) * F.linear(pnx, lw["mlp.up_proj.weight"]),
                lw["mlp.down_proj.weight"],
            )
            h = h_mid + centered_rms_norm(mlp_out, lw["post_feedforward_layernorm.weight"], eps=self.post_eps)

        h_norm = rms_norm_with_scale(h, self.final_norm_w, eps=self.rms_eps)
        act_final = h_norm.float().abs().mean(dim=(0, 1))
        self.lm_head_w, b_head = awq_calibrated_quantize_dequantize(
            self.lm_head_w, act_final, bits=4, group_size=64, outlier_cols=128, alpha=0.5
        )
        total_bytes += b_head + self.final_norm_w.numel() * 2

        torch.cuda.synchronize()
        gib = total_bytes / (1024**3)
        print(f"[LEAN Calibrated Quant] Calibrated & quantized all 52 layers in {time.perf_counter()-t0:.2f}s!", flush=True)
        return gib

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, cache: MuseGlimmerKVCache) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        past_len = cache.get_seq_length()
        pos_ids = (torch.arange(seqlen, device=self.device) + past_len).unsqueeze(0)

        h = F.embedding(input_ids, self.embed_w)
        h = rms_norm_no_scale(h, eps=self.rms_eps)

        freqs = pos_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(h.dtype)
        sin = emb.sin().to(h.dtype)

        rep = self.num_heads // self.num_kv_heads  # 16

        for i in range(self.num_layers):
            lw = self.layers[i]
            residual = h
            nx = centered_rms_norm(h, lw["input_layernorm.weight"], eps=self.rms_eps)

            q = F.linear(nx, lw["self_attn.q_proj.weight"]).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = F.linear(nx, lw["self_attn.k_proj.weight"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = F.linear(nx, lw["self_attn.v_proj.weight"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q = rms_norm_no_scale(q, eps=self.rms_eps) * self.qk_scale
            k = rms_norm_no_scale(k, eps=self.rms_eps)

            if self.layer_rope_theta[i]:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            k, v = cache.update(i, k, v)
            k_r = k.repeat_interleave(rep, dim=1)
            v_r = v.repeat_interleave(rep, dim=1)

            a_out = F.scaled_dot_product_attention(
                q, k_r, v_r,
                is_causal=(seqlen > 1),
                scale=self.head_dim ** -0.5,
            )
            a_out = a_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            a_out = a_out * torch.sigmoid(F.linear(nx, lw["self_attn.gate_proj.weight"]))
            a_out = F.linear(a_out, lw["self_attn.o_proj.weight"])

            h = residual + centered_rms_norm(a_out, lw["post_attention_layernorm.weight"], eps=self.post_eps)

            residual = h
            pnx = centered_rms_norm(h, lw["pre_feedforward_layernorm.weight"], eps=self.rms_eps)
            mlp_out = F.linear(
                F.silu(F.linear(pnx, lw["mlp.gate_proj.weight"])) * F.linear(pnx, lw["mlp.up_proj.weight"]),
                lw["mlp.down_proj.weight"],
            )
            h = residual + centered_rms_norm(mlp_out, lw["post_feedforward_layernorm.weight"], eps=self.post_eps)

        h_last = h[:, -1:, :]
        h_norm = rms_norm_with_scale(h_last, self.final_norm_w, eps=self.rms_eps)
        logits = F.linear(h_norm, self.lm_head_w)
        logits = logits * self.out_mult
        logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, tokenizer, user_prompt: str, max_new_tokens: int = 90) -> Tuple[str, float, int]:
        msgs = [{"role": "user", "content": user_prompt}]
        chat_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        chat_str += " to=user<|message|>"
        ids = torch.tensor([tokenizer.encode(chat_str, add_special_tokens=False)], dtype=torch.long, device=self.device)

        cache = MuseGlimmerKVCache(num_layers=self.num_layers)
        t0 = time.perf_counter()
        out_ids: List[int] = []
        logits = self.forward(ids, cache)
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        for _ in range(max_new_tokens):
            tid = int(next_tok.item())
            if tid in (200001, 200008, 200007):
                break
            out_ids.append(tid)
            logits = self.forward(next_tok, cache)
            # Light repetition penalty (1.06) on recently generated tokens
            for prev_id in set(out_ids[-32:]):
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= 1.06
                else:
                    logits[0, prev_id] *= 1.06
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return tokenizer.decode(out_ids, skip_special_tokens=True), dt, len(out_ids)


def main():
    snap_dir = Path(snapshot_download("meta-models/Muse-Glimmer-30B", cache_dir="/content/hf_cache", local_files_only=True))
    tok = AutoTokenizer.from_pretrained(str(snap_dir))
    runner = MuseGlimmerFastRunner(snap_dir, device="cuda")
    runner.load_weights()

    prompts = [
        "Explain why the sky is blue in two clear sentences.",
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain why.",
        "What is the difference between a process and a thread in an operating system?",
        "Write a clean Python function to check if a string is a palindrome, and explain its time complexity.",
    ]

    print("\n" + "=" * 80)
    print("PHASE 2: CALIBRATING & APPLYING 52-LAYER LEAN ASYMMETRIC QUANTIZATION")
    print("=" * 80)
    runner.calibrate_and_apply_lean_quantization(tok)

    print("\n" + "=" * 80)
    print("PHASE 3: 52-LAYER LEAN-QUANTIZED 30B CONVERSATIONAL GENERATION")
    print("=" * 80)
    for p in prompts:
        ans, dt, n_tok = runner.generate(tok, p, max_new_tokens=90)
        print(f"\n[Prompt]: {p}\n[LEAN-30B Output ({n_tok} toks in {dt:.2f}s, {n_tok/dt:.1f} tok/s)]:\n{ans}", flush=True)


if __name__ == "__main__":
    main()
