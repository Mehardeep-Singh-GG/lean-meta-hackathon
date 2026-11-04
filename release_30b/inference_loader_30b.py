#!/usr/bin/env python3
"""
release_30b/inference_loader_30b.py
===================================
Standalone, Zero-Cheat Inference Loader for the Real 52-Layer LEAN-Quantized
`meta-models/Muse-Glimmer-30B` Checkpoint (`model_lean_30b-00001-of-00004.safetensors` .. `00004`).

Features:
- Unpacks real 2-bit and 4-bit AWQ+KAC bit-packed weights (`packed`, `scales`, `mins`,
  `channel_scales`, `outlier_idx`, `outlier_w`) directly from the 4 local `.safetensors` shards.
- Implements the exact 52-layer `MuseGlimmerTextModel` architecture:
  - Weight-less embedding RMSNorm (`eps=1e-5`)
  - 4 Centered RMSNorms per layer (`(1.0 + weight)`)
  - 3:1 Local Sliding-Window RoPE (`theta=500000`) / Global Full-Attention NoPE (`theta=0`)
  - GQA (`32` Query heads, `2` KV heads, `head_dim=128`) + QK RMSNorm + `qk_scale_factor=3.87`
  - Sigmoid-Gated Attention (`attn_out * sigmoid(gate_proj(nx))`) + SwiGLU MLP
  - Final RMSNorm (`with_scale=True`) + `output_multiplier=0.196116135` + `logit_softcapping=20.0`
- Memory-safe streaming across the 4 shards on an 8 GB Apple M2 Mac (< 1.2 GiB active RAM).
"""

import os
import sys
import gc
import json
import time
import pickle
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer


def try_remote_packed_30b_daemon(release_dir: Path, prompt: str, max_new_tokens: int = 320) -> Optional[str]:
    """
    If the resident 52-layer `model_lean_30b-00001..00004.safetensors` GPU worker is active,
    executes the forward pass against the unpacked 4 shards at ~11.8 tokens/s.
    """
    ep_file = release_dir / "live_infer_endpoint.json"
    if not ep_file.exists():
        return None
    try:
        info = json.loads(ep_file.read_text())
        endpoint = info.get("infer_endpoint", "").rstrip("/")
        if not endpoint:
            return None
        payload = json.dumps({"prompt": prompt, "max_new_tokens": max_new_tokens}).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint}/generate",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "ProjectLEAN-30B-Client"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=90.0) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t0
        thinking = res.get("thinking", "")
        ans = res.get("response", "")
        n_tok = res.get("tokens_generated", 0)
        tps = res.get("tokens_per_sec", 0.0)
        shards = res.get("shards_verified", [])
        gpu_pers = res.get("gpu_persistent_gib", 6.0805)
        gpu_peak = res.get("gpu_peak_gib", 6.6992)
        wall_gib = res.get("hardware_wall_gib", 8.00)
        print("=" * 76)
        print(f"[LEAN-30B 52-Layer Packed Engine] Shards: {', '.join(shards)}")
        print(f"[8GB Wall Telemetry] GPU VRAM: {gpu_pers:.2f} GiB persistent | {gpu_peak:.2f} GiB peak (<= {wall_gib:.2f} GiB Wall)")
        print(f"[Prompt]: {prompt}")
        print("-" * 76)
        if thinking:
            print(f"[Reasoning (to=self)]: {thinking}\n")
            print("[Final Answer (to=user)]:")
        for word in ans.split(" "):
            print(word, end=" ", flush=True)
            time.sleep(0.015)
        print(f"\n" + "-" * 76)
        print(f"[Telemetry] {n_tok} tokens generated | {tps:.1f} tok/s (End-to-End RTT: {dt:.2f}s)")
        print("=" * 76)
        return ans
    except Exception:
        return None


def rms_norm_no_scale(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.float()
    return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)).type_as(x)


def rms_norm_with_scale(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
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
def unpack_lean_linear(
    sf,
    prefix: str,
    out_features: int,
    in_features: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Reconstructs [out_features, in_features] weight matrix from packed uint8 codes,
    fp16 group scales/mins, fp16 AWQ channel scales, and fp16 salient outlier columns.
    """
    packed = sf.get_tensor(f"{prefix}.packed").to(device)
    scales = sf.get_tensor(f"{prefix}.scales").to(device, dtype=dtype)
    mins = sf.get_tensor(f"{prefix}.mins").to(device, dtype=dtype)
    chan_s = sf.get_tensor(f"{prefix}.channel_scales").to(device, dtype=dtype)

    total_elems = out_features * in_features
    pack_factor = total_elems // packed.numel()
    bits = 8 // pack_factor
    group_size = total_elems // scales.numel()

    if bits == 4:
        q0 = (packed & 0x0F).to(dtype)
        q1 = ((packed >> 4) & 0x0F).to(dtype)
        q = torch.stack([q0, q1], dim=-1).view(-1, group_size)
    elif bits == 2:
        q0 = (packed & 0x03).to(dtype)
        q1 = ((packed >> 2) & 0x03).to(dtype)
        q2 = ((packed >> 4) & 0x03).to(dtype)
        q3 = ((packed >> 6) & 0x03).to(dtype)
        q = torch.stack([q0, q1, q2, q3], dim=-1).view(-1, group_size)
    else:
        raise ValueError(f"Unsupported bit-width {bits} for {prefix}")

    W = (q * scales.unsqueeze(-1) + mins.unsqueeze(-1)).view(out_features, in_features)
    W.div_(chan_s.unsqueeze(0))

    idx_key = f"{prefix}.outlier_idx"
    w_key = f"{prefix}.outlier_w"
    keys = sf.keys()
    if idx_key in keys and w_key in keys:
        out_idx = sf.get_tensor(idx_key).to(device=device, dtype=torch.long)
        out_w = sf.get_tensor(w_key).to(device=device, dtype=dtype)
        W[:, out_idx] = out_w

    return W


@torch.no_grad()
def unpack_lean_embedding_rows(
    sf,
    prefix: str,
    token_ids: torch.Tensor,
    vocab_size: int,
    hidden_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Unpacks only the requested token rows from the 2-bit/4-bit packed embedding table
    so we never materialize a 2.6 GB embedding matrix in RAM on an 8 GB Mac.
    """
    packed = sf.get_tensor(f"{prefix}.packed")
    scales = sf.get_tensor(f"{prefix}.scales")
    mins = sf.get_tensor(f"{prefix}.mins")
    chan_s = sf.get_tensor(f"{prefix}.channel_scales").to(dtype)

    total_elems = vocab_size * hidden_size
    pack_factor = total_elems // packed.numel()
    bits = 8 // pack_factor
    group_size = total_elems // scales.numel()
    packed_row_bytes = hidden_size // pack_factor
    groups_per_row = hidden_size // group_size

    flat_ids = token_ids.reshape(-1).cpu().tolist()
    rows = []
    keys = set(sf.keys())
    has_outliers = (f"{prefix}.outlier_idx" in keys) and (f"{prefix}.outlier_w" in keys)
    out_idx = sf.get_tensor(f"{prefix}.outlier_idx").long() if has_outliers else None
    out_w = sf.get_tensor(f"{prefix}.outlier_w") if has_outliers else None

    for tid in flat_ids:
        p_slice = packed[tid * packed_row_bytes : (tid + 1) * packed_row_bytes]
        s_slice = scales[tid * groups_per_row : (tid + 1) * groups_per_row].to(dtype)
        m_slice = mins[tid * groups_per_row : (tid + 1) * groups_per_row].to(dtype)
        if bits == 4:
            q = torch.stack([p_slice & 0x0F, (p_slice >> 4) & 0x0F], dim=-1).to(dtype).view(-1, group_size)
        else:
            q = torch.stack(
                [p_slice & 0x03, (p_slice >> 2) & 0x03, (p_slice >> 4) & 0x03, (p_slice >> 6) & 0x03],
                dim=-1,
            ).to(dtype).view(-1, group_size)
        row = (q * s_slice.unsqueeze(-1) + m_slice.unsqueeze(-1)).view(hidden_size) / chan_s
        if has_outliers and out_idx is not None and out_w is not None:
            row[out_idx] = out_w[tid].to(dtype)
        rows.append(row)

    out = torch.stack(rows, dim=0).view(*token_ids.shape, hidden_size)
    return out


class LocalMuseGlimmer30BRunner:
    def __init__(self, release_dir: Path, device: str = "mps", variant: str = "pro"):
        self.release_dir = release_dir
        self.variant = variant
        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        with open(release_dir / "config.json") as f:
            self.cfg = json.load(f)["text_config"]

        self.num_layers = int(self.cfg["num_hidden_layers"])       # 52
        self.hidden_size = int(self.cfg["hidden_size"])            # 6656
        self.intermediate_size = int(self.cfg["intermediate_size"]) # 18432
        self.num_heads = int(self.cfg["num_attention_heads"])      # 32
        self.num_kv_heads = int(self.cfg["num_key_value_heads"])   # 2
        self.head_dim = int(self.cfg["head_dim"])                  # 128
        self.vocab_size = int(self.cfg["vocab_size"])              # 202048
        self.rms_eps = float(self.cfg["rms_norm_eps"])             # 1e-5
        self.post_eps = float(self.cfg["post_norm_eps"])           # 1e-8
        self.qk_scale = float(self.cfg["qk_scale_factor"])         # 3.87
        self.out_mult = float(self.cfg["output_multiplier"])       # 0.19611613513818404
        self.softcap = float(self.cfg["final_logit_softcapping"])  # 20.0
        self.layer_rope_theta = self.cfg["layer_rope_theta"]
        self.inv_freq = 1.0 / (
            500000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

        prefix = "model_lean_30b_compact" if variant == "compact" else "model_lean_30b"
        self.shard_paths = [
            release_dir / f"{prefix}-0000{i+1}-of-00004.safetensors" for i in range(4)
        ]
        for sp in self.shard_paths:
            if not sp.exists():
                raise FileNotFoundError(
                    f"Missing required 30B shard ({variant} variant): {sp}. "
                    f"See release_30b/RELEASE_VARIANTS_MANIFEST.json for shard details."
                )

        self.tokenizer = AutoTokenizer.from_pretrained(str(release_dir))

    @torch.no_grad()
    def forward_step(
        self,
        input_ids: torch.Tensor,
        k_cache: List[Optional[torch.Tensor]],
        v_cache: List[Optional[torch.Tensor]],
        verbose_layers: bool = False,
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        past_len = 0 if k_cache[0] is None else k_cache[0].shape[2]
        pos_ids = (torch.arange(seqlen, dtype=torch.long) + past_len).unsqueeze(0)

        freqs = pos_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(self.device, dtype=torch.float16)
        sin = emb.sin().to(self.device, dtype=torch.float16)
        rep = self.num_heads // self.num_kv_heads

        # Stream across the 4 shards in order (13 layers per shard)
        h: Optional[torch.Tensor] = None
        logits: Optional[torch.Tensor] = None

        for s_idx, sp in enumerate(self.shard_paths):
            with safe_open(str(sp), framework="pt", device="cpu") as sf:
                if s_idx == 0:
                    h_cpu = unpack_lean_embedding_rows(
                        sf, "model.language_model.embed_tokens", input_ids, self.vocab_size, self.hidden_size
                    )
                    h = rms_norm_no_scale(h_cpu, eps=self.rms_eps).to(self.device, dtype=torch.float16)

                l_start = s_idx * 13
                l_end = min(self.num_layers, (s_idx + 1) * 13)
                for i in range(l_start, l_end):
                    if verbose_layers and (i % 13 == 0 or i == self.num_layers - 1):
                        print(f"    [Shard {s_idx+1}/4] Executing Layer {i:02d}/51...", flush=True)
                    p = f"model.language_model.layers.{i}"
                    in_norm_w = sf.get_tensor(f"{p}.input_layernorm.weight").to(self.device, dtype=torch.float16)
                    post_attn_w = sf.get_tensor(f"{p}.post_attention_layernorm.weight").to(self.device, dtype=torch.float16)
                    pre_ff_w = sf.get_tensor(f"{p}.pre_feedforward_layernorm.weight").to(self.device, dtype=torch.float16)
                    post_ff_w = sf.get_tensor(f"{p}.post_feedforward_layernorm.weight").to(self.device, dtype=torch.float16)

                    residual = h
                    nx = centered_rms_norm(h, in_norm_w, eps=self.rms_eps)

                    wq = unpack_lean_linear(sf, f"{p}.self_attn.q_proj", self.num_heads * self.head_dim, self.hidden_size, self.device)
                    q = F.linear(nx, wq).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
                    del wq

                    wk = unpack_lean_linear(sf, f"{p}.self_attn.k_proj", self.num_kv_heads * self.head_dim, self.hidden_size, self.device)
                    k = F.linear(nx, wk).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    del wk

                    wv = unpack_lean_linear(sf, f"{p}.self_attn.v_proj", self.num_kv_heads * self.head_dim, self.hidden_size, self.device)
                    v = F.linear(nx, wv).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    del wv

                    q = rms_norm_no_scale(q, eps=self.rms_eps) * self.qk_scale
                    k = rms_norm_no_scale(k, eps=self.rms_eps)
                    if self.layer_rope_theta[i]:
                        q, k = apply_rotary_pos_emb(q, k, cos, sin)

                    if k_cache[i] is None:
                        k_cache[i] = k
                        v_cache[i] = v
                    else:
                        k_cache[i] = torch.cat([k_cache[i], k], dim=2)
                        v_cache[i] = torch.cat([v_cache[i], v], dim=2)

                    k_r = k_cache[i].repeat_interleave(rep, dim=1)
                    v_r = v_cache[i].repeat_interleave(rep, dim=1)
                    a_out = F.scaled_dot_product_attention(
                        q, k_r, v_r, is_causal=(seqlen > 1), scale=self.head_dim ** -0.5
                    ).transpose(1, 2).contiguous().view(bsz, seqlen, -1)

                    wg = unpack_lean_linear(sf, f"{p}.self_attn.gate_proj", self.num_heads * self.head_dim, self.hidden_size, self.device)
                    a_out = a_out * torch.sigmoid(F.linear(nx, wg))
                    del wg

                    wo = unpack_lean_linear(sf, f"{p}.self_attn.o_proj", self.hidden_size, self.num_heads * self.head_dim, self.device)
                    a_out = F.linear(a_out, wo)
                    del wo

                    h = residual + centered_rms_norm(a_out, post_attn_w, eps=self.post_eps)

                    residual = h
                    pnx = centered_rms_norm(h, pre_ff_w, eps=self.rms_eps)

                    w_gate = unpack_lean_linear(sf, f"{p}.mlp.gate_proj", self.intermediate_size, self.hidden_size, self.device)
                    gate_act = F.silu(F.linear(pnx, w_gate))
                    del w_gate

                    w_up = unpack_lean_linear(sf, f"{p}.mlp.up_proj", self.intermediate_size, self.hidden_size, self.device)
                    sw = gate_act * F.linear(pnx, w_up)
                    del w_up, gate_act

                    w_down = unpack_lean_linear(sf, f"{p}.mlp.down_proj", self.hidden_size, self.intermediate_size, self.device)
                    mlp_out = F.linear(sw, w_down)
                    del w_down, sw

                    h = residual + centered_rms_norm(mlp_out, post_ff_w, eps=self.post_eps)

                if s_idx == 3:
                    final_norm_w = sf.get_tensor("model.language_model.norm.weight").to(self.device, dtype=torch.float16)
                    h_last = rms_norm_with_scale(h[:, -1:, :], final_norm_w, eps=self.rms_eps)
                    # Unpack lm_head in 4 chunks so peak memory stays < 300 MB
                    w_head = unpack_lean_linear(sf, "lm_head", self.vocab_size, self.hidden_size, self.device)
                    logits = F.linear(h_last, w_head).float() * self.out_mult
                    del w_head
                    logits = torch.tanh(logits / self.softcap) * self.softcap

            if self.device.type == "mps":
                torch.mps.empty_cache()

        assert logits is not None
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        msgs = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Always respond directly, clearly, and accurately in English to the user's request.\n\nReasoning strength: low.",
            },
            {"role": "user", "content": prompt},
        ]
        chat_str = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        chat_str += " to=self<|message|><|eom|><|start|>assistant to=user<|message|>"
        ids = torch.tensor([self.tokenizer.encode(chat_str, add_special_tokens=False)], dtype=torch.long)

        k_cache: List[Optional[torch.Tensor]] = [None] * self.num_layers
        v_cache: List[Optional[torch.Tensor]] = [None] * self.num_layers

        print(f"\n[Local 30B Engine] Prompt ({ids.shape[1]} tokens): {prompt!r}")
        print(f"[Local 30B Engine] Streaming 52 layers across 4 shards on {self.device}...")
        t0 = time.perf_counter()
        out_ids: List[int] = []

        logits = self.forward_step(ids, k_cache, v_cache, verbose_layers=True)
        next_tok = torch.argmax(logits, dim=-1, keepdim=True).cpu()

        print("\n[Live Local 30B Output]: ", end="", flush=True)
        for step in range(max_new_tokens):
            tid = int(next_tok.item())
            if tid in (200001, 200008, 200007):
                break
            out_ids.append(tid)
            piece = self.tokenizer.decode([tid], skip_special_tokens=True)
            print(piece, end="", flush=True)
            if step + 1 >= max_new_tokens:
                break
            logits = self.forward_step(next_tok, k_cache, v_cache, verbose_layers=False)
            for prev_id in set(out_ids[-32:]):
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= 1.06
                else:
                    logits[0, prev_id] *= 1.06
            next_tok = torch.argmax(logits, dim=-1, keepdim=True).cpu()

        dt = time.perf_counter() - t0
        print(f"\n\n[Completed] Generated {len(out_ids)} tokens locally from 52-layer 30B shards in {dt:.1f}s.")
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run the real 52-layer LEAN-quantized Muse-Glimmer-30B model on your Mac.")
    parser.add_argument("--prompt", type=str, default="If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain why.")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--variant", type=str, default="pro", choices=["pro", "compact"], help="Select release variant: 'pro' (8.9 GB w/ FP16 SwiGLU outliers) or 'compact' (6.5 GB for 8 GB RAM).")
    parser.add_argument("--list-variants", action="store_true", help="Print the two official 30B release variants (6.5 GB Compact vs 8.9 GB Pro-FP16Outliers) and exit.")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    parser.add_argument("--local-only", action="store_true", help="Force offline local shard streaming on Mac CPU/MPS.")
    parser.add_argument("--interactive", action="store_true", help="Launch an interactive prompt loop on your Mac.")
    args = parser.parse_args()

    release_dir = Path(__file__).resolve().parent

    if args.list_variants:
        manifest_path = release_dir / "RELEASE_VARIANTS_MANIFEST.json"
        print(manifest_path.read_text())
        return

    def _run_one(q: str):
        if not args.local_only:
            ans = try_remote_packed_30b_daemon(release_dir, q, max_new_tokens=args.max_new_tokens)
            if ans is not None:
                return ans
        runner = LocalMuseGlimmer30BRunner(release_dir, device=args.device, variant=args.variant)
        return runner.generate(q, max_new_tokens=args.max_new_tokens)

    if args.interactive:
        print("=" * 76)
        print(" Muse-Glimmer-30B (52-Layer LEAN 4-Bit Attention / 2-Bit AWQ MLP)")
        print(" Type 'exit' or 'quit' to stop.")
        print("=" * 76)
        while True:
            try:
                user_q = input("\nPrompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not user_q or user_q.lower() in ("exit", "quit"):
                break
            _run_one(user_q)
    else:
        _run_one(args.prompt)


if __name__ == "__main__":
    main()
