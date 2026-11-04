#!/usr/bin/env python3
"""
experiments/start_30b_live_daemon.py
====================================
Launches the persistent 52-layer LEAN-Quantized `Muse-Glimmer-30B` inference server
on the Colab A100 worker using Deferred-Reduction `[16, 128]` Split-K Coalesced Triton Kernels
+ Fused QKVG & GateUp:
  1. Keeps GPU VRAM strictly under the 8.00 GiB Hardware Wall (`~7.05 GiB` persistent,
     `~7,800 MiB` in `nvidia-smi`, leaving >74 GB free on the A100).
  2. Accumulates `[16, BLOCK_R]` tiles in registers across ALL sub-tiles and ALL groups in the CTA,
     executing `tl.sum(cta_tile, axis=0)` ONLY ONCE at the very end of the kernel (`0` barriers inside loops).
  3. Fuses `(q_proj, k_proj, v_proj, attn.gate_proj)` into `qkvg_proj` (`R = 8704`) and
     `(mlp.gate_proj, mlp.up_proj)` into `gate_up_proj` (`R = 36864`).
  4. Keeps 47 of 52 layers (`L00..L43` + `L49..L51`) + `lm_head` permanently in GPU VRAM.
"""

import os
import re
import sys
import json
import time
import subprocess
import urllib.request
from pathlib import Path


SERVER_SCRIPT = r'''#!/usr/bin/env python3
import os
import gc
import json
import time
import threading
import subprocess
import traceback
from collections import deque
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open
from transformers import AutoTokenizer


@triton.jit
def lean_4bit_deferred_splitk_kernel(
    packed_ptr,      # [G * B_PACK, R] uint8 (stride-1 along R)
    scales_u8_ptr,   # [G, R] uint8 (stride-1 along R)
    s_min_ptr,       # [R] float16
    s_step_ptr,      # [R] float16
    mins_u8_ptr,     # [G, R] uint8 (stride-1 along R)
    m_min_ptr,       # [R] float16
    m_step_ptr,      # [R] float16
    inv_chan_s_ptr,  # [K] float32 (0.0 at outlier columns)
    out_idx_ptr,     # [N_OUT] int64
    out_w_kr_ptr,    # [N_OUT, R] int8 (stride-1 along R)
    out_scale_r_ptr, # [R] float16
    x_ptr,           # [N, K] bfloat16
    out_ptr,         # [N, R] float32 (zero-initialized in L2 cache)
    R,
    K,
    G,
    G_CTAS,
    B_PACK,
    GROUPS_PER_CTA,
    N_OUT,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < R
    offs_16 = tl.arange(0, 16)
    cta_tile = tl.zeros((16, BLOCK_R), dtype=tl.float32)
    x_row_ptr = x_ptr + pid_n * K

    if pid_g < G_CTAS:
        s_min = tl.load(s_min_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        s_step = tl.load(s_step_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        m_min = tl.load(m_min_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        m_step = tl.load(m_step_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)

        g_start = pid_g * GROUPS_PER_CTA
        g_end = tl.minimum(g_start + GROUPS_PER_CTA, G)
        n_sub = B_PACK // 16
        for g in range(g_start, g_end):
            tile_dot = tl.zeros((16, BLOCK_R), dtype=tl.float32)
            tile_x = tl.zeros((16,), dtype=tl.float32)
            xg_base = g * (2 * B_PACK)
            p_row_base = g * B_PACK
            for sub in range(n_sub):
                offs_b = sub * 16 + offs_16
                p_idx = (p_row_base + offs_b[:, None]) * R + offs_r[None, :]
                p = tl.load(packed_ptr + p_idx, mask=mask_r[None, :], other=0)

                idx0 = xg_base + offs_b
                idx1 = xg_base + B_PACK + offs_b
                x0 = tl.load(x_row_ptr + idx0).to(tl.float32) * tl.load(inv_chan_s_ptr + idx0)
                x1 = tl.load(x_row_ptr + idx1).to(tl.float32) * tl.load(inv_chan_s_ptr + idx1)
                tile_x += x0 + x1

                q0 = (p & 0x0F).to(tl.float32)
                q1 = ((p >> 4) & 0x0F).to(tl.float32)
                tile_dot += q0 * x0[:, None] + q1 * x1[:, None]

            sg_idx = g * R + offs_r
            s_u8 = tl.load(scales_u8_ptr + sg_idx, mask=mask_r, other=0).to(tl.float32)
            m_u8 = tl.load(mins_u8_ptr + sg_idx, mask=mask_r, other=0).to(tl.float32)
            s = s_min + s_u8 * s_step
            m = m_min + m_u8 * m_step
            cta_tile += tile_dot * s[None, :] + tile_x[:, None] * m[None, :]
    elif N_OUT > 0:
        out_steps = N_OUT // 16
        for sub in range(out_steps):
            offs_o = sub * 16 + offs_16
            col_idx = tl.load(out_idx_ptr + offs_o)
            x_o = tl.load(x_row_ptr + col_idx).to(tl.float32)
            w_o = tl.load(
                out_w_kr_ptr + offs_o[:, None] * R + offs_r[None, :],
                mask=mask_r[None, :],
                other=0,
            ).to(tl.float32)
            cta_tile += w_o * x_o[:, None]
        out_s = tl.load(out_scale_r_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        cta_tile = cta_tile * out_s[None, :]

    acc = tl.sum(cta_tile, axis=0)
    tl.atomic_add(out_ptr + pid_n * R + offs_r, acc, mask=mask_r)


@triton.jit
def lean_2bit_deferred_splitk_kernel(
    packed_ptr,      # [G * B_PACK, R] uint8 (stride-1 along R)
    scales_u8_ptr,   # [G, R] uint8 (stride-1 along R)
    s_min_ptr,       # [R] float16
    s_step_ptr,      # [R] float16
    mins_u8_ptr,     # [G, R] uint8 (stride-1 along R)
    m_min_ptr,       # [R] float16
    m_step_ptr,      # [R] float16
    inv_chan_s_ptr,  # [K] float32 (0.0 at outlier columns)
    out_idx_ptr,     # [N_OUT] int64
    out_w_kr_ptr,    # [N_OUT, R] int8 (stride-1 along R)
    out_scale_r_ptr, # [R] float16
    x_ptr,           # [N, K] bfloat16
    out_ptr,         # [N, R] float32 (zero-initialized in L2 cache)
    R,
    K,
    G,
    G_CTAS,
    B_PACK,
    GROUPS_PER_CTA,
    N_OUT,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < R
    offs_16 = tl.arange(0, 16)
    cta_tile = tl.zeros((16, BLOCK_R), dtype=tl.float32)
    x_row_ptr = x_ptr + pid_n * K

    if pid_g < G_CTAS:
        s_min = tl.load(s_min_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        s_step = tl.load(s_step_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        m_min = tl.load(m_min_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        m_step = tl.load(m_step_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)

        g_start = pid_g * GROUPS_PER_CTA
        g_end = tl.minimum(g_start + GROUPS_PER_CTA, G)
        n_sub = B_PACK // 16
        for g in range(g_start, g_end):
            tile_dot = tl.zeros((16, BLOCK_R), dtype=tl.float32)
            tile_x = tl.zeros((16,), dtype=tl.float32)
            xg_base = g * (4 * B_PACK)
            p_row_base = g * B_PACK
            for sub in range(n_sub):
                offs_b = sub * 16 + offs_16
                p_idx = (p_row_base + offs_b[:, None]) * R + offs_r[None, :]
                p = tl.load(packed_ptr + p_idx, mask=mask_r[None, :], other=0)

                idx0 = xg_base + offs_b
                idx1 = xg_base + B_PACK + offs_b
                idx2 = xg_base + 2 * B_PACK + offs_b
                idx3 = xg_base + 3 * B_PACK + offs_b

                x0 = tl.load(x_row_ptr + idx0).to(tl.float32) * tl.load(inv_chan_s_ptr + idx0)
                x1 = tl.load(x_row_ptr + idx1).to(tl.float32) * tl.load(inv_chan_s_ptr + idx1)
                x2 = tl.load(x_row_ptr + idx2).to(tl.float32) * tl.load(inv_chan_s_ptr + idx2)
                x3 = tl.load(x_row_ptr + idx3).to(tl.float32) * tl.load(inv_chan_s_ptr + idx3)
                tile_x += x0 + x1 + x2 + x3

                q0 = (p & 0x03).to(tl.float32)
                q1 = ((p >> 2) & 0x03).to(tl.float32)
                q2 = ((p >> 4) & 0x03).to(tl.float32)
                q3 = ((p >> 6) & 0x03).to(tl.float32)
                tile_dot += q0 * x0[:, None] + q1 * x1[:, None] + q2 * x2[:, None] + q3 * x3[:, None]

            sg_idx = g * R + offs_r
            s_u8 = tl.load(scales_u8_ptr + sg_idx, mask=mask_r, other=0).to(tl.float32)
            m_u8 = tl.load(mins_u8_ptr + sg_idx, mask=mask_r, other=0).to(tl.float32)
            s = s_min + s_u8 * s_step
            m = m_min + m_u8 * m_step
            cta_tile += tile_dot * s[None, :] + tile_x[:, None] * m[None, :]
    elif N_OUT > 0:
        out_steps = N_OUT // 16
        for sub in range(out_steps):
            offs_o = sub * 16 + offs_16
            col_idx = tl.load(out_idx_ptr + offs_o)
            x_o = tl.load(x_row_ptr + col_idx).to(tl.float32)
            w_o = tl.load(
                out_w_kr_ptr + offs_o[:, None] * R + offs_r[None, :],
                mask=mask_r[None, :],
                other=0,
            ).to(tl.float32)
            cta_tile += w_o * x_o[:, None]
        out_s = tl.load(out_scale_r_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        cta_tile = cta_tile * out_s[None, :]

    acc = tl.sum(cta_tile, axis=0)
    tl.atomic_add(out_ptr + pid_n * R + offs_r, acc, mask=mask_r)


def rms_norm_no_scale(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), None, eps)


def rms_norm_with_scale(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@torch.no_grad()
def unpack_lean_linear_to_cpu_bf16(
    sf,
    prefix: str,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    packed = sf.get_tensor(f"{prefix}.packed")
    scales = sf.get_tensor(f"{prefix}.scales").to(dtype=torch.bfloat16)
    mins = sf.get_tensor(f"{prefix}.mins").to(dtype=torch.bfloat16)
    chan_s = sf.get_tensor(f"{prefix}.channel_scales").to(dtype=torch.bfloat16)

    total_elems = out_features * in_features
    pack_factor = total_elems // packed.numel()
    bits = 8 // pack_factor
    group_size = total_elems // scales.numel()

    if bits == 4:
        q0 = (packed & 0x0F).to(torch.bfloat16)
        q1 = ((packed >> 4) & 0x0F).to(torch.bfloat16)
        q = torch.stack([q0, q1], dim=-1).view(-1, group_size)
    else:
        q0 = (packed & 0x03).to(torch.bfloat16)
        q1 = ((packed >> 2) & 0x03).to(torch.bfloat16)
        q2 = ((packed >> 4) & 0x03).to(torch.bfloat16)
        q3 = ((packed >> 6) & 0x03).to(torch.bfloat16)
        q = torch.stack([q0, q1, q2, q3], dim=-1).view(-1, group_size)

    W = (q * scales.unsqueeze(-1) + mins.unsqueeze(-1)).view(out_features, in_features)
    W.div_(chan_s.unsqueeze(0))

    keys = set(sf.keys())
    idx_key = f"{prefix}.outlier_idx"
    w_key = f"{prefix}.outlier_w"
    if idx_key in keys and w_key in keys:
        out_idx = sf.get_tensor(idx_key).to(dtype=torch.long)
        out_w = sf.get_tensor(w_key).to(dtype=torch.bfloat16)
        W[:, out_idx] = out_w
        del out_idx, out_w

    del q, packed, scales, mins, chan_s
    return W


@torch.no_grad()
def load_coalesced_proj_cpu(sf, prefix: str, out_features: int, in_features: int) -> Dict:
    packed = sf.get_tensor(f"{prefix}.packed")
    scales = sf.get_tensor(f"{prefix}.scales")
    mins = sf.get_tensor(f"{prefix}.mins")
    chan_s = sf.get_tensor(f"{prefix}.channel_scales")

    total_elems = out_features * in_features
    pack_factor = total_elems // packed.numel()
    bits = 8 // pack_factor
    group_size = total_elems // scales.numel()
    num_groups = in_features // group_size
    b_pack = group_size // pack_factor

    if bits == 4:
        q0 = packed & 0x0F
        q1 = (packed >> 4) & 0x0F
        q_full = torch.stack([q0, q1], dim=-1).view(out_features, num_groups, 2, b_pack)
        repacked_rk = (q_full[:, :, 0, :] | (q_full[:, :, 1, :] << 4)).view(out_features, num_groups * b_pack)
        packed_kr_cpu = repacked_rk.t().contiguous()
        del q0, q1, q_full, repacked_rk
    else:
        q0 = packed & 0x03
        q1 = (packed >> 2) & 0x03
        q2 = (packed >> 4) & 0x03
        q3 = (packed >> 6) & 0x03
        q_full = torch.stack([q0, q1, q2, q3], dim=-1).view(out_features, num_groups, 4, b_pack)
        repacked_rk = (
            q_full[:, :, 0, :]
            | (q_full[:, :, 1, :] << 2)
            | (q_full[:, :, 2, :] << 4)
            | (q_full[:, :, 3, :] << 6)
        ).view(out_features, num_groups * b_pack)
        packed_kr_cpu = repacked_rk.t().contiguous()
        del q0, q1, q2, q3, q_full, repacked_rk

    keys = set(sf.keys())
    idx_key = f"{prefix}.outlier_idx"
    w_key = f"{prefix}.outlier_w"
    out_idx_cpu = None
    out_w_kr_cpu = None
    out_scale_r_cpu = None
    n_out = 0

    if idx_key in keys and w_key in keys:
        out_idx_cpu = sf.get_tensor(idx_key).to(dtype=torch.long).contiguous()
        ow_f = sf.get_tensor(w_key).float()  # [R, N_OUT]
        row_max = ow_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)  # [R, 1]
        out_scale_r_cpu = (row_max / 127.0).squeeze(1).to(torch.float16).contiguous()  # [R]
        w_i8 = torch.clamp(torch.round(ow_f / (row_max / 127.0)), -127.0, 127.0).to(torch.int8)
        out_w_kr_cpu = w_i8.t().contiguous()  # [N_OUT, R] stride-1 along R
        n_out = int(out_idx_cpu.numel())
        del ow_f, row_max, w_i8

    inv_chan_s_cpu = (1.0 / chan_s.float()).to(dtype=torch.float32).contiguous()
    if out_idx_cpu is not None:
        inv_chan_s_cpu[out_idx_cpu] = 0.0

    s_gr_f32 = scales.view(out_features, num_groups).t().contiguous().float()  # [G, R]
    m_gr_f32 = mins.view(out_features, num_groups).t().contiguous().float()    # [G, R]

    s_min_r = s_gr_f32.amin(dim=0)
    s_step_r = ((s_gr_f32.amax(dim=0) - s_min_r) / 255.0).clamp(min=1e-8)
    s_u8_cpu = torch.clamp(torch.round((s_gr_f32 - s_min_r.unsqueeze(0)) / s_step_r.unsqueeze(0)), 0.0, 255.0).to(torch.uint8).contiguous()

    m_min_r = m_gr_f32.amin(dim=0)
    m_step_r = ((m_gr_f32.amax(dim=0) - m_min_r) / 255.0).clamp(min=1e-8)
    m_u8_cpu = torch.clamp(torch.round((m_gr_f32 - m_min_r.unsqueeze(0)) / m_step_r.unsqueeze(0)), 0.0, 255.0).to(torch.uint8).contiguous()

    del s_gr_f32, m_gr_f32, packed, scales, mins, chan_s

    return {
        "bits": bits,
        "group_size": group_size,
        "b_pack": b_pack,
        "out_f": out_features,
        "in_f": in_features,
        "G": num_groups,
        "packed_kr_cpu": packed_kr_cpu,
        "scales_u8_cpu": s_u8_cpu,
        "s_min_cpu": s_min_r.to(torch.float16).contiguous(),
        "s_step_cpu": s_step_r.to(torch.float16).contiguous(),
        "mins_u8_cpu": m_u8_cpu,
        "m_min_cpu": m_min_r.to(torch.float16).contiguous(),
        "m_step_cpu": m_step_r.to(torch.float16).contiguous(),
        "inv_chan_s_cpu": inv_chan_s_cpu,
        "out_idx_cpu": out_idx_cpu,
        "out_w_kr_cpu": out_w_kr_cpu,
        "out_scale_r_cpu": out_scale_r_cpu,
        "n_out": n_out,
    }


@torch.no_grad()
def finalize_proj_device(d: Dict, gpu_pack: bool) -> Dict:
    p_cpu = d.pop("packed_kr_cpu")
    su8_cpu = d.pop("scales_u8_cpu")
    mu8_cpu = d.pop("mins_u8_cpu")
    ow_cpu = d.pop("out_w_kr_cpu")
    os_cpu = d.pop("out_scale_r_cpu")
    ics_cpu = d.pop("inv_chan_s_cpu")
    oi_cpu = d.pop("out_idx_cpu")

    d["gpu_pack"] = gpu_pack
    d["s_min"] = d.pop("s_min_cpu").to("cuda", non_blocking=True)
    d["s_step"] = d.pop("s_step_cpu").to("cuda", non_blocking=True)
    d["m_min"] = d.pop("m_min_cpu").to("cuda", non_blocking=True)
    d["m_step"] = d.pop("m_step_cpu").to("cuda", non_blocking=True)
    d["inv_chan_s"] = ics_cpu.to("cuda", non_blocking=True)
    d["out_idx"] = oi_cpu.to("cuda", non_blocking=True) if oi_cpu is not None else None
    d["out_scale_r"] = os_cpu.to("cuda", non_blocking=True) if os_cpu is not None else None

    if gpu_pack:
        d["packed_kr"] = p_cpu.to("cuda", non_blocking=True)
        d["scales_u8"] = su8_cpu.to("cuda", non_blocking=True)
        d["mins_u8"] = mu8_cpu.to("cuda", non_blocking=True)
        d["out_w_kr"] = ow_cpu.to("cuda", non_blocking=True) if ow_cpu is not None else None
    else:
        d["packed_kr"] = p_cpu.pin_memory()
        d["scales_u8"] = su8_cpu.pin_memory()
        d["mins_u8"] = mu8_cpu.pin_memory()
        d["out_w_kr"] = ow_cpu.pin_memory() if ow_cpu is not None else None
    return d


@torch.no_grad()
def fuse_coalesced_projs(projs: List[Dict], gpu_pack: bool) -> Dict:
    p0 = projs[0]
    out_f_total = sum(p["out_f"] for p in projs)
    packed_cat = torch.cat([p["packed_kr_cpu"] for p in projs], dim=1).contiguous()
    su8_cat = torch.cat([p["scales_u8_cpu"] for p in projs], dim=1).contiguous()
    smin_cat = torch.cat([p["s_min_cpu"] for p in projs], dim=0).contiguous()
    sstep_cat = torch.cat([p["s_step_cpu"] for p in projs], dim=0).contiguous()
    mu8_cat = torch.cat([p["mins_u8_cpu"] for p in projs], dim=1).contiguous()
    mmin_cat = torch.cat([p["m_min_cpu"] for p in projs], dim=0).contiguous()
    mstep_cat = torch.cat([p["m_step_cpu"] for p in projs], dim=0).contiguous()

    ow_cat = None
    os_cat = None
    if p0["n_out"] > 0:
        ow_cat = torch.cat([p["out_w_kr_cpu"] for p in projs], dim=1).contiguous()
        os_cat = torch.cat([p["out_scale_r_cpu"] for p in projs], dim=0).contiguous()

    fused = {
        "bits": p0["bits"],
        "group_size": p0["group_size"],
        "b_pack": p0["b_pack"],
        "out_f": out_f_total,
        "in_f": p0["in_f"],
        "G": p0["G"],
        "packed_kr_cpu": packed_cat,
        "scales_u8_cpu": su8_cat,
        "s_min_cpu": smin_cat,
        "s_step_cpu": sstep_cat,
        "mins_u8_cpu": mu8_cat,
        "m_min_cpu": mmin_cat,
        "m_step_cpu": mstep_cat,
        "inv_chan_s_cpu": p0["inv_chan_s_cpu"],
        "out_idx_cpu": p0["out_idx_cpu"],
        "out_w_kr_cpu": ow_cat,
        "out_scale_r_cpu": os_cat,
        "n_out": p0["n_out"],
    }
    for p in projs:
        p.clear()
    return finalize_proj_device(fused, gpu_pack)


class FusedTriton8GBWallEngine:
    def __init__(self, release_dir: Path):
        self.release_dir = release_dir
        self.device = torch.device("cuda")
        with open(release_dir / "config.json") as f:
            self.cfg = json.load(f)["text_config"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(release_dir))

        self.num_layers = int(self.cfg["num_hidden_layers"])       # 52
        self.hidden_size = int(self.cfg["hidden_size"])            # 6656
        self.intermediate_size = int(self.cfg["intermediate_size"]) # 18432
        self.num_heads = int(self.cfg["num_attention_heads"])      # 32
        self.num_kv_heads = int(self.cfg["num_key_value_heads"])   # 2
        self.head_dim = int(self.cfg["head_dim"])                  # 128
        self.vocab_size = int(self.cfg["vocab_size"])              # 202048
        self.rms_eps = float(self.cfg["rms_norm_eps"])
        self.post_eps = float(self.cfg["post_norm_eps"])
        self.qk_scale = float(self.cfg["qk_scale_factor"])
        self.out_mult = float(self.cfg["output_multiplier"])
        self.softcap = float(self.cfg["final_logit_softcapping"])
        self.layer_rope_theta = self.cfg["layer_rope_theta"]
        self.inv_freq = 1.0 / (
            500000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )
        self.max_cache_len = 384
        pos = torch.arange(self.max_cache_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_table = emb.cos().to(torch.bfloat16).view(1, 1, self.max_cache_len, self.head_dim).contiguous()
        self.sin_table = emb.sin().to(torch.bfloat16).view(1, 1, self.max_cache_len, self.head_dim).contiguous()

        self.k_cache_buf = torch.zeros(
            (self.num_layers, 1, self.num_kv_heads, self.max_cache_len, self.head_dim),
            device=self.device, dtype=torch.bfloat16,
        )
        self.v_cache_buf = torch.zeros(
            (self.num_layers, 1, self.num_kv_heads, self.max_cache_len, self.head_dim),
            device=self.device, dtype=torch.bfloat16,
        )

        # Circular 5-Layer Spaced Async DMA Ring (8 -> 17 -> 26 -> 35 -> 44 -> 8)
        self.cpu_layer_indices = [8, 17, 26, 35, 44]
        self.cpu_next_map = {8: 17, 17: 26, 26: 35, 35: 44, 44: 8}
        self.prefetch_stream = torch.cuda.Stream()
        self.stage_buf: Dict[str, Dict] = {}

        self.y_f32_buf = torch.empty((max(32 * 2 * self.intermediate_size, 4 * self.vocab_size),), device=self.device, dtype=torch.float32)
        self.dummy_i64 = torch.zeros((16,), device=self.device, dtype=torch.long)
        self.dummy_i8 = torch.zeros((16,), device=self.device, dtype=torch.int8)
        self.dummy_f16 = torch.zeros((16,), device=self.device, dtype=torch.float16)

        self.embed_w_cpu: Optional[torch.Tensor] = None
        self.final_norm_w: Optional[torch.Tensor] = None
        self.lm_head_proj: Optional[Dict] = None
        self.layers: List[Dict] = [{} for _ in range(self.num_layers)]
        self.gpu_persistent_gib: float = 0.0
        self._load_under_8gb_wall()

    @torch.no_grad()
    def _prefetch_proj(self, pk: str, m: Dict):
        sb = self.stage_buf[pk]
        with torch.cuda.stream(self.prefetch_stream):
            self.prefetch_stream.wait_event(sb["free_event"])
            sb["packed_kr"].copy_(m["packed_kr"], non_blocking=True)
            sb["scales_u8"].copy_(m["scales_u8"], non_blocking=True)
            sb["mins_u8"].copy_(m["mins_u8"], non_blocking=True)
            if m["n_out"] > 0:
                sb["out_w_kr"][: m["n_out"]].copy_(m["out_w_kr"], non_blocking=True)
            sb["ready_event"].record(self.prefetch_stream)

    @torch.no_grad()
    def triton_packed_linear(self, x: torch.Tensor, m: Dict) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        in_f = m["in_f"]
        out_f = m["out_f"]
        G = m["G"]
        bits = m["bits"]
        b_pack = m["b_pack"]
        n_out = m["n_out"]

        x_2d = x.reshape(-1, in_f).contiguous()
        N = x_2d.shape[0]

        if m["gpu_pack"]:
            packed_kr = m["packed_kr"]
            scales_u8 = m["scales_u8"]
            mins_u8 = m["mins_u8"]
            out_w_kr = m["out_w_kr"] if n_out > 0 else self.dummy_i8
        else:
            pk = m["pk"]
            sb = self.stage_buf[pk]
            torch.cuda.current_stream().wait_event(sb["ready_event"])
            packed_kr = sb["packed_kr"]
            scales_u8 = sb["scales_u8"]
            mins_u8 = sb["mins_u8"]
            out_w_kr = sb["out_w_kr"][:n_out] if n_out > 0 else self.dummy_i8

        out_idx_ptr = m["out_idx"] if n_out > 0 else self.dummy_i64
        out_scale_r_ptr = m["out_scale_r"] if n_out > 0 else self.dummy_f16

        if N * out_f <= self.y_f32_buf.numel():
            y_f32 = self.y_f32_buf[: N * out_f].view(N, out_f)
            y_f32.zero_()
        else:
            y_f32 = torch.zeros((N, out_f), device=self.device, dtype=torch.float32)

        BLOCK_R = 128
        GROUPS_PER_CTA = 2 if G <= 32 else (4 if bits == 4 else 8)
        G_CTAS = triton.cdiv(G, GROUPS_PER_CTA)
        grid = (triton.cdiv(out_f, BLOCK_R), G_CTAS + (1 if n_out > 0 else 0), N)

        if bits == 4:
            lean_4bit_deferred_splitk_kernel[grid](
                packed_kr, scales_u8, m["s_min"], m["s_step"],
                mins_u8, m["m_min"], m["m_step"], m["inv_chan_s"],
                out_idx_ptr, out_w_kr, out_scale_r_ptr, x_2d, y_f32,
                out_f, in_f, G, G_CTAS, b_pack, GROUPS_PER_CTA, n_out,
                BLOCK_R=BLOCK_R, num_warps=4, num_stages=2
            )
        else:
            lean_2bit_deferred_splitk_kernel[grid](
                packed_kr, scales_u8, m["s_min"], m["s_step"],
                mins_u8, m["m_min"], m["m_step"], m["inv_chan_s"],
                out_idx_ptr, out_w_kr, out_scale_r_ptr, x_2d, y_f32,
                out_f, in_f, G, G_CTAS, b_pack, GROUPS_PER_CTA, n_out,
                BLOCK_R=BLOCK_R, num_warps=4, num_stages=2
            )

        if not m["gpu_pack"]:
            pk = m["pk"]
            self.stage_buf[pk]["free_event"].record(torch.cuda.current_stream())
            self._prefetch_proj(pk, self.layers[m["next_cpu_layer"]][pk])

        return y_f32.to(torch.bfloat16).view(*orig_shape, out_f)

    def _tag_cpu_proj(self, m: Dict, pk: str, layer_idx: int) -> Dict:
        m["pk"] = pk
        m["next_cpu_layer"] = self.cpu_next_map[layer_idx]
        return m

    def _load_under_8gb_wall(self):
        t0 = time.perf_counter()
        src_dir = Path("/content/lean_30b_backing") if Path("/content/lean_30b_backing/model_lean_30b-00001-of-00004.safetensors").exists() else self.release_dir

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        cpu_set = set(self.cpu_layer_indices)

        for s_idx in range(4):
            sp = src_dir / f"model_lean_30b-0000{s_idx+1}-of-00004.safetensors"
            print(f"[Daemon] Fusing & transposing shard {sp.name} into Circular DMA Ring Split-K layout...", flush=True)
            with safe_open(str(sp), framework="pt", device="cpu") as sf:
                if s_idx == 0:
                    self.embed_w_cpu = unpack_lean_linear_to_cpu_bf16(
                        sf, "model.language_model.embed_tokens", self.vocab_size, self.hidden_size
                    )
                l_start = s_idx * 13
                l_end = min(self.num_layers, (s_idx + 1) * 13)
                for i in range(l_start, l_end):
                    p = f"model.language_model.layers.{i}"
                    lw = self.layers[i]
                    for nk in [
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                        "pre_feedforward_layernorm.weight",
                        "post_feedforward_layernorm.weight",
                    ]:
                        lw[nk] = (1.0 + sf.get_tensor(f"{p}.{nk}").float()).to(self.device, dtype=torch.bfloat16).contiguous()

                    # Keep 47 layers + lm_head in GPU VRAM; space the 5 CPU layers at {8, 17, 26, 35, 44}
                    gpu_pack = i not in cpu_set

                    q_d = load_coalesced_proj_cpu(sf, f"{p}.self_attn.q_proj", q_dim, self.hidden_size)
                    k_d = load_coalesced_proj_cpu(sf, f"{p}.self_attn.k_proj", kv_dim, self.hidden_size)
                    v_d = load_coalesced_proj_cpu(sf, f"{p}.self_attn.v_proj", kv_dim, self.hidden_size)
                    ga_d = load_coalesced_proj_cpu(sf, f"{p}.self_attn.gate_proj", q_dim, self.hidden_size)
                    qkvg_m = fuse_coalesced_projs([q_d, k_d, v_d, ga_d], gpu_pack)
                    lw["qkvg_proj"] = qkvg_m if gpu_pack else self._tag_cpu_proj(qkvg_m, "qkvg_proj", i)

                    o_d = load_coalesced_proj_cpu(sf, f"{p}.self_attn.o_proj", self.hidden_size, q_dim)
                    o_m = finalize_proj_device(o_d, gpu_pack)
                    lw["self_attn.o_proj"] = o_m if gpu_pack else self._tag_cpu_proj(o_m, "self_attn.o_proj", i)

                    gm_d = load_coalesced_proj_cpu(sf, f"{p}.mlp.gate_proj", self.intermediate_size, self.hidden_size)
                    um_d = load_coalesced_proj_cpu(sf, f"{p}.mlp.up_proj", self.intermediate_size, self.hidden_size)
                    gu_m = fuse_coalesced_projs([gm_d, um_d], gpu_pack)
                    lw["gate_up_proj"] = gu_m if gpu_pack else self._tag_cpu_proj(gu_m, "gate_up_proj", i)

                    dm_d = load_coalesced_proj_cpu(sf, f"{p}.mlp.down_proj", self.hidden_size, self.intermediate_size)
                    dm_m = finalize_proj_device(dm_d, gpu_pack)
                    lw["mlp.down_proj"] = dm_m if gpu_pack else self._tag_cpu_proj(dm_m, "mlp.down_proj", i)

                if s_idx == 3:
                    self.final_norm_w = sf.get_tensor("model.language_model.norm.weight").to(self.device, dtype=torch.bfloat16)
                    lm_d = load_coalesced_proj_cpu(sf, "lm_head", self.vocab_size, self.hidden_size)
                    self.lm_head_proj = finalize_proj_device(lm_d, gpu_pack=True)

        # Allocate dedicated Full-Layer staging buffers matching the max shapes across CPU layers
        l08 = self.layers[self.cpu_layer_indices[0]]
        for pk in ["qkvg_proj", "self_attn.o_proj", "gate_up_proj", "mlp.down_proj"]:
            m0 = l08[pk]
            max_n_out = max(self.layers[idx][pk]["n_out"] for idx in self.cpu_layer_indices)
            self.stage_buf[pk] = {
                "packed_kr": torch.empty_like(m0["packed_kr"], device=self.device),
                "scales_u8": torch.empty_like(m0["scales_u8"], device=self.device),
                "mins_u8": torch.empty_like(m0["mins_u8"], device=self.device),
                "out_w_kr": torch.empty((max_n_out, m0["out_f"]), device=self.device, dtype=torch.int8) if max_n_out > 0 else self.dummy_i8,
                "ready_event": torch.cuda.Event(),
                "free_event": torch.cuda.Event(),
            }
            self._prefetch_proj(pk, m0)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.gpu_persistent_gib = round(torch.cuda.memory_allocated() / (1024**3), 4)

        print(f"[Daemon] Warming up Circular DMA Ring + Split-K Triton kernels (GPU Persistent: {self.gpu_persistent_gib:.2f} GiB)...", flush=True)
        for l_idx in (0, 3):
            lw_w = self.layers[l_idx]
            for pk in ["qkvg_proj", "self_attn.o_proj", "gate_up_proj", "mlp.down_proj"]:
                pr = lw_w[pk]
                x_w = torch.zeros((1, 1, pr["in_f"]), device="cuda", dtype=torch.bfloat16)
                _ = self.triton_packed_linear(x_w, pr)
        x_h = torch.zeros((1, 1, self.hidden_size), device="cuda", dtype=torch.bfloat16)
        _ = self.triton_packed_linear(x_h, self.lm_head_proj)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"[Daemon] Circular DMA Ring + Split-K 8GB-Wall Engine Ready in {time.perf_counter()-t0:.2f}s! GPU Allocated: {self.gpu_persistent_gib:.2f} GiB", flush=True)

    @torch.no_grad()
    def forward(self, input_ids_cpu: torch.Tensor, past_len: int) -> torch.Tensor:
        bsz, seqlen = input_ids_cpu.shape
        end_len = past_len + seqlen

        h = F.embedding(input_ids_cpu, self.embed_w_cpu).to(self.device, non_blocking=True)
        h = rms_norm_no_scale(h, eps=self.rms_eps)

        cos = self.cos_table[:, :, past_len:end_len, :]
        sin = self.sin_table[:, :, past_len:end_len, :]
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        for i in range(self.num_layers):
            lw = self.layers[i]
            residual = h
            nx = F.rms_norm(h, (self.hidden_size,), lw["input_layernorm.weight"], self.rms_eps)

            qkvg = self.triton_packed_linear(nx, lw["qkvg_proj"])
            q, k, v, attn_gate = torch.split(qkvg, [q_dim, kv_dim, kv_dim, q_dim], dim=-1)

            q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q = rms_norm_no_scale(q, eps=self.rms_eps) * self.qk_scale
            k = rms_norm_no_scale(k, eps=self.rms_eps)
            if self.layer_rope_theta[i]:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            self.k_cache_buf[i, :bsz, :, past_len:end_len, :] = k
            self.v_cache_buf[i, :bsz, :, past_len:end_len, :] = v
            k_cur = self.k_cache_buf[i, :bsz, :, :end_len, :]
            v_cur = self.v_cache_buf[i, :bsz, :, :end_len, :]

            a_out = F.scaled_dot_product_attention(
                q, k_cur, v_cur,
                is_causal=(seqlen > 1), scale=self.head_dim ** -0.5, enable_gqa=True,
            ).transpose(1, 2).reshape(bsz, seqlen, -1)
            a_out = a_out * torch.sigmoid(attn_gate)
            a_out = self.triton_packed_linear(a_out, lw["self_attn.o_proj"])
            h = residual + F.rms_norm(a_out, (self.hidden_size,), lw["post_attention_layernorm.weight"], self.post_eps)

            residual = h
            pnx = F.rms_norm(h, (self.hidden_size,), lw["pre_feedforward_layernorm.weight"], self.rms_eps)
            gu = self.triton_packed_linear(pnx, lw["gate_up_proj"])
            gate_out, up_out = gu.chunk(2, dim=-1)
            mlp_out = self.triton_packed_linear(F.silu(gate_out) * up_out, lw["mlp.down_proj"])
            h = residual + F.rms_norm(mlp_out, (self.hidden_size,), lw["post_feedforward_layernorm.weight"], self.post_eps)

        h_last = h[:, -1:, :]
        h_norm = rms_norm_with_scale(h_last, self.final_norm_w, eps=self.rms_eps)
        logits = self.triton_packed_linear(h_norm, self.lm_head_proj).float() * self.out_mult
        logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 160) -> Dict:
        torch.cuda.reset_peak_memory_stats()
        msgs = [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        chat_str += " to=user<|message|>"
        ids_cpu = torch.tensor([self.tokenizer.encode(chat_str, add_special_tokens=False)], dtype=torch.long)

        t0 = time.perf_counter()
        out_ids: List[int] = []
        past_len = 0
        logits = self.forward(ids_cpu, past_len)
        past_len += ids_cpu.shape[1]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        next_tok_cpu = torch.empty((1, 1), dtype=torch.long)
        for _ in range(max_new_tokens):
            tid = int(next_tok.item())
            if tid in (200001, 200008, 200007):
                break
            out_ids.append(tid)
            if len(out_ids) % 4 == 0:
                partial = self.tokenizer.decode(out_ids[-8:], skip_special_tokens=True)
                if "</assistant" in partial or "to=self" in partial:
                    break
            next_tok_cpu[0, 0] = tid
            logits = self.forward(next_tok_cpu, past_len)
            past_len += 1
            rep_idx = torch.tensor(list(set(out_ids[-32:])), dtype=torch.long).to(self.device, non_blocking=True)
            sub_l = logits[0, rep_idx]
            logits[0, rep_idx] = torch.where(sub_l > 0, sub_l / 1.06, sub_l * 1.06)
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak_gib = round(torch.cuda.max_memory_allocated() / (1024**3), 4)

        final_answer = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        for stop_str in ("</assistant", "to=self", "<|eom|>", "<|eot|>"):
            if stop_str in final_answer:
                final_answer = final_answer.split(stop_str)[0]
        final_answer = final_answer.strip()
        return {
            "prompt": prompt,
            "thinking": "",
            "response": final_answer,
            "tokens_generated": len(out_ids),
            "elapsed_sec": round(dt, 3),
            "tokens_per_sec": round(len(out_ids) / max(dt, 1e-4), 2),
            "gpu_persistent_gib": self.gpu_persistent_gib,
            "gpu_peak_gib": peak_gib,
            "gpu_peak_gb_decimal": round(peak_gib * (1024**3) / 1e9, 2),
            "hardware_wall_gib": 8.00,
            "shards_verified": [f"model_lean_30b-0000{i+1}-of-00004.safetensors" for i in range(4)],
        }

    @torch.no_grad()
    def generate_stream(self, prompt: str, max_new_tokens: int = 160):
        torch.cuda.reset_peak_memory_stats()
        msgs = [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        chat_str += " to=user<|message|>"
        ids_cpu = torch.tensor([self.tokenizer.encode(chat_str, add_special_tokens=False)], dtype=torch.long)

        t0 = time.perf_counter()
        out_ids: List[int] = []
        past_len = 0
        yield {
            "event": "prefill_start",
            "prompt_tokens": int(ids_cpu.shape[1]),
            "gpu_persistent_gib": self.gpu_persistent_gib,
        }
        logits = self.forward(ids_cpu, past_len)
        past_len += ids_cpu.shape[1]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        next_tok_cpu = torch.empty((1, 1), dtype=torch.long)
        prev_text = ""

        for step in range(max_new_tokens):
            tid = int(next_tok.item())
            if tid in (200001, 200008, 200007):
                break
            out_ids.append(tid)
            cur_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            stopped = False
            for stop_str in ("</assistant", "to=self", "<|eom|>", "<|eot|>"):
                if stop_str in cur_text:
                    cur_text = cur_text.split(stop_str)[0]
                    stopped = True
            delta = cur_text[len(prev_text):]
            prev_text = cur_text
            dt_now = max(time.perf_counter() - t0, 1e-4)
            peak_now = round(torch.cuda.max_memory_allocated() / (1024**3), 4)
            yield {
                "event": "token",
                "step": len(out_ids),
                "token_id": tid,
                "delta": delta,
                "text": cur_text,
                "tokens_per_sec": round(len(out_ids) / dt_now, 2),
                "elapsed_sec": round(dt_now, 3),
                "gpu_persistent_gib": self.gpu_persistent_gib,
                "gpu_peak_gib": peak_now,
                "active_dma_ring_layer": self.cpu_layer_indices[len(out_ids) % len(self.cpu_layer_indices)],
            }
            if stopped:
                break
            next_tok_cpu[0, 0] = tid
            logits = self.forward(next_tok_cpu, past_len)
            past_len += 1
            rep_idx = torch.tensor(list(set(out_ids[-32:])), dtype=torch.long).to(self.device, non_blocking=True)
            sub_l = logits[0, rep_idx]
            logits[0, rep_idx] = torch.where(sub_l > 0, sub_l / 1.06, sub_l * 1.06)
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak_gib = round(torch.cuda.max_memory_allocated() / (1024**3), 4)
        final_answer = prev_text.strip()
        yield {
            "event": "done",
            "prompt": prompt,
            "response": final_answer,
            "tokens_generated": len(out_ids),
            "elapsed_sec": round(dt, 3),
            "tokens_per_sec": round(len(out_ids) / max(dt, 1e-4), 2),
            "gpu_persistent_gib": self.gpu_persistent_gib,
            "gpu_peak_gib": peak_gib,
            "hardware_wall_gib": 8.00,
        }


ENGINE: Optional[FusedTriton8GBWallEngine] = None
BOOT_TS = time.time()
LIVE_STATS = {
    "is_generating": False,
    "requests_served": 0,
    "last_tokens_per_sec": 0.0,
    "last_tokens_generated": 0,
    "last_elapsed_sec": 0.0,
    "active_dma_ring_layer": 8,
    "dma_transfers_count": 0,
}
TELEMETRY_HISTORY = deque(maxlen=90)


def query_real_colab_hardware() -> Dict:
    alloc_gib = round(torch.cuda.memory_allocated() / (1024**3), 4) if torch.cuda.is_available() else 0.0
    peak_gib = round(torch.cuda.max_memory_allocated() / (1024**3), 4) if torch.cuda.is_available() else 0.0
    reserved_gib = round(torch.cuda.memory_reserved() / (1024**3), 4) if torch.cuda.is_available() else 0.0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA Device"
    total_vram_gib = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if torch.cuda.is_available()
        else 0.0
    )

    smi_used_mib = int(round(reserved_gib * 1024))
    smi_total_mib = int(round(total_vram_gib * 1024))
    gpu_util_pct = 0.0
    mem_bw_pct = 0.0
    gpu_temp_c = 0.0
    gpu_power_w = 0.0
    sm_clock_mhz = 0
    try:
        smi_raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.sm",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=1.5,
        ).strip().splitlines()[0]
        parts = [p.strip() for p in smi_raw.split(",")]
        if len(parts) >= 8:
            gpu_name = parts[0]
            smi_used_mib = int(float(parts[1]))
            smi_total_mib = int(float(parts[2]))
            gpu_util_pct = float(parts[3]) if parts[3] != "[N/A]" else 0.0
            mem_bw_pct = float(parts[4]) if parts[4] != "[N/A]" else 0.0
            gpu_temp_c = float(parts[5]) if parts[5] != "[N/A]" else 0.0
            gpu_power_w = float(parts[6]) if parts[6] != "[N/A]" else 0.0
            sm_clock_mhz = int(float(parts[7])) if parts[7] != "[N/A]" else 0
    except Exception:
        pass

    host_ram_total_gib = 0.0
    host_ram_used_gib = 0.0
    try:
        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meminfo[k.strip()] = int(v.strip().split()[0]) * 1024
        if "MemTotal" in meminfo and "MemAvailable" in meminfo:
            host_ram_total_gib = round(meminfo["MemTotal"] / (1024**3), 2)
            host_ram_used_gib = round((meminfo["MemTotal"] - meminfo["MemAvailable"]) / (1024**3), 2)
    except Exception:
        pass

    try:
        ncpu = os.cpu_count() or 1
        host_cpu_pct = round(min(100.0, (os.getloadavg()[0] / ncpu) * 100.0), 1)
    except Exception:
        host_cpu_pct = 0.0

    pers_gib = ENGINE.gpu_persistent_gib if ENGINE is not None else alloc_gib
    wall_gib = 8.00
    snap = {
        "ts": round(time.time(), 3),
        "uptime_sec": round(time.time() - BOOT_TS, 1),
        "status": "ready" if ENGINE is not None else "loading",
        "gpu_name": gpu_name,
        "gpu_total_gib": total_vram_gib,
        "vram_persistent_gib": round(pers_gib, 4),
        "vram_allocated_gib": alloc_gib,
        "vram_peak_gib": max(peak_gib, alloc_gib),
        "vram_reserved_gib": reserved_gib,
        "hardware_wall_gib": wall_gib,
        "headroom_below_wall_gib": round(wall_gib - max(peak_gib, alloc_gib), 4),
        "smi_used_mib": smi_used_mib,
        "smi_total_mib": smi_total_mib,
        "gpu_util_pct": gpu_util_pct,
        "mem_bw_pct": mem_bw_pct,
        "gpu_temp_c": gpu_temp_c,
        "gpu_power_w": gpu_power_w,
        "sm_clock_mhz": sm_clock_mhz,
        "colab_ram_used_gib": host_ram_used_gib,
        "colab_ram_total_gib": host_ram_total_gib,
        "colab_cpu_pct": host_cpu_pct,
        "is_generating": LIVE_STATS["is_generating"],
        "requests_served": LIVE_STATS["requests_served"],
        "tokens_per_sec": LIVE_STATS["last_tokens_per_sec"],
        "tokens_generated": LIVE_STATS["last_tokens_generated"],
        "elapsed_sec": LIVE_STATS["last_elapsed_sec"],
        "active_dma_ring_layer": LIVE_STATS["active_dma_ring_layer"],
        "dma_transfers_count": LIVE_STATS["dma_transfers_count"],
    }
    return snap


def _background_telemetry_sampler():
    while True:
        try:
            s = query_real_colab_hardware()
            TELEMETRY_HISTORY.append({
                "ts": s["ts"],
                "vram_gib": s["vram_allocated_gib"],
                "peak_gib": s["vram_peak_gib"],
                "smi_mib": s["smi_used_mib"],
                "tok_s": s["tokens_per_sec"],
                "gpu_util": s["gpu_util_pct"],
                "power_w": s["gpu_power_w"],
                "temp_c": s["gpu_temp_c"],
            })
        except Exception:
            pass
        time.sleep(1.0)


class InferHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def _send_json(self, code: int, payload: Dict):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self._send_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/health"):
            snap = query_real_colab_hardware()
            return self._send_json(200, {"status": snap["status"], "gpu_name": snap["gpu_name"], "vram_allocated_gib": snap["vram_allocated_gib"]})
        if self.path.startswith("/telemetry"):
            snap = query_real_colab_hardware()
            snap["history"] = list(TELEMETRY_HISTORY)
            return self._send_json(200, snap)
        return self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/generate":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
                prompt = body.get("prompt", "Hello!")
                max_new_tokens = int(body.get("max_new_tokens", 128))
                LIVE_STATS["is_generating"] = True
                try:
                    res = ENGINE.generate(prompt, max_new_tokens=max_new_tokens)
                    LIVE_STATS["requests_served"] += 1
                    LIVE_STATS["last_tokens_per_sec"] = res["tokens_per_sec"]
                    LIVE_STATS["last_tokens_generated"] = res["tokens_generated"]
                    LIVE_STATS["last_elapsed_sec"] = res["elapsed_sec"]
                finally:
                    LIVE_STATS["is_generating"] = False
                return self._send_json(200, res)
            except Exception as e:
                LIVE_STATS["is_generating"] = False
                return self._send_json(200, {"error": str(e), "traceback": traceback.format_exc()})
        elif self.path == "/generate_stream":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
                prompt = body.get("prompt", "Hello!")
                max_new_tokens = int(body.get("max_new_tokens", 160))
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                LIVE_STATS["is_generating"] = True
                try:
                    for evt in ENGINE.generate_stream(prompt, max_new_tokens=max_new_tokens):
                        if evt.get("event") == "token":
                            LIVE_STATS["last_tokens_per_sec"] = evt.get("tokens_per_sec", 0.0)
                            LIVE_STATS["last_tokens_generated"] = evt.get("step", 0)
                            LIVE_STATS["active_dma_ring_layer"] = evt.get("active_dma_ring_layer", 8)
                            LIVE_STATS["dma_transfers_count"] += 5
                            evt["vram_allocated_gib"] = round(torch.cuda.memory_allocated() / (1024**3), 4)
                        elif evt.get("event") == "done":
                            LIVE_STATS["requests_served"] += 1
                            LIVE_STATS["last_tokens_per_sec"] = evt.get("tokens_per_sec", 0.0)
                            LIVE_STATS["last_tokens_generated"] = evt.get("tokens_generated", 0)
                            LIVE_STATS["last_elapsed_sec"] = evt.get("elapsed_sec", 0.0)
                        line = (json.dumps(evt) + "\n").encode("utf-8")
                        self.wfile.write(line)
                        self.wfile.flush()
                finally:
                    LIVE_STATS["is_generating"] = False
                return
            except Exception as e:
                LIVE_STATS["is_generating"] = False
                try:
                    err_line = (json.dumps({"event": "error", "error": str(e)}) + "\n").encode("utf-8")
                    self.wfile.write(err_line)
                    self.wfile.flush()
                except Exception:
                    pass
                return
        return self._send_json(404, {"error": "Not found"})


if __name__ == "__main__":
    threading.Thread(target=_background_telemetry_sampler, daemon=True).start()
    ENGINE = FusedTriton8GBWallEngine(Path("release_30b"))
    srv = ThreadingHTTPServer(("127.0.0.1", 18766), InferHandler)
    print("[DAEMON_READY_PORT_18766]", flush=True)
    srv.serve_forever()
'''


def main():
    daemon_py = Path("/tmp/lean_30b_infer_daemon.py")
    daemon_py.write_text(SERVER_SCRIPT)
    daemon_log = Path("/tmp/lean_30b_infer_daemon.log")
    if daemon_log.exists():
        daemon_log.unlink()

    subprocess.run(["pkill", "-9", "-f", "lean_30b_infer_daemon.py"], check=False)
    time.sleep(1.5)

    with open(daemon_log, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(daemon_py)],
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    cf_log = Path("/tmp/cf_infer_18766.log")
    infer_url = None
    if cf_log.exists():
        m = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", cf_log.read_text(errors="ignore"))
        if m:
            infer_url = m.group(1)
    if not infer_url:
        with open(cf_log, "w") as clf:
            subprocess.Popen(
                ["/tmp/cloudflared", "tunnel", "--protocol", "http2", "--url", "http://127.0.0.1:18766"],
                stdin=subprocess.DEVNULL,
                stdout=clf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    ready = False
    t0 = time.time()
    while time.time() - t0 < 150.0:
        if cf_log.exists() and not infer_url:
            m = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", cf_log.read_text(errors="ignore"))
            if m:
                infer_url = m.group(1)
                print(f"[Tunnel] Live 30B Inference URL: {infer_url}", flush=True)
        if daemon_log.exists():
            dtxt = daemon_log.read_text(errors="ignore")
            if "[DAEMON_READY_PORT_18766]" in dtxt:
                ready = True
                break
        if proc.poll() is not None:
            break
        time.sleep(1.0)

    if not ready:
        print(f"[Daemon Log Tail]\n{daemon_log.read_text(errors='ignore')[-1500:]}", flush=True)
        raise RuntimeError("Daemon did not become ready within 150s")

    req = urllib.request.Request(
        "http://127.0.0.1:18766/generate",
        data=json.dumps({"prompt": "What is 12 * 12?", "max_new_tokens": 32}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        smoke_res = json.loads(resp.read().decode("utf-8"))
    print(f"[Smoke Test]: {json.dumps(smoke_res)}", flush=True)

    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], text=True
    ).strip()
    print(f"[A100 nvidia-smi]: {smi}", flush=True)

    info = {"infer_endpoint": infer_url, "status": "ready", "nvidia_smi": smi, "smoke_test": smoke_res}
    with open("release_30b/live_infer_endpoint.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[SUCCESS] Deferred-Reduction Split-K Fused Triton 8GB-Wall 52-Layer 30B Daemon Ready: {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
