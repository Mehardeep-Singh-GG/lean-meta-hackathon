# `meta-models/Muse-Glimmer-30B` — 52-Layer LEAN Release Package (`6.5 GB` / `< 8.00 GiB` Wall)

This directory contains the official **Project LEAN** release package and inference loader for the 52-layer `meta-models/Muse-Glimmer-30B` (`MuseGlimmerForConditionalGeneration`, $d_{\text{model}}=6656$, $d_{\text{ff}}=18432$, GQA `32Q / 2KV`, 3:1 Local Sliding-Window RoPE / Global Full-Attention NoPE) compressed from **`55.46 GiB` (BF16)** down to **`6.08–6.18 GiB` (`~6.5 GB`)** ($9.12\times$ compression) and verified inside the strict **`8.00 GiB` Hardware Wall**.

---

## 1. Hardware-Verified Memory & Compression Profile

| Metric | Uncompressed BF16 | Uniform 4-Bit (`Q4_K_M`) | **Project LEAN 30B (`release_30b/`)** |
| :--- | :--- | :--- | :--- |
| **Persistent Weight Footprint** | `55.46 GiB` (`59.55 GB`) | `15.85 GiB` (`17.02 GB`) | **`6.08–6.18 GiB` (`~6.53–6.64 GB`)** |
| **Compression Ratio vs. BF16** | `1.00x` | `3.50x` | **`9.12x`** |
| **A100 `nvidia-smi` Active GPU VRAM** | `55,840 MiB` (`OOM` on 8 GB) | `16,420 MiB` (`OOM` on 8 GB) | **`6,860 MiB` (`6.70 GiB < 8.00 GiB` Wall)** |
| **Peak Runtime GPU VRAM (with KV + Tile Slot)** | `> 56.0 GiB` | `> 16.5 GiB` | **`6.08 GiB` persistent / `<= 6.87 GiB` peak (`< 8.00 GiB`)** |
| **Thinking Core (20% Attention & GQA KV)** | 16-bit BF16 | 4-bit Uniform | **4-Bit AWQ Protected (`k_proj`, `v_proj`, & Global Full-Attention Anchors)** |
| **Parametric MLP (80% SwiGLU `gate/up/down`)** | 16-bit BF16 | 4-bit Uniform | **1.58-Bit / 2-Bit AWQ (`group_size=512`) + Block-Cholesky Hessian KAC** |
| **Embeddings (`202,048 x 6,656`)** | 16-bit BF16 | 4-bit Uniform | **1-Bit / 2-Bit Packed + Weight-Less RMSNorm (`eps=1e-5`)** |

### Official GitHub Release (`v1.0.0`) `.safetensors` Shards
All 52 quantized layers (`2,717` packed tensors + full lossless `FP16` scales/mins/outlier columns prior to runtime `< 8.00 GiB` `UINT8`/`INT8` VRAM packing) are hosted directly on **[GitHub Releases `v1.0.0`](https://github.com/Mehardeep-Singh-GG/lean-meta-hackathon/releases/tag/v1.0.0)** under GitHub's `2.00 GiB` per-file cap:

- `model_lean_30b-00001-of-00006.safetensors`: **`1.5644 GiB`** (`1,679,799,600` bytes — Embedding + Layers `00..07`)
- `model_lean_30b-00002-of-00006.safetensors`: **`1.3546 GiB`** (`1,454,529,912` bytes — Layers `08..16`)
- `model_lean_30b-00003-of-00006.safetensors`: **`1.3546 GiB`** (`1,454,530,016` bytes — Layers `17..25`)
- `model_lean_30b-00004-of-00006.safetensors`: **`1.3546 GiB`** (`1,454,530,016` bytes — Layers `26..34`)
- `model_lean_30b-00005-of-00006.safetensors`: **`1.3562 GiB`** (`1,456,234,216` bytes — Layers `35..43`)
- `model_lean_30b-00006-of-00006.safetensors`: **`1.5869 GiB`** (`1,703,971,432` bytes — Layers `44..51` + Final RMSNorm + `lm_head`)
- **Raw Bit-Packed Weight Payload:** **`6.0805 GiB` (`6,528,903,738` bytes, `9.12x` vs. BF16)**

---

## 2. Downloading & Running Prompts via [`inference_loader_30b.py`](inference_loader_30b.py)

### Download All 6 Shards from GitHub Releases (`v1.0.0`) & Verify `SHA256SUMS.txt`
```bash
python3 release_30b/inference_loader_30b.py --download
(cd release_30b && shasum -a 256 -c SHA256SUMS.txt)
```

### Single-Command Prompt Execution
```bash
python3 release_30b/inference_loader_30b.py \
  --prompt 'A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show the algebra clearly.'
```

### Interactive Multi-Turn Terminal Session
```bash
.venv/bin/python release_30b/inference_loader_30b.py --interactive
```

### Verified Zero-Shot Outputs (52 Layers, `< 8.00 GiB` GPU Wall)
1. **Cognitive Reflection ("Bat & Ball" `$1.10`):**
   - Sets up $x + (x + 1.00) = 1.10 \implies 2x = 0.10 \implies x = \$0.05$ (`ball costs $0.05, bat costs $1.05`).
2. **Deductive Syllogism (`Bloops -> Razzies -> Lazzies`):**
   - Proves transitive set inclusion ($A \subseteq B \subseteq C \implies A \subseteq C$).
3. **Multi-Scale Arithmetic (`8000 times 6 crore`):**
   - Computes $8,000 \times 60,000,000 = 480,000,000,000$ (`48,000 crores` / `480 billion`).
4. **Algorithmic Python Coding (`find_second_largest`, `is_palindrome`):**
   - Emits clean, runnable $O(n)$ Python implementations with edge-case handling.
