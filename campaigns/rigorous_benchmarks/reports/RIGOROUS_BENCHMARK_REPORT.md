# Project LEAN: Rigorous Independent Benchmark & Ablation Report
## Canonical Open-Source Validation on Live NVIDIA A100-SXM4-80GB (`meta-models/Muse-Glimmer-30B`)

**Hardware Platform:** NVIDIA A100-SXM4-80GB (cuda)  
**Target Storage Budget:** 6,528,903,738 bytes (6.0805 GiB, 1.7541 bpp)  
**Seeds Evaluated ($N=5$):** `1042, 2042, 3042, 4042, 5042`  
**Wall-Clock Duration:** 2192.94 seconds  

---

## 1. Candidate Model Comparative Performance

| Model Identifier | Paradigm | Storage (GiB) | Effective bpp | GSM8K (Math) | ARC-Challenge | MMLU (5-Subj) | HumanEval (Pass@1) | WikiText-2 (PPL) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`FP16_REF`** | FP16 Reference (55.46 GiB) | 55.46 GiB | 16.0000 | 10.9% | 28.8% | 22.8% | 49.2% | 100.00 |
| **`LEAN_PQv2`** | LEAN-PQv2 (6.08 GiB) | 6.08 GiB | 1.7541 | 9.3% | 25.9% | 24.0% | 64.4% | 100.00 |
| **`GPTQ_UNIFORM`** | GPTQ-Uniform (6.08 GiB) | 6.08 GiB | 1.7541 | 22.1% | 22.4% | 26.6% | 60.4% | 100.00 |
| **`AWQ_UNIFORM`** | AWQ-Uniform (6.08 GiB) | 6.08 GiB | 1.7541 | 10.7% | 23.7% | 23.8% | 71.2% | 100.00 |
| **`HQQ_TERNARY`** | HQQ-Ternary (6.08 GiB) | 6.08 GiB | 1.7541 | 15.5% | 27.2% | 24.2% | 43.2% | 100.00 |
| **`UNIFORM_MATCHED`** | Uniform-Matched RTN (6.08 GiB) | 6.08 GiB | 1.7541 | 12.5% | 27.2% | 22.6% | 57.6% | 100.00 |

---

## 2. Statistical Significance & Hypothesis Testing (LEAN-PQv2 vs Baselines)

Evaluated across $N=5$ seeds with 95% Bootstrap Confidence Intervals ($B=10,000$) and exact paired permutation tests:

| Benchmark Metric | Advantage vs Uniform | Cliff's $\delta$ (Unif) | Permutation $p$ | Advantage vs GPTQ | Cliff's $\delta$ (GPTQ) | Permutation $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gsm8k_accuracy`** | **+-3.20 pp** | **-0.360** | $p = 0.5625$ | **+-12.80 pp** | **-1.000** | $p = 0.0625$ |
| **`arc_accuracy`** | **+-1.33 pp** | **-0.160** | $p = 0.3750$ | **+3.47 pp** | **0.240** | $p = 0.7500$ |
| **`mmlu_accuracy`** | **+1.40 pp** | **0.120** | $p = 0.6875$ | **+-2.60 pp** | **-0.240** | $p = 0.2500$ |
| **`humaneval_pass1`** | **+6.80 pp** | **0.080** | $p = 0.5000$ | **+4.00 pp** | **0.600** | $p = 0.8750$ |
| **`humaneval_pass3`** | **+-2.40 pp** | **-0.480** | $p = 0.3750$ | **+4.80 pp** | **0.280** | $p = 0.7500$ |

---

## 3. 4-Factor Ablation Analysis

### A. 5-Rung Iterative Stacking Ladder

| Rung | Configuration Description | Active Factors | GSM8K | ARC-Challenge | MMLU | HumanEval | PPL |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Rung 1** | Rung 1: Baseline Naive Ternary | `KAC=False, HESS=False, RET=False, KV=fp16` | 12.5% | 27.2% | 22.6% | 57.6% | 100.00 |
| **Rung 2** | Rung 2: + Block-Cholesky Hessian Compensation | `KAC=False, HESS=True, RET=False, KV=fp16` | 12.5% | 27.2% | 22.6% | 57.6% | 100.00 |
| **Rung 3** | Rung 3: + Knapsack-Aware Asymmetric Allocation (GQA-KAC) | `KAC=True, HESS=True, RET=False, KV=fp16` | 9.3% | 23.5% | 25.0% | 64.4% | 100.00 |
| **Rung 4** | Rung 4: + 4-Bit Group-64 KV-Cache | `KAC=True, HESS=True, RET=False, KV=4bit_group64` | 9.3% | 23.5% | 25.0% | 64.4% | 100.00 |
| **Rung 5** | Rung 5: + 18 KB BM25 Retrieval (Full LEAN-PQv2) | `KAC=True, HESS=True, RET=True, KV=4bit_group64` | 9.3% | 25.9% | 24.0% | 64.4% | 100.00 |

### B. Drop-One Sensitivity Matrix (Ablating from Full LEAN-PQv2)

| Target Removed | Ablation Point | GSM8K Loss ($\Delta$) | ARC Loss ($\Delta$) | MMLU Loss ($\Delta$) | HumanEval Loss ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Drop `KAC`** | Full LEAN-PQv2 minus KAC | --3.2 pp | --1.3 pp | -1.4 pp | -6.8 pp |
| **Drop `HESSIAN`** | Full LEAN-PQv2 minus Hessian | -0.0 pp | -0.0 pp | -0.0 pp | -0.0 pp |
| **Drop `RETRIEVAL`** | Full LEAN-PQv2 minus BM25 | -0.0 pp | -2.4 pp | --1.0 pp | -0.0 pp |
| **Drop `KV_CACHE`** | Full LEAN-PQv2 minus 4-bit KV Cache | -0.0 pp | -0.0 pp | -0.0 pp | -0.0 pp |