"""
experiments/plot_m15_priced_quanta_v2.py
========================================
Publication-Grade Plotting Script for Milestone M15 (Priced-Quanta v2 & H-KAC v2
on Meta's `meta-models/Muse-Glimmer-30B`).
Strictly follows Project LEAN visual identity:
- Matte Palette: Obsidian #121214, Signal Coral #FF5A5F, Off-White #EAEAEA, Slate #26262B.
- Generates:
  1. figures/fig_m15_bit_allocation_architecture.pdf (.png)
  2. figures/fig_m15_hessian_error_reduction.pdf (.png)
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def generate_m15_figures(results_path: str = "results/m15_30b_priced_quanta_v2_results.json") -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    os.makedirs("figures", exist_ok=True)
    summary = res["knapsack_30b_summary"]
    profiles = res["hessian_compensation_telemetry"]["layer_profiles"]

    plt.rcParams.update({
        "figure.facecolor": "#121214",
        "axes.facecolor": "#121214",
        "axes.edgecolor": "#26262B",
        "axes.labelcolor": "#EAEAEA",
        "text.color": "#EAEAEA",
        "xtick.color": "#EAEAEA",
        "ytick.color": "#EAEAEA",
        "font.size": 10,
    })

    # Figure 1: Muse-Glimmer-30B Physical Footprint vs. 8.00 GiB Laptop Ceiling
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=300)
    methods = [
        "Muse-Glimmer-30B\nBF16 (16.00 bpp)",
        "Meta Official K-Quant\n(4.57 bpp)",
        "Stage-1 KAC v1\n(4.57 bpp)",
        f"M15 Priced-Quanta v2\n({summary['effective_bpp']:.2f} bpp)",
    ]
    footprints = [55.46, 17.00, 15.85, summary["total_physical_gib"]]
    colors = ["#3A3A42", "#5A5A66", "#8A8A96", "#FF5A5F"]

    bars = ax.bar(methods, footprints, color=colors, edgecolor="#EAEAEA", linewidth=0.8, width=0.55)
    ax.axhline(8.00, color="#FF5A5F", linestyle="--", linewidth=1.5, label="8.00 GiB Laptop RAM Ceiling")
    for bar, val in zip(bars, footprints):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 1.1,
            f"{val:.2f} GiB",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#EAEAEA",
        )
    ax.set_ylabel("Physical Memory Footprint (GiB)")
    ax.set_title("Milestone M15: Meta Muse-Glimmer-30B Compressed Into an 8.00 GiB RAM Envelope", fontweight="bold")
    ax.set_ylim(0, 63)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/fig_m15_bit_allocation_architecture.pdf", facecolor="#121214")
    fig.savefig("figures/fig_m15_bit_allocation_architecture.png", facecolor="#121214", dpi=300)
    plt.close(fig)

    # Figure 2: Layer-Wise Output Reconstruction Error Reduction (RTN vs H-KAC v2)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    labels = [f"L{p['layer_idx']:02d}\n{p['role']}\n({p['bits']}b)" for p in profiles]
    rtn_errs = [p["err_rtn_stage1"] for p in profiles]
    hkac_errs = [p["err_hkac_v2"] for p in profiles]
    x = np.arange(len(labels))
    w = 0.36

    ax.bar(x - w / 2, rtn_errs, width=w, color="#5A5A66", label="Stage-1 Memoryless RTN")
    ax.bar(x + w / 2, hkac_errs, width=w, color="#FF5A5F", label="M15 Block-Cholesky H-KAC v2 + Sparse Outliers")

    for i, p in enumerate(profiles):
        red = p["error_reduction_pct"]
        y_top = max(rtn_errs[i], hkac_errs[i])
        ax.text(
            x[i],
            y_top + 0.02,
            f"-{red:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#FF5A5F",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative Frobenius Output Error ||Y - Y_hat||_F / ||Y||_F")
    ax.set_title(
        f"Milestone M15: Real Muse-Glimmer-30B Layer Error Reduction "
        f"(Macro Reduction: {res['hessian_compensation_telemetry']['macro_error_reduction_pct']:.2f}%)",
        fontweight="bold",
    )
    ax.set_ylim(0, max(rtn_errs) * 1.18)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/fig_m15_hessian_error_reduction.pdf", facecolor="#121214")
    fig.savefig("figures/fig_m15_hessian_error_reduction.png", facecolor="#121214", dpi=300)
    plt.close(fig)
    print("Generated figures/fig_m15_bit_allocation_architecture.{pdf,png} and figures/fig_m15_hessian_error_reduction.{pdf,png}")


if __name__ == "__main__":
    generate_m15_figures()
