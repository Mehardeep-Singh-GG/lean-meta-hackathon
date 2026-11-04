"""
Milestone M16 Publication-Grade Plotting Script:
Renders:
1. `figures/m16/fig_m16_accuracy_improvement_ladder.{pdf,png}`:
   5-Rung Iterative Accuracy Recovery Ladder on `meta-models/Muse-Glimmer-30B`
   (Raw Zero-Context Cloze, RAG Retrieval Cloze with 18 KB BM25 Pocket Index, and Procedural Skill Retention).
2. `figures/m16/fig_m16_layerwise_kl_divergence.{pdf,png}`:
   Monotonic Layer-by-Layer Relative L2 Drift & Logit KL-Divergence Collapse across Rung 0 -> Rung 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def render_m16_figures(input_json: Path, out_dir: Path) -> Dict[str, str]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    rungs_dict = payload["rungs"]
    ladder = [rungs_dict[f"rung_{i}"] for i in range(5)]

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "#121214",
            "axes.facecolor": "#121214",
            "savefig.facecolor": "#121214",
            "text.color": "#EAEAEA",
            "axes.labelcolor": "#EAEAEA",
            "xtick.color": "#EAEAEA",
            "ytick.color": "#EAEAEA",
            "axes.edgecolor": "#26262B",
            "grid.color": "#26262B",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        }
    )

    rungs = [r["rung_id"] for r in ladder]
    labels = [
        "Rung 0:\nUncalibrated\nMin-Max (6.04 GiB)",
        "Rung 1:\n+Multi-Domain\nAWQ v1 (6.05 GiB)",
        "Rung 2:\n+Block-Cholesky\nH-KAC v2 (6.08 GiB)",
        "Rung 3:\n+GQA KV-4b &\nRoPE Shield (6.08 GiB)",
        "Rung 4:\n+Closed-Form Bias\nmu-Folding (6.08 GiB)",
    ]
    raw_acc = [r["raw_cloze_acc_pct"] for r in ladder]
    rag_acc = [r["retrieval_cloze_acc_pct"] for r in ladder]
    proc_acc = [r["procedural_skill_retention_pct"] for r in ladder]

    # Figure 1: 5-Rung Iterative Accuracy Improvement Ladder
    fig1, ax1 = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    x = np.arange(len(rungs))
    width = 0.25

    ax1.bar(x - width, raw_acc, width, color="#6E7078", edgecolor="#121214", label="Raw Closed-Book Cloze (0-Search, %)")
    bars_proc = ax1.bar(x, proc_acc, width, color="#EAEAEA", edgecolor="#121214", label="Procedural & Deductive Skill Retention (%)")
    bars_rag = ax1.bar(
        x + width,
        rag_acc,
        width,
        color="#FF5A5F",
        edgecolor="#EAEAEA",
        linewidth=1.0,
        label="RAG Cloze (+18 KB BM25 Pocket Index, %)",
    )

    ax1.axhline(80.0, color="#FF5A5F", linestyle="--", linewidth=1.3, label="80.0% North-Star Target")

    for b, val in zip(bars_rag, rag_acc):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            val + 1.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#FF5A5F",
        )
    for b, val in zip(bars_proc, proc_acc):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            val + 1.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#EAEAEA",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("Measured Accuracy on Muse-Glimmer-30B (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 112)
    ax1.set_title(
        "Milestone M16: 5-Rung Iterative Calibration & >=80% Accuracy Recovery on Meta Muse-Glimmer-30B (6.08 GiB)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, facecolor="#121214", edgecolor="#26262B", fontsize=8.5)

    fig1.tight_layout()
    f1_pdf = out_dir / "fig_m16_accuracy_improvement_ladder.pdf"
    f1_png = out_dir / "fig_m16_accuracy_improvement_ladder.png"
    fig1.savefig(f1_pdf, bbox_inches="tight")
    fig1.savefig(f1_png, bbox_inches="tight", dpi=300)
    plt.close(fig1)

    # Figure 2: Layer-Wise Relative L2 Drift & Logit KL-Divergence Collapse
    fig2, (ax_drift, ax_kl) = plt.subplots(1, 2, figsize=(11.8, 4.6), dpi=300)
    rung_colors = ["#6E7078", "#8D99AE", "#3A86FF", "#EAEAEA", "#FF5A5F"]
    for r_idx, r_data in enumerate(ladder):
        curve = r_data["layer_drift_curve"]
        l_ids = [c["layer_idx"] + 1 for c in curve]
        l2_vals = [c["relative_l2_error"] for c in curve]
        ax_drift.plot(
            l_ids,
            l2_vals,
            marker="o",
            linewidth=2.2 if r_idx == 4 else 1.6,
            color=rung_colors[r_idx],
            label=f"Rung {r_idx}",
        )

    ax_drift.set_xlabel("Evaluated Muse-Glimmer-30B Layer Index", fontweight="bold")
    ax_drift.set_ylabel("Relative Hidden L2 Error ||H - H_hat|| / ||H||", fontweight="bold")
    ax_drift.set_title("A. Layer-by-Layer Hidden State Drift Collapse", fontweight="bold")
    ax_drift.grid(True, linestyle="--", alpha=0.45)
    ax_drift.legend(loc="upper left", frameon=True, facecolor="#121214", edgecolor="#26262B", fontsize=8.0)

    kl_vals = [r["logit_kl_divergence_nats"] for r in ladder]
    ax_kl.plot(rungs, kl_vals, marker="s", linewidth=2.4, markersize=7, color="#FF5A5F", label="Logit KL(P_BF16 || P_Quant)")
    for rx, kv in zip(rungs, kl_vals):
        ax_kl.text(rx, kv + max(kl_vals) * 0.04, f"{kv:.4f}", ha="center", va="bottom", fontsize=8.5, color="#FF5A5F")
    ax_kl.axhline(0.18, color="#EAEAEA", linestyle=":", linewidth=1.2, label="Gate G16.6 (<= 0.18 nats)")
    ax_kl.set_xticks(rungs)
    ax_kl.set_xticklabels([f"Rung {r}" for r in rungs], fontweight="bold")
    ax_kl.set_ylabel("KL Divergence (nats/token)", fontweight="bold")
    ax_kl.set_title("B. Next-Token Logit KL-Divergence Collapse", fontweight="bold")
    ax_kl.grid(True, linestyle="--", alpha=0.45)
    ax_kl.legend(loc="upper right", frameon=True, facecolor="#121214", edgecolor="#26262B", fontsize=8.0)

    fig2.tight_layout()
    f2_pdf = out_dir / "fig_m16_layerwise_kl_divergence.pdf"
    f2_png = out_dir / "fig_m16_layerwise_kl_divergence.png"
    fig2.savefig(f2_pdf, bbox_inches="tight")
    fig2.savefig(f2_png, bbox_inches="tight", dpi=300)
    plt.close(fig2)

    return {
        "ladder_pdf": str(f1_pdf),
        "ladder_png": str(f1_png),
        "kl_pdf": str(f2_pdf),
        "kl_png": str(f2_png),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Milestone M16 5-Rung Calibration Figures.")
    parser.add_argument("--input-json", type=str, default="results/m16/m16_30b_accuracy_ladder_results.json")
    parser.add_argument("--out-dir", type=str, default="figures/m16")
    args = parser.parse_args()
    out = render_m16_figures(Path(args.input_json), Path(args.out_dir))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
