"""
Publication-Grade Plotting Script for Milestone M17:
Generates:
1. `figures/fig_m17_agentic_horizon_survival.{pdf,png}`
2. `figures/fig_m17_5domain_radar_and_ci.{pdf,png}`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def render_m17_figures(input_json: Path, out_dir: Path) -> Dict[str, str]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    per_seed = data["per_seed_results"]
    storage = data["storage_parity_report"]

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

    turns = [1, 2, 3, 4, 5, 6]
    palette = {
        "FP16_REF": ("#EAEAEA", "o", "--", "BF16 Reference (55.46 GiB Uncompressed)"),
        "LEAN_PQv2": ("#FF5A5F", "s", "-", "LEAN Priced-Quanta v2 (6.08 GiB + BM25)"),
        "UNIFORM_MATCHED": ("#6E7078", "^", "-.", "Matched Uniform Low-Bit (6.08 GiB, ΔBytes=0%)"),
    }

    for m_key, (color, marker, ls, label) in palette.items():
        curves = np.array(
            [
                [r["domain1_survival_curve"][f"turn_{t}"] * 100.0 for t in turns]
                for r in per_seed[m_key]
            ]
        )
        mean_c = curves.mean(axis=0)
        std_c = curves.std(axis=0, ddof=1)
        ci95 = 1.96 * std_c / np.sqrt(max(1, len(curves)))
        ax1.plot(turns, mean_c, color=color, marker=marker, linestyle=ls, linewidth=2.4, markersize=6, label=label)
        ax1.fill_between(turns, np.clip(mean_c - ci95, 0, 100), np.clip(mean_c + ci95, 0, 100), color=color, alpha=0.16)

    ax1.set_title("A. Multi-Turn ReAct Agentic Survival Horizon (N=5 Seeds)", fontweight="bold", pad=10)
    ax1.set_xlabel("Agentic Loop Turn Horizon (K Turns)")
    ax1.set_ylabel("Trajectory Survival Rate (%)")
    ax1.set_xticks(turns)
    ax1.set_ylim(0, 106)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="lower left", frameon=True, facecolor="#121214", edgecolor="#26262B", fontsize=8.0)

    depths = [10, 25, 50, 75, 90]
    for m_key, (color, marker, ls, label) in palette.items():
        d_mat = np.array(
            [
                [r["domain5_needle_by_depth"][f"depth_{d}pct"] * 100.0 for d in depths]
                for r in per_seed[m_key]
            ]
        )
        mean_d = d_mat.mean(axis=0)
        std_d = d_mat.std(axis=0, ddof=1)
        ci95_d = 1.96 * std_d / np.sqrt(max(1, len(d_mat)))
        ax2.plot(depths, mean_d, color=color, marker=marker, linestyle=ls, linewidth=2.4, markersize=6, label=label)
        ax2.fill_between(depths, np.clip(mean_d - ci95_d, 0, 100), np.clip(mean_d + ci95_d, 0, 100), color=color, alpha=0.16)

    ax2.set_title("B. Long-Context Needle-in-a-Haystack by Context Depth", fontweight="bold", pad=10)
    ax2.set_xlabel("Needle Burial Depth in Context Window (%)")
    ax2.set_ylabel("Exact Passcode Retrieval Accuracy (%)")
    ax2.set_xticks(depths)
    ax2.set_ylim(0, 106)
    ax2.grid(True, linestyle="--", alpha=0.45)

    fig.tight_layout()
    f1_pdf = out_dir / "fig_m17_agentic_horizon_survival.pdf"
    f1_png = out_dir / "fig_m17_agentic_horizon_survival.png"
    fig.savefig(f1_pdf, bbox_inches="tight")
    fig.savefig(f1_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(11.5, 5.0), dpi=300)
    domain_keys = [
        ("domain1_react_completion", "D1: 6-Turn\nReAct Agent"),
        ("domain2_code_pass_at_1", "D2a: Code\nPass@1"),
        ("domain2_code_pass_at_3", "D2b: Code Pass@3\n(Self-Repair)"),
        ("domain3_deductive_math_acc", "D3: Deductive\n& Math (0-Search)"),
        ("domain4_factual_offload_acc", "D4: BM25 Pocket\nIndex Offload"),
        ("domain5_needle_mean_acc", "D5: Long-Ctx\nNeedle (10-90%)"),
    ]

    x = np.arange(len(domain_keys))
    width = 0.26

    for idx, (m_key, (color, _, _, label)) in enumerate(palette.items()):
        means = []
        cis = []
        for k, _ in domain_keys:
            vals = np.array([r[k] * 100.0 for r in per_seed[m_key]])
            means.append(vals.mean())
            cis.append(1.96 * vals.std(ddof=1) / np.sqrt(max(1, len(vals))))
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=cis,
            capsize=3.5,
            color=color,
            edgecolor="#121214",
            linewidth=0.8,
            label=label,
        )
        if m_key == "LEAN_PQv2":
            for b, m_val in zip(bars, means):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    m_val + 2.8,
                    f"{m_val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#FF5A5F",
                )

    ax.set_title(
        f"Milestone M17: 5-Domain Generalization on Meta Muse-Glimmer-30B (N=5 Seeds, Byte Delta = {storage['byte_delta_pct']:.3f}%)",
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in domain_keys], fontweight="bold")
    ax.set_ylabel("Accuracy / Task Completion Rate (%)")
    ax.set_ylim(0, 114)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.legend(loc="upper right", frameon=True, facecolor="#121214", edgecolor="#26262B", fontsize=8.5)

    fig2.tight_layout()
    f2_pdf = out_dir / "fig_m17_5domain_radar_and_ci.pdf"
    f2_png = out_dir / "fig_m17_5domain_radar_and_ci.png"
    fig2.savefig(f2_pdf, bbox_inches="tight")
    fig2.savefig(f2_png, bbox_inches="tight", dpi=300)
    plt.close(fig2)

    return {
        "horizon_pdf": str(f1_pdf),
        "horizon_png": str(f1_png),
        "domains_pdf": str(f2_pdf),
        "domains_png": str(f2_png),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Milestone M17 5-Domain Agentic Figures.")
    parser.add_argument("--input-json", type=str, default="results/m17_30b_agentic_multidomain_results.json")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()
    paths = render_m17_figures(Path(args.input_json), Path(args.out_dir))
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
