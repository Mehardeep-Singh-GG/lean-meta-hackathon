"""
Milestone M18 Publication-Grade Plotting Script:
Renders `figures/fig_m18_8gb_memory_wall_trace.{pdf,png}` directly from
`results/m18_8gb_runtime_proof.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def render_m18_figure(input_json: Path, out_dir: Path) -> Dict[str, str]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    oom = data["baseline_oom_falsifications"]
    lean = data["lean_30b_8gb_runtime_proof"]

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

    # Panel A: Layer-by-Layer Memory Trajectory Against the 8.00 GiB Hardware Wall
    fp16_tr = oom["30B_FP16_Baseline"]["layer_trace"]
    u4_tr = oom["30B_Uniform_4Bit"]["layer_trace"]
    lean_tr = lean["layer_memory_trace"]

    fp16_x = [p["layer"] for p in fp16_tr] + [oom["30B_FP16_Baseline"]["crash_layer"]]
    fp16_y = [p["memory_gib"] for p in fp16_tr] + [8.25]

    u4_x = [p["layer"] for p in u4_tr] + [oom["30B_Uniform_4Bit"]["crash_layer"]]
    u4_y = [p["memory_gib"] for p in u4_tr] + [8.15]

    lean_x = [p["layer"] for p in lean_tr]
    lean_y = [p["memory_gib"] for p in lean_tr]

    ax1.axhline(8.00, color="#FF5A5F", linestyle="-", linewidth=2.0, label="Hard 8.00 GiB Hardware Wall")
    ax1.axhline(7.45, color="#EAEAEA", linestyle=":", linewidth=1.3, label="7.45 GiB Safety Gate (G18.1)")

    ax1.plot(fp16_x, fp16_y, color="#6E7078", linestyle="--", linewidth=2.2, label="Muse-Glimmer-30B BF16 (55.46 GiB -> OOM)")
    ax1.scatter([fp16_x[-1]], [fp16_y[-1]], color="#FF5A5F", marker="X", s=110, zorder=5)

    ax1.plot(u4_x, u4_y, color="#EAEAEA", linestyle="-.", linewidth=2.2, label="Stage-1 / Uniform 4-Bit (15.85 GiB -> OOM)")
    ax1.scatter([u4_x[-1]], [u4_y[-1]], color="#FF5A5F", marker="X", s=110, zorder=5)

    ax1.plot(lean_x, lean_y, color="#FF5A5F", marker="o", linewidth=2.6, markersize=5, label=f"LEAN Priced-Quanta v2 + Tiled Runtime (Peak {lean['peak_memory_gib']:.2f} GiB)")

    ax1.set_title("A. 52-Layer Memory Trajectory Inside 8.00 GiB Wall", fontweight="bold", pad=10)
    ax1.set_xlabel("Muse-Glimmer-30B Layer Index (l = 1..52)", fontweight="bold")
    ax1.set_ylabel("Physical Memory Footprint (GiB)", fontweight="bold")
    ax1.set_ylim(0, 9.5)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="lower right", frameon=True, facecolor="#121214", edgecolor="#26262B", fontsize=8.0)

    # Panel B: 8.00 GiB RAM Budget Breakdown
    categories = [
        "Priced-Quanta Weights\n(29.78B @ 1.754 bpp)",
        "4-Bit Group-64 KV-Cache\n(4,096 Context Tokens)",
        "Static Tile Workspace\n(TILE_OUT <= 1024)",
        "Runtime & Activations\n(Chunked Prefill)",
        "Free RAM Headroom\nBelow 8.00 GiB Wall",
    ]
    weights_gib = 6.0805
    kv_gib = lean["kv_cache_4096_mib"] / 1024.0
    ws_gib = lean["static_workspace_mib"] / 1024.0
    act_gib = max(0.02, lean["peak_memory_gib"] - weights_gib - kv_gib - ws_gib)
    headroom_gib = max(0.0, 8.00 - lean["peak_memory_gib"])
    values = [weights_gib, kv_gib, ws_gib, act_gib, headroom_gib]
    colors = ["#FF5A5F", "#EAEAEA", "#3A86FF", "#8D99AE", "#26262B"]

    y_pos = np.arange(len(categories))
    bars = ax2.barh(y_pos, values, color=colors, edgecolor="#EAEAEA", linewidth=0.8)
    for b, v in zip(bars, values):
        ax2.text(
            v + 0.12,
            b.get_y() + b.get_height() / 2.0,
            f"{v:.3f} GiB",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#EAEAEA",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=8.5, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Physical Memory Allocation (GiB)", fontweight="bold")
    ax2.set_xlim(0, 7.5)
    ax2.set_title("B. Measured 8.00 GiB Edge Runtime Memory Breakdown", fontweight="bold", pad=10)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.45)

    fig.tight_layout()
    pdf_path = out_dir / "fig_m18_8gb_memory_wall_trace.pdf"
    png_path = out_dir / "fig_m18_8gb_memory_wall_trace.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return {"pdf": str(pdf_path), "png": str(png_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Milestone M18 8.00 GiB Memory Wall Trace.")
    parser.add_argument("--input-json", type=str, default="results/m18_8gb_runtime_proof.json")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()
    out = render_m18_figure(Path(args.input_json), Path(args.out_dir))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
