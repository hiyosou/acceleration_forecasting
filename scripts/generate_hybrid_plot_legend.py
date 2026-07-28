"""Generate a standalone legend used by the hybrid evaluation plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def create_legend(output_path: Path, *, dpi: int = 200) -> None:
    plt.rcParams["font.family"] = "MS Gothic"

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=7,
               markerfacecolor="deepskyblue", markeredgecolor="deepskyblue",
               alpha=0.75, label="実測最大加速度（色：走行速度）"),
        Line2D([], [], linestyle="--", color="tab:blue", linewidth=1.5,
               label="類似ガイド1"),
        Line2D([], [], linestyle="--", color="tab:green", linewidth=1.5,
               label="類似ガイド2"),
        Line2D([], [], linestyle="--", color="tab:purple", linewidth=1.5,
               label="類似ガイド3"),
        Patch(facecolor="red", edgecolor="none", alpha=0.15,
              label="予測 p10–p90"),
        Line2D([], [], linestyle="-", marker="o", color="red", linewidth=2,
               markersize=4, label="予測中央値"),
        Line2D([], [], linestyle="-.", marker="x", color="darkorange",
               linewidth=2, markersize=5, label="生成例"),
        Line2D([], [], linestyle="--", color="firebrick", linewidth=1,
               label="p10 / p90 境界"),
        Line2D([], [], linestyle="-", marker="o", color="black", linewidth=1.5,
               markerfacecolor="white", markersize=5, label="未来正解"),
        Line2D([], [], linestyle="-", color="steelblue", linewidth=1.5,
               label="予測起点"),
        Line2D([], [], linestyle="--", color="navy", linewidth=2.5,
               label="予測開始"),
        Line2D([], [], linestyle="--", color="0.35", linewidth=1,
               label="施工日"),
    ]

    fig = plt.figure(figsize=(11.5, 2.15))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="0.65",
        fontsize=11,
        handlelength=2.8,
        columnspacing=1.8,
        borderpad=0.9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=True, pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation_one_anchor/hybrid_plot_legend.png"),
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    create_legend(args.output, dpi=args.dpi)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
