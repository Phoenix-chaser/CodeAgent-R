#!/usr/bin/env python3
"""重画 README 里的三张图，规格同 docs/images 里原有的图：宽 1784 像素、200 dpi、白底、DejaVu Sans。

数字都写在下面的常量里，和 README「Qwen3.5-9B 上的结果」一节逐个对应；实验结果变了就两处一起改。
只依赖 matplotlib（不在 dev 依赖里）：python scripts/make_readme_charts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # matplotlib.use() 要在导入 pyplot 之前
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
DPI = 200
WIDTH_IN = 8.92                                                # 1784 像素，同 method-comparison.png
BASE, VERIFIED, CONTROL = "#b0c4de", "#4169e1", "#ff8c00"       # lightsteelblue / royalblue / darkorange
HAND_WRITTEN, SYNTHETIC = "#b0c4de", "#87ceeb"                 # lightsteelblue / skyblue
INK, MUTED, LINE = "#1f2328", "#57606a", "#8c959f"

# 扩大题池后的 RFT：dev 165 题上贪心解码的通过数（README 里那张表的最后两列）
RFT_GROUPS = (("Human-written · 66 tasks", (29, 30, 31)), ("Synthetic · 99 tasks", (39, 49, 50)))
ARMS = ("Base", "A-verified", "C-control")
ARM_COLORS = (BASE, VERIFIED, CONTROL)

# 基座在训练集上每题采 8 次：(通过的样本数, 采样总数)
PASS_RATE = ((3, 192), (146, 880), (88, 424), (1122, 2392))
TIER_LABELS = ("Hand-written\nmulti-defect", "Hand-written\nsingle-defect", "Synthetic\nmulti-defect",
               "Synthetic\nsingle-defect")

PIPELINE = ("Frozen task pool", "Grading by\nexecution", "QLoRA SFT\nDPO · RFT", "Selection\non dev",
            "Paired test\nvs Base")


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": LINE, "xtick.color": INK, "ytick.color": MUTED, "text.color": INK,
        "axes.labelcolor": MUTED, "axes.labelsize": 9.5,
    })


def grid(ax) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#d0d7de")
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", length=0)


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.35))
    span = 2.2
    for index, label in enumerate(PIPELINE):
        x = index * span
        ax.add_patch(FancyBboxPatch((x - 0.92, -0.44), 1.84, 0.88,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=0.8, edgecolor=LINE, facecolor="#f3f5f7"))
        ax.text(x, 0, label, ha="center", va="center", fontsize=9.5, linespacing=1.35)
        if index:
            ax.add_patch(FancyArrowPatch((x - span + 0.94, 0), (x - 0.94, 0), arrowstyle="-|>",
                                         mutation_scale=9, linewidth=0.8, color=LINE))
    ax.set_xlim(-1.1, (len(PIPELINE) - 1) * span + 1.1)
    ax.set_ylim(-0.56, 0.56)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "pipeline-overview.png", dpi=DPI)
    plt.close(fig)


def rft_after_expanding() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
    xs, heights = [], []
    for group, (_, values) in enumerate(RFT_GROUPS):
        start = group * 4.3
        xs.extend(start + offset for offset in range(len(values)))
        heights.extend(values)
        ax.hlines(values[0], start - 0.42, start + 2.42, colors=MUTED, linestyles="dashed", linewidth=0.8)
    ax.bar(xs, heights, width=0.78, color=list(ARM_COLORS) * len(RFT_GROUPS))
    for x, height in zip(xs, heights):
        ax.text(x, height + 0.7, str(height), ha="center", va="bottom", fontsize=10, fontweight="bold")
    for group, (title, _) in enumerate(RFT_GROUPS):
        ax.text(group * 4.3 + 1, 59, title, ha="center", va="bottom", fontsize=10)
    ax.text(-0.42, 50, "dashed line: the Base level", ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(ARMS) * len(RFT_GROUPS))
    ax.set_ylim(0, 66)
    ax.set_ylabel("tasks passed (dev, greedy decoding)")
    ax.set_title("RFT after expanding the task pool · Qwen3.5-9B")
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "qwen35-rft-expanded.png", dpi=DPI)
    plt.close(fig)


def base_pass_rate() -> None:
    rates = [passed / total * 100 for passed, total in PASS_RATE]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.9))
    ax.bar(range(len(PASS_RATE)), rates, width=0.58,
           color=[HAND_WRITTEN, HAND_WRITTEN, SYNTHETIC, SYNTHETIC])
    for index, rate in enumerate(rates):
        ax.text(index, rate + 1.0, f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(PASS_RATE)))
    ax.set_xticklabels([f"{label}\n{passed}/{total} samples"
                        for label, (passed, total) in zip(TIER_LABELS, PASS_RATE)], fontsize=9)
    ax.set_ylim(0, 58)
    ax.set_ylabel("pass rate per sample (%)")
    ax.set_title("Base model on the training tasks · Qwen3.5-9B, 8 samples per task")
    ax.legend(handles=[Patch(color=HAND_WRITTEN, label="hand-written tasks"),
                       Patch(color=SYNTHETIC, label="synthetic tasks")],
              loc="upper left", frameon=False, fontsize=9)
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "qwen35-base-pass-rate.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    style()
    pipeline()
    rft_after_expanding()
    base_pass_rate()
    for name in ("pipeline-overview.png", "qwen35-rft-expanded.png", "qwen35-base-pass-rate.png"):
        print(OUT / name)


if __name__ == "__main__":
    main()
