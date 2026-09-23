"""Create slide-ready figures from the committed reproduction summary."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_COLOR = "#6B7280"
OURS_COLOR = "#2563EB"
ACCENT_COLOR = "#D97706"


def label_bars(ax, bars, *, digits: int = 1) -> None:
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.{digits}f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def accuracy_comparison(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), dpi=180)
    x = np.arange(len(data))
    width = 0.37
    panels = (
        (axes[0], "paper_snli", "ours_snli", "SNLI test accuracy", (89, 94)),
        (axes[1], "paper_ood", "ours_ood", "NLI Diagnostics accuracy", (58, 66)),
    )
    for ax, paper_col, ours_col, title, ylim in panels:
        paper = ax.bar(x - width / 2, data[paper_col], width, label="Paper", color=PAPER_COLOR)
        ours = ax.bar(x + width / 2, data[ours_col], width, label="Reproduction", color=OURS_COLOR)
        label_bars(ax, paper)
        label_bars(ax, ours)
        ax.set_title(title, fontsize=17, weight="bold", pad=14)
        ax.set_xticks(x, data["label"], rotation=18, ha="right")
        ax.set_ylim(*ylim)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Dataset Cartography reproduction: paper vs. one-seed run", fontsize=20, weight="bold")
    fig.text(0.5, 0.015, "RoBERTa-large on SNLI • seed 93078 • effective batch size 96", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def selection_comparison(data: pd.DataFrame, output: Path) -> None:
    subsets = data[data.subset != "full"].copy()
    full = data.loc[data.subset == "full"].iloc[0]
    x = np.arange(len(subsets))
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=180)
    bars = ax.bar(x, subsets.ours_snli, width=0.62, color=[PAPER_COLOR, "#DC2626", OURS_COLOR])
    label_bars(ax, bars, digits=2)
    ax.axhline(full.ours_snli, color=ACCENT_COLOR, linewidth=2.5, linestyle="--", label=f"Full data: {full.ours_snli:.2f}%")
    ax.set_xticks(x, subsets.label)
    ax.set_ylim(89.5, 93.2)
    ax.set_ylabel("SNLI test accuracy (%)", fontsize=14)
    ax.set_title("Which 33% of the training data should we keep?", fontsize=21, weight="bold", pad=15)
    ax.text(
        0.5,
        0.94,
        "Ambiguous examples retain nearly all full-data performance",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/reproduction_metrics.csv")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    required = {"subset", "label", "paper_snli", "ours_snli", "paper_ood", "ours_ood"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accuracy_comparison(data, output / "accuracy_comparison.png")
    selection_comparison(data, output / "selection_comparison.png")


if __name__ == "__main__":
    main()
