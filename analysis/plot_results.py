#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", data)
    required = {"accuracy", "unfaithful_rate"}
    missing = required - metrics.keys()
    if missing:
        raise KeyError(f"Missing metrics {sorted(missing)} in {path}")
    return metrics


def build_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    specs = [
        ("Qwen3", "Baseline", args.baseline_qwen),
        ("Qwen3", "SDF", args.sft_qwen),
        ("DeepSeek", "Baseline", args.baseline_deepseek),
        ("DeepSeek", "SDF", args.sft_deepseek),
    ]
    records = []
    for model, condition, path in specs:
        metrics = load_metrics(path)
        records.append(
            {
                "model": model,
                "condition": condition,
                "accuracy": metrics["accuracy"],
                "unfaithful_rate": metrics["unfaithful_rate"],
            }
        )
    return pd.DataFrame.from_records(records)


def add_inside_labels(
    ax: plt.Axes, bars, values: np.ndarray, offset: float = 0.02, fontsize: int = 13
) -> None:
    for bar, val in zip(bars, values):
        height = bar.get_height()
        y = max(0.0, height - offset)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{int(round(val * 100))}%",
            ha="center",
            va="top",
            fontsize=fontsize,
            color="white",
            fontweight="bold",
        )


def plot_accuracy(
    df: pd.DataFrame, out_prefix: Path, title: str | None, dpi: int
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    models = ["Qwen3", "DeepSeek"]
    conditions = ["Baseline", "SDF"]
    acc = (
        df.pivot(index="model", columns="condition", values="accuracy")
        .reindex(models)[conditions]
    )

    x = np.arange(len(models))
    width = 0.36

    colors = {"Baseline": "#4C78A8", "SDF": "#E45756"}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    baseline_bars = ax.bar(
        x - width / 2,
        acc["Baseline"],
        width,
        label="Baseline",
        color=colors["Baseline"],
    )
    sdf_bars = ax.bar(
        x + width / 2,
        acc["SDF"],
        width,
        label="SDF",
        color=colors["SDF"],
    )

    ax.set_xticks(x, models)
    ax.set_ylabel("Task accuracy")
    ax.set_ylim(0, 1.0)
    ax.margins(y=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.spines["top"].set_visible(False)
    add_inside_labels(ax, baseline_bars, acc["Baseline"].to_numpy())
    add_inside_labels(ax, sdf_bars, acc["SDF"].to_numpy())

    if title:
        ax.set_title(title)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_faithfulness(
    df: pd.DataFrame, out_prefix: Path, title: str | None, dpi: int
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    models = ["Qwen3", "DeepSeek"]
    conditions = ["Baseline", "SDF"]
    unfaithful = (
        df.pivot(index="model", columns="condition", values="unfaithful_rate")
        .reindex(models)[conditions]
    )
    faithful = 1.0 - unfaithful

    x = np.arange(len(models))
    width = 0.36

    colors = {"Baseline": "#4C78A8", "SDF": "#E45756"}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    baseline_bars = ax.bar(
        x - width / 2,
        faithful["Baseline"],
        width,
        label="Baseline",
        color=colors["Baseline"],
    )
    sdf_bars = ax.bar(
        x + width / 2,
        faithful["SDF"],
        width,
        label="SDF",
        color=colors["SDF"],
    )
    ax.set_xticks(x, models)
    ax.set_ylabel("Faithfulness rate")
    ax.set_ylim(0.0, 1.0)
    ax.margins(y=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.spines["top"].set_visible(False)
    add_inside_labels(ax, baseline_bars, faithful["Baseline"].to_numpy())
    add_inside_labels(ax, sdf_bars, faithful["SDF"].to_numpy())

    if title:
        ax.set_title(title)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy and unfaithfulness.")
    parser.add_argument("--baseline_qwen", required=True)
    parser.add_argument("--sft_qwen", required=True)
    parser.add_argument("--baseline_deepseek", required=True)
    parser.add_argument("--sft_deepseek", required=True)
    parser.add_argument("--out_dir", default="analysis")
    parser.add_argument("--out_prefix", default="results_figure")
    parser.add_argument("--title_accuracy", default=None)
    parser.add_argument("--title_faithfulness", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataframe(args)
    accuracy_prefix = out_dir / f"{args.out_prefix}_accuracy"
    faithful_prefix = out_dir / f"{args.out_prefix}_faithfulness"
    plot_accuracy(df, accuracy_prefix, args.title_accuracy, args.dpi)
    plot_faithfulness(df, faithful_prefix, args.title_faithfulness, args.dpi)


if __name__ == "__main__":
    main()
