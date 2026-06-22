#!/usr/bin/env python3
"""
Create report figures for the agentic-architecture project from exported CSV files.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ARCH_ORDER = [
    "Level 1: Single-Agent Baseline",
    "Level 2A: Planner + Executor",
    "Level 2B: Solver + Critic",
    "Level 3: Adaptive Memory Multi-Agent",
]

ARCH_SHORT = {
    "Level 1: Single-Agent Baseline": "L1\nSingle",
    "Level 2A: Planner + Executor": "L2A\nPlanner-Exec",
    "Level 2B: Solver + Critic": "L2B\nSolver-Critic",
    "Level 3: Adaptive Memory Multi-Agent": "L3\nAdaptive",
}

TASK_LABELS = {
    "cheminformatics_easy": "Chem easy",
    "cheminformatics_hard": "Chem hard",
    "materials_easy": "Materials easy",
    "materials_hard": "Materials hard",
}

HARD_TASK_LABELS = {
    "cheminformatics_hard": "ChEMBL selectivity",
    "materials_hard": "Dimensional polymorphs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication figures from agentic architecture CSV exports."
    )
    parser.add_argument("--input-dir", default=".", help="Directory containing the CSV files.")
    parser.add_argument("--output-dir", default="figures", help="Directory where figures will be written.")
    parser.add_argument("--score-matrix", default=None, help="Path to agentic_architecture_score_matrix CSV.")
    parser.add_argument("--efficiency", default=None, help="Path to agentic_architecture_efficiency CSV.")
    parser.add_argument("--summary", default=None, help="Path to agentic_architecture_results_summary CSV.")
    parser.add_argument("--raw", default=None, help="Path to agentic_architecture_results_raw CSV.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving.")
    return parser.parse_args()


def _numeric_suffix_rank(path: Path) -> int:
    """Prefer files like name(6).csv over older name(5).csv and unsuffixed files."""
    match = re.search(r"\((\d+)\)\.csv$", path.name)
    if match:
        return int(match.group(1))
    return -1


def newest_matching(input_dir: Path, pattern: str) -> Path:
    matches = [Path(p) for p in glob.glob(str(input_dir / pattern))]
    if not matches:
        raise FileNotFoundError(f"No files matched {input_dir / pattern}")
    # In ChatGPT/download folders, file modification times can be identical.
    # Prefer the highest parenthesized export suffix, e.g. score_matrix(6).csv.
    return max(matches, key=lambda p: (_numeric_suffix_rank(p), p.stat().st_mtime, p.name))


def resolve_path(value: Optional[str], input_dir: Path, pattern: str) -> Path:
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = input_dir / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    return newest_matching(input_dir, pattern)


def ensure_columns(df: pd.DataFrame, required: Iterable[str], file_label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label} missing required columns: {missing}. Columns found: {list(df.columns)}")


def order_architectures(df: pd.DataFrame) -> pd.DataFrame:
    if "architecture" not in df.columns:
        return df
    order = {name: i for i, name in enumerate(ARCH_ORDER)}
    out = df.copy()
    out["_arch_order"] = out["architecture"].map(order).fillna(999)
    return out.sort_values(["_arch_order", "architecture"]).drop(columns=["_arch_order"])


def short_arch_labels(architectures: Iterable[str]) -> List[str]:
    return [ARCH_SHORT.get(a, re.sub(r"\s+", "\n", str(a), count=1)) for a in architectures]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def annotate_bars(ax, bars, fmt="{:.2f}", y_offset=0.01):
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_hard_scores(score_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    ensure_columns(score_df, ["architecture", "cheminformatics_hard", "materials_hard"], "score matrix")
    df = order_architectures(score_df)
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    chem = df["cheminformatics_hard"].astype(float).to_numpy()
    mat = df["materials_hard"].astype(float).to_numpy()

    b1 = ax.bar(x - width / 2, chem, width, label=HARD_TASK_LABELS["cheminformatics_hard"])
    b2 = ax.bar(x + width / 2, mat, width, label=HARD_TASK_LABELS["materials_hard"])
    annotate_bars(ax, b1, "{:.3f}", y_offset=0.01)
    annotate_bars(ax, b2, "{:.3f}", y_offset=0.01)

    ax.set_title("Hard-task performance by architecture")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_arch_labels(df["architecture"]), fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "report_expert_hard_scores", dpi)


def plot_complexity(eff_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    ensure_columns(eff_df, ["architecture", "mean_steps", "mean_llm_calls"], "efficiency")
    df = order_architectures(eff_df)
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    steps = df["mean_steps"].astype(float).to_numpy()
    calls = df["mean_llm_calls"].astype(float).to_numpy()
    b1 = ax.bar(x - width / 2, steps, width, label="Mean workflow steps")
    b2 = ax.bar(x + width / 2, calls, width, label="Mean LLM calls")
    annotate_bars(ax, b1, "{:.1f}", y_offset=0.05)
    annotate_bars(ax, b2, "{:.1f}", y_offset=0.05)

    ax.set_title("Agentic complexity")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(short_arch_labels(df["architecture"]), fontsize=9)
    ymax = max(np.nanmax(steps), np.nanmax(calls)) if len(df) else 1
    ax.set_ylim(0, ymax + 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "report_expert_complexity", dpi)


def _token_column(df: pd.DataFrame) -> str:
    if "mean_effective_total_tokens" in df.columns:
        return "mean_effective_total_tokens"
    if "mean_estimated_total_tokens" in df.columns:
        return "mean_estimated_total_tokens"
    return "mean_estimated_tokens"


def plot_tradeoff(eff_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    token_col = _token_column(eff_df)
    ensure_columns(
        eff_df,
        ["architecture", "mean_score", token_col, "pass_rate", "mean_runtime_seconds"],
        "efficiency",
    )
    df = order_architectures(eff_df)

    tokens = df[token_col].astype(float).to_numpy()
    scores = df["mean_score"].astype(float).to_numpy()
    pass_rate = df["pass_rate"].astype(float).to_numpy()
    runtime = df["mean_runtime_seconds"].astype(float).to_numpy()

    # Marker area reflects runtime while keeping readable bounds.
    if np.nanmax(runtime) > np.nanmin(runtime):
        sizes = 80 + 320 * (runtime - np.nanmin(runtime)) / (np.nanmax(runtime) - np.nanmin(runtime))
    else:
        sizes = np.full_like(runtime, 180.0)

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    scatter = ax.scatter(tokens, scores, s=sizes, alpha=0.85)

    for _, row in df.iterrows():
        label = ARCH_SHORT.get(row["architecture"], row["architecture"]).replace("\n", " ")
        ax.annotate(
            label,
            (float(row[token_col]), float(row["mean_score"])),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )

    ax.set_title("Score-token trade-off")
    ax.set_xlabel("Mean effective tokens")
    ax.set_ylabel("Mean score")
    lower = max(0.0, np.nanmin(scores) - 0.02)
    upper = min(1.02, np.nanmax(scores) + 0.02)
    ax.set_ylim(lower, upper)
    ax.grid(alpha=0.3)

    # Bubble-size note rather than a legend, to keep it compact for LaTeX.
    ax.text(
        0.02,
        0.02,
        "Bubble area ≈ mean runtime",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "report_expert_tradeoff", dpi)


def plot_token_accounting(eff_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    required = [
        "architecture",
        "mean_estimated_total_tokens",
        "mean_actual_total_tokens",
        "mean_effective_total_tokens",
    ]
    if not all(col in eff_df.columns for col in required):
        if "mean_estimated_tokens" in eff_df.columns:
            required = ["architecture", "mean_estimated_tokens"]
        else:
            ensure_columns(eff_df, required, "efficiency")
    ensure_columns(eff_df, required, "efficiency")

    df = order_architectures(eff_df)
    x = np.arange(len(df))
    width = 0.24

    if "mean_effective_total_tokens" in df.columns:
        estimated = df["mean_estimated_total_tokens"].astype(float).to_numpy()
        actual = df["mean_actual_total_tokens"].astype(float).to_numpy()
        series = [
            ("Estimated", estimated, x - width),
            ("Actual", actual, x),
        ]
    else:
        estimated = df["mean_estimated_tokens"].astype(float).to_numpy()
        series = [("Estimated", estimated, x)]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = []
    for label, values, positions in series:
        bars.append(ax.bar(positions, values, width, label=label))
    for bar_group in bars:
        annotate_bars(ax, bar_group, "{:.0f}", y_offset=max(np.nanmax(estimated) * 0.01, 5))

    ax.set_title("Token accounting by architecture")
    ax.set_ylabel("Mean tokens per task")
    ax.set_xticks(x)
    ax.set_xticklabels(short_arch_labels(df["architecture"]), fontsize=9)
    ymax = max(np.nanmax(values) for _, values, _ in series) if len(df) else 1
    ax.set_ylim(0, ymax * 1.18 + 1)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir, "report_expert_token_accounting", dpi)


def plot_score_heatmap(score_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    ensure_columns(score_df, ["architecture"], "score matrix")
    task_cols = [c for c in ["cheminformatics_easy", "cheminformatics_hard", "materials_easy", "materials_hard"] if c in score_df.columns]
    if not task_cols:
        return
    df = order_architectures(score_df)
    matrix = df[task_cols].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_title("Score matrix")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(short_arch_labels(df["architecture"]), fontsize=8)
    ax.set_xticks(np.arange(len(task_cols)))
    ax.set_xticklabels([TASK_LABELS.get(c, c) for c in task_cols], rotation=25, ha="right")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean score")
    fig.tight_layout()
    save_figure(fig, output_dir, "report_expert_score_heatmap", dpi)


def plot_pass_rate(eff_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    ensure_columns(eff_df, ["architecture", "pass_rate"], "efficiency")
    df = order_architectures(eff_df)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    bars = ax.bar(x, df["pass_rate"].astype(float).to_numpy())
    annotate_bars(ax, bars, "{:.2f}", y_offset=0.01)
    ax.set_title("Strict pass rate by architecture")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1.10)
    ax.set_xticks(x)
    ax.set_xticklabels(short_arch_labels(df["architecture"]), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "report_expert_pass_rate", dpi)


def plot_task_scores(summary_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    ensure_columns(summary_df, ["architecture", "domain", "difficulty", "mean_score"], "summary")
    df = summary_df.copy()
    df["task_label"] = df["domain"].str.capitalize() + " " + df["difficulty"].str.capitalize()
    pivot = df.pivot_table(index="architecture", columns="task_label", values="mean_score", aggfunc="mean")
    pivot = pivot.reindex(ARCH_ORDER)
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(pivot))
    cols = list(pivot.columns)
    width = min(0.8 / max(len(cols), 1), 0.22)
    offsets = (np.arange(len(cols)) - (len(cols) - 1) / 2) * width
    for offset, col in zip(offsets, cols):
        ax.bar(x + offset, pivot[col].astype(float).to_numpy(), width, label=col)

    ax.set_title("Mean score by task")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_arch_labels(pivot.index), fontsize=9)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "report_expert_task_scores", dpi)


def plot_failure_counts(raw_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    ensure_columns(raw_df, ["task_id", "passed"], "raw results")
    df = raw_df.copy()
    # Accept booleans or strings from CSV.
    passed = df["passed"].astype(str).str.lower().isin(["true", "1", "yes"])
    failed_df = df.loc[~passed].copy()
    if failed_df.empty:
        # Still create an empty-success figure so the LaTeX build has a stable target.
        fig, ax = plt.subplots(figsize=(6.3, 3.6))
        ax.axis("off")
        ax.text(0.5, 0.5, "No strict validation failures", ha="center", va="center", fontsize=14)
        fig.tight_layout()
        save_figure(fig, output_dir, "report_expert_failure_counts", dpi)
        return

    counts = failed_df.groupby("task_id").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    y = np.arange(len(counts))
    ax.barh(y, counts.to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels([str(x).replace("_", " ") for x in counts.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Failed rows")
    ax.set_title("Strict-validation failures by task")
    ax.grid(axis="x", alpha=0.3)
    for i, value in enumerate(counts.to_numpy()):
        ax.text(value + 0.05, i, str(value), va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "report_expert_failure_counts", dpi)


def write_latex_snippets(output_dir: Path, figure_stems: List[str]) -> None:
    lines = [
        "% Auto-generated LaTeX figure snippets. Copy the blocks you want into your report.",
        "% Put the figures directory next to your .tex file or update the paths below.",
        "",
    ]
    for stem in figure_stems:
        lines.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.85\linewidth]{{figures/{stem}.png}}",
                rf"\caption{{TODO: Caption for {stem.replace('_', ' ')}.}}",
                rf"\label{{fig:{stem}}}",
                r"\end{figure}",
                "",
            ]
        )
    (output_dir / "latex_figure_snippets.tex").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_dir: Path, inputs: dict, figure_stems: List[str]) -> None:
    lines = ["Generated report figures", "========================", "", "Input files:"]
    for name, path in inputs.items():
        lines.append(f"- {name}: {path}")
    lines.extend(["", "Figures:"])
    for stem in figure_stems:
        lines.append(f"- {stem}.png")
        lines.append(f"- {stem}.pdf")
    lines.extend(
        [
            "",
            "The report generated earlier directly expects these file names:",
            "- report_expert_hard_scores.png",
            "- report_expert_complexity.png",
            "- report_expert_tradeoff.png",
            "- report_expert_token_accounting.png",
        ]
    )
    (output_dir / "figure_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    score_path = resolve_path(args.score_matrix, input_dir, "agentic_architecture_score_matrix.csv")
    efficiency_path = resolve_path(args.efficiency, input_dir, "agentic_architecture_efficiency.csv")
    summary_path = resolve_path(args.summary, input_dir, "agentic_architecture_results_summary.csv")
    raw_path = resolve_path(args.raw, input_dir, "agentic_architecture_results_raw.csv")

    score_df = pd.read_csv(score_path)
    eff_df = pd.read_csv(efficiency_path)
    summary_df = pd.read_csv(summary_path)
    raw_df = pd.read_csv(raw_path)

    generated_stems = [
        "report_expert_hard_scores",
        "report_expert_complexity",
        "report_expert_tradeoff",
        "report_expert_token_accounting",
        "report_expert_score_heatmap",
        "report_expert_pass_rate",
        "report_expert_task_scores",
        "report_expert_failure_counts",
    ]

    plot_hard_scores(score_df, output_dir, args.dpi)
    plot_complexity(eff_df, output_dir, args.dpi)
    plot_tradeoff(eff_df, output_dir, args.dpi)
    plot_token_accounting(eff_df, output_dir, args.dpi)
    plot_score_heatmap(score_df, output_dir, args.dpi)
    plot_pass_rate(eff_df, output_dir, args.dpi)
    plot_task_scores(summary_df, output_dir, args.dpi)
    plot_failure_counts(raw_df, output_dir, args.dpi)

    write_latex_snippets(output_dir, generated_stems)
    write_manifest(
        output_dir,
        {
            "score_matrix": score_path,
            "efficiency": efficiency_path,
            "summary": summary_path,
            "raw": raw_path,
        },
        generated_stems,
    )

    print(f"Figures written to: {output_dir}")
    print("Generated files:")
    for stem in generated_stems:
        print(f"  - {stem}.png")
        print(f"  - {stem}.pdf")
    print("  - latex_figure_snippets.tex")
    print("  - figure_manifest.txt")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
