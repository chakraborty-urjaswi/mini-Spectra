"""
src/data/eda.py
---------------
Exploratory Data Analysis script for the Project Skylark data pipeline.

Generates a set of diagnostic figures for any downloaded dataset:
  1. GPS flight path scatter coloured by altitude
  2. Blur score histogram with threshold line
  3. Exposure vs. sharpness 2-D scatter
  4. Per-image flag heatmap (blur / exposure / GPS / altitude missing)
  5. Dataset comparison bar chart (when multiple datasets are available)

Usage:
    python -m src.data.eda data/raw/odm_samples/aukerman --out reports/figures/aukerman
    python -m src.data.eda data/raw/odm_samples/aukerman --qa-csv data/raw/odm_samples/aukerman/qa_report.csv
    python -m src.data.eda --compare data/raw/odm_samples   # compare all datasets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.data.inspect import inspect_dataset, BLUR_THRESHOLD
from src.utils.viz import (
    plot_gps_scatter,
    plot_blur_histogram,
    plot_exposure_scatter,
    save_fig,
    ACCENT,
)

# ---------------------------------------------------------------------------
# Dark theme for all figures
# ---------------------------------------------------------------------------
DARK_BG = "#0F1117"
DARK_AX = "#1A1D2E"
TEXT = "#E8E8F0"
GRID = "#2A2D3E"

def _apply_dark(fig: plt.Figure, axes) -> None:
    fig.patch.set_facecolor(DARK_BG)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_color(GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
        if ax.get_legend() is not None:
            ax.get_legend().get_frame().set_facecolor(DARK_AX)
            for text in ax.get_legend().get_texts():
                text.set_color(TEXT)


# ---------------------------------------------------------------------------
# Figure 1: GPS flight path
# ---------------------------------------------------------------------------

def fig_gps_path(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    lats = df["lat"].dropna().values
    lons = df["lon"].dropna().values
    alts = df.loc[df["lat"].notna(), "alt_m"].values if "alt_m" in df.columns else None

    if len(lats) == 0:
        print(f"  [skip] No GPS data in {dataset_name}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax = plot_gps_scatter(lats, lons, alts, ax=ax,
                          title=f"Flight path — {dataset_name}")
    _apply_dark(fig, [ax])
    save_fig(fig, out_dir / "01_gps_path.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Blur histogram
# ---------------------------------------------------------------------------

def fig_blur_hist(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    scores = df["blur_score"].dropna().values
    fig, ax = plt.subplots(figsize=(7, 4))
    ax = plot_blur_histogram(scores, BLUR_THRESHOLD, ax=ax,
                             title=f"Sharpness distribution — {dataset_name}")
    _apply_dark(fig, [ax])
    save_fig(fig, out_dir / "02_blur_histogram.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Exposure vs. sharpness
# ---------------------------------------------------------------------------

def fig_exposure(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    bm = df["brightness_mean"].values
    bs = df["blur_score"].values
    fig, ax = plt.subplots(figsize=(7, 5))
    ax = plot_exposure_scatter(bm, bs, ax=ax,
                               title=f"Exposure vs. Sharpness — {dataset_name}")
    _apply_dark(fig, [ax])
    save_fig(fig, out_dir / "03_exposure_sharpness.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Per-image flag heatmap
# ---------------------------------------------------------------------------

def fig_flag_heatmap(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    flag_cols = {
        "is_blurry": "Blurry",
        "underexposed": "Underexposed",
        "overexposed": "Overexposed",
        "file_ok": "File OK",
        "has_gps": "Has GPS",
        "has_altitude": "Has Alt",
    }
    # Invert file_ok so that True = problem
    existing = [c for c in flag_cols if c in df.columns]
    matrix = df[existing].copy()
    if "file_ok" in matrix.columns:
        matrix["file_ok"] = ~matrix["file_ok"].astype(bool)

    labels = [flag_cols[c] for c in existing]
    data = matrix.values.astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(4, len(df) * 0.3)))
    im = ax.imshow(data.T, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        [Path(f).stem for f in df["filename"]], rotation=45, ha="right", fontsize=7
    )
    ax.set_title(f"QA flags — {dataset_name}", pad=10)
    plt.colorbar(im, ax=ax, label="Flag (1=issue)", shrink=0.6)
    _apply_dark(fig, [ax])
    plt.tight_layout()
    save_fig(fig, out_dir / "04_qa_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Dataset comparison (multi-dataset)
# ---------------------------------------------------------------------------

def fig_dataset_comparison(datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Bar chart comparing key QA stats across multiple datasets."""
    names = list(datasets.keys())
    metrics = {
        "Total images": [len(df) for df in datasets.values()],
        "% GPS": [df["has_gps"].mean() * 100 if "has_gps" in df.columns else 0 for df in datasets.values()],
        "% Blurry": [df["is_blurry"].mean() * 100 if "is_blurry" in df.columns else 0 for df in datasets.values()],
        "% File OK": [df["file_ok"].mean() * 100 if "file_ok" in df.columns else 100 for df in datasets.values()],
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    colors = [ACCENT, "#4FC3F7", "#EF5350", "#66BB6A"]
    for ax, (metric, vals), color in zip(axes, metrics.items(), colors):
        ax.bar(names, vals, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(metric)
        ax.set_ylim(0, max(vals) * 1.2 + 1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    fig.suptitle("Dataset Comparison", fontsize=14, y=1.02, color=TEXT)
    _apply_dark(fig, axes)
    plt.tight_layout()
    save_fig(fig, out_dir / "05_dataset_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main EDA runner
# ---------------------------------------------------------------------------

def run_eda(
    dataset_dir: Path,
    qa_csv: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run full EDA for one dataset. Returns the QA DataFrame."""
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name

    # Find images sub-folder
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        images_dir = dataset_dir  # fallback to root

    if qa_csv and Path(qa_csv).exists():
        print(f"Loading existing QA report: {qa_csv}")
        df = pd.read_csv(qa_csv)
    else:
        print(f"Running QA inspection on {images_dir} …")
        df = inspect_dataset(images_dir)

    if out_dir is None:
        out_dir = dataset_dir / "eda_figures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating EDA figures → {out_dir}")
    fig_gps_path(df, dataset_name, out_dir)
    fig_blur_hist(df, dataset_name, out_dir)
    fig_exposure(df, dataset_name, out_dir)
    fig_flag_heatmap(df, dataset_name, out_dir)

    print(f"\n✓ EDA complete — {len(list(out_dir.glob('*.png')))} figures saved to {out_dir}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDA for Project Skylark datasets")
    parser.add_argument("dataset_dir", nargs="?",
                        help="Path to a single dataset folder")
    parser.add_argument("--qa-csv", default=None,
                        help="Path to existing QA CSV (skip re-running inspect)")
    parser.add_argument("--out", default=None, help="Output directory for figures")
    parser.add_argument("--compare", default=None,
                        help="Directory containing multiple dataset folders to compare")
    args = parser.parse_args()

    if args.compare:
        # Multi-dataset comparison
        compare_root = Path(args.compare)
        datasets = {}
        for d in sorted(compare_root.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                images_dir = d / "images" if (d / "images").exists() else d
                try:
                    datasets[d.name] = inspect_dataset(images_dir)
                except FileNotFoundError:
                    pass
        out_dir = Path(args.out) if args.out else compare_root / "comparison_figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dataset_comparison(datasets, out_dir)
        print(f"Comparison figure saved to {out_dir}")

    elif args.dataset_dir:
        run_eda(
            dataset_dir=Path(args.dataset_dir),
            qa_csv=Path(args.qa_csv) if args.qa_csv else None,
            out_dir=Path(args.out) if args.out else None,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
