"""
src/utils/viz.py
----------------
Standard visualisation helpers for the Project Skylark pipeline.

All functions accept matplotlib Axes and return them for chaining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Default palette (dark-mode-friendly)
SKYLARK_CMAP = "magma"
DIFF_CMAP = "RdYlGn"
ACCENT = "#F5A623"  # amber


# ---------------------------------------------------------------------------
# Dataset EDA
# ---------------------------------------------------------------------------

def plot_gps_scatter(
    lats: Sequence[float],
    lons: Sequence[float],
    alts: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Flight path (GPS)",
) -> plt.Axes:
    """Scatter of GPS positions coloured by altitude (if given)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    c = alts if alts is not None else ACCENT
    sc = ax.scatter(lons, lats, c=c, cmap="plasma" if alts is not None else None,
                    s=18, alpha=0.85, linewidths=0)
    if alts is not None:
        plt.colorbar(sc, ax=ax, label="Altitude (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_blur_histogram(
    blur_scores: Sequence[float],
    threshold: float,
    ax: Optional[plt.Axes] = None,
    title: str = "Image sharpness distribution",
) -> plt.Axes:
    """Histogram of Laplacian variance scores with threshold line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    valid = [s for s in blur_scores if s >= 0]
    ax.hist(valid, bins=30, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, color="crimson", linewidth=1.5, linestyle="--",
               label=f"Threshold = {threshold}")
    ax.set_xlabel("Laplacian variance (sharpness)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_exposure_scatter(
    brightness_means: Sequence[float],
    blur_scores: Sequence[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Exposure vs. Sharpness",
) -> plt.Axes:
    """2-D scatter to find images that are both blurry and poorly exposed."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    bm = np.array(brightness_means, dtype=float)
    bs = np.array(blur_scores, dtype=float)
    mask = np.isfinite(bm) & np.isfinite(bs)
    ax.scatter(bm[mask], bs[mask], s=14, alpha=0.7, color=ACCENT, linewidths=0)
    ax.axvline(30, color="steelblue", linestyle=":", label="Under-exposure limit")
    ax.axvline(220, color="tomato", linestyle=":", label="Over-exposure limit")
    ax.set_xlabel("Brightness mean (0–255)")
    ax.set_ylabel("Laplacian variance")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# DSM / elevation
# ---------------------------------------------------------------------------

def plot_dsm(
    dsm: np.ndarray,
    transform=None,
    ax: Optional[plt.Axes] = None,
    title: str = "Digital Surface Model",
    cmap: str = SKYLARK_CMAP,
    no_data: float = -9999.0,
) -> plt.Axes:
    """Imshow of a DSM array with a colourbar."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    display = dsm.astype(float)
    display[display == no_data] = np.nan
    im = ax.imshow(display, cmap=cmap, origin="upper",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_dsm_profile(
    elevations: np.ndarray,
    distances: np.ndarray,
    ax: Optional[plt.Axes] = None,
    pile_mask: Optional[np.ndarray] = None,
    base_elevations: Optional[np.ndarray] = None,
    title: str = "Elevation cross-section",
) -> plt.Axes:
    """Cross-section line with optional pile shading and base surface."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(distances, elevations, color=ACCENT, linewidth=1.6, label="Surface")
    if base_elevations is not None:
        ax.plot(distances, base_elevations, color="steelblue", linewidth=1.2,
                linestyle="--", label="Base surface")
    if pile_mask is not None:
        ax.fill_between(distances, elevations, base_elevations or 0,
                        where=pile_mask.astype(bool), alpha=0.25,
                        color=ACCENT, label="Pile region")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Volume uncertainty
# ---------------------------------------------------------------------------

def plot_volume_distribution(
    volumes: Sequence[float],
    true_vol: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    label: str = "Monte Carlo volumes",
    title: str = "Volume uncertainty distribution",
) -> plt.Axes:
    """Histogram of MC volume samples with true value and 95% CI."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    v = np.array(volumes)
    lo, hi = np.percentile(v, [2.5, 97.5])
    ax.hist(v, bins=40, color=ACCENT, alpha=0.8, edgecolor="white", linewidth=0.3,
            label=label)
    ax.axvline(v.mean(), color="white", linewidth=1.5, linestyle="-", label=f"Mean = {v.mean():.1f} m³")
    ax.axvline(lo, color="crimson", linewidth=1.2, linestyle="--", label=f"95% CI: [{lo:.1f}, {hi:.1f}]")
    ax.axvline(hi, color="crimson", linewidth=1.2, linestyle="--")
    if true_vol is not None:
        ax.axvline(true_vol, color="lime", linewidth=1.5, linestyle="-.", label=f"True = {true_vol:.1f} m³")
    ax.set_xlabel("Volume (m³)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Convenience save
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save figure to path, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figure saved → {path}")
