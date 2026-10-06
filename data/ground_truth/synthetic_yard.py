"""
data/ground_truth/synthetic_yard.py
-------------------------------------
Generate synthetic DSM (Digital Surface Model) GeoTIFFs with known-volume
stockpile piles on controlled ground surfaces.

Purpose
-------
Every other module in the pipeline depends on having ground truth volumes
to validate against. This generator creates synthetic yards instantly,
without needing Blender or a real drone, by:

  1. Creating a configurable ground surface (flat, tilted, bumpy, uneven pad)
  2. Placing conical / ellipsoidal / irregular piles of KNOWN volume on it
  3. Optionally adding a doming artefact (see Module D)
  4. Saving the result as a GeoTIFF with a realistic UTM coordinate system

The exact volume formula for each pile shape is used to record ground truth,
so every experiment has a reference to compare against.

Usage
-----
    python data/ground_truth/synthetic_yard.py --out data/ground_truth \\
        --piles 3 --ground bumpy --dome 0.05

    python data/ground_truth/synthetic_yard.py --demo   # quick validation plot
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NODATA = -9999.0

@dataclass
class PileConfig:
    """Configuration for a single synthetic pile."""
    shape: Literal["cone", "ellipsoid", "truncated_cone"] = "cone"
    centre_x: float = 50.0          # metres from left edge of raster
    centre_y: float = 50.0          # metres from bottom edge of raster
    radius_x: float = 10.0          # metres
    radius_y: float = 10.0          # metres (allows elliptical footprint)
    height: float = 5.0             # metres above local ground
    random_seed: Optional[int] = None  # for irregular piles

    @property
    def true_volume_m3(self) -> float:
        """Analytic volume of the pile shape (ignores ground curvature)."""
        if self.shape == "cone":
            return (1 / 3) * math.pi * self.radius_x * self.radius_y * self.height
        elif self.shape == "ellipsoid":
            # Half-ellipsoid above ground
            return (2 / 3) * math.pi * self.radius_x * self.radius_y * self.height
        elif self.shape == "truncated_cone":
            r1 = min(self.radius_x, self.radius_y)
            r2 = r1 * 0.3
            return (1 / 3) * math.pi * self.height * (r1**2 + r1 * r2 + r2**2)
        return 0.0


@dataclass
class YardConfig:
    """Configuration for a synthetic yard DSM."""
    # Raster dimensions
    width_m: float = 200.0          # metres east-west
    height_m: float = 200.0         # metres north-south
    resolution_m: float = 0.1       # metres per pixel (10 cm = typical drone output)
    # Ground surface type
    ground_type: Literal["flat", "tilted", "bumpy", "uneven_pad"] = "flat"
    ground_base_elev: float = 10.0  # base elevation in metres
    ground_tilt_deg: float = 2.0    # slope in degrees (for "tilted")
    ground_roughness: float = 0.15  # std of random noise (for "bumpy")
    # Piles
    piles: list[PileConfig] = field(default_factory=list)
    # Doming artefact (Module D)
    dome_amplitude: float = 0.0     # metres; positive = bowl-up, negative = bowl-down
    # Coordinate system
    origin_easting: float = 400_000.0   # UTM easting of top-left corner
    origin_northing: float = 3_300_000.0  # UTM northing of top-left corner
    epsg: int = 32643               # WGS84 / UTM zone 43N (central India)
    # Noise on surface model (simulates reconstruction noise)
    reconstruction_noise_std: float = 0.0  # metres; 0 = perfect DSM


# ---------------------------------------------------------------------------
# Ground surface generators
# ---------------------------------------------------------------------------

def _make_ground(cfg: YardConfig, rows: int, cols: int) -> np.ndarray:
    """Generate a 2-D ground elevation array (before piles)."""
    y_idx = np.arange(rows)  # row 0 = north
    x_idx = np.arange(cols)
    xx, yy = np.meshgrid(x_idx, y_idx)

    x_m = xx * cfg.resolution_m
    y_m = (rows - yy) * cfg.resolution_m  # invert so y increases northward

    ground = np.full((rows, cols), cfg.ground_base_elev, dtype=np.float64)

    if cfg.ground_type == "flat":
        pass

    elif cfg.ground_type == "tilted":
        slope = math.tan(math.radians(cfg.ground_tilt_deg))
        ground += x_m * slope  # slope in x direction

    elif cfg.ground_type == "bumpy":
        rng = np.random.default_rng(42)
        # Low-frequency random bumps (simulate uneven ground pad)
        bump_grid_rows = max(3, rows // 20)
        bump_grid_cols = max(3, cols // 20)
        raw = rng.normal(0, cfg.ground_roughness, (bump_grid_rows, bump_grid_cols))
        from scipy.ndimage import zoom
        scale_r = rows / bump_grid_rows
        scale_c = cols / bump_grid_cols
        bumps = zoom(raw, (scale_r, scale_c), order=3)[:rows, :cols]
        ground += bumps

    elif cfg.ground_type == "uneven_pad":
        rng = np.random.default_rng(7)
        # Coarser bumps + a flat ramp
        slope = math.tan(math.radians(cfg.ground_tilt_deg / 2))
        ground += x_m * slope
        raw = rng.normal(0, cfg.ground_roughness * 2, (rows // 30 + 3, cols // 30 + 3))
        from scipy.ndimage import zoom
        bumps = zoom(raw, (rows / raw.shape[0], cols / raw.shape[1]), order=3)[:rows, :cols]
        ground += bumps

    return ground


# ---------------------------------------------------------------------------
# Pile generators
# ---------------------------------------------------------------------------

def _add_pile(surface: np.ndarray, ground: np.ndarray,
              pile: PileConfig, cfg: YardConfig,
              rows: int, cols: int) -> None:
    """Add a single pile (in-place) on top of the ground surface."""
    cx_px = pile.centre_x / cfg.resolution_m
    cy_px = (cfg.height_m - pile.centre_y) / cfg.resolution_m  # row 0 = north
    rx_px = pile.radius_x / cfg.resolution_m
    ry_px = pile.radius_y / cfg.resolution_m

    row_idx = np.arange(rows)
    col_idx = np.arange(cols)
    cc, rr = np.meshgrid(col_idx, row_idx)

    # Normalised radial distance from pile centre
    r_norm = np.sqrt(((cc - cx_px) / rx_px) ** 2 + ((rr - cy_px) / ry_px) ** 2)
    inside = r_norm <= 1.0

    local_ground = ground[inside]

    if pile.shape == "cone":
        z = pile.height * (1 - r_norm[inside])
    elif pile.shape == "ellipsoid":
        z = pile.height * np.sqrt(np.maximum(0.0, 1 - r_norm[inside] ** 2))
    elif pile.shape == "truncated_cone":
        z = pile.height * (1 - r_norm[inside] * 0.7)
    else:
        z = np.zeros_like(r_norm[inside])

    # Add to local ground height
    surface[inside] = np.maximum(surface[inside], local_ground + z)


# ---------------------------------------------------------------------------
# Doming artefact
# ---------------------------------------------------------------------------

def _add_dome(surface: np.ndarray, amplitude: float, rows: int, cols: int) -> None:
    """
    Add a smooth bowl / dome distortion to simulate the SfM doming artefact.
    amplitude > 0 → surface bows upward at centre (dome)
    amplitude < 0 → surface bows downward at centre (bowl)
    """
    cx, cy = cols / 2, rows / 2
    r_max = math.sqrt(cx ** 2 + cy ** 2)
    row_idx = np.arange(rows)
    col_idx = np.arange(cols)
    cc, rr = np.meshgrid(col_idx, row_idx)
    r_norm = np.sqrt((cc - cx) ** 2 + (rr - cy) ** 2) / r_max
    dome = amplitude * (1 - r_norm ** 2)
    surface += dome


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_yard(cfg: YardConfig, out_dir: Path, name: str = "synthetic_yard") -> dict:
    """
    Generate a synthetic DSM and save as GeoTIFF + metadata JSON.

    Returns a dict with:
      'dsm_path'   : path to GeoTIFF
      'meta_path'  : path to JSON sidecar
      'pile_vols'  : list of true pile volumes (m³)
      'total_vol'  : sum of pile volumes
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = int(cfg.height_m / cfg.resolution_m)
    cols = int(cfg.width_m / cfg.resolution_m)

    # 1. Ground surface
    print(f"Generating ground ({rows}×{cols} px, {cfg.ground_type}) …")
    ground = _make_ground(cfg, rows, cols)

    # 2. Surface starts as ground
    surface = ground.copy()

    # 3. Add piles
    pile_vols = []
    for i, pile in enumerate(cfg.piles):
        print(f"  Adding pile {i+1}/{len(cfg.piles)} ({pile.shape}, "
              f"r={pile.radius_x}m, h={pile.height}m) …")
        _add_pile(surface, ground, pile, cfg, rows, cols)
        pile_vols.append(pile.true_volume_m3)

    # 4. Doming artefact
    if abs(cfg.dome_amplitude) > 0:
        print(f"  Applying doming artefact ({cfg.dome_amplitude:+.3f} m) …")
        _add_dome(surface, cfg.dome_amplitude, rows, cols)

    # 5. Reconstruction noise
    if cfg.reconstruction_noise_std > 0:
        rng = np.random.default_rng(99)
        surface += rng.normal(0, cfg.reconstruction_noise_std, surface.shape)

    # 6. Save GeoTIFF
    dsm_path = out_dir / f"{name}_dsm.tif"
    if HAS_RASTERIO:
        transform = from_origin(
            west=cfg.origin_easting,
            north=cfg.origin_northing,
            xsize=cfg.resolution_m,
            ysize=cfg.resolution_m,
        )
        with rasterio.open(
            dsm_path, "w",
            driver="GTiff",
            height=rows, width=cols,
            count=1, dtype=np.float32,
            crs=f"EPSG:{cfg.epsg}",
            transform=transform,
            nodata=NODATA,
        ) as dst:
            dst.write(surface.astype(np.float32), 1)
        print(f"DSM saved → {dsm_path}")
    else:
        # Fallback: save as numpy array
        dsm_path = out_dir / f"{name}_dsm.npy"
        np.save(dsm_path, surface.astype(np.float32))
        print(f"[warn] rasterio not available — saved raw array → {dsm_path}")

    # 7. Save metadata
    meta = {
        "name": name,
        "config": {
            **{k: v for k, v in asdict(cfg).items() if k != "piles"},
            "piles": [asdict(p) for p in cfg.piles],
        },
        "raster": {"rows": rows, "cols": cols, "resolution_m": cfg.resolution_m},
        "ground_truth": {
            "pile_volumes_m3": pile_vols,
            "total_volume_m3": sum(pile_vols),
        },
    }
    meta_path = out_dir / f"{name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nGround truth volumes:")
    for i, v in enumerate(pile_vols):
        print(f"  Pile {i+1}: {v:,.2f} m³")
    print(f"  Total : {sum(pile_vols):,.2f} m³")

    return {
        "dsm_path": dsm_path,
        "meta_path": meta_path,
        "ground": ground,
        "surface": surface,
        "pile_vols": pile_vols,
        "total_vol": sum(pile_vols),
        "cfg": cfg,
    }


# ---------------------------------------------------------------------------
# Preset yard configurations
# ---------------------------------------------------------------------------

def make_validation_yard() -> YardConfig:
    """
    A simple yard for pipeline validation.
    3 cones of known volume on flat ground.
    Easy to compute the exact answer by hand.
    """
    return YardConfig(
        width_m=200, height_m=200, resolution_m=0.1,
        ground_type="flat",
        piles=[
            PileConfig("cone",   50,  50, 12, 12, 6),   # V = π/3 × 12² × 6 = 904.8 m³
            PileConfig("cone",  130, 100, 10, 10, 5),   # V = π/3 × 10² × 5 = 523.6 m³
            PileConfig("ellipsoid", 70, 150, 15, 8, 4), # V = 2π/3 × 15 × 8 × 4 = 1005.3 m³
        ]
    )


def make_hard_yard() -> YardConfig:
    """
    A hard yard that tests Module A.
    Mixed shapes on uneven ground, piles close together.
    """
    return YardConfig(
        width_m=300, height_m=300, resolution_m=0.1,
        ground_type="uneven_pad",
        ground_tilt_deg=3.0,
        ground_roughness=0.20,
        piles=[
            PileConfig("cone",          80, 80,  20, 20, 8),
            PileConfig("truncated_cone", 100, 100, 15, 12, 7),  # overlapping!
            PileConfig("ellipsoid",     200, 150, 18, 10, 5),
            PileConfig("cone",          220, 200, 8,  8,  3),
        ],
        reconstruction_noise_std=0.05,
    )


def make_doming_yard(dome_amplitude: float = 0.3) -> YardConfig:
    """
    Flat yard with doming artefact — for Module D (flight design study).
    """
    return YardConfig(
        width_m=400, height_m=400, resolution_m=0.1,
        ground_type="flat",
        dome_amplitude=dome_amplitude,
        piles=[
            PileConfig("cone", 200, 200, 25, 25, 10),
        ],
    )


# ---------------------------------------------------------------------------
# Quick demo / validation
# ---------------------------------------------------------------------------

def demo():
    """Generate and visualise the validation yard."""
    cfg = make_validation_yard()
    result = generate_yard(cfg, Path("data/ground_truth"), name="validation_yard")

    surface = result["surface"]
    ground = result["ground"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0F1117")
    for ax in axes:
        ax.set_facecolor("#1A1D2E")

    im0 = axes[0].imshow(surface, cmap="magma", origin="upper")
    axes[0].set_title("Surface DSM", color="white")
    plt.colorbar(im0, ax=axes[0], label="Elevation (m)")

    im1 = axes[1].imshow(surface - ground, cmap="hot", origin="upper")
    axes[1].set_title("Height above ground (pile only)", color="white")
    plt.colorbar(im1, ax=axes[1], label="Height (m)")

    for ax in axes:
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#2A2D3E")

    plt.tight_layout()
    fig.savefig("data/ground_truth/validation_yard_preview.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("\nDemo saved → data/ground_truth/validation_yard_preview.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic stockpile yard DSMs")
    parser.add_argument("--out", default="data/ground_truth",
                        help="Output directory")
    parser.add_argument("--preset", choices=["validation", "hard", "doming"],
                        default="validation", help="Yard preset")
    parser.add_argument("--name", default=None, help="Output file name prefix")
    parser.add_argument("--dome", type=float, default=0.0,
                        help="Doming amplitude (m) for 'doming' preset")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo mode and exit")
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if args.preset == "validation":
        cfg = make_validation_yard()
    elif args.preset == "hard":
        cfg = make_hard_yard()
    elif args.preset == "doming":
        cfg = make_doming_yard(args.dome)
    else:
        cfg = make_validation_yard()

    name = args.name or args.preset
    generate_yard(cfg, Path(args.out), name=name)


if __name__ == "__main__":
    main()
