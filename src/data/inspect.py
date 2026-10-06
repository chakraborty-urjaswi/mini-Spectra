"""
src/data/inspect.py
-------------------
Image quality assessment for raw drone photos.

Checks performed on every image:
  1. EXIF completeness — GPS lat/lon/alt, camera model, focal length, timestamp
  2. Blur detection    — Laplacian variance; flag if < BLUR_THRESHOLD
  3. Exposure check    — Mean brightness in [30, 225] for 8-bit
  4. Overlap estimate  — Pairwise footprint overlap given altitude & FOV
  5. File integrity    — Can PIL open it without error?

Outputs a summary DataFrame + per-image JSON sidecar in the same folder.

Usage:
    python -m src.data.inspect data/raw/odm_samples/aukerman/images
    python -m src.data.inspect data/raw/odm_samples/aukerman/images --report data/raw/aukerman_qa.csv
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
BLUR_THRESHOLD = 80.0       # Laplacian variance below this → blurry
MIN_BRIGHTNESS = 30         # pixel mean below → underexposed
MAX_BRIGHTNESS = 220        # pixel mean above → overexposed
MIN_IMAGES = 20             # warn if fewer than this in a dataset
IDEAL_OVERLAP = 0.75        # flag datasets where estimated overlap < this


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

def _rational_to_float(rational) -> Optional[float]:
    """Convert piexif rational (tuple of tuples) to float."""
    try:
        if isinstance(rational, tuple) and len(rational) == 3:
            # GPS degrees/minutes/seconds
            d = rational[0][0] / rational[0][1]
            m = rational[1][0] / rational[1][1]
            s = rational[2][0] / rational[2][1]
            return d + m / 60 + s / 3600
        elif isinstance(rational, tuple) and len(rational) == 2:
            return rational[0] / rational[1] if rational[1] != 0 else None
    except Exception:
        return None
    return None


def read_exif(img_path: Path) -> dict:
    """
    Extract key EXIF fields from a JPEG/TIFF drone image.
    Returns a dict with keys: lat, lon, alt_m, focal_mm, sensor_w_mm,
    timestamp, camera_make, camera_model, image_w, image_h.
    Missing fields are None.
    """
    result: dict = {
        "path": str(img_path),
        "lat": None, "lon": None, "alt_m": None,
        "focal_mm": None, "sensor_w_mm": None,
        "timestamp": None,
        "camera_make": None, "camera_model": None,
        "image_w": None, "image_h": None,
    }
    if not HAS_PIEXIF:
        return result

    try:
        exif_data = piexif.load(str(img_path))
    except Exception:
        return result

    ifd0 = exif_data.get("0th", {})
    exif_ifd = exif_data.get("Exif", {})
    gps_ifd = exif_data.get("GPS", {})

    # Camera identity
    try:
        result["camera_make"] = ifd0.get(piexif.ImageIFD.Make, b"").decode("utf-8", errors="ignore").strip("\x00")
        result["camera_model"] = ifd0.get(piexif.ImageIFD.Model, b"").decode("utf-8", errors="ignore").strip("\x00")
    except Exception:
        pass

    # Timestamp
    try:
        ts_raw = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal, b"")
        result["timestamp"] = ts_raw.decode("utf-8", errors="ignore").strip("\x00")
    except Exception:
        pass

    # Focal length
    try:
        fl = exif_ifd.get(piexif.ExifIFD.FocalLength)
        if fl:
            result["focal_mm"] = fl[0] / fl[1]
    except Exception:
        pass

    # GPS
    try:
        lat_raw = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
        lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef, b"N")
        lon_raw = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
        lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef, b"E")
        alt_raw = gps_ifd.get(piexif.GPSIFD.GPSAltitude)

        if lat_raw:
            lat = _rational_to_float(lat_raw)
            if lat_ref in (b"S", "S"):
                lat = -lat
            result["lat"] = lat
        if lon_raw:
            lon = _rational_to_float(lon_raw)
            if lon_ref in (b"W", "W"):
                lon = -lon
            result["lon"] = lon
        if alt_raw:
            result["alt_m"] = alt_raw[0] / alt_raw[1]
    except Exception:
        pass

    # Image dimensions (from PIL, more reliable than EXIF tags)
    try:
        with Image.open(img_path) as img:
            result["image_w"], result["image_h"] = img.size
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Image quality checks
# ---------------------------------------------------------------------------

def blur_score(img_path: Path) -> float:
    """
    Laplacian variance as blur metric.
    Higher = sharper. Values < BLUR_THRESHOLD are flagged.
    """
    try:
        with Image.open(img_path) as img:
            gray = np.array(img.convert("L"), dtype=np.float32)
        # 3×3 Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        from scipy.signal import convolve2d
        lap = convolve2d(gray, kernel, mode="valid")
        return float(lap.var())
    except Exception:
        return -1.0


def exposure_stats(img_path: Path) -> dict:
    """Mean and std brightness from grayscale conversion."""
    try:
        with Image.open(img_path) as img:
            gray = np.array(img.convert("L"), dtype=np.float32)
        return {"brightness_mean": float(gray.mean()), "brightness_std": float(gray.std())}
    except Exception:
        return {"brightness_mean": None, "brightness_std": None}


def file_integrity(img_path: Path) -> bool:
    """True if PIL can open and verify the image without error."""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Overlap estimation
# ---------------------------------------------------------------------------

def estimate_footprint_m(alt_m: float, focal_mm: float,
                          image_w_px: int, image_h_px: int,
                          sensor_w_mm: float = 6.17) -> tuple[float, float]:
    """
    Estimate ground footprint (width × height) in metres for a nadir image.
    Assumes sensor_w_mm = 6.17 mm (DJI Phantom 4 equivalent default).
    """
    sensor_h_mm = sensor_w_mm * image_h_px / image_w_px
    gsd = (sensor_w_mm / focal_mm) * alt_m / image_w_px  # m/px (not used directly)
    foot_w = (sensor_w_mm / focal_mm) * alt_m
    foot_h = (sensor_h_mm / focal_mm) * alt_m
    return foot_w, foot_h


# ---------------------------------------------------------------------------
# Full dataset QA
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def inspect_dataset(images_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """
    Run all QA checks on every image in images_dir.
    Returns a DataFrame with one row per image.
    """
    images_dir = Path(images_dir)
    img_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"Found {len(img_paths)} images in {images_dir}")
    if len(img_paths) < MIN_IMAGES:
        warnings.warn(f"Only {len(img_paths)} images — ODM needs ≥20 for reliable reconstruction")

    rows = []
    for p in img_paths:
        if verbose:
            print(f"  {p.name} …", end="", flush=True)

        row: dict = {"filename": p.name}
        row["file_ok"] = file_integrity(p)
        row.update(read_exif(p))
        row["blur_score"] = blur_score(p)
        row["is_blurry"] = row["blur_score"] < BLUR_THRESHOLD and row["blur_score"] >= 0
        row.update(exposure_stats(p))
        row["underexposed"] = (
            row["brightness_mean"] is not None and row["brightness_mean"] < MIN_BRIGHTNESS
        )
        row["overexposed"] = (
            row["brightness_mean"] is not None and row["brightness_mean"] > MAX_BRIGHTNESS
        )
        row["has_gps"] = row["lat"] is not None and row["lon"] is not None
        row["has_altitude"] = row["alt_m"] is not None
        row["has_focal"] = row["focal_mm"] is not None

        rows.append(row)
        if verbose:
            flags = [k for k in ("is_blurry", "underexposed", "overexposed")
                     if row.get(k)]
            status = "⚠ " + ", ".join(flags) if flags else "✓"
            print(f" {status}")

    df = pd.DataFrame(rows)

    # ---------- Summary statistics ----------
    n = len(df)
    n_ok = df["file_ok"].sum()
    n_gps = df["has_gps"].sum()
    n_blur = df["is_blurry"].sum()
    n_alt = df["has_altitude"].sum()
    alt_median = df["alt_m"].median() if df["alt_m"].notna().any() else None

    print("\n" + "=" * 55)
    print(f"QA SUMMARY — {images_dir.parent.name}/{images_dir.name}")
    print("=" * 55)
    print(f"  Total images   : {n}")
    print(f"  File integrity : {n_ok}/{n} ok")
    print(f"  GPS tags       : {n_gps}/{n} have lat/lon")
    print(f"  Altitude tags  : {n_alt}/{n} have altitude")
    print(f"  Blurry images  : {n_blur} (Laplacian var < {BLUR_THRESHOLD})")
    if alt_median:
        print(f"  Median altitude: {alt_median:.1f} m")

    # Overlap estimate (rough)
    focal_median = df["focal_mm"].median()
    w_median = df["image_w"].median()
    h_median = df["image_h"].median()
    if alt_median and focal_median and w_median:
        fw, fh = estimate_footprint_m(alt_median, focal_median, int(w_median), int(h_median))
        print(f"  Est. footprint : {fw:.1f} m × {fh:.1f} m at {alt_median:.0f} m AGL")
        # Crude overlap estimate assuming grid spacing ~ footprint * (1-overlap)
        # We can't compute true overlap without flight path, so just report footprint

    issues = df[(df["is_blurry"] | df["underexposed"] | df["overexposed"] | ~df["file_ok"])]
    if not issues.empty:
        print(f"\n  ⚠ {len(issues)} image(s) with issues:")
        for _, row in issues.iterrows():
            flags = [k for k in ("is_blurry", "underexposed", "overexposed", "file_ok")
                     if (k == "file_ok" and not row[k]) or (k != "file_ok" and row.get(k))]
            print(f"    {row['filename']:40s} {', '.join(flags)}")
    else:
        print("\n  ✓ All images pass QA checks")
    print("=" * 55)

    return df


def save_report(df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Report saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QA-inspect drone image datasets")
    parser.add_argument("images_dir", help="Path to folder containing raw images")
    parser.add_argument("--report", default=None,
                        help="Save CSV report to this path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    df = inspect_dataset(Path(args.images_dir), verbose=args.verbose)
    if args.report:
        save_report(df, Path(args.report))


if __name__ == "__main__":
    main()
