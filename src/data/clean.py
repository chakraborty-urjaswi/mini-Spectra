"""
src/data/clean.py
-----------------
Image cleaning pipeline for raw drone photos before ODM processing.

Steps performed:
  1. Reject corrupt files (PIL verify fails)
  2. Remove blurry images (Laplacian variance < threshold)
  3. Remove extreme-exposure images (underexposed or overexposed)
  4. Detect near-duplicate frames (cosine similarity of downsampled thumbnails)
  5. Flag images with missing GPS (ODM can still use them, but warn)
  6. Produce a cleaned/ sub-folder with symlinks (not copies) to passing images
  7. Write a cleaning_report.json with kept/rejected counts and reasons

Usage:
    python -m src.data.clean data/raw/odm_samples/aukerman/images
    python -m src.data.clean data/raw/odm_samples/aukerman/images \\
        --blur-thresh 60 --duplicate-thresh 0.98 --out data/raw/aukerman_clean
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Reuse QA helpers from inspect
from src.data.inspect import (
    IMAGE_EXTENSIONS,
    BLUR_THRESHOLD,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    blur_score,
    exposure_stats,
    file_integrity,
    read_exif,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
THUMBNAIL_SIZE = (64, 64)          # For duplicate detection
DUPLICATE_COSINE_THRESH = 0.995    # Cosine similarity ≥ this → near-duplicate


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _thumbnail_vector(img_path: Path, size=THUMBNAIL_SIZE) -> Optional[np.ndarray]:
    """Flatten a grayscale thumbnail to a normalised 1-D vector."""
    try:
        with Image.open(img_path) as img:
            thumb = img.convert("L").resize(size, Image.LANCZOS)
        vec = np.array(thumb, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    except Exception:
        return None


def find_duplicates(img_paths: list[Path], threshold: float = DUPLICATE_COSINE_THRESH
                    ) -> list[tuple[Path, Path, float]]:
    """
    Return list of (img_a, img_b, similarity) pairs that are near-duplicates.
    O(n²) — acceptable for typical dataset sizes (< 500 images).
    """
    vectors = {p: _thumbnail_vector(p) for p in img_paths}
    duplicates = []
    paths = [p for p, v in vectors.items() if v is not None]
    for i, pa in enumerate(paths):
        for pb in paths[i + 1:]:
            sim = float(np.dot(vectors[pa], vectors[pb]))
            if sim >= threshold:
                duplicates.append((pa, pb, sim))
    return duplicates


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def clean_dataset(
    images_dir: Path,
    out_dir: Optional[Path] = None,
    blur_thresh: float = BLUR_THRESHOLD,
    dup_thresh: float = DUPLICATE_COSINE_THRESH,
    copy_files: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Run full cleaning pipeline.

    Parameters
    ----------
    images_dir  : Folder containing raw images
    out_dir     : Where to put cleaned images (default: images_dir/../cleaned)
    blur_thresh : Laplacian variance threshold; below = blurry
    dup_thresh  : Cosine similarity threshold for duplicates
    copy_files  : If True, copy files; otherwise symlink (saves disk space)
    dry_run     : Report what would be done without actually doing it

    Returns
    -------
    dict with summary statistics and per-image decisions
    """
    images_dir = Path(images_dir)
    if out_dir is None:
        out_dir = images_dir.parent / "cleaned"

    img_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"\n{'='*55}")
    print(f"CLEANING PIPELINE — {len(img_paths)} images")
    print(f"{'='*55}")

    decisions: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Pass 1: Per-image checks
    # ------------------------------------------------------------------
    print("Pass 1/3: Per-image QA checks …")
    for p in img_paths:
        d: dict = {"path": str(p), "keep": True, "reasons": []}

        if not file_integrity(p):
            d["keep"] = False
            d["reasons"].append("corrupt_file")
            decisions[p.name] = d
            continue

        bs = blur_score(p)
        d["blur_score"] = bs
        if bs >= 0 and bs < blur_thresh:
            d["keep"] = False
            d["reasons"].append(f"blurry (score={bs:.1f} < {blur_thresh})")

        exp = exposure_stats(p)
        d["brightness_mean"] = exp["brightness_mean"]
        bm = exp["brightness_mean"]
        if bm is not None:
            if bm < MIN_BRIGHTNESS:
                d["keep"] = False
                d["reasons"].append(f"underexposed (mean={bm:.1f})")
            elif bm > MAX_BRIGHTNESS:
                d["keep"] = False
                d["reasons"].append(f"overexposed (mean={bm:.1f})")

        exif = read_exif(p)
        d["has_gps"] = exif["lat"] is not None
        if not d["has_gps"]:
            d["reasons"].append("missing_gps (warning only)")  # don't reject

        decisions[p.name] = d

    n_pass1 = sum(1 for d in decisions.values() if d["keep"])
    n_fail1 = len(decisions) - n_pass1
    print(f"  {n_pass1} pass, {n_fail1} failed (blur/exposure/corrupt)")

    # ------------------------------------------------------------------
    # Pass 2: Duplicate detection (on images that passed pass 1)
    # ------------------------------------------------------------------
    print("Pass 2/3: Duplicate detection …")
    good_paths = [img_paths[i] for i, p in enumerate(img_paths)
                  if decisions[p.name]["keep"]]
    dups = find_duplicates(good_paths, threshold=dup_thresh)

    # For each duplicate pair, keep the first, reject the second
    rejected_as_dup = set()
    for pa, pb, sim in dups:
        if pb.name not in rejected_as_dup:
            decisions[pb.name]["keep"] = False
            decisions[pb.name]["reasons"].append(
                f"near_duplicate of {pa.name} (sim={sim:.4f})"
            )
            rejected_as_dup.add(pb.name)

    n_dup = len(rejected_as_dup)
    print(f"  {n_dup} near-duplicates removed (cosine ≥ {dup_thresh})")

    # ------------------------------------------------------------------
    # Pass 3: Write output
    # ------------------------------------------------------------------
    kept = [p for p in img_paths if decisions[p.name]["keep"]]
    rejected = [p for p in img_paths if not decisions[p.name]["keep"]]

    print(f"\nPass 3/3: Output → {out_dir}")
    print(f"  Kept   : {len(kept)}")
    print(f"  Removed: {len(rejected)}")

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in kept:
            dst = out_dir / p.name
            if dst.exists():
                dst.unlink()
            if copy_files:
                shutil.copy2(p, dst)
            else:
                # Symlink using absolute path
                dst.symlink_to(p.resolve())

        # Write rejection log
        reject_log = out_dir / "rejected_images.json"
        reject_info = {
            p.name: decisions[p.name]["reasons"]
            for p in rejected
        }
        reject_log.write_text(json.dumps(reject_info, indent=2))

        # Write full cleaning report
        report = {
            "source_dir": str(images_dir),
            "out_dir": str(out_dir),
            "total_input": len(img_paths),
            "total_kept": len(kept),
            "total_rejected": len(rejected),
            "rejected_corrupt": n_fail1,
            "rejected_duplicates": n_dup,
            "blur_threshold": blur_thresh,
            "duplicate_threshold": dup_thresh,
            "per_image": decisions,
        }
        (out_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))
        print(f"  Reports written to {out_dir}")
    else:
        print("  [DRY RUN — no files written]")
        report = {"dry_run": True, "would_keep": len(kept), "would_reject": len(rejected)}

    return report


# ---------------------------------------------------------------------------
# Cleaning report summary printer
# ---------------------------------------------------------------------------

def print_cleaning_report(report_path: Path) -> None:
    """Pretty-print an existing cleaning_report.json."""
    report = json.loads(Path(report_path).read_text())
    print(f"\n{'='*55}")
    print(f"CLEANING REPORT — {Path(report['source_dir']).name}")
    print(f"{'='*55}")
    print(f"  Input     : {report['total_input']} images")
    print(f"  Kept      : {report['total_kept']} images")
    print(f"  Rejected  : {report['total_rejected']}")
    print(f"    Corrupt/blur/exposure : {report['rejected_corrupt']}")
    print(f"    Near-duplicates       : {report['rejected_duplicates']}")
    print(f"  Blur threshold          : {report['blur_threshold']}")
    print(f"  Duplicate threshold     : {report['duplicate_threshold']}")

    per_image = report.get("per_image", {})
    rejected = {k: v for k, v in per_image.items() if not v.get("keep", True)}
    if rejected:
        print(f"\n  Rejected images:")
        for fname, info in rejected.items():
            reasons = "; ".join(info.get("reasons", []))
            print(f"    {fname:40s} {reasons}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean drone image dataset for ODM")
    parser.add_argument("images_dir", help="Path to raw image folder")
    parser.add_argument("--out", default=None,
                        help="Output folder (default: images_dir/../cleaned)")
    parser.add_argument("--blur-thresh", type=float, default=BLUR_THRESHOLD,
                        help=f"Laplacian variance threshold (default {BLUR_THRESHOLD})")
    parser.add_argument("--duplicate-thresh", type=float, default=DUPLICATE_COSINE_THRESH,
                        help=f"Cosine similarity threshold (default {DUPLICATE_COSINE_THRESH})")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be done, don't write files")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else None
    clean_dataset(
        images_dir=Path(args.images_dir),
        out_dir=out_dir,
        blur_thresh=args.blur_thresh,
        dup_thresh=args.duplicate_thresh,
        copy_files=args.copy,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
