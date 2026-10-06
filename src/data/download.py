"""
data/download.py
----------------
Fetch OpenDroneMap benchmark datasets and community photogrammetry datasets.

Datasets targeted:
  - ODM Brighton Beach  (good coastal scene, varied textures)
  - ODM Aukerman        (agricultural, flat ground → easy baseline)
  - ODM Toledo          (larger urban scene, good for occlusion tests)
  - natowi CSV index    (filter for 'aerial' + 'construction'/'quarry')

Usage:
    python -m src.data.download --dataset aukerman --out data/raw/odm_samples
    python -m src.data.download --list          # show all available datasets
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    name: str
    url: str
    description: str
    n_images: int
    has_gcp: bool = False
    tags: list[str] = field(default_factory=list)

# Official ODM benchmark datasets — each dataset lives in its OWN GitHub repo.
# Zip URL format: https://github.com/<org>/<repo>/archive/refs/heads/master.zip
ODM_DATASETS: dict[str, Dataset] = {
    "aukerman": Dataset(
        name="aukerman",
        url="https://github.com/OpenDroneMap/odm_data_aukerman/archive/refs/heads/master.zip",
        description="Agricultural field, flat terrain — ideal baseline for volume pipeline",
        n_images=77,
        has_gcp=False,
        tags=["agriculture", "flat", "nadir"],
    ),
    "brighton": Dataset(
        name="brighton",
        url="https://github.com/pierotofy/drone_dataset_brighton_beach/archive/refs/heads/master.zip",
        description="Brighton Beach coastal scene — textured, varied elevation",
        n_images=18,
        has_gcp=False,
        tags=["coastal", "mixed-terrain", "nadir"],
    ),
    "banana": Dataset(
        name="banana",
        url="https://github.com/pierotofy/dataset_banana/archive/refs/heads/master.zip",
        description="Small 16-image dataset — fast test of ODM pipeline",
        n_images=16,
        has_gcp=False,
        tags=["small", "3d-model", "nadir"],
    ),
    "boruszyn": Dataset(
        name="boruszyn",
        url="https://github.com/merkato/odm_boruszyn_kap/archive/refs/heads/master.zip",
        description="Polish field with GCP file — tests GCP workflow",
        n_images=46,
        has_gcp=True,
        tags=["agriculture", "gcp", "kap"],
    ),
}

# natowi community CSV (350+ datasets)
NATOWI_CSV_URL = (
    "https://raw.githubusercontent.com/natowi/photogrammetry_datasets/master/datasets.csv"
)

NATOWI_STOCKPILE_TAGS = {"quarry", "stockpile", "construction", "mine", "earthwork"}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _stream_download(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file with a progress bar. Returns dest path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as fh,
        tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))
    return dest


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip archive, flattening the top-level folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        # Detect common top-level prefix to strip
        top_dirs = {m.split("/")[0] for m in members if "/" in m}
        prefix = top_dirs.pop() + "/" if len(top_dirs) == 1 else ""
        for member in tqdm(members, desc="Extracting"):
            target_name = member[len(prefix):] if prefix and member.startswith(prefix) else member
            if not target_name:
                continue
            target = out_dir / target_name
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())


# ---------------------------------------------------------------------------
# ODM dataset download
# ---------------------------------------------------------------------------

def download_odm_dataset(name: str, out_root: Path, force: bool = False) -> Path:
    """
    Download an ODM dataset by name. Returns the folder containing raw images.

    Each ODM dataset lives in its own GitHub repo.
    The zip unpacks to a single top-level folder (repo-name-master/).
    We strip that prefix and place everything under out_root/<name>/.
    """
    if name not in ODM_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(ODM_DATASETS)}")

    ds = ODM_DATASETS[name]
    out_dir = out_root / name

    # Detect images folder (ODM convention: images/ or image/)
    for candidate in ["images", "image", ""]:
        images_dir = out_dir / candidate if candidate else out_dir
        if images_dir.exists() and any(
            p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"}
        ):
            print(f"[skip] {name} already downloaded at {out_dir}")
            return out_dir

    # Each dataset has its own zip — use a per-dataset cache file
    zip_cache = out_root / "_cache" / f"{name}-master.zip"
    if not zip_cache.exists() or force:
        size_mb = ds.n_images * 7  # rough estimate
        print(f"Downloading {name} (~{size_mb} MB) …")
        _stream_download(ds.url, zip_cache, desc=f"{name}-master.zip")

    # The zip top-level folder is <repo-name>-master/
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting dataset '{name}' …")
    with zipfile.ZipFile(zip_cache) as zf:
        members = zf.namelist()
        # Find the single top-level directory prefix
        top_dirs = {m.split("/")[0] for m in members if "/" in m}
        prefix = (top_dirs.pop() + "/") if len(top_dirs) == 1 else ""

        for member in tqdm(members, desc=f"Extracting {name}"):
            rel = member[len(prefix):] if prefix and member.startswith(prefix) else member
            if not rel:
                continue
            target = out_dir / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    # Save dataset metadata sidecar
    meta = {
        "name": ds.name,
        "description": ds.description,
        "n_images_expected": ds.n_images,
        "has_gcp": ds.has_gcp,
        "tags": ds.tags,
        "source_url": ds.url,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] {name} → {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# natowi community CSV filter
# ---------------------------------------------------------------------------

def fetch_natowi_stockpile_links(out_root: Path) -> list[dict]:
    """
    Download the natowi CSV index and return rows relevant to stockpile/quarry.
    Saves a filtered CSV to out_root/natowi_stockpile.csv.
    """
    csv_cache = out_root / "_cache" / "natowi_datasets.csv"
    if not csv_cache.exists():
        _stream_download(NATOWI_CSV_URL, csv_cache, desc="natowi_datasets.csv")

    results = []
    with open(csv_cache, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Combine all text fields for tag matching
            combined = " ".join(row.values()).lower()
            if any(tag in combined for tag in NATOWI_STOCKPILE_TAGS):
                results.append(row)

    out_csv = out_root / "natowi_stockpile.csv"
    if results:
        keys = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"[ok] Found {len(results)} natowi datasets → {out_csv}")
    else:
        print("[warn] No natowi datasets matched stockpile tags")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_datasets():
    print("\nAvailable ODM datasets:")
    print(f"{'Name':<15} {'Images':>7}  {'GCP':>4}  Tags")
    print("-" * 60)
    for name, ds in ODM_DATASETS.items():
        gcp = "yes" if ds.has_gcp else "no"
        tags = ", ".join(ds.tags)
        print(f"{name:<15} {ds.n_images:>7}  {gcp:>4}  {tags}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Project Skylark")
    parser.add_argument("--dataset", choices=list(ODM_DATASETS) + ["all"],
                        help="Which ODM dataset to download (or 'all')")
    parser.add_argument("--natowi", action="store_true",
                        help="Filter natowi CSV for stockpile-relevant datasets")
    parser.add_argument("--out", default="data/raw/odm_samples",
                        help="Output directory (default: data/raw/odm_samples)")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="List available datasets and exit")
    args = parser.parse_args()

    if args.list_only:
        _list_datasets()
        sys.exit(0)

    out_root = Path(args.out)

    if args.natowi:
        fetch_natowi_stockpile_links(out_root)

    if args.dataset:
        names = list(ODM_DATASETS) if args.dataset == "all" else [args.dataset]
        for name in names:
            download_odm_dataset(name, out_root, force=args.force)


if __name__ == "__main__":
    main()
