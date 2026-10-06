"""
src/utils/geo.py
----------------
Coordinate reference system helpers for the Project Skylark pipeline.

Provides:
  - latlon_to_utm  : find the best UTM zone for a set of GPS points and reproject
  - pixel_to_geo   : convert raster pixel (row, col) → (lon, lat) using a rasterio transform
  - haversine      : great-circle distance between two GPS points (m)
  - bbox_from_points : bounding box (minx, miny, maxx, maxy) in WGS84

No GDAL/PROJ binary required — uses pyproj if available, falls back to a
simplified spherical model for haversine-only use cases.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    from pyproj import CRS, Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

EARTH_RADIUS_M = 6_371_000.0  # mean radius


# ---------------------------------------------------------------------------
# UTM zone helpers
# ---------------------------------------------------------------------------

def utm_zone_epsg(lat: float, lon: float) -> int:
    """
    Return the EPSG code for the UTM zone covering the given lat/lon.
    E.g., EPSG:32643 for UTM zone 43N (India).
    """
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone_number   # Northern hemisphere
    else:
        return 32700 + zone_number   # Southern hemisphere


def latlon_to_utm(
    lats: np.ndarray,
    lons: np.ndarray,
    epsg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Convert WGS84 lat/lon arrays to UTM easting/northing.

    Parameters
    ----------
    lats, lons : 1-D arrays of decimal degrees
    epsg       : Force a specific EPSG code; if None, auto-detect from centre

    Returns
    -------
    (eastings, northings, epsg_used)
    """
    if not HAS_PYPROJ:
        raise ImportError("pyproj is required for CRS reprojection. pip install pyproj")
    if epsg is None:
        epsg = utm_zone_epsg(float(np.nanmedian(lats)), float(np.nanmedian(lons)))
    wgs84 = CRS("EPSG:4326")
    utm = CRS(f"EPSG:{epsg}")
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)
    eastings, northings = transformer.transform(lons, lats)
    return np.array(eastings), np.array(northings), epsg


# ---------------------------------------------------------------------------
# Raster pixel ↔ geographic coordinate
# ---------------------------------------------------------------------------

def pixel_to_geo(row: int, col: int, transform) -> Tuple[float, float]:
    """
    Convert raster pixel position (row, col) to geographic coordinates.
    'transform' is a rasterio Affine transform or a 6-element tuple (GDAL order).

    Returns (x, y) which are (lon, lat) for geographic CRS or (easting, northing) for projected.
    """
    try:
        # rasterio Affine transform: (col, row) → (x, y)
        x, y = transform * (col + 0.5, row + 0.5)
    except TypeError:
        # GDAL GeoTransform: (x_origin, pixel_width, 0, y_origin, 0, -pixel_height)
        gt = transform
        x = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
        y = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]
    return x, y


def geo_to_pixel(x: float, y: float, transform) -> Tuple[int, int]:
    """
    Convert geographic coordinate (x, y) to raster pixel (row, col).
    Inverse of pixel_to_geo.
    """
    try:
        col_f, row_f = ~transform * (x, y)
    except Exception:
        gt = transform
        col_f = (x - gt[0]) / gt[1]
        row_f = (y - gt[3]) / gt[5]
    return int(row_f), int(col_f)


# ---------------------------------------------------------------------------
# Distance and geometry
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bbox_from_points(
    lats: np.ndarray, lons: np.ndarray
) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) bounding box."""
    return (
        float(np.nanmin(lons)),
        float(np.nanmin(lats)),
        float(np.nanmax(lons)),
        float(np.nanmax(lats)),
    )


def flight_extent_m(
    lats: np.ndarray, lons: np.ndarray
) -> Tuple[float, float]:
    """
    Approximate extent of the flight area in metres (width × height).
    Uses haversine on the bounding box corners.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_from_points(lats, lons)
    width_m = haversine(min_lat, min_lon, min_lat, max_lon)
    height_m = haversine(min_lat, min_lon, max_lat, min_lon)
    return width_m, height_m


# ---------------------------------------------------------------------------
# GSD calculation
# ---------------------------------------------------------------------------

def compute_gsd(
    altitude_m: float,
    focal_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
) -> float:
    """
    Ground Sampling Distance in cm/pixel.
    GSD = (sensor_width_mm / focal_mm) * altitude_m / image_width_px * 100
    """
    return (sensor_width_mm / focal_mm) * altitude_m / image_width_px * 100.0
