"""Gallery construction for blind retrieval evaluation (SPEC §6; suppl. D.1).

Per region (city) with sizable training data: take the bounding box of the
*train* coordinates with outliers dropped (percentile clip, GUESSES.md #26),
pad it by a constant margin, and cover it with a uniform grid whose step is
the chosen resolution in km on both axes (lat step constant; lon step scaled
by cos(latitude) at the region center). Resolution: 0.1 km (MSLS), 0.5 km
(GAMa). Built from the train split only — the model stays blind to val
coordinates.

CLI: python -m vidtag.data.gallery --coords-csv coords.csv \
        --resolution-km 0.1 --out gallery.npy
where the CSV has lat, lon, region columns.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQUATOR = 111.320


def _inclusive_arange(lo: float, hi: float, step: float) -> np.ndarray:
    """np.arange over [lo, hi] inclusive of both edges: append hi when the
    open-ended arange stops short of it."""
    vals = np.arange(lo, hi, step)
    if vals.size == 0 or vals[-1] < hi - 1e-12:
        vals = np.append(vals, hi)
    return vals


def build_region_grid(
    coords: np.ndarray,
    resolution_km: float,
    padding_deg: float = 0.02,
    outlier_pct: float = 0.5,
) -> np.ndarray:
    """coords: (N, 2) degrees (lat, lon) of one region's train coordinates ->
    (G, 2) uniform gallery grid over the outlier-clipped, padded bbox.

    Outliers are dropped by clipping each axis to its
    [outlier_pct, 100 - outlier_pct] percentiles before padding (suppl. D.1
    leaves both knobs unquantified; defaults per GUESSES.md #26).
    """
    coords = np.asarray(coords, dtype=np.float64)
    lat_lo, lat_hi = np.percentile(coords[:, 0], [outlier_pct, 100 - outlier_pct])
    lon_lo, lon_hi = np.percentile(coords[:, 1], [outlier_pct, 100 - outlier_pct])
    lat_lo, lat_hi = lat_lo - padding_deg, lat_hi + padding_deg
    lon_lo, lon_hi = lon_lo - padding_deg, lon_hi + padding_deg

    center_lat = 0.5 * (lat_lo + lat_hi)
    dlat = resolution_km / KM_PER_DEG_LAT
    dlon = resolution_km / (KM_PER_DEG_LON_EQUATOR * np.cos(np.radians(center_lat)))
    lats = _inclusive_arange(lat_lo, lat_hi, dlat)
    lons = _inclusive_arange(lon_lo, lon_hi, dlon)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    return np.stack((grid_lat.ravel(), grid_lon.ravel()), axis=1)


def build_uniform_grid(
    coords_by_region: dict[str, np.ndarray],
    resolution_km: float,
    **kw,
) -> np.ndarray:
    """Concatenate per-region grids and dedup exact duplicate rows -> (G, 2).

    kw is forwarded to build_region_grid (padding_deg, outlier_pct).
    """
    grids = [
        build_region_grid(coords, resolution_km, **kw)
        for coords in coords_by_region.values()
    ]
    return np.unique(np.concatenate(grids, axis=0), axis=0)


def save_gallery(path: str, grid: np.ndarray) -> None:
    np.save(path, np.asarray(grid, dtype=np.float64))


def load_gallery(path: str) -> np.ndarray:
    return np.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a retrieval gallery grid from train coordinates (SPEC §6)."
    )
    parser.add_argument(
        "--coords-csv", required=True, help="CSV with lat, lon, region columns"
    )
    parser.add_argument(
        "--resolution-km",
        type=float,
        required=True,
        help="grid step in km (paper: 0.1 for MSLS, 0.5 for GAMa)",
    )
    parser.add_argument("--out", required=True, help="output .npy path")
    parser.add_argument("--padding-deg", type=float, default=0.02)
    parser.add_argument("--outlier-pct", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.coords_csv)
    coords_by_region = {
        str(region): group[["lat", "lon"]].to_numpy(dtype=np.float64)
        for region, group in df.groupby("region")
    }
    grid = build_uniform_grid(
        coords_by_region,
        args.resolution_km,
        padding_deg=args.padding_deg,
        outlier_pct=args.outlier_pct,
    )
    save_gallery(args.out, grid)
    print(
        f"{grid.shape[0]} gallery points "
        f"({len(coords_by_region)} regions, {args.resolution_km} km) -> {args.out}"
    )


if __name__ == "__main__":
    main()
