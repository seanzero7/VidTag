"""Tests for gallery grid construction (SPEC §6; suppl. D.1)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from vidtag.data.gallery import (
    build_region_grid,
    build_uniform_grid,
    load_gallery,
    save_gallery,
)

EARTH_RADIUS_KM = 6371.0088
RESOLUTION_KM = 0.1
PADDING_DEG = 0.02


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def make_coords(
    center=(47.37, 8.54), spread_deg=0.02, n=1000, seed=0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.uniform(-spread_deg, spread_deg, size=(n, 2))


def test_grid_covers_padded_bbox():
    coords = make_coords()
    grid = build_region_grid(coords, RESOLUTION_KM, padding_deg=PADDING_DEG)
    assert grid.ndim == 2 and grid.shape[1] == 2

    lat_lo, lat_hi = np.percentile(coords[:, 0], [0.5, 99.5])
    lon_lo, lon_hi = np.percentile(coords[:, 1], [0.5, 99.5])
    eps = 1e-9
    assert grid[:, 0].min() <= lat_lo - PADDING_DEG + eps
    assert grid[:, 0].max() >= lat_hi + PADDING_DEG - eps
    assert grid[:, 1].min() <= lon_lo - PADDING_DEG + eps
    assert grid[:, 1].max() >= lon_hi + PADDING_DEG - eps


def test_grid_spacing_matches_resolution_km():
    coords = make_coords()
    grid = build_region_grid(coords, RESOLUTION_KM)
    lats = np.unique(grid[:, 0])
    lons = np.unique(grid[:, 1])
    assert len(lats) >= 3 and len(lons) >= 3

    # Interior steps only: the final step may be a short edge-closing remainder.
    lat_steps = haversine_km(lats[:-2], lons[0], lats[1:-1], lons[0])
    center_lat = lats[len(lats) // 2]
    lon_steps = haversine_km(center_lat, lons[:-2], center_lat, lons[1:-1])
    assert np.all(np.abs(lat_steps - RESOLUTION_KM) <= 0.05 * RESOLUTION_KM)
    assert np.all(np.abs(lon_steps - RESOLUTION_KM) <= 0.05 * RESOLUTION_KM)


def test_outliers_excluded_from_bbox():
    inliers = make_coords()
    outliers = np.array([[48.5, 9.5], [46.0, 7.5]])  # ~1 deg away from cluster
    coords = np.concatenate([inliers, outliers], axis=0)
    grid = build_region_grid(coords, RESOLUTION_KM, padding_deg=PADDING_DEG)

    # Grid stays near the inlier cluster, far from the planted outliers.
    assert grid[:, 0].max() < 48.5 - 0.5
    assert grid[:, 0].min() > 46.0 + 0.5
    assert grid[:, 1].max() < 9.5 - 0.5
    assert grid[:, 1].min() > 7.5 + 0.5
    # And still hugs the inlier bbox (within padding + one percentile clip).
    slack = PADDING_DEG + 0.005
    assert grid[:, 0].max() <= inliers[:, 0].max() + slack
    assert grid[:, 0].min() >= inliers[:, 0].min() - slack


def test_uniform_grid_dedups_identical_regions():
    coords = make_coords()
    single = build_region_grid(coords, RESOLUTION_KM)
    merged = build_uniform_grid({"a": coords, "b": coords.copy()}, RESOLUTION_KM)
    assert merged.shape == single.shape
    assert np.array_equal(np.unique(single, axis=0), merged)


def test_uniform_grid_concats_disjoint_regions():
    zurich = make_coords(center=(47.37, 8.54), seed=1)
    tokyo = make_coords(center=(35.68, 139.77), seed=2)
    g_z = build_region_grid(zurich, RESOLUTION_KM)
    g_t = build_region_grid(tokyo, RESOLUTION_KM)
    merged = build_uniform_grid({"zurich": zurich, "tokyo": tokyo}, RESOLUTION_KM)
    assert merged.shape[0] == g_z.shape[0] + g_t.shape[0]


def test_save_load_roundtrip(tmp_path):
    grid = build_region_grid(make_coords(), RESOLUTION_KM)
    path = tmp_path / "gallery.npy"
    save_gallery(str(path), grid)
    loaded = load_gallery(str(path))
    assert np.array_equal(loaded, grid)


def test_cli_builds_gallery(tmp_path):
    import pandas as pd

    coords = make_coords(n=200)
    df = pd.DataFrame(
        {"lat": coords[:, 0], "lon": coords[:, 1], "region": ["zrh"] * len(coords)}
    )
    csv_path = tmp_path / "coords.csv"
    out_path = tmp_path / "gallery.npy"
    df.to_csv(csv_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "vidtag.data.gallery",
            "--coords-csv",
            str(csv_path),
            "--resolution-km",
            "0.1",
            "--out",
            str(out_path),
        ],
        check=True,
    )
    grid = load_gallery(str(out_path))
    expected = build_uniform_grid({"zrh": coords}, 0.1)
    assert np.array_equal(grid, expected)
