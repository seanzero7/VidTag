"""Evaluation metrics (paper §4.3, suppl. E).

All inputs are numpy arrays of (lat, lon) in **degrees**. Geodesic distances
use the haversine formula on a sphere of mean Earth radius, in kilometers.
MRD converts degree ranges to km via local metric scales (GUESSES.md #30).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQUATOR = 111.320

DEFAULT_THRESHOLDS_KM = (0.5, 1.0, 5.0, 25.0)


def hierarchy_accuracy(
    pred_cities: Sequence[str],
    gt_cities: Sequence[str],
    labels,
) -> dict[str, float]:
    """CityGuessr68k city-level protocol (paper §5.3, Table 3): % of videos
    whose predicted city matches the GT at each hierarchy level
    (City/State/Country/Continent), via the labels_list.csv mapping
    (a pandas DataFrame with those four columns)."""
    lookup = {r.City: (r.City, r.State, r.Country, r.Continent) for r in labels.itertuples()}
    levels = ("city", "state", "country", "continent")
    hits = dict.fromkeys(levels, 0)
    for p, g in zip(pred_cities, gt_cities):
        ph, gh = lookup[p], lookup[g]
        for i, lvl in enumerate(levels):
            hits[lvl] += int(ph[i] == gh[i])
    n = max(len(gt_cities), 1)
    return {f"{lvl}_acc": 100.0 * hits[lvl] / n for lvl in levels}


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between (lat, lon) degree pairs.

    a, b: (..., 2); leading dims broadcast elementwise (numpy rules).
    """
    a = np.deg2rad(np.asarray(a, dtype=np.float64))
    b = np.deg2rad(np.asarray(b, dtype=np.float64))
    dlat = b[..., 0] - a[..., 0]
    dlon = b[..., 1] - a[..., 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _acc_key(threshold_km: float) -> str:
    return f"acc@{threshold_km:g}km"


def frame_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    thresholds_km: Sequence[float] = DEFAULT_THRESHOLDS_KM,
) -> dict[str, float]:
    """Frame-wise accuracy (§4.3): % of frames with geodesic error under each
    threshold. pred, gt: (N, 2). Returns {"acc@0.5km": pct in [0, 100], ...}."""
    d = haversine_km(pred, gt)
    return {_acc_key(t): float((d < t).mean() * 100.0) for t in thresholds_km}


def median_distance_km(pred: np.ndarray, gt: np.ndarray) -> float:
    """Median geodesic distance error in km over frames (§4.3). pred, gt: (N, 2)."""
    return float(np.median(haversine_km(pred, gt)))


def _video_distances_km(
    preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]
) -> np.ndarray:
    """Per-video representative error per suppl. E.1: the video's GT is the
    centroid of its GT labels; the representative prediction is the one
    closest to the *prediction* centroid (suppresses outlier frames)."""
    dists = []
    for p, g in zip(preds, gts, strict=True):
        gt_centroid = g.mean(axis=0)
        pred_centroid = p.mean(axis=0)
        rep = p[int(np.argmin(haversine_km(p, pred_centroid)))]
        dists.append(haversine_km(rep, gt_centroid))
    return np.asarray(dists, dtype=np.float64)


def video_accuracy(
    preds: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    thresholds_km: Sequence[float] = DEFAULT_THRESHOLDS_KM,
) -> dict[str, float]:
    """Video-level accuracy (suppl. E.1) over a list of (T_i, 2) sequences.

    Returns {"acc@<t>km": pct, ..., "median_km": median representative error}."""
    d = _video_distances_km(preds, gts)
    out = {_acc_key(t): float((d < t).mean() * 100.0) for t in thresholds_km}
    out["median_km"] = float(np.median(d))
    return out


def discrete_frechet_km(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Fréchet distance in km between polylines a (Ta, 2) and b (Tb, 2)
    of (lat, lon) degrees (suppl. E). Classic iterative DP (Eiter & Mannila)
    over the haversine distance matrix."""
    d = haversine_km(a[:, None, :], b[None, :, :])  # (Ta, Tb)
    ca = np.empty_like(d)
    ca[0, 0] = d[0, 0]
    for j in range(1, d.shape[1]):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, d.shape[0]):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
        for j in range(1, d.shape[1]):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d[i, j])
    return float(ca[-1, -1])


def mean_range_difference_km(
    preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]
) -> float:
    """Mean Range Difference adapted to 2-D (suppl. E, GUESSES.md #30).

    Per sequence: |range(pred) - range(gt)| separately for lat and lon, in km
    (lat via 110.574 km/deg; lon via 111.320*cos(mean GT lat) km/deg), the two
    components averaged. Returns the mean over sequences."""
    vals = []
    for p, g in zip(preds, gts, strict=True):
        lat_diff = abs(np.ptp(p[:, 0]) - np.ptp(g[:, 0])) * KM_PER_DEG_LAT
        lon_scale = KM_PER_DEG_LON_EQUATOR * np.cos(np.deg2rad(g[:, 0].mean()))
        lon_diff = abs(np.ptp(p[:, 1]) - np.ptp(g[:, 1])) * lon_scale
        vals.append(0.5 * (lat_diff + lon_diff))
    return float(np.mean(vals))


def evaluate_sequences(
    preds: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    thresholds_km: Sequence[float] = DEFAULT_THRESHOLDS_KM,
) -> dict[str, float]:
    """All §8 metrics over a list of per-video (T_i, 2) prediction/GT arrays.

    Returns a flat dict: "frame_acc@<t>km" + "frame_median_km" (over all frames
    concatenated), "video_acc@<t>km" + "video_median_km" (suppl. E.1),
    "dfd_km" (mean per-sequence discrete Fréchet), and "mrd_km"."""
    pred_all = np.concatenate(list(preds), axis=0)
    gt_all = np.concatenate(list(gts), axis=0)
    out = {f"frame_{k}": v for k, v in frame_accuracy(pred_all, gt_all, thresholds_km).items()}
    out["frame_median_km"] = median_distance_km(pred_all, gt_all)
    out.update({f"video_{k}": v for k, v in video_accuracy(preds, gts, thresholds_km).items()})
    out["dfd_km"] = float(
        np.mean([discrete_frechet_km(p, g) for p, g in zip(preds, gts, strict=True)])
    )
    out["mrd_km"] = mean_range_difference_km(preds, gts)
    return out
