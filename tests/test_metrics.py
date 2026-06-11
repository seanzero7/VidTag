"""Tests for vidtag.metrics (SPEC §8) with hand-computed references."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vidtag.metrics import (
    EARTH_RADIUS_KM,
    discrete_frechet_km,
    evaluate_sequences,
    frame_accuracy,
    haversine_km,
    mean_range_difference_km,
    median_distance_km,
    video_accuracy,
)

# 1 degree of longitude at the equator (and of great-circle arc generally).
DEG_KM = 2 * math.pi * EARTH_RADIUS_KM / 360.0  # ~111.195 km


class TestHaversine:
    def test_veness_worked_example(self):
        # Movable Type Scripts reference: 50°03'59"N 005°42'53"W to
        # 58°38'38"N 003°04'12"W = 968.9 km.
        a = np.array([50.06639, -5.71472])
        b = np.array([58.64389, -3.07000])
        assert haversine_km(a, b) == pytest.approx(968.9, rel=0.005)

    def test_nyc_to_la(self):
        a = np.array([40.7128, -74.0060])
        b = np.array([34.0522, -118.2437])
        assert haversine_km(a, b) == pytest.approx(3935.7, rel=0.005)

    def test_one_degree_longitude_at_equator(self):
        d = haversine_km(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        assert d == pytest.approx(111.195, rel=0.005)

    def test_zero_and_antipodal(self):
        p = np.array([12.3, 45.6])
        assert haversine_km(p, p) == pytest.approx(0.0, abs=1e-9)
        d = haversine_km(np.array([0.0, 0.0]), np.array([0.0, 180.0]))
        assert d == pytest.approx(math.pi * EARTH_RADIUS_KM, rel=1e-6)

    def test_symmetry(self):
        a = np.array([48.8566, 2.3522])
        b = np.array([51.5074, -0.1278])
        assert haversine_km(a, b) == pytest.approx(haversine_km(b, a), rel=1e-12)

    def test_broadcasting(self):
        a = np.zeros((3, 2))
        b = np.array([0.0, 1.0])
        d = haversine_km(a, b)
        assert d.shape == (3,)
        np.testing.assert_allclose(d, DEG_KM, rtol=1e-9)
        # (2,1,2) vs (3,2) -> (2,3)
        assert haversine_km(np.zeros((2, 1, 2)), np.zeros((3, 2))).shape == (2, 3)


class TestFrameMetrics:
    def setup_method(self):
        # Distances ~ 0.111, 0.890, 3.336, 11.12, 111.19 km from GT at origin.
        self.gt = np.zeros((5, 2))
        self.pred = np.array(
            [[0.0, 0.001], [0.0, 0.008], [0.0, 0.03], [0.0, 0.1], [0.0, 1.0]]
        )

    def test_frame_accuracy_thresholds(self):
        acc = frame_accuracy(self.pred, self.gt)
        assert acc == {
            "acc@0.5km": pytest.approx(20.0),
            "acc@1km": pytest.approx(40.0),
            "acc@5km": pytest.approx(60.0),
            "acc@25km": pytest.approx(80.0),
        }

    def test_median_distance(self):
        # Median of the five distances is the middle one: 0.03 deg.
        assert median_distance_km(self.pred, self.gt) == pytest.approx(
            0.03 * DEG_KM, rel=1e-6
        )


class TestVideoAccuracy:
    def test_e1_representative_ignores_outlier(self):
        # GT centroid = (0, 0.01). Prediction centroid = (0, 0.340667) is
        # dragged toward the outlier; the prediction closest to it is
        # (0, 0.012) -- NOT the outlier, NOT the centroid itself, and NOT the
        # prediction closest to the GT centroid (which would be (0, 0.01),
        # giving distance 0).
        gt = np.array([[0.0, 0.0], [0.0, 0.01], [0.0, 0.02]])
        pred = np.array([[0.0, 0.01], [0.0, 0.012], [0.0, 1.0]])
        out = video_accuracy([pred], [gt])
        expected = 0.002 * DEG_KM  # ~0.2224 km: representative (0,0.012)
        assert out["median_km"] == pytest.approx(expected, rel=1e-6)
        assert out["median_km"] > 0.1  # rules out closest-to-GT-centroid logic
        assert out["median_km"] < 1.0  # rules out using the prediction centroid (~36.8 km)
        assert out["acc@0.5km"] == pytest.approx(100.0)

    def test_threshold_percentages_over_videos(self):
        gt_good = np.array([[0.0, 0.0], [0.0, 0.01], [0.0, 0.02]])
        pred_good = np.array([[0.0, 0.01], [0.0, 0.012], [0.0, 1.0]])
        gt_bad = np.zeros((3, 2))
        pred_bad = np.full((3, 2), 10.0)  # ~1569 km off
        out = video_accuracy([pred_good, pred_bad], [gt_good, gt_bad])
        assert out["acc@0.5km"] == pytest.approx(50.0)
        assert out["acc@25km"] == pytest.approx(50.0)


class TestDiscreteFrechet:
    def test_identical_polylines(self):
        a = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        assert discrete_frechet_km(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_point_vs_segment(self):
        # Every coupling pairs the single point with both endpoints; the
        # farther endpoint (0.5 deg away) dominates.
        a = np.array([[0.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.0, 0.5]])
        expected = 0.5 * DEG_KM  # ~55.597 km
        assert discrete_frechet_km(a, b) == pytest.approx(expected, rel=1e-6)
        assert discrete_frechet_km(b, a) == pytest.approx(expected, rel=1e-6)

    def test_three_vs_two_points(self):
        # a's middle point (0,1) must pair with (0,0) or (0,2): 1 deg either
        # way; endpoints pair at 0. DFD = 1 deg.
        a = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        b = np.array([[0.0, 0.0], [0.0, 2.0]])
        assert discrete_frechet_km(a, b) == pytest.approx(DEG_KM, rel=1e-6)

    def test_backtracking_polyline(self):
        # Hand-traced DP: d (deg) = [[1,0,2],[1,2,0]] ->
        # ca = [[1,1,2],[1,2,1]] -> DFD = 1 deg (the leash cannot avoid the
        # 1-deg pairing while b backtracks through (0,0)).
        a = np.array([[0.0, 0.0], [0.0, 2.0]])
        b = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 2.0]])
        assert discrete_frechet_km(a, b) == pytest.approx(DEG_KM, rel=1e-6)


class TestMeanRangeDifference:
    def test_rectangles_at_equator(self):
        # pred: lat range 1, lon range 2. gt: lat range 0.5, lon range 1,
        # mean GT lat 0 -> lon scale exactly 111.320 km/deg.
        pred = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]])
        gt = np.array([[-0.25, 0.0], [0.25, 0.0], [0.25, 1.0], [-0.25, 1.0]])
        expected = 0.5 * (0.5 * 110.574 + 1.0 * 111.320)  # 83.3035 km
        assert mean_range_difference_km([pred], [gt]) == pytest.approx(expected, rel=1e-9)

    def test_cosine_latitude_scaling(self):
        # Mean GT lat 60 -> cos = 0.5; only lon ranges differ (by 1 deg).
        gt = np.array([[59.5, 10.0], [60.5, 10.0], [60.5, 11.0], [59.5, 11.0]])
        pred = np.array([[59.5, 10.0], [60.5, 10.0], [60.5, 12.0], [59.5, 12.0]])
        expected = 0.5 * (1.0 * 111.320 * 0.5)  # 27.83 km
        assert mean_range_difference_km([pred], [gt]) == pytest.approx(expected, rel=1e-9)

    def test_mean_over_sequences(self):
        pred1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]])
        gt1 = np.array([[-0.25, 0.0], [0.25, 0.0], [0.25, 1.0], [-0.25, 1.0]])
        gt2 = np.array([[59.5, 10.0], [60.5, 10.0], [60.5, 11.0], [59.5, 11.0]])
        out = mean_range_difference_km([pred1, gt2], [gt1, gt2])
        assert out == pytest.approx(0.5 * 83.3035, rel=1e-6)  # second seq contributes 0


class TestEvaluateSequences:
    def test_flat_dict_consistency(self):
        rng = np.random.default_rng(0)
        gts = [np.column_stack([rng.uniform(40, 41, t), rng.uniform(2, 3, t)]) for t in (5, 7)]
        preds = [g + rng.normal(0, 0.01, g.shape) for g in gts]
        out = evaluate_sequences(preds, gts)

        pred_all = np.concatenate(preds)
        gt_all = np.concatenate(gts)
        for k, v in frame_accuracy(pred_all, gt_all).items():
            assert out[f"frame_{k}"] == pytest.approx(v)
        assert out["frame_median_km"] == pytest.approx(median_distance_km(pred_all, gt_all))
        for k, v in video_accuracy(preds, gts).items():
            assert out[f"video_{k}"] == pytest.approx(v)
        dfd = np.mean([discrete_frechet_km(p, g) for p, g in zip(preds, gts)])
        assert out["dfd_km"] == pytest.approx(dfd)
        assert out["mrd_km"] == pytest.approx(mean_range_difference_km(preds, gts))
        assert all(isinstance(v, float) for v in out.values())

    def test_mismatched_lengths_raise(self):
        seq = np.zeros((4, 2))
        with pytest.raises(ValueError):
            video_accuracy([seq, seq], [seq])
        with pytest.raises(ValueError):
            mean_range_difference_km([seq], [seq, seq])
