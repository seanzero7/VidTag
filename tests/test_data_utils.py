"""Tests for vidtag.data: sample_indices, frames_to_tensor, SyntheticSequences."""

import numpy as np
import torch
from PIL import Image

from vidtag.data.sequences import sample_indices
from vidtag.data.synthetic import SyntheticSequences, collate_sequences
from vidtag.data.transforms import frames_to_tensor, load_image

T = 16


# ------------------------------------------------------------ sample_indices
def test_sample_indices_bounds_and_dtype():
    for n in (17, 40, 300, 1000):
        for train in (True, False):
            idx = sample_indices(n, T, train=train, rng=np.random.default_rng(0))
            assert idx.dtype == np.int64
            assert idx.shape == (T,)
            assert idx.min() >= 0 and idx.max() < n


def test_sample_indices_monotonic():
    rng = np.random.default_rng(1)
    for n in (17, 20, 64, 500):
        train_idx = sample_indices(n, T, train=True, rng=rng)
        assert np.all(np.diff(train_idx) >= 0)  # cells may touch after floor
        eval_idx = sample_indices(n, T, train=False)
        assert np.all(np.diff(eval_idx) >= 1)  # stride > 1 -> strictly increasing


def test_sample_indices_train_vs_eval_determinism():
    eval_a = sample_indices(300, T, train=False)
    eval_b = sample_indices(300, T, train=False)
    np.testing.assert_array_equal(eval_a, eval_b)

    train_a = sample_indices(300, T, train=True, rng=np.random.default_rng(0))
    train_b = sample_indices(300, T, train=True, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(train_a, train_b)
    train_c = sample_indices(300, T, train=True, rng=np.random.default_rng(1))
    assert not np.array_equal(train_a, train_c)
    assert not np.array_equal(train_a, eval_a)


def test_sample_indices_short_sequence():
    for n in (5, T):
        for train in (True, False):
            idx = sample_indices(n, T, train=train, rng=np.random.default_rng(0))
            np.testing.assert_array_equal(idx, np.arange(n, dtype=np.int64))


# --------------------------------------------------------- frames_to_tensor
def test_frames_to_tensor_shape_and_range():
    images = [
        Image.new("RGB", (320, 240), (255, 0, 0)),
        Image.new("L", (100, 400), 128),  # grayscale -> RGB conversion
        Image.new("RGB", (224, 224), (0, 128, 255)),
    ]
    out = frames_to_tensor(images, size=224)
    assert out.shape == (3, 3, 224, 224)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Solid red stays solid red after resize.
    assert torch.allclose(out[0, 0], torch.ones(224, 224))
    assert torch.allclose(out[0, 1:], torch.zeros(2, 224, 224))


def test_load_image_rgb(tmp_path):
    p = tmp_path / "gray.png"
    Image.new("L", (10, 10), 200).save(p)
    img = load_image(p)
    assert img.mode == "RGB"
    assert img.size == (10, 10)


# ------------------------------------------------------- SyntheticSequences
def test_synthetic_features_shapes_and_norms():
    ds = SyntheticSequences(num_videos=8, frames_per_seq=T, mode="features", seed=0)
    assert len(ds) == 8
    item = ds[3]
    assert item["fused"].shape == (T, 1792)
    assert item["coords"].shape == (T, 2)
    assert item["fused"].dtype == torch.float32
    assert item["coords"].dtype == torch.float32
    assert item["video_id"] == 3
    assert item["n_frames"] == T
    norms = item["fused"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(T), atol=1e-5)
    lat, lon = item["coords"][:, 0], item["coords"][:, 1]
    assert lat.abs().max() <= 61 and lon.abs().max() <= 171


def test_synthetic_feature_determinism():
    a = SyntheticSequences(num_videos=4, frames_per_seq=T, mode="features", seed=0)
    b = SyntheticSequences(num_videos=4, frames_per_seq=T, mode="features", seed=0)
    c = SyntheticSequences(num_videos=4, frames_per_seq=T, mode="features", seed=1)
    torch.testing.assert_close(a[2]["fused"], b[2]["fused"])
    torch.testing.assert_close(a[2]["coords"], b[2]["coords"])
    assert not torch.equal(a[2]["fused"], c[2]["fused"])


def test_synthetic_features_correlate_with_trajectory():
    # Frames of one video must be mutually closer than frames of different
    # videos, otherwise contrastive smoke training is meaningless.
    ds = SyntheticSequences(num_videos=16, frames_per_seq=T, mode="features", seed=0)
    fused = torch.stack([ds[i]["fused"] for i in range(len(ds))])  # (V, T, D)
    flat = fused.flatten(0, 1)
    sim = flat @ flat.T  # cosine (rows are unit-norm)
    same = torch.block_diag(*[torch.ones(T, T, dtype=torch.bool)] * len(ds))
    off_diag = ~torch.eye(len(flat), dtype=torch.bool)
    within = sim[same & off_diag].mean()
    across = sim[~same].mean()
    assert within > across + 0.2


def test_synthetic_frames_mode():
    ds = SyntheticSequences(
        num_videos=3, frames_per_seq=4, mode="frames", seed=0, image_size=64
    )
    item = ds[1]
    assert "fused" not in item
    assert item["frames"].shape == (4, 3, 64, 64)
    assert item["frames"].dtype == torch.float32
    assert item["frames"].min() >= 0.0 and item["frames"].max() <= 1.0
    ds2 = SyntheticSequences(
        num_videos=3, frames_per_seq=4, mode="frames", seed=0, image_size=64
    )
    torch.testing.assert_close(item["frames"], ds2[1]["frames"])


def test_collate_sequences():
    ds = SyntheticSequences(num_videos=4, frames_per_seq=T, mode="features", seed=0)
    batch = collate_sequences([ds[0], ds[2], ds[3]])
    assert batch["fused"].shape == (3, T, 1792)
    assert batch["coords"].shape == (3, T, 2)
    assert batch["video_id"].tolist() == [0, 2, 3]
    assert batch["n_frames"].dtype == torch.int64
    assert batch["n_frames"].tolist() == [T, T, T]
