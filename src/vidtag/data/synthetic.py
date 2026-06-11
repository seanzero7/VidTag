"""Synthetic sequence dataset for smoke tests (SPEC §10) — no disk access.

Each video gets a random city center and a smooth random-walk trajectory
(~5e-4 deg steps). In 'features' mode the fused embeddings are a fixed,
smooth function of the trajectory coordinates — sin/cos expansions of
(lat, lon) projected through a frozen random matrix, plus tiny noise, then
L2-normalized — so a contrastive loss CAN learn the coordinate mapping
(pure noise would make smoke training meaningless). In 'frames' mode raw
random [0, 1] pixel tensors are generated lazily per item.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

STEP_DEG = 5e-4  # trajectory step magnitude (degrees per frame)
HEADING_STD = 0.1  # per-frame heading jitter (radians) — keeps walks smooth
FEATURE_NOISE_STD = 0.01  # pre-normalization per-dim noise on features
_FREQS = 2.0 ** np.arange(10)  # sin/cos expansion frequencies (cycles/deg)


def _fourier_features(coords: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """(..., 2) degrees -> (..., 4 * len(freqs)) sin/cos expansion."""
    ang = 2.0 * np.pi * coords[..., None] * freqs  # (..., 2, F)
    feats = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)  # (..., 2, 2F)
    return feats.reshape(*coords.shape[:-1], -1)


class SyntheticSequences(Dataset):
    def __init__(
        self,
        num_videos: int = 64,
        frames_per_seq: int = 16,
        mode: str = "features",
        seed: int = 0,
        image_size: int = 224,
        feature_dim: int = 1792,
    ):
        if mode not in ("features", "frames"):
            raise ValueError(f"mode must be 'features' or 'frames', got {mode!r}")
        self.mode = mode
        self.seed = seed
        self.image_size = image_size

        rng = np.random.default_rng(seed)
        V, T = num_videos, frames_per_seq
        centers = np.stack(
            [rng.uniform(-60.0, 60.0, V), rng.uniform(-170.0, 170.0, V)], axis=-1
        )
        heading = rng.uniform(0.0, 2.0 * np.pi, (V, 1)) + np.cumsum(
            rng.normal(0.0, HEADING_STD, (V, T)), axis=1
        )
        steps = STEP_DEG * np.stack([np.cos(heading), np.sin(heading)], axis=-1)
        coords = centers[:, None, :] + np.cumsum(steps, axis=1)  # (V, T, 2)
        self.coords = torch.from_numpy(coords.astype(np.float32))

        if mode == "features":
            phi = _fourier_features(coords, _FREQS)  # (V, T, 4F)
            # The feature->coordinate mapping must be SHARED across dataset
            # instances (train/val) or generalization is impossible by
            # construction; only trajectories vary with `seed`.
            proj_rng = np.random.default_rng(12345)
            proj = proj_rng.normal(0.0, phi.shape[-1] ** -0.5, (phi.shape[-1], feature_dim))
            feats = phi @ proj + FEATURE_NOISE_STD * rng.normal(size=(V, T, feature_dim))
            feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
            self.fused = torch.from_numpy(feats.astype(np.float32))

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int) -> dict:
        T = self.coords.shape[1]
        item: dict = {"coords": self.coords[idx], "video_id": idx, "n_frames": T}
        if self.mode == "features":
            item["fused"] = self.fused[idx]
        else:
            rng = np.random.default_rng((self.seed, idx))
            item["frames"] = torch.from_numpy(
                rng.random((T, 3, self.image_size, self.image_size), dtype=np.float32)
            )
        return item


def collate_sequences(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack a list of dataset items (all same T in train mode) into batch
    tensors; scalar fields (video_id, n_frames) become 1-D int64 tensors."""
    out: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        vals = [item[key] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = torch.tensor(vals, dtype=torch.int64)
    return out
