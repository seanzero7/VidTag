"""Frame index sampling for fixed-length training sequences (GUESSES.md #20).

The paper only says "16 frames were sampled" from longer videos (suppl. A);
we use t uniformly spaced stride cells over [0, n_frames): training draws one
uniformly random index per cell, evaluation takes cell centers.
"""

from __future__ import annotations

import numpy as np


def sample_indices(
    n_frames: int,
    t: int = 16,
    train: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample t frame indices from a video of n_frames, sorted, int64.

    If n_frames <= t, returns arange(n_frames) — eval keeps natural length;
    training datasets must pre-filter to sequences with >= t frames.
    """
    if n_frames <= t:
        return np.arange(n_frames, dtype=np.int64)
    edges = np.linspace(0.0, float(n_frames), t + 1)
    if train:
        rng = rng if rng is not None else np.random.default_rng()
        pos = rng.uniform(edges[:-1], edges[1:])
    else:
        pos = (edges[:-1] + edges[1:]) / 2.0
    return np.minimum(pos.astype(np.int64), n_frames - 1)
