"""Frame index sampling for fixed-length training sequences (GUESSES.md #20).

The paper only says "16 frames were sampled" from longer videos (suppl. A);
we use t stride cells over [0, n_frames) with one index per cell. Cells are
made DISJOINT in integer space so the t indices are always unique — naive
floor() of fractional cells duplicates indices for n_frames close to t
(measured: 98% of draws at n=17, t=16), which would feed identical
(frame, GPS) pairs to the contrastive loss as mutual negatives.
"""

from __future__ import annotations

import numpy as np


def _integer_cells(n_frames: int, t: int) -> tuple[np.ndarray, np.ndarray]:
    """t disjoint, non-empty integer ranges [lo_i, hi_i) covering [0, n)."""
    edges = np.linspace(0.0, float(n_frames), t + 1)
    lo = np.ceil(edges[:-1]).astype(np.int64)
    hi = np.ceil(edges[1:]).astype(np.int64)
    # guarantee every cell holds at least one integer (possible because n >= t)
    for i in range(t):
        if hi[i] <= lo[i]:
            hi[i] = lo[i] + 1
        if i + 1 < t and lo[i + 1] < hi[i]:
            lo[i + 1] = hi[i]
    hi[-1] = min(hi[-1], n_frames)
    return lo, hi


def sample_indices(
    n_frames: int,
    t: int = 16,
    train: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample t unique frame indices from a video of n_frames, sorted, int64.

    If n_frames <= t, returns arange(n_frames) — eval keeps natural length;
    training datasets must pre-filter to sequences with >= t frames.
    """
    if n_frames <= t:
        return np.arange(n_frames, dtype=np.int64)
    lo, hi = _integer_cells(n_frames, t)
    if train:
        rng = rng if rng is not None else np.random.default_rng()
        return (lo + (rng.random(t) * (hi - lo)).astype(np.int64)).astype(np.int64)
    return ((lo + hi - 1) // 2).astype(np.int64)
