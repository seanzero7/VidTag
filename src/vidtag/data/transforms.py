"""Frame preprocessing (suppl. A; GUESSES.md #21).

Frames are resized directly (non-aspect-preserving) to 224x224 and kept as
raw [0, 1] RGB tensors. Backbone-specific normalization (CLIP / ImageNet
stats) happens inside DualFrameEncoder, so tensors produced here stay
backbone-agnostic and cacheable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    """Open an image file and convert to RGB."""
    return Image.open(path).convert("RGB")


def frames_to_tensor(images: list[Image.Image], size: int = 224) -> torch.Tensor:
    """PIL images -> (T, 3, size, size) float32 RGB in [0, 1].

    Direct bilinear resize with antialias (no aspect preservation, no crop —
    paper: "resized to 224x224"); no normalization (see module docstring).
    """
    out: list[torch.Tensor] = []
    for img in images:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        if t.shape[-2:] != (size, size):
            t = F.interpolate(
                t[None],
                size=(size, size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        out.append(t)
    return torch.stack(out)
