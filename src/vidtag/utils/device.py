"""Device resolution and mixed-precision policy.

cuda -> bf16 autocast (Blackwell); mps/cpu -> fp32 (MPS autocast support is
incomplete; correctness first on the Mac proof-of-concept runs).
"""

from __future__ import annotations

import contextlib

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_for(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
