"""Checkpoint save/load with atomic writes (SPEC §10: save/resume)."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write a checkpoint (temp file + rename, no torn files)."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "extra": extra or {},
    }
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore model (+ optional optimizer/scheduler); returns epoch/step/extra."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return {
        "epoch": payload.get("epoch", 0),
        "step": payload.get("step", 0),
        "extra": payload.get("extra", {}),
    }
