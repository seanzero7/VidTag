"""LR schedule: linear warmup into per-epoch StepLR decay.

Paper suppl. A: 1000 warmup steps, then StepLR with decay 0.99 (Phase I) or
0.95 (Phase II). Decay cadence is per epoch and warmup is linear from 0 to
the base LR (GUESSES.md #16, #17); the two factors compose multiplicatively.
"""

from __future__ import annotations

from typing import Any

import torch


class WarmupStepLR:
    """Call ``step_batch()`` after every optimizer step and ``step_epoch()``
    at each epoch end. LR = base_lr * min(step/warmup_steps, 1) * gamma^epoch."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, gamma: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.step_count = 0
        self.epoch_count = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._apply()

    def _factor(self) -> float:
        warmup = 1.0
        if self.warmup_steps > 0:
            # step_count+1: the k-th optimizer step runs at k/W of base LR
            # (plain step_count would make the very first step run at LR=0).
            warmup = min((self.step_count + 1) / self.warmup_steps, 1.0)
        return warmup * self.gamma**self.epoch_count

    def _apply(self) -> None:
        factor = self._factor()
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor

    def step_batch(self) -> None:
        self.step_count += 1
        self._apply()

    def step_epoch(self) -> None:
        self.epoch_count += 1
        self._apply()

    @property
    def current_lrs(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
            "base_lrs": self.base_lrs,
            "warmup_steps": self.warmup_steps,
            "gamma": self.gamma,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_count = state["step_count"]
        self.epoch_count = state["epoch_count"]
        self.base_lrs = list(state["base_lrs"])
        self.warmup_steps = state["warmup_steps"]
        self.gamma = state["gamma"]
        self._apply()
