"""Tests for the config system and training utilities."""

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from vidtag.config import load_config
from vidtag.utils import WarmupStepLR, load_checkpoint, save_checkpoint, set_seed


def test_config_load_override_attr(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "train:\n  lr: 5e-5\n  epochs: 600\nmodel:\n  embed_dim: 512\n"
    )
    cfg = load_config(str(cfg_file), overrides=["train.lr=1e-4", "data.root=/tmp/x"])

    assert cfg.train.lr == pytest.approx(1e-4)  # override wins, YAML-parsed float
    assert cfg.train.epochs == 600
    assert cfg.model.embed_dim == 512
    assert cfg.data.root == "/tmp/x"  # override may create new sections

    assert cfg.get("train.epochs") == 600
    assert cfg.get("missing.key", 7) == 7
    assert cfg.to_dict()["train"]["epochs"] == 600

    with pytest.raises(AttributeError, match="nope"):
        _ = cfg.train.nope


def test_warmup_step_lr_trajectory():
    base = 1e-2
    opt = torch.optim.Adam(nn.Linear(4, 4).parameters(), lr=base)
    sched = WarmupStepLR(opt, warmup_steps=10, gamma=0.5)
    assert sched.current_lrs == [0.0]  # warmup starts from 0

    lrs = []
    for _ in range(10):
        sched.step_batch()
        lrs.append(sched.current_lrs[0])
    assert lrs[0] == pytest.approx(base / 10)
    assert all(b > a for a, b in zip(lrs, lrs[1:]))  # linear ramp
    assert lrs[-1] == pytest.approx(base)  # base lr reached at end of warmup

    sched.step_epoch()
    assert sched.current_lrs[0] == pytest.approx(base * 0.5)
    sched.step_batch()  # post-warmup batch steps do not change the lr
    assert sched.current_lrs[0] == pytest.approx(base * 0.5)
    sched.step_epoch()
    assert sched.current_lrs[0] == pytest.approx(base * 0.25)  # decay compounds

    opt2 = torch.optim.Adam(nn.Linear(4, 4).parameters(), lr=base)
    sched2 = WarmupStepLR(opt2, warmup_steps=10, gamma=0.5)
    sched2.load_state_dict(sched.state_dict())
    assert sched2.current_lrs == sched.current_lrs


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = WarmupStepLR(opt, warmup_steps=5, gamma=0.9)
    model(torch.randn(4, 3)).sum().backward()
    opt.step()
    sched.step_batch()

    path = tmp_path / "ckpt" / "model.pt"  # parent dir created by save
    save_checkpoint(str(path), model, opt, sched, epoch=3, step=17, extra={"k": "v"})

    model2 = nn.Linear(3, 2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    sched2 = WarmupStepLR(opt2, warmup_steps=5, gamma=0.9)
    meta = load_checkpoint(str(path), model2, opt2, sched2)

    assert meta == {"epoch": 3, "step": 17, "extra": {"k": "v"}}
    for p, q in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p, q)
    assert sched2.step_count == 1
    assert sched2.current_lrs == sched.current_lrs
    assert not path.with_suffix(".pt.tmp").exists()  # atomic write cleaned up


def test_set_seed_determinism():
    set_seed(123)
    a = (torch.randn(8), np.random.rand(8), random.random())
    set_seed(123)
    b = (torch.randn(8), np.random.rand(8), random.random())
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]
