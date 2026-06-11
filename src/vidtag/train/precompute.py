"""Precompute fused CLIP+DINOv2 features for any frames-mode dataset.

The backbones are frozen (paper §3.1), so features are computed once and
cached as one float16 ``(n_frames, 1792)`` .npy per sequence — the paper's
own fast path (suppl. I / Table 14: 68h -> 2.75h). Training then runs in
``data.mode: features``.

  PYTHONPATH=src python -m vidtag.train.precompute \
      --config configs/msls_phase1_full.yaml --split train \
      --out /data/PaperRepro/datasets/msls/features/train

NOTE: for MSLS the dataset must be constructed with train=False semantics so
every frame of every sequence is featurized (no 16-frame subsampling); this
module handles that internally.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import load_config
from ..models.frame_encoder import DualFrameEncoder
from ..utils import get_logger, resolve_device
from .common import build_dataset


class _FlatFrames(Dataset):
    """Flattens a sequence dataset into per-frame items for batched encoding."""

    def __init__(self, seq_dataset):
        self.ds = seq_dataset
        self.items: list[tuple[int, int]] = []
        for si in range(len(seq_dataset)):
            n = seq_dataset.sequence_length(si)
            self.items.extend((si, fi) for fi in range(n))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        si, fi = self.items[i]
        return self.ds.load_frame(si, fi), si, fi


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute fused backbone features")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--override", action="append", default=[], metavar="K=V")
    args = ap.parse_args()

    cfg = load_config(args.config, [f"data.mode=frames", *args.override])
    device = resolve_device(cfg.get("run.device", "auto"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = get_logger("vidtag.precompute")

    ds, _ = build_dataset(cfg, args.split)
    if not hasattr(ds, "load_frame") or not hasattr(ds, "sequence_length"):
        raise TypeError(
            f"{type(ds).__name__} does not expose load_frame/sequence_length; "
            "precompute supports per-frame access (see data/msls.py)"
        )
    flat = _FlatFrames(ds)
    loader = DataLoader(flat, batch_size=args.batch_size, num_workers=args.num_workers)
    logger.info("%s: %d sequences, %d frames", args.split, len(ds), len(flat))

    enc = DualFrameEncoder(
        cfg.get("model.clip_name", "openai/clip-vit-large-patch14"),
        cfg.get("model.dino_name", "facebook/dinov2-large"),
    ).to(device)

    bufs = {
        si: np.zeros((ds.sequence_length(si), 1792), dtype=np.float16)
        for si in range(len(ds))
    }
    done, t0 = 0, time.time()
    with torch.no_grad():
        for frames, sis, fis in loader:
            feats = enc(frames.to(device)).cpu().numpy().astype(np.float16)
            for f, si, fi in zip(feats, sis.tolist(), fis.tolist()):
                bufs[si][fi] = f
            done += len(feats)
            if done % (args.batch_size * 50) < args.batch_size:
                logger.info("%d/%d (%.1f img/s)", done, len(flat), done / (time.time() - t0))

    rows = []
    for si, arr in bufs.items():
        sid = ds.seq_id(si)
        np.save(out / f"{sid}.npy", arr)
        rows.append({"seq_id": sid, "n_frames": len(arr)})
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    logger.info("done: %d sequences -> %s", len(rows), out)


if __name__ == "__main__":
    main()
