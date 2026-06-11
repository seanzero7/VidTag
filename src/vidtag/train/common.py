"""Shared training-loop plumbing for Phase I/II (SPEC §4, §10)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import Config, load_config
from ..models import VidTAG
from ..utils import (
    WarmupStepLR,
    get_logger,
    log_jsonl,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def cli_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    p.add_argument("--override", action="append", default=[], metavar="K=V")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    return p


def build_dataset(cfg: Config, split: str):
    kind = cfg.data.kind
    if kind == "synthetic":
        from ..data.synthetic import SyntheticSequences, collate_sequences

        ds = SyntheticSequences(
            num_videos=cfg.data.num_videos if split == "train" else max(cfg.data.num_videos // 4, 8),
            frames_per_seq=cfg.data.frames_per_seq,
            mode=cfg.data.mode,
            seed=0 if split == "train" else 1,
        )
        return ds, collate_sequences
    if kind == "msls":
        from ..data.msls import MSLSequences, collate_padded

        index_path = cfg.data.train_index if split == "train" else cfg.data.val_index
        df = pd.read_csv(index_path)
        cities = cfg.get(f"data.{split}_cities")
        if cities:
            df = df[df.city.isin(list(cities))]
        ds = MSLSequences(
            cfg.data.msls_root,
            df,
            frames_per_seq=cfg.data.frames_per_seq,
            train=(split == "train"),
            mode=cfg.data.mode,
            features_dir=cfg.get(f"data.{split}_features_dir"),
            image_size=cfg.get("data.image_size", 224),
            max_len=cfg.get("model.max_len", 512),
        )
        return ds, collate_padded
    if kind == "gama":
        from ..data.gama import GamaSequences, collate_padded

        ds = GamaSequences(
            cfg.data.gama_root,
            split=split,
            frames_per_seq=cfg.data.frames_per_seq,
            train=(split == "train"),
            mode=cfg.data.mode,
            features_dir=cfg.get(f"data.{split}_features_dir"),
        )
        return ds, collate_padded
    if kind == "cityguessr":
        from ..data.cityguessr import CityGuessrSequences, collate_padded

        ds = CityGuessrSequences(
            cfg.data.cityguessr_root,
            split=split,
            frames_per_seq=cfg.data.frames_per_seq,
            train=(split == "train"),
            mode=cfg.data.mode,
            features_dir=cfg.get(f"data.{split}_features_dir"),
        )
        return ds, collate_padded
    raise ValueError(f"unknown data.kind: {kind}")


def _worker_init(worker_id: int) -> None:
    """Give each DataLoader worker an independent sampling RNG — workers
    clone the dataset, so without this every worker draws identical frame
    jitter (verified empirically)."""
    import numpy as np

    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32 + worker_id)


def build_loader(cfg: Config, ds, collate, device: torch.device, train: bool = True) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size if train else cfg.get("eval.batch_size", 4),
        shuffle=train,
        num_workers=cfg.get("data.num_workers", 4),
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        drop_last=train,
        persistent_workers=cfg.get("data.num_workers", 4) > 0,
        worker_init_fn=_worker_init,
    )


def build_model(cfg: Config, device: torch.device) -> VidTAG:
    model = VidTAG(
        sigma=tuple(cfg.model.sigma),
        tempgeo_layers=cfg.model.tempgeo_layers,
        tempgeo_heads=cfg.model.tempgeo_heads,
        tempgeo_ff=cfg.get("model.tempgeo_ff", 2400),
        tempgeo_dropout=cfg.model.tempgeo_dropout,
        refiner_encoder_layers=cfg.model.refiner_encoder_layers,
        refiner_decoder_layers=cfg.model.refiner_decoder_layers,
        refiner_heads=cfg.model.refiner_heads,
        refiner_ff=cfg.model.refiner_ff,
        refiner_dropout=cfg.model.refiner_dropout,
        max_len=cfg.model.max_len,
        embed_dim=cfg.model.embed_dim,
        with_backbones=(cfg.data.mode == "frames"),
        clip_name=cfg.get("model.clip_name", "openai/clip-vit-large-patch14"),
        dino_name=cfg.get("model.dino_name", "facebook/dinov2-large"),
    )
    if cfg.get("model.geoclip_init", False):
        model.load_geoclip_init(
            cfg.model.geoclip_location_weights,
            cfg.get("model.geoclip_logit_scale_weights"),
        )
    return model.to(device)


class TrainContext:
    """Bundles everything a phase loop needs; created by each phase's main()."""

    def __init__(self, cfg: Config, phase: str):
        self.cfg = cfg
        self.phase = phase
        set_seed(cfg.get("run.seed", 0))
        self.device = resolve_device(cfg.get("run.device", "auto"))
        self.run_dir = Path(cfg.run.dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f"vidtag.{phase}", str(self.run_dir / f"{phase}.log"))
        self.log_path = str(self.run_dir / f"{phase}_metrics.jsonl")
        self.logger.info("device: %s", self.device)

    def log_step(self, **record):
        log_jsonl(self.log_path, {"t": time.time(), **record})

    def save(self, model, optimizer, scheduler, epoch, step, name=None):
        path = self.run_dir / (name or f"{self.phase}_epoch{epoch:04d}.pt")
        save_checkpoint(
            str(path), model, optimizer, scheduler, epoch, step,
            extra={"config": self.cfg.to_dict(), "phase": self.phase},
        )
        latest = self.run_dir / f"{self.phase}_latest.pt"
        save_checkpoint(
            str(latest), model, optimizer, scheduler, epoch, step,
            extra={"config": self.cfg.to_dict(), "phase": self.phase},
        )
        self.logger.info("checkpoint -> %s", path)


def parse_and_load(description: str) -> tuple[Config, argparse.Namespace]:
    args = cli_parser(description).parse_args()
    cfg = load_config(args.config, args.override)
    return cfg, args
