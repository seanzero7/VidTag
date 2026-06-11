"""Phase I training (paper Fig. 3a, §3.5, suppl. A).

Trains TempGeo + MLP head + location encoder (+ logit scale) contrastively;
CLIP/DINOv2 stay frozen (or are bypassed entirely in cached-feature mode).

  PYTHONPATH=src python -m vidtag.train.phase1 --config configs/msls_phase1_smoke.yaml
"""

from __future__ import annotations

import math

import torch

from ..losses import phase1_contrastive
from ..utils import WarmupStepLR, load_checkpoint
from .common import TrainContext, build_dataset, build_loader, build_model, parse_and_load

MAX_LOGIT_SCALE = math.log(100.0)


def run_phase1(cfg, resume: str | None = None) -> dict:
    ctx = TrainContext(cfg, "phase1")
    model = build_model(cfg, ctx.device)
    ds, collate = build_dataset(cfg, "train")
    loader = build_loader(cfg, ds, collate, ctx.device, train=True)
    ctx.logger.info("train sequences: %d, steps/epoch: %d", len(ds), len(loader))

    params = list(model.phase1_trainable_parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.train.lr)
    scheduler = WarmupStepLR(optimizer, cfg.train.warmup_steps, cfg.train.lr_decay)

    start_epoch = global_step = 0
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch, global_step = state["epoch"], state["step"]
        ctx.logger.info("resumed from %s at epoch %d step %d", resume, start_epoch, global_step)

    first_loss = last_loss = None
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch in loader:
            coords = batch["coords"].to(ctx.device)
            mask = batch.get("key_padding_mask")
            mask = mask.to(ctx.device) if mask is not None else None
            if "fused" in batch:
                fused = batch["fused"].to(ctx.device)
            else:
                with torch.no_grad():
                    fused = model.encode_frames_raw(batch["frames"].to(ctx.device))

            frame_emb, gps_emb = model.forward_phase1(fused, coords, mask)
            if mask is not None:  # drop PAD rows from the contrastive matrix
                keep = ~mask.flatten(0, 1)
                frame_emb, gps_emb = frame_emb[keep], gps_emb[keep]
            loss = phase1_contrastive(
                frame_emb, gps_emb, model.logit_scale,
                symmetric=cfg.train.symmetric_ce,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("train.grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            optimizer.step()
            scheduler.step_batch()
            with torch.no_grad():
                model.logit_scale.clamp_(max=MAX_LOGIT_SCALE)

            global_step += 1
            running += loss.item()
            n_batches += 1
            if first_loss is None:
                first_loss = loss.item()
            if global_step % cfg.run.log_every == 0:
                ctx.log_step(epoch=epoch, step=global_step, loss=loss.item(),
                             lr=scheduler.current_lrs[0],
                             logit_scale=model.logit_scale.exp().item())
        last_loss = running / max(n_batches, 1)
        ctx.logger.info("epoch %d: mean loss %.4f", epoch, last_loss)
        ctx.log_step(epoch=epoch, step=global_step, epoch_mean_loss=last_loss)
        scheduler.step_epoch()
        if (epoch + 1) % cfg.get("run.ckpt_every_epochs", 1) == 0 or epoch + 1 == cfg.train.epochs:
            ctx.save(model, optimizer, scheduler, epoch + 1, global_step)

    return {"first_loss": first_loss, "final_epoch_loss": last_loss,
            "ckpt": str(ctx.run_dir / "phase1_latest.pt")}


def main() -> None:
    cfg, args = parse_and_load("VidTAG Phase I contrastive training")
    run_phase1(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
