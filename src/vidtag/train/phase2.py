"""Phase II training (paper Fig. 3b, §3.4-3.5, suppl. A).

Loads the Phase-I checkpoint, freezes everything except GeoRefiner, and
trains it to denoise GPS embeddings of corrupted GT sequences against the
frame context, with the weighted Hinge loss (alpha=10, beta=1).

  PYTHONPATH=src python -m vidtag.train.phase2 \
      --config configs/msls_phase2_smoke.yaml \
      --override train.phase1_ckpt=runs/msls_smoke/phase1_latest.pt
"""

from __future__ import annotations

import torch

from ..losses import phase2_weighted_hinge
from ..models.georefiner import GPSNoiser
from ..utils import WarmupStepLR, load_checkpoint
from .common import TrainContext, build_dataset, build_loader, build_model, parse_and_load


def run_phase2(cfg, resume: str | None = None) -> dict:
    ctx = TrainContext(cfg, "phase2")
    model = build_model(cfg, ctx.device)
    load_checkpoint(cfg.train.phase1_ckpt, model, strict=False)
    ctx.logger.info("loaded phase1 weights from %s", cfg.train.phase1_ckpt)

    # Freeze everything but GeoRefiner (paper: Phase II trains GeoRefiner only).
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.phase2_trainable_parameters():
        p.requires_grad_(True)
    model.tempgeo.eval()
    model.location_encoder.eval()

    ds, collate = build_dataset(cfg, "train")
    loader = build_loader(cfg, ds, collate, ctx.device, train=True)
    ctx.logger.info("train sequences: %d, steps/epoch: %d", len(ds), len(loader))

    noiser = GPSNoiser(
        collapse_prob=cfg.noise.collapse_prob,
        jitter_range=tuple(cfg.noise.jitter_range),
        shift_range=tuple(cfg.noise.shift_range),
    )
    params = [p for p in model.phase2_trainable_parameters()]
    optimizer = torch.optim.Adam(params, lr=cfg.train.lr)
    scheduler = WarmupStepLR(optimizer, cfg.train.warmup_steps, cfg.train.lr_decay)

    start_epoch = global_step = 0
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch, global_step = state["epoch"], state["step"]

    first_loss = last_loss = None
    for epoch in range(start_epoch, cfg.train.epochs):
        model.georefiner.train()
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

            noisy = noiser(coords, n_frames=batch.get("n_frames"))
            refined = model.forward_phase2(fused, noisy, mask)  # (B,T,512)
            with torch.no_grad():
                gt_emb = model.encode_gps(coords)  # (B,T,512)

            B, T = coords.shape[:2]
            video_ids = batch["video_id"].to(ctx.device).repeat_interleave(T)
            refined_f, gt_f = refined.flatten(0, 1), gt_emb.flatten(0, 1)
            if mask is not None:
                keep = ~mask.flatten(0, 1)
                refined_f, gt_f, video_ids = refined_f[keep], gt_f[keep], video_ids[keep]
            losses = phase2_weighted_hinge(
                refined_f, gt_f, video_ids,
                alpha=cfg.loss.alpha, beta=cfg.loss.beta,
            )
            loss = losses["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("train.grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            optimizer.step()
            scheduler.step_batch()

            global_step += 1
            running += loss.item()
            n_batches += 1
            if first_loss is None:
                first_loss = loss.item()
            if global_step % cfg.run.log_every == 0:
                ctx.log_step(epoch=epoch, step=global_step, loss=loss.item(),
                             loss_frame=losses["loss_frame"].item(),
                             loss_video=losses["loss_video"].item(),
                             lr=scheduler.current_lrs[0])
        last_loss = running / max(n_batches, 1)
        ctx.logger.info("epoch %d: mean loss %.4f", epoch, last_loss)
        ctx.log_step(epoch=epoch, step=global_step, epoch_mean_loss=last_loss)
        scheduler.step_epoch()
        if (epoch + 1) % cfg.get("run.ckpt_every_epochs", 1) == 0 or epoch + 1 == cfg.train.epochs:
            ctx.save(model, optimizer, scheduler, epoch + 1, global_step)

    return {"first_loss": first_loss, "final_epoch_loss": last_loss,
            "ckpt": str(ctx.run_dir / "phase2_latest.pt")}


def main() -> None:
    cfg, args = parse_and_load("VidTAG Phase II GeoRefiner training")
    run_phase2(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
