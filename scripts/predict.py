#!/usr/bin/env python
"""Standalone inference: video (or frame folder) -> per-frame GPS trajectory.

Reconstructs the model from the checkpoint's embedded config, embeds the
gallery, runs the full two-stage VidTAG inference (initial retrieval ->
GeoRefiner -> second retrieval), and writes a CSV trajectory.

  PYTHONPATH=src python scripts/predict.py \
      --ckpt runs/msls_full/phase2_latest.pt \
      --gallery runs/msls_full/gallery_train_grid.npy \
      --frames-dir /path/to/frames    # or --video clip.mp4 [--sample-fps 1] \
      --out trajectory.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vidtag.data.gallery import load_gallery
from vidtag.data.transforms import frames_to_tensor, load_image
from vidtag.models import VidTAG
from vidtag.utils import resolve_device


def load_frames_dir(d: Path, size: int) -> torch.Tensor:
    paths = sorted(d.glob("*.[jp][pn]g"), key=lambda p: (len(p.stem), p.stem))
    if not paths:
        raise FileNotFoundError(f"no .jpg/.png frames under {d}")
    return frames_to_tensor([load_image(p) for p in paths], size)


def load_video(path: Path, size: int, sample_fps: float | None) -> torch.Tensor:
    import av

    images = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate or 30.0)
        step = max(int(round(native_fps / sample_fps)), 1) if sample_fps else 1
        for i, frame in enumerate(container.decode(stream)):
            if i % step == 0:
                images.append(frame.to_image())
    if not images:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames_to_tensor(images, size)


def main() -> None:
    ap = argparse.ArgumentParser(description="VidTAG inference on one video")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gallery", required=True, help=".npy of (G,2) gallery coords")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", type=Path)
    src.add_argument("--video", type=Path)
    ap.add_argument("--sample-fps", type=float, default=None,
                    help="sample the video at this fps (default: every frame)")
    ap.add_argument("--out", default="trajectory.csv")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--gallery-chunk", type=int, default=65536)
    args = ap.parse_args()

    device = resolve_device()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = payload.get("extra", {}).get("config", {}).get("model", {})
    model = VidTAG(
        sigma=tuple(mc.get("sigma", (1, 8, 256))),
        tempgeo_layers=mc.get("tempgeo_layers", 2),
        tempgeo_heads=mc.get("tempgeo_heads", 8),
        tempgeo_ff=mc.get("tempgeo_ff", 2400),
        tempgeo_dropout=mc.get("tempgeo_dropout", 0.1),
        refiner_encoder_layers=mc.get("refiner_encoder_layers", 1),
        refiner_decoder_layers=mc.get("refiner_decoder_layers", 2),
        refiner_heads=mc.get("refiner_heads", 8),
        refiner_ff=mc.get("refiner_ff", 2048),
        refiner_dropout=mc.get("refiner_dropout", 0.1),
        max_len=mc.get("max_len", 512),
        embed_dim=mc.get("embed_dim", 512),
        with_backbones=True,
        clip_name=mc.get("clip_name", "openai/clip-vit-large-patch14"),
        dino_name=mc.get("dino_name", "facebook/dinov2-large"),
    ).to(device).eval()
    model.load_state_dict(payload["model"], strict=False)
    print(f"model from {args.ckpt} (phase: {payload.get('extra', {}).get('phase', '?')})")

    size = 224
    frames = (
        load_frames_dir(args.frames_dir, size) if args.frames_dir
        else load_video(args.video, size, args.sample_fps)
    )
    T = frames.shape[0]
    max_len = mc.get("max_len", 512)
    if T > max_len:
        print(f"warning: {T} frames > max_len {max_len}; truncating")
        frames, T = frames[:max_len], max_len
    print(f"{T} frames @ {size}x{size}")

    grid = load_gallery(args.gallery)
    gallery_coords = torch.tensor(grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        parts = [
            model.encode_gps(gallery_coords[i : i + args.gallery_chunk])
            for i in range(0, len(grid), args.gallery_chunk)
        ]
        gallery_emb = torch.cat(parts)
        fused = model.encode_frames_raw(frames[None].to(device))
        out = model.predict(fused, gallery_coords, gallery_emb,
                            refine=not args.no_refine,
                            gallery_chunk=args.gallery_chunk)

    initial = out["initial_coords"][0].cpu().numpy()
    refined = out.get("refined_coords")
    refined = refined[0].cpu().numpy() if refined is not None else None
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "lat", "lon", "initial_lat", "initial_lon"])
        for i in range(T):
            best = refined[i] if refined is not None else initial[i]
            w.writerow([i, f"{best[0]:.6f}", f"{best[1]:.6f}",
                        f"{initial[i][0]:.6f}", f"{initial[i][1]:.6f}"])
    print(f"trajectory ({T} points) -> {args.out}")
    print("first point:", initial[0] if refined is None else refined[0])


if __name__ == "__main__":
    main()
