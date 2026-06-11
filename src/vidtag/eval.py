"""End-to-end evaluation (paper Fig. 5, §4.2-4.3, suppl. D-E).

Builds (or loads) the uniform-grid gallery from TRAIN coordinates, embeds it
with the trained location encoder, runs initial frame->GPS retrieval, refines
with GeoRefiner, re-retrieves, and reports all paper metrics for both stages.

  PYTHONPATH=src python -m vidtag.eval --config configs/msls_phase2_smoke.yaml \
      --ckpt runs/msls_smoke/phase2_latest.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data.gallery import build_uniform_grid, load_gallery, save_gallery
from .metrics import evaluate_sequences
from .train.common import build_dataset, build_loader, build_model
from .utils import get_logger, load_checkpoint, resolve_device, set_seed


def gallery_from_cfg(cfg, logger) -> np.ndarray:
    """Build/load the retrieval gallery. cfg.eval.gallery_source:
      train_grid   (default) — uniform grid over train coords (suppl. D.1)
      val_coords   — GT coordinates of the val split (suppl. D.2, Table 11)
      city_centers — the CityGuessr 166 city centers (§5.3 city-level task)
    """
    source = cfg.get("eval.gallery_source", "train_grid")
    if source == "city_centers":
        centers = pd.read_csv(Path(cfg.data.cityguessr_root) / "meta" / "city_centers.csv")
        logger.info("gallery = %d city centers (city-level protocol)", len(centers))
        return centers[["lat", "lon"]].to_numpy()
    if source == "val_coords":
        ds, _ = build_dataset(cfg, "val")
        coords = np.concatenate([ds[i]["coords"].numpy() for i in range(len(ds))])
        grid = np.unique(np.round(coords, 6), axis=0)
        logger.info("gallery = %d unique val GT coords (suppl. D.2 mode)", len(grid))
        return grid

    gpath = Path(cfg.eval.gallery_path)
    if gpath.exists():
        grid = load_gallery(str(gpath))
        logger.info("loaded gallery %s (%d points)", gpath, len(grid))
        return grid
    coords_by_region: dict[str, np.ndarray] = {}
    if cfg.data.kind == "msls":
        df = pd.read_csv(cfg.data.train_index)
        cities = cfg.get("data.train_cities")
        if cities:
            df = df[df.city.isin(list(cities))]
        for city, g in df.groupby("city"):
            coords_by_region[city] = g[["lat", "lon"]].to_numpy()
    elif cfg.data.kind == "synthetic":
        # Synthetic trajectories live at unrelated random locations per split,
        # so cover BOTH splits' regions; blind retrieval is not the point of
        # the synthetic smoke (the MSLS run is the real generalization test).
        for split_name in ("train", "val"):
            ds, _ = build_dataset(cfg, split_name)
            for i in range(len(ds)):
                coords_by_region[f"{split_name}_v{i}"] = ds[i]["coords"].numpy()
    else:
        ds, _ = build_dataset(cfg, "train")
        all_coords = np.concatenate([ds[i]["coords"].numpy() for i in range(len(ds))])
        coords_by_region["all"] = all_coords
    grid = build_uniform_grid(coords_by_region, cfg.eval.resolution_km,
                              padding_deg=cfg.get("eval.padding_deg", 0.02))
    gpath.parent.mkdir(parents=True, exist_ok=True)
    save_gallery(str(gpath), grid)
    logger.info("built gallery: %d points at %.2f km", len(grid), cfg.eval.resolution_km)
    return grid


@torch.no_grad()
def embed_gallery(model, grid: np.ndarray, device, chunk: int) -> torch.Tensor:
    parts = []
    for i in range(0, len(grid), chunk):
        coords = torch.tensor(grid[i : i + chunk], dtype=torch.float32, device=device)
        parts.append(model.encode_gps(coords))
    return torch.cat(parts)


@torch.no_grad()
def run_eval(cfg, ckpt: str, refine: bool = True, limit: int | None = None,
             split: str = "val") -> dict:
    set_seed(cfg.get("run.seed", 0))
    device = resolve_device(cfg.get("run.device", "auto"))
    run_dir = Path(cfg.run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("vidtag.eval", str(run_dir / "eval.log"))

    model = build_model(cfg, device)
    state = load_checkpoint(ckpt, model, strict=False)
    model.eval()
    logger.info("loaded %s", ckpt)
    if refine and state.get("extra", {}).get("phase") == "phase1":
        logger.warning(
            "checkpoint %s is PHASE 1 — the 'refined' stage below uses an "
            "UNTRAINED GeoRefiner; pass --no-refine for meaningful numbers", ckpt
        )

    grid = gallery_from_cfg(cfg, logger)
    gallery_coords = torch.tensor(grid, dtype=torch.float32, device=device)
    gallery_emb = embed_gallery(model, grid, device, cfg.get("eval.gallery_chunk", 8192))

    ds, collate = build_dataset(cfg, split)
    loader = build_loader(cfg, ds, collate, device, train=False)
    logger.info("%s sequences: %d", split, len(ds))

    initial_preds, refined_preds, gts, seq_keys = [], [], [], []
    seen = 0
    for batch in loader:
        coords = batch["coords"].to(device)
        mask = batch.get("key_padding_mask")
        mask = mask.to(device) if mask is not None else None
        if "fused" in batch:
            fused = batch["fused"].to(device)
        else:
            fused = model.encode_frames_raw(batch["frames"].to(device))
        out = model.predict(fused, gallery_coords, gallery_emb, mask, refine=refine)
        n_frames = batch["n_frames"]
        for b in range(coords.shape[0]):
            n = int(n_frames[b])
            gts.append(coords[b, :n].cpu().numpy())
            seq_keys.append(batch["seq_key"][b])
            initial_preds.append(out["initial_coords"][b, :n].cpu().numpy())
            if refine:
                refined_preds.append(out["refined_coords"][b, :n].cpu().numpy())
        seen += coords.shape[0]
        if limit and seen >= limit:
            break

    results = {"initial": evaluate_sequences(initial_preds, gts)}
    if refine:
        results["refined"] = evaluate_sequences(refined_preds, gts)

    if cfg.data.kind == "cityguessr":
        # City-level protocol (paper §5.3, Table 3): per-frame nearest city
        # center, per-video majority vote, hierarchy accuracy (GUESSES #31).
        from .data.cityguessr import _video_city
        from .metrics import haversine_km, hierarchy_accuracy

        meta = Path(cfg.data.cityguessr_root) / "meta"
        centers = pd.read_csv(meta / "city_centers.csv")
        labels = pd.read_csv(meta / "labels_list.csv")
        center_xy = centers[["lat", "lon"]].to_numpy()

        def to_cities(per_video_preds):
            out = []
            for arr in per_video_preds:
                d = haversine_km(arr[:, None, :], center_xy[None, :, :])
                votes = centers.city.to_numpy()[d.argmin(axis=1)]
                vals, counts = np.unique(votes, return_counts=True)
                out.append(vals[counts.argmax()])
            return out

        gt_cities = [_video_city(k) for k in seq_keys]
        stages = [("initial", initial_preds)]
        if refine:
            stages.append(("refined", refined_preds))
        for stage, preds in stages:
            results[stage].update(hierarchy_accuracy(to_cities(preds), gt_cities, labels))

    cols = list(results["initial"].keys())
    logger.info("%-9s " + " ".join(f"{c:>14}" for c in cols), "stage")
    for stage, m in results.items():
        logger.info("%-9s " + " ".join(f"{m[c]:14.3f}" for c in cols), stage)

    out_json = cfg.get("run.out_json", str(run_dir / "eval_results.json"))
    with open(out_json, "w") as f:
        json.dump({"ckpt": ckpt, "gallery_points": int(len(grid)), **results}, f, indent=2)
    logger.info("results -> %s", out_json)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="VidTAG end-to-end evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--override", action="append", default=[], metavar="K=V")
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="max val sequences")
    args = p.parse_args()
    cfg = load_config(args.config, args.override)
    run_eval(cfg, args.ckpt, refine=not args.no_refine, limit=args.limit)


if __name__ == "__main__":
    main()
