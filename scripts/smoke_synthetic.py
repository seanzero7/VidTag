#!/usr/bin/env python
"""Full-stack synthetic smoke test (SPEC §10; task: prove the pipeline).

Runs on the current device (MPS on the Mac): Phase I training must reduce the
contrastive loss, Phase II must reduce the hinge loss, the gallery +
inference + metrics path must run end-to-end, and checkpoints must
round-trip. Exits non-zero on any failure.

  PYTHONPATH=src python scripts/smoke_synthetic.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vidtag.config import load_config
from vidtag.eval import run_eval
from vidtag.train.common import build_model
from vidtag.train.phase1 import run_phase1
from vidtag.train.phase2 import run_phase2
from vidtag.utils import load_checkpoint, resolve_device

CFG = Path(__file__).resolve().parent.parent / "configs" / "synthetic_smoke.yaml"


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def main() -> int:
    run_dir = Path("runs/synthetic_smoke")
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cfg = load_config(str(CFG), ["train.epochs=14", "train.warmup_steps=10"])
    device = resolve_device("auto")
    print(f"device: {device}", flush=True)
    all_ok = True

    # --- Phase I: loss must drop ---------------------------------------
    p1 = run_phase1(cfg)
    all_ok &= check(
        "phase1 loss decreases",
        p1["final_epoch_loss"] < 0.85 * p1["first_loss"],
        f"(first {p1['first_loss']:.3f} -> final epoch mean {p1['final_epoch_loss']:.3f})",
    )

    # --- Phase II: loss must drop ---------------------------------------
    cfg2 = load_config(str(CFG), [f"train.phase1_ckpt={p1['ckpt']}",
                                  "train.lr=1e-3", "train.epochs=3"])
    p2 = run_phase2(cfg2)
    all_ok &= check(
        "phase2 loss decreases",
        p2["final_epoch_loss"] < p2["first_loss"],
        f"(first {p2['first_loss']:.3f} -> final epoch mean {p2['final_epoch_loss']:.3f})",
    )

    # --- checkpoint round-trip ------------------------------------------
    m1 = build_model(cfg, device)
    load_checkpoint(p2["ckpt"], m1, strict=False)
    m1.eval()
    m2 = build_model(cfg, device)
    load_checkpoint(p2["ckpt"], m2, strict=False)
    m2.eval()
    with torch.no_grad():
        fused = torch.randn(2, 16, 1792, device=device)
        fused = torch.nn.functional.normalize(fused, dim=-1)
        coords = torch.rand(2, 16, 2, device=device) * 10
        a = m1.forward_phase2(fused, coords)
        b = m2.forward_phase2(fused, coords)
    all_ok &= check("checkpoint round-trip deterministic", torch.allclose(a, b, atol=1e-6))

    # --- end-to-end eval (gallery -> retrieval -> refine -> metrics) -----
    # Eval on the TRAIN split: with a random-init location encoder and ~100
    # optimizer steps, the fair plumbing bar is "memorizes seen trajectories"
    # (generalization is proven on MSLS with GeoCLIP init, not here).
    results = run_eval(cfg2, p2["ckpt"], refine=True, split="train")
    needed = {"frame_acc@1km", "frame_median_km", "video_acc@1km", "dfd_km", "mrd_km"}
    have = set(results["initial"]) | set(results["refined"])
    med = results["initial"]["frame_median_km"]
    all_ok &= check("eval produced all metric families", needed <= have,
                    f"(initial median {med:.2f} km)")
    all_ok &= check("retrieval memorizes seen trajectories", med < 500.0,
                    f"(median {med:.1f} km; random globe would be ~10,000 km)")
    print(json.dumps(results, indent=2)[:800], flush=True)

    print("SMOKE", "PASSED" if all_ok else "FAILED", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
