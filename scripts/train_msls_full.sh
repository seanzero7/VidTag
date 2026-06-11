#!/usr/bin/env bash
# One-shot paper-scale MSLS run: precompute -> Phase I -> Phase II -> eval.
# Usage: bash scripts/train_msls_full.sh /data/PaperRepro
# Each stage is resumable; re-running skips/continues sensibly
# (precompute overwrites per-sequence .npy files; trainers accept --resume).
set -euo pipefail

ROOT="${1:?usage: train_msls_full.sh <data-root>}"
export PYTHONPATH=src

echo "== Stage 0: precompute frozen-backbone features (paper suppl. I) =="
for split in train val; do
  python -m vidtag.train.precompute \
      --config configs/msls_phase1_full.yaml --split "$split" \
      --out "$ROOT/datasets/msls/features/$split"
done

echo "== Stage 1: Phase I contrastive (600 epochs, bs128, lr 5e-5) =="
RESUME=""
[ -f runs/msls_full/phase1_latest.pt ] && RESUME="--resume runs/msls_full/phase1_latest.pt"
python -m vidtag.train.phase1 --config configs/msls_phase1_full.yaml $RESUME

echo "== Stage 2: Phase II GeoRefiner (100 epochs, lr 1e-4) =="
RESUME=""
[ -f runs/msls_full/phase2_latest.pt ] && RESUME="--resume runs/msls_full/phase2_latest.pt"
python -m vidtag.train.phase2 --config configs/msls_phase2_full.yaml $RESUME

echo "== Stage 3: evaluate both stages on the uniform-grid gallery =="
python -m vidtag.eval --config configs/msls_phase2_full.yaml \
    --ckpt runs/msls_full/phase2_latest.pt

echo "done — results in runs/msls_full/eval_results.json"
