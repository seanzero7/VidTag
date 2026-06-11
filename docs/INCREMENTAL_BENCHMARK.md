# Incremental benchmark ladder (DGX Spark)

Validate the pipeline in cheap stages — each gate is a go/no-go so you never
burn hours of training just to find a setup problem. Written for the actual
deployment: code mounted at `/app` (flat layout — `vidtag/` package at repo
root, no `src/`), the 8TB drive mounted into the container with
`PaperRepro/` on it, `--shm-size=48g` (plenty; workers peak at a few GB).

## 0. One-time path setup

The configs ship with two stale path prefixes (`/data/PaperRepro` in the
full configs, `/Volumes/8TBExternal/PaperRepro` in the smoke configs from
Mac development). Point them all at the drive's location **inside the
container**, then verify:

```bash
DATA=/media/honeywell/8TBExternal/PaperRepro   # adjust to the CONTAINER-side path
ls $DATA/datasets/msls/index/train.csv          # must exist before proceeding

sed -i "s#/data/PaperRepro#$DATA#g; s#/Volumes/8TBExternal/PaperRepro#$DATA#g" configs/*.yaml
```

Notes for the flat layout:
- Ignore/drop any `PYTHONPATH=src` seen in older docs/scripts — with
  `vidtag/` at the repo root, running from `/app` resolves the package via
  CWD (this is why `pytest tests/ -q` passes as-is).
- Use a distinct `run.dir` per probe (`--override run.dir=...`) so
  experiments never clobber each other.
- `--override key=value` changes any config value without editing YAMLs.

## Gate 1 — GPU smoke on real data (~10 min)

The drive already carries cached features + indexes for 6 MSLS cities, so
this runs immediately:

```bash
python -m vidtag.train.phase1 --config configs/msls_phase1_smoke.yaml
python -m vidtag.train.phase2 --config configs/msls_phase2_smoke.yaml
python -m vidtag.eval --config configs/msls_phase2_smoke.yaml \
    --ckpt runs/msls_smoke/phase2_latest.pt
```

**Pass:** Phase-1 epoch-mean loss falls ~6.0 → ~3.1; eval frame median
**≤ 3 km** and @25km **≥ 95%** on the subset. (Reference: the identical run
on Apple-Silicon development hardware scored median ≈ 2.2 km, 99.2% @25km.)
If the GPU run agrees with those numbers, the whole stack is healthy.

## Gate 2 — extract + precompute full MSLS (a few hours, one-time, USB-bound)

```bash
python scripts/extract_msls.py \
    --zips-dir $DATA/datasets/msls --out $DATA/datasets/msls/extracted

python -m vidtag.train.precompute --config configs/msls_phase1_full.yaml \
    --split train --out $DATA/datasets/msls/features/train
python -m vidtag.train.precompute --config configs/msls_phase1_full.yaml \
    --split val   --out $DATA/datasets/msls/features/val
```

**Pass:** extraction completes without `BadZipFile`/CRC errors; precompute
logs end with ~**16,588** train / ~**2,537** val sequences.

## Gate 3 — short full-scale probe (~1 h) ← the cheap "is it learning?" check

```bash
python -m vidtag.train.phase1 --config configs/msls_phase1_full.yaml \
    --override train.epochs=50 --override run.dir=runs/msls_e50

python -m vidtag.eval --config configs/msls_phase1_full.yaml \
    --ckpt runs/msls_e50/phase1_latest.pt --no-refine \
    --override run.dir=runs/msls_e50 \
    --override eval.gallery_path=runs/msls_e50/gallery.npy
```

**Pass:** at only 50/600 epochs you should already be in
**GeoCLIP-FineTuned territory** (paper Table 1 baseline: 22.5% @1km,
median 2.97 km, 93.9% @25km). Far below that → STOP, do not run Gate 4;
triage with GUESSES.md (see bottom).

## Gate 4 — full Phase I (600 epochs)

```bash
python -m vidtag.train.phase1 --config configs/msls_phase1_full.yaml
```

Checkpoints land every 10 epochs (`runs/msls_full/phase1_epochXXXX.pt`).
Monitor mid-run by pointing the Gate-3 eval command at any intermediate
checkpoint (use yet another `run.dir`/`gallery_path` override) — metrics
should climb monotonically-ish with epochs. Resumable:
`--resume runs/msls_full/phase1_latest.pt`.

## Gate 5 — Phase II + final benchmark

```bash
python -m vidtag.train.phase2 --config configs/msls_phase2_full.yaml
python -m vidtag.eval --config configs/msls_phase2_full.yaml \
    --ckpt runs/msls_full/phase2_latest.pt
```

**Pass = paper Table 1 (frame-wise):**

| | @0.5km | @1km | @5km | @25km | median | DFD | MRD |
|---|---|---|---|---|---|---|---|
| GeoCLIP-FT baseline | 8.3 | 22.5 | 63.0 | 93.9 | 2.97 | 22.52 | 2.82 |
| **VidTAG target** | **21.5** | **41.0** | **76.7** | **97.9** | **1.35** | **3.87** | **1.07** |

…and **refined must beat initial** across the board (that flip was already
reproduced in direction during development: at 60 Phase-II epochs on a
6-city subset, median 3.47→2.76 km, @25km 83.0→91.5).

Sanity probe if numbers look odd: `--override eval.gallery_source=val_coords`
(suppl. D.2 mode) should score notably HIGHER than the blind grid — if it
doesn't, the model (not the gallery) is the problem.

## If a gate fails — triage order

1. **GUESSES.md #9** — GeoRefiner width: retry Phase II with
   `--override model.refiner_ff=2048` variants / the documented 1792-wide
   alternative (affects Gate 5 refined-vs-initial only).
2. **GUESSES.md #8** — CE direction:
   `--override train.symmetric_ce=false` (re-run Gate 3; cheap).
3. **GUESSES.md #4** — TempGeo FFN: `--override model.tempgeo_ff=3584`
   (re-run Gate 3).
4. Then the rest of GUESSES.md top-down.

## After MSLS: CityGuessr68k

Same ladder shape: extract the four video archives (frame folders) →
precompute → 100-epoch Phase I probe → full train → city-level eval with
`--override eval.gallery_source=city_centers`. Target (paper Table 3):
**94.9 / 95.5 / 96.8 / 98.5** City/State/Country/Continent. GAMa is
deferred until the BDD100k videos are downloaded (Berkeley account).
