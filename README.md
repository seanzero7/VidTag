# VidTAG — Reproduction

Unofficial PyTorch reproduction of **VidTAG: Temporally Aligned Video to GPS
Geolocalization with Denoising Sequence Prediction at a Global Scale**
(Kulkarni, Gupta, Chhipa, Shah — arXiv:2604.12159). The authors did not
release code; this implementation is built from the paper + supplementary.

- [SPEC.md](SPEC.md) — the full implementation spec extracted from the paper.
- [GUESSES.md](GUESSES.md) — **every decision the paper leaves unspecified**
  (30+ entries), what we chose, and why. Read this before tuning anything.
- [docs/NIGHT_REPORT.md](docs/NIGHT_REPORT.md) — build/validation narrative.

## Status & TODO

**Done** (validated on Apple Silicon, 2026-06-11):

- [x] Full pipeline implemented, 56 unit tests, 3-agent adversarial review
- [x] End-to-end proofs on real data (see table below) — incl. GeoRefiner
      gains reproducing in direction at scale
- [x] All data acquired & verified on the transport drive (`PaperRepro/`):
      MSLS (MD5-checked), CityGuessr68k, GAMa lists + BDD GPS info, GeoCLIP
      weights, **and the CLIP/DINOv2 HF caches (offline-load verified)** —
      the drive is fully self-contained; no internet needed for weights
- [x] Docker image defined (x86 + DGX Spark/aarch64 bases)

**TODO on the DGX Spark** (in order; commands in the recipe below):

- [ ] Clone this repo; `docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3 -t vidtag .`
- [ ] Copy the drive: `rsync -ah --progress /mnt/<drive>/PaperRepro/ /data/PaperRepro/`
      (one tree: datasets + weights + hf_cache)
- [ ] Prep MSLS (extract remaining cities + build index — Step 1 commands)
- [ ] `pytest tests/ -q` + synthetic smoke in the container
- [ ] MSLS: precompute → Phase I (600 ep) → Phase II (100 ep) → eval
      → compare against Table 1 targets
- [ ] CityGuessr68k: extract archives → precompute → train → city-level eval
- [ ] (deferred) GAMa — requires the BDD100k videos (Berkeley account)
- [ ] If metrics fall short: work GUESSES.md top-down (#9 GeoRefiner width
      first, then #8 CE direction)

## Architecture at a glance

```
                 ┌─ CLIP ViT-L/14 (frozen) ── 768-d ─┐
video frames ────┤                                    ├─ concat → 1792-d, L2-norm
(16 × 224×224)   └─ DINOv2 ViT-L/14 (frozen) ─ 1024-d ┘        │
                                                               ▼
                                       TempGeo: +temporal pos-emb, 2-layer
                                       pre-norm transformer (d=1792, FFN 2400)
                                                               │
                                       MLP 1792→1024→768→512 (Mish) → frame emb
                                                               │
 (lat,lon) ── EqualEarth → RFF σ={2⁰,2³,2⁸} → per-σ MLPs ──────┤ Phase I:
              (GeoCLIP-init location encoder, 512-d)           │ contrastive
                                                               ▼ CE(V·Gᵀ, I)
 Phase II (everything above frozen):
   GT coords → noise (10% collapse / jitter / shift) → location encoder
        └→ GeoRefiner (1-enc/2-dec transformer, d=512, no causal mask)
           cross-attends GPS queries to frame embeddings → refined GPS emb
           trained with weighted Hinge loss (α=10 off-diag, β=1 diag)

 Inference: frame emb → nearest gallery GPS → embed prediction → GeoRefiner
            → second (GPS-to-GPS) retrieval → final per-frame coordinates.
 Gallery  : uniform grid over train coords per city (0.1 km MSLS / 0.5 GAMa).
```

## What's verified (proof-of-concept on Apple Silicon / MPS)

| Check | Result |
|---|---|
| Synthetic end-to-end smoke (`scripts/smoke_synthetic.py`) | PASS (6/6 stages) |
| Unit tests (`pytest tests/`) | 55 passed |
| MSLS Phase I (3-city subset, 30 epochs, cached features) | loss 6.07 → 3.14 |
| MSLS Phase II (GeoRefiner, 15 epochs) | loss ~1.9 → 1.24 |
| Full eval on real MSLS val (gallery 59k pts @0.1km, two-stage) | initial median 2.2 km, 99.2% @25km |
| GAMa pipeline (real release lists: 45,029/3,103 videos; real BDD GPS; full trainer path) | PASS |
| CityGuessr68k real-frame training (124 release videos, 30 epochs) | loss 5.26 → 4.44 |
| 6-city / 300+60-epoch run: **GeoRefiner improves every metric** (median 3.47→2.76 km, @25km 83.0→91.5) | direction matches paper |
| Paper-scale probes: batch 128×16 (0.18s/step, 1.2GB MPS); full 24-city gallery = 2.02M pts (4.1GB) | PASS |

(For scale: the paper's full-training MSLS numbers are 41.0% @1km, median
1.35 km — reached with 22 cities × 600 epochs on an RTX A6000. The smoke
runs above exist to prove correctness, not to claim numbers.)

## Repository layout

```
src/vidtag/
  models/        # DualFrameEncoder (CLIP+DINOv2), TempGeo, LocationEncoder,
                 # GeoRefiner (+GPSNoiser), VidTAG assembly (vidtag.py)
  data/          # msls.py, gama.py, cityguessr.py, gallery.py, transforms,
                 # sequences (16-frame sampling), synthetic (smoke data)
  train/         # phase1.py, phase2.py, precompute.py, common.py
  losses.py      # Phase-I contrastive (Eq.1), Phase-II weighted hinge (Eq.2-5)
  metrics.py     # acc@{0.5,1,5,25}km, median, video metrics (E.1), DFD, MRD
  eval.py        # gallery -> retrieval -> GeoRefiner -> retrieval -> metrics
configs/         # *_smoke (Mac/MPS proofs) and *_full (paper-scale, Linux)
scripts/         # dataset download/extract/index/geocode + smoke test
data_static/     # geocoded CityGuessr city centers (committed)
tests/           # unit tests, no datasets required
```

## Assumptions not explicit in the paper

[GUESSES.md](GUESSES.md) is the complete registry; these are the ones most
likely to matter if your results differ from the paper's:

| # | Assumption | Basis |
|---|---|---|
| 4 | **TempGeo FFN width = 2400** (not the transformer-default 4×d=7168) | Reverse-engineered from the paper's own Table 8 trainable-parameter budget (56.3M; 4×d would give 90.5M). Configurable as `model.tempgeo_ff`. |
| 24 | **Train/val split is within-city** (10% of each city's sequences to val) | The paper builds its gallery from *train* coords yet reports 97.9% @25km on val — impossible with held-out cities. The 10% fraction is our choice. |
| 9/10 | **GeoRefiner width = 512**, fed by post-MLP frame embeddings | "Width matches TempGeo" can't be literal (1792) without extra projections; 512 is the only dimension that "integrates cleanly" with GPS embeddings and gallery retrieval. Fallback: try d=1792 + in/out projections. |
| 28b | **Eq. 2 reads G′Gᵀ (refined vs GT)**, not the printed G′G′ᵀ | The printed form has identically-1 diagonal for unit vectors (zero loss); §3.4 says refined embeddings align with GT embeddings. |
| 7/8 | Learnable logit scale (CLIP-style, GeoCLIP init), symmetric CE | Unspecified; GeoCLIP lineage. One-directional CE is a config flag. |
| 1 | CLIP feature = 768-d *projected* image embedding | Suppl. B.2 gives 768-d, which only the projection layer produces for ViT-L/14. |
| 20 | 16-frame sampling = one index per disjoint integer stride cell | "16 frames were sampled" is all the paper says; this preserves trajectory coverage, varies across epochs, and guarantees unique frames. |
| 15 | Video-level embedding = mean over frame embeddings, re-normalized | "Organized at the video level" is all we get. |
| 13/14 | Phase-II noise: jitter + per-axis shift combined in the 90% branch; collapse picks a uniformly random own point | Suppl. A describes them sequentially; collapse target unstated. |
| 27 | CityGuessr city centers = Nominatim-geocoded (166 cities, committed CSV) | §5.3 says "city center GPS coordinates", source unstated. |
| 16/17 | LR: per-epoch ×0.99/×0.95 decay, linear 1000-step warmup | "StepLR + 1000 steps warmup" without cadence; per-step decay would kill the LR inside one epoch. |
| 30 | DFD/MRD computed in kilometers | Units unstated; the paper's magnitudes (DFD≈3.9, MRD≈1.1) only make sense in km. |

Also worth knowing (verified empirically, not a guess): **GeoCLIP
initialization of the location encoder is required.** A random-init RFF
encoder is not geographically smooth, and gallery retrieval degenerates to
near-random (8,805 km vs 2.5 km median in our controlled test).

## Replication guide

### Step 0 — Environment

```bash
git clone <this repo> && cd vidtag-repro
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # torch w/ CUDA, transformers, av, ...
pytest tests/ -q                         # 55 tests, no datasets needed
```

Device selection is automatic (`cuda` → `mps` → `cpu`). On CUDA you can add
bf16 autocast / `torch.compile` in `train/common.py` (left off by default
for MPS parity). All commands below assume `PYTHONPATH=src` (or `pip install
-e .` equivalent of your choosing).

### Step 1 — Data (~2.5TB total with BDD videos)

Prereqs: `apt install aria2 squashfs-tools unzip` (squashfs-tools ≥ 4.4 for
zstd archives). Everything with a public source is one command (resumable —
re-run it after any interruption):

```bash
bash scripts/download_datasets.sh /data/PaperRepro
```

**Alternative: seed from the prepared drive.** A `PaperRepro/` tree on a
drive (as prepared for this project) holds everything the script downloads
*plus* the CLIP/DINOv2 HF caches (offline-load verified) — copy the whole
tree and no downloads are needed at all:

```bash
# adjust the mount point; on Linux an NTFS drive mounts via the kernel ntfs3 driver
rsync -ah --progress /mnt/8TBExternal/PaperRepro/ /data/PaperRepro/
```

Either way, continue with the preparation commands below (they skip
already-extracted files, so partial trees are fine).

| Dataset | Source | Auth |
|---|---|---|
| MSLS (~57GB) | HF mirror `deansmile123/msls` (verified against official MD5s) | none |
| GeoCLIP weights | github.com/VicenteVivan/geo-clip | none |
| CLIP ViT-L/14, DINOv2-L | HuggingFace | none |
| GAMa aerial+lists (88GB) | crcv.ucf.edu/data1/GAMa | none |
| CityGuessr68k (374GB) | crcv.ucf.edu/data1/CityGuessr68k | none |
| BDD100k GPS info (5GB) | dl.yf.io/bdd100k (slow server, be patient) | none |
| **BDD100k videos (1.8TB)** | **bdd-data.berkeley.edu — register (free), download `bdd100k_videos.zip`, unzip into `datasets/gama/videos/{train,val}/`. Only needed for GAMa training; there is no usable public mirror.** | account |

Then prepare each dataset (all verified against the real releases):

```bash
ROOT=/data/PaperRepro   # same root you passed to download_datasets.sh

# MSLS: extract, index (within-city 90/10 split — see GUESSES.md #24)
python scripts/extract_msls.py --zips-dir $ROOT/datasets/msls --out $ROOT/datasets/msls/extracted
PYTHONPATH=src python scripts/build_msls_index.py \
    --msls-root $ROOT/datasets/msls/extracted --out-dir $ROOT/datasets/msls/index
# expect: "train: ~1.25M frames, ~16.6k sequences | val: ~140k frames, ~2.5k sequences"

# GAMa: GPS info + split lists
unzip $ROOT/datasets/bdd100k/bdd100k_info.zip -d $ROOT/datasets/bdd100k/info_extracted
mkdir -p $ROOT/datasets/gama/splits
ln -s $ROOT/datasets/bdd100k/info_extracted $ROOT/datasets/gama/info
unsquashfs -d $ROOT/datasets/gama/gama_meta $ROOT/datasets/gama/GAMa_dataset-zstd.sq "list" "Readme.txt"
cp $ROOT/datasets/gama/gama_meta/list/train.list       $ROOT/datasets/gama/splits/train.txt  # 45,029 videos
cp $ROOT/datasets/gama/gama_meta/list/val_day_vid.list $ROOT/datasets/gama/splits/val.txt    # 3,103 videos

# CityGuessr68k: meta + frame folders (<split>/<City>/<video_id>/<n>.jpg)
unsquashfs -d $ROOT/datasets/cityguessr68k/meta $ROOT/datasets/cityguessr68k/CityGuessr68k-meta_files.sq
cp data_static/cityguessr_city_centers.csv $ROOT/datasets/cityguessr68k/meta/city_centers.csv
for f in ac dk lo pz; do
  unsquashfs -f -d $ROOT/datasets/cityguessr68k/videos $ROOT/datasets/cityguessr68k/CityGuessr68k-$f.sq
done
```

### Step 2 — Sanity proof on your machine (~5 min)

```bash
sed -i "s#/Volumes/8TBExternal/PaperRepro#$ROOT#" configs/synthetic_smoke.yaml
PYTHONPATH=src python scripts/smoke_synthetic.py     # must print SMOKE PASSED
```

### Step 3 — Precompute backbone features (MSLS / CityGuessr)

The backbones are frozen, so CLIP+DINOv2 features are computed once
(paper suppl. I: turns a 68h train into ~3h):

```bash
for split in train val; do
  PYTHONPATH=src python -m vidtag.train.precompute \
      --config configs/msls_phase1_full.yaml --split $split \
      --out $ROOT/datasets/msls/features/$split
done
```

### Step 4 — Train (paper settings live in the configs)

```bash
# Phase I: contrastive (600 epochs, bs128, lr 5e-5, StepLR 0.99/epoch, warmup 1k)
PYTHONPATH=src python -m vidtag.train.phase1 --config configs/msls_phase1_full.yaml

# Phase II: GeoRefiner (100 epochs, lr 1e-4, StepLR 0.95/epoch)
PYTHONPATH=src python -m vidtag.train.phase2 --config configs/msls_phase2_full.yaml
```

What to expect while training: per-step JSONL metrics in
`runs/msls_full/phase{1,2}_metrics.jsonl` (loss, lr, logit_scale), epoch
summaries in `phase{1,2}.log`, checkpoints every N epochs plus
`phase{1,2}_latest.pt`. Resume any run with `--resume
runs/msls_full/phase1_latest.pt`. Override anything without editing configs:
`--override train.batch_size=256 --override run.dir=runs/exp2`.

Phase-I loss should fall from ~ln(batch·16) and grind toward ~2-3 over
hundreds of epochs (within-video frames are near-duplicates at GPS scale, so
it never approaches 0 — that's expected, see GUESSES #20).

### Step 5 — Evaluate

```bash
PYTHONPATH=src python -m vidtag.eval --config configs/msls_phase2_full.yaml \
    --ckpt runs/msls_full/phase2_latest.pt          # add --no-refine to skip GeoRefiner
```

First run builds + saves the uniform-grid gallery (0.1 km over all train
coords; takes a while, cached at `eval.gallery_path` after). Prints a
two-row table (initial vs refined) with all paper metrics and writes
`runs/msls_full/eval_results.json`. Paper targets (Table 1, frame-wise):

| | @0.5km | @1km | @5km | @25km | median | DFD | MRD |
|---|---|---|---|---|---|---|---|
| GeoCLIP-FT baseline | 8.3 | 22.5 | 63.0 | 93.9 | 2.97 | 22.52 | 2.82 |
| VidTAG (paper) | 21.5 | 41.0 | 76.7 | 97.9 | 1.35 | 3.87 | 1.07 |

### Step 6 — GAMa / CityGuessr68k

Same commands with `configs/gama_*.yaml` / `configs/cityguessr_*.yaml`
(paper: 100 epochs, LR decay 0.95, everything else identical). The
precompute fast path supports MSLS and CityGuessr (frame folders); GAMa
trains in `data.mode: frames` (decodes .mov via PyAV). Pre-build their eval
galleries per the comments in those configs — auto-building one bbox over
all of BDD100k is intractable.

One-shot wrapper for the whole MSLS flow (precompute → both phases → eval,
resumable): `bash scripts/train_msls_full.sh /data/PaperRepro`.

### Inference on a new video (script)

```bash
PYTHONPATH=src python scripts/predict.py \
    --ckpt runs/msls_full/phase2_latest.pt \
    --gallery runs/msls_full/gallery_train_grid.npy \
    --video clip.mp4 --sample-fps 1 \      # or --frames-dir /path/to/frames
    --out trajectory.csv
```

Rebuilds the model from the checkpoint's embedded config, runs the two-stage
retrieval, writes `frame,lat,lon,initial_lat,initial_lon` per row.

### Using a trained model in Python

```python
import torch
from vidtag.models import VidTAG
from vidtag.utils import load_checkpoint, resolve_device
from vidtag.data.transforms import frames_to_tensor, load_image
from vidtag.data.gallery import load_gallery

device = resolve_device()
model = VidTAG(with_backbones=True).to(device).eval()
load_checkpoint("runs/msls_full/phase2_latest.pt", model, strict=False)

frames = frames_to_tensor([load_image(p) for p in frame_paths])  # (T,3,224,224)
grid = load_gallery("runs/msls_full/gallery_train_grid.npy")
gallery_coords = torch.tensor(grid, dtype=torch.float32, device=device)
gallery_emb = model.encode_gps(gallery_coords)                   # chunk if huge

with torch.no_grad():
    fused = model.encode_frames_raw(frames[None].to(device))
    out = model.predict(fused, gallery_coords, gallery_emb)
print(out["refined_coords"][0])   # (T, 2) lat/lon trajectory
```

## The recipe: reproducing the paper's accuracy

Everything below assumes data is prepared per Step 1 and lives at
`/data/PaperRepro`. Change the paths at the top of each `configs/*_full.yaml`
(or pass `--override`), then run the commands — nothing else to edit. Times
are projections for a Blackwell-class GPU from the paper's A6000 figures
(suppl. I) and our measured throughputs; treat as ballpark.

| Step | Command (after `export PYTHONPATH=src`) | Est. time | Target (paper) |
|---|---|---|---|
| MSLS precompute (×2 splits) | `python -m vidtag.train.precompute --config configs/msls_phase1_full.yaml --split {train,val} --out /data/PaperRepro/datasets/msls/features/{train,val}` | 1–3 h | — |
| **MSLS Phase I** — 600 epochs, bs 128, lr 5e-5, decay 0.99 | `python -m vidtag.train.phase1 --config configs/msls_phase1_full.yaml` | 1–3 h | loss plateaus ~2 |
| **MSLS Phase II** — 100 epochs, lr 1e-4, decay 0.95 | `python -m vidtag.train.phase2 --config configs/msls_phase2_full.yaml` | <1 h | refined > initial |
| **MSLS eval** — 0.1 km grid (~2M pts) | `python -m vidtag.eval --config configs/msls_phase2_full.yaml --ckpt runs/msls_full/phase2_latest.pt` | ~0.5 h | **21.5 / 41.0 / 76.7 / 97.9 @ 0.5/1/5/25 km, median 1.35, DFD 3.87, MRD 1.07** |
| GAMa Phase I+II — 100+100 epochs, decay 0.95 (needs BDD videos; frames mode) | same commands with `configs/gama_phase{1,2}_full.yaml` | 1–2 days (video decode dominates) | **35.4 / 53.1 / 77.8 / 94.4, median 0.88** (0.5 km gallery) |
| CityGuessr precompute + Phase I+II — 100+100 epochs, decay 0.95 | `precompute --config configs/cityguessr_phase1_full.yaml ...` then phase1/phase2 | 4–8 h precompute, then ~2 h | — |
| CityGuessr city-level eval | `python -m vidtag.eval --config configs/cityguessr_phase2_full.yaml --ckpt ... --override eval.gallery_source=city_centers` | minutes | **94.9 / 95.5 / 96.8 / 98.5 City/State/Country/Continent** |

Notes that matter for hitting the numbers:

- **Order:** MSLS first — it's the paper's primary benchmark, the fastest to
  train (precomputed features), and validates your setup before the
  video-heavy datasets.
- **Don't skip GeoCLIP init** (`model.geoclip_init: true` in the Phase-I
  configs) — verified to be the difference between working retrieval and
  random-globe predictions.
- Phase II always loads `train.phase1_ckpt` — train phases in order.
- Eval galleries: MSLS auto-builds (cached to `eval.gallery_path`);
  GAMa/CityGuessr should be pre-built per the comments in their configs.
- If results fall short, work through [GUESSES.md](GUESSES.md) top-down —
  GeoRefiner width 1792 (#9) and one-directional CE (#8) are the first two
  ablations to try.
- Suppl. D.2 sanity mode (`--override eval.gallery_source=val_coords`)
  should score notably higher than the blind grid (paper Table 11); if it
  doesn't, the model — not the gallery — is the problem.

## Docker

The repo ships a CUDA [Dockerfile](Dockerfile) (Blackwell-ready:
torch 2.7 + cu128 carries sm_120 kernels):

```bash
docker build -t vidtag .

# data prep (downloads resume across runs)
docker run --rm -v /data/PaperRepro:/data/PaperRepro vidtag \
    bash scripts/download_datasets.sh /data/PaperRepro

# full MSLS pipeline
docker run --gpus all --ipc=host --shm-size=16g \
    -v /data/PaperRepro:/data/PaperRepro \
    -v $PWD/runs:/workspace/runs \
    vidtag bash scripts/train_msls_full.sh /data/PaperRepro
```

**Running data directly off the external drive:** no config edits needed —
mount the drive's tree AS `/data/PaperRepro`:

```bash
docker run --gpus all --ipc=host --shm-size=16g -it \
    -v /media/honeywell/8TBExternal/PaperRepro:/data/PaperRepro \
    -v $PWD/runs:/workspace/runs \
    vidtag
```

(Check the host mount driver first: `mount | grep 8TB` — if it says `fuseblk`
the drive auto-mounted via slow FUSE ntfs-3g; remount with the kernel driver:
`sudo umount /media/honeywell/8TBExternal && sudo mount -t ntfs3 /dev/sdX1 /media/honeywell/8TBExternal`.)

**DGX Spark / Grace (aarch64):** build with the NGC base instead — the
default PyTorch Docker images are x86-only:

```bash
docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3 -t vidtag .
```

**No external storage required.** All paths are config values; running
`bash scripts/download_datasets.sh /data/PaperRepro` on the box itself
rebuilds the full data tree from public sources (plus the manual BDD100k
videos download). Disk budget if everything lives on internal NVMe, with
archives deleted after extraction: MSLS ~70GB (extracted + features),
CityGuessr ~430GB extracted, GAMa lists ~1GB (the 90GB aerial archive is
NOT needed by this pipeline — only its `list/` directory), BDD100k videos
~1.8TB (the dominant item; defer it until the GAMa phase). MSLS-only ≈
**70GB**; everything except BDD videos ≈ **510GB**; absolutely everything ≈
**2.4TB** — plan accordingly on a 4TB Spark (keep zip + extracted
simultaneously only if you have headroom).

Container notes:
- `--ipc=host --shm-size=16g` is required — DataLoader workers use shared
  memory (the default 64MB shm will crash `num_workers: 12`).
- `HF_HOME` is set to `/data/PaperRepro/hf_cache` inside the image so model
  weights persist on the mounted volume. Keep that volume on a **Linux
  filesystem (ext4/xfs)**; HF's file locks misbehave on network/odd mounts.
  Mounting the NTFS data drive via the kernel `ntfs3` driver is fine.
- Mount `runs/` (checkpoints/logs) to the host or they die with the container.
- Config paths (`/data/PaperRepro/...`) match the mount point above — change
  both together if you relocate.

## Known deviations / open items

- The within-city 90/10 split fraction is our choice (GUESSES #24); the
  paper's exact split is unpublished.
- The smoke runs show GeoRefiner not yet beating initial retrieval —
  expected at 15 epochs on 3 cities; the paper's gains come from full-scale
  Phase II. If they don't materialize at full scale, revisit GUESSES #9-15
  (GeoRefiner width 1792 is the first alternative to try).
- GAMa requires the BDD100k videos from the official source (free
  registration). All GPS/list plumbing is verified against the real release.
- Trained-checkpoint compatibility: checkpoints embed their config; loading
  with mismatched `model.*` dimensions fails loudly (by design).

## License

MIT — see [LICENSE](LICENSE). The datasets and the VidTAG paper have their
own licenses/terms; respect them (MSLS is CC-BY-NC-SA; BDD100k and
CityGuessr68k have academic-use terms).
