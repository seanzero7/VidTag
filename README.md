# VidTAG — Reproduction

Unofficial PyTorch reproduction of **VidTAG: Temporally Aligned Video to GPS
Geolocalization with Denoising Sequence Prediction at a Global Scale**
(Kulkarni, Gupta, Chhipa, Shah — arXiv:2604.12159). The authors did not
release code; this implementation is built from the paper + supplementary.

- [SPEC.md](SPEC.md) — the full implementation spec extracted from the paper.
- [GUESSES.md](GUESSES.md) — **every decision the paper leaves unspecified**,
  what we chose, and why. Read this before tuning anything.

## What's verified (proof-of-concept on Apple Silicon / MPS)

| Check | Result |
|---|---|
| Synthetic end-to-end smoke (`scripts/smoke_synthetic.py`) | PASS (5/5 stages) |
| Unit tests (`pytest tests/`) | 55 passed |
| MSLS Phase I (3-city subset, 30 epochs, cached features) | loss 6.07 → 3.09 |
| MSLS Phase II (GeoRefiner, 15 epochs) | loss 1.97 → 1.27 |
| Full eval on real MSLS val (gallery 59k pts @0.1km) | median 1.43 km, 35.6% @1km, 98.7% @25km |
| GAMa pipeline (real BDD GPS JSONs + video decode) | PASS |
| CityGuessr68k pipeline (real meta + city centers) | PASS |

(For scale: the paper's full-training MSLS numbers are 41.0% @1km, median
1.35 km — reached with 22 cities × 600 epochs on an RTX A6000.)

## Layout

```
src/vidtag/
  models/        # DualFrameEncoder (CLIP+DINOv2), TempGeo, LocationEncoder,
                 # GeoRefiner (+GPSNoiser), VidTAG assembly
  data/          # msls.py, gama.py, cityguessr.py, gallery.py, transforms,
                 # sequences (16-frame sampling), synthetic (smoke data)
  train/         # phase1.py, phase2.py, precompute.py, common.py
  losses.py      # Phase-I contrastive (Eq.1), Phase-II weighted hinge (Eq.2-5)
  metrics.py     # acc@{0.5,1,5,25}km, median, video metrics (E.1), DFD, MRD
  eval.py        # gallery -> retrieval -> GeoRefiner -> retrieval -> metrics
configs/         # *_smoke (Mac/MPS proofs) and *_full (paper-scale, Linux)
scripts/         # dataset download/extract/index/geocode + smoke test
data_static/     # geocoded CityGuessr city centers (committed)
```

## Setup (Linux / Blackwell)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # torch w/ CUDA, transformers, av, ...
pytest tests/ -q                         # 55+ tests, no datasets needed
```

After the GeoCLIP weights are downloaded (next section), prove the full
stack on your device:

```bash
# point the smoke config's weight paths at your data root, then run it
sed -i 's#/Volumes/8TBExternal/PaperRepro#/data/PaperRepro#' configs/synthetic_smoke.yaml
PYTHONPATH=src python scripts/smoke_synthetic.py
```

Device selection is automatic (`cuda` → `mps` → `cpu`). On CUDA you can add
bf16 autocast / `torch.compile` in `train/common.py` (left off by default for
MPS parity).

## Datasets

Prereqs: `apt install aria2 squashfs-tools unzip` (squashfs-tools ≥ 4.4 for
the zstd archives). One-stop script (downloads everything that has a public
source):

```bash
bash scripts/download_datasets.sh /data/PaperRepro
```

| Dataset | Source | Auth |
|---|---|---|
| MSLS (~57GB) | HF mirror `deansmile123/msls` (matches official MD5s) | none |
| GeoCLIP weights | github.com/VicenteVivan/geo-clip | none |
| CLIP ViT-L/14, DINOv2-L | HuggingFace | none |
| GAMa aerial+lists (88GB) | crcv.ucf.edu/data1/GAMa | none |
| CityGuessr68k (374GB) | crcv.ucf.edu/data1/CityGuessr68k | none |
| BDD100k GPS info (5GB) | dl.yf.io/bdd100k (slow) | none |
| **BDD100k videos (1.8TB)** | **bdd-data.berkeley.edu — register (free) and download `bdd100k_videos.zip`; the public mirror is unusably slow** | account |

Then:

```bash
ROOT=/data/PaperRepro   # same root you passed to download_datasets.sh

# MSLS: extract, index (within-city 90/10 split — see GUESSES.md #24)
python scripts/extract_msls.py --zips-dir $ROOT/datasets/msls --out $ROOT/datasets/msls/extracted
PYTHONPATH=src python scripts/build_msls_index.py \
    --msls-root $ROOT/datasets/msls/extracted --out-dir $ROOT/datasets/msls/index

# GAMa: GPS info + split lists
unzip $ROOT/datasets/bdd100k/bdd100k_info.zip -d $ROOT/datasets/bdd100k/info_extracted
mkdir -p $ROOT/datasets/gama
ln -s $ROOT/datasets/bdd100k/info_extracted $ROOT/datasets/gama/info  # loader wants gama/info/100k/{train,val}/*.json
unsquashfs -d $ROOT/datasets/gama/extracted $ROOT/datasets/gama/GAMa_dataset-zstd.sq
mkdir -p $ROOT/datasets/gama/splits   # copy GAMa's selected-video lists here as
# train.txt / val.txt (they ship inside the GAMa archive with the aerial data;
# without them the loader falls back to "all videos with info JSONs")

# CityGuessr68k: meta + videos (archives are zstd squashfs; loop-mount also works)
unsquashfs -d $ROOT/datasets/cityguessr68k/meta $ROOT/datasets/cityguessr68k/CityGuessr68k-meta_files.sq
cp data_static/cityguessr_city_centers.csv $ROOT/datasets/cityguessr68k/meta/city_centers.csv
for f in ac dk lo pz; do
  unsquashfs -f -d $ROOT/datasets/cityguessr68k/videos \
      $ROOT/datasets/cityguessr68k/CityGuessr68k-$f.sq
done
# loader expects flat files videos/<City>_<idx>.<ext>; if the archives hold
# per-video folders (of frames), flatten or adapt data/cityguessr.py:_find_video
```

## Training (paper-scale)

```bash
# 0) cache frozen-backbone features once (paper suppl. I: 68h -> 2.75h)
PYTHONPATH=src python -m vidtag.train.precompute \
    --config configs/msls_phase1_full.yaml --split train --out $ROOT/datasets/msls/features/train
PYTHONPATH=src python -m vidtag.train.precompute \
    --config configs/msls_phase1_full.yaml --split val --out $ROOT/datasets/msls/features/val

# 1) Phase I: contrastive (600 epochs, bs128, lr 5e-5, StepLR 0.99, warmup 1k)
PYTHONPATH=src python -m vidtag.train.phase1 --config configs/msls_phase1_full.yaml

# 2) Phase II: GeoRefiner (100 epochs, lr 1e-4, StepLR 0.95)
PYTHONPATH=src python -m vidtag.train.phase2 --config configs/msls_phase2_full.yaml

# 3) Evaluate (uniform-grid gallery @0.1km from train coords; both stages)
PYTHONPATH=src python -m vidtag.eval --config configs/msls_phase2_full.yaml \
    --ckpt runs/msls_full/phase2_latest.pt
```

Every hyperparameter lives in the YAML; any value can be overridden:
`--override train.lr=1e-4 --override train.batch_size=256`.

GAMa and CityGuessr68k train with the same phase1/phase2/eval commands using
`configs/gama_*.yaml` and `configs/cityguessr_*.yaml`, but in
`data.mode: frames` — the precompute fast path currently supports MSLS only.
(Paper: 100 epochs, LR decay 0.95 for both; everything else identical —
suppl. A.) Their eval galleries should be pre-built (see comments in those
configs); auto-building one bbox over all of BDD100k is intractable.

## Known deviations / open items

- The MSLS train/val split is a within-city 90/10 sequence split — the only
  reading consistent with the paper's gallery construction; fraction is our
  choice (GUESSES.md #24).
- GeoRefiner width = 512 (GUESSES.md #9 explains the "width matches TempGeo"
  ambiguity and the 1792-wide alternative to try if results fall short).
- The smoke runs show GeoRefiner not yet improving over initial retrieval —
  expected at 15 epochs on 3 cities; the paper's gains come from full-scale
  Phase II. If they don't materialize at full scale, revisit GUESSES #9-15.
- GAMa needs BDD100k *videos* from the official source (registration). All
  GPS/info plumbing is already proven against the real info JSONs.
