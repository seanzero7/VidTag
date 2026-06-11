# Overnight build report — 2026-06-10 18:30 → 06-11 morning

Goal: a GitHub-ready reproduction of VidTAG (arXiv:2604.12159, no official
code) that, pulled onto a Linux/Blackwell box with the datasets, should
train to results comparable to the paper. Mac/MPS runs prove correctness,
not convergence.

## What exists now

- **Complete codebase** (`src/vidtag/`): dual frozen CLIP-L/14 + DINOv2-L
  encoder, TempGeo, GeoCLIP-faithful location encoder (loads their released
  weights), GeoRefiner + GPS noise model, both losses, both training phases,
  feature precompute, uniform-grid gallery, two-stage inference, all four
  metric families. Config-driven, device-agnostic, checkpoint/resume.
- **All proofs pass on MPS** (see README table): synthetic smoke 6/6, 55
  unit tests, real-MSLS Phase I/II training with decreasing losses, full
  two-stage eval on real validation sequences producing sane geolocalization
  (initial median ~2.2 km on a 3-city smoke subset).
- **Datasets on the external drive** (`/Volumes/8TBExternal/PaperRepro/`):
  MSLS complete + MD5-verified + indexed (16,588 train / 2,537 val seqs);
  BDD100k GPS info complete (80k videos' GPS extracted); GAMa archive
  complete with split lists staged (45,029 train / 3,103 val); CityGuessr68k
  meta + city centers done, video archives downloading; CLIP/DINOv2/GeoCLIP
  weights cached.
- **Three-agent adversarial review applied** (fidelity vs paper, correctness
  bug-hunt, fresh-clone repro): all findings fixed, most notably TempGeo
  FFN=2400 (pinned by the paper's Table 8 parameter budget — the
  transformer-default 4×d would be 90.5M trainable vs the paper's 56.3M)
  and unique 16-frame sampling (the naive version drew duplicate frames on
  29% of MSLS train sequences).

## Decisions a future reader should know

1. Every unspecified hyperparameter is in GUESSES.md with rationale (30+
   entries). The empirically-derived ones: FFN width (#4), within-city
   train/val split (#24), Eq. 2 typo reading (#28b).
2. GeoCLIP init is **required**, not optional — verified 8,805 km vs 2.5 km
   median with identical training.
3. The paper's GeoRefiner gains appear only at full scale; at smoke scale it
   slightly trails initial retrieval (expected; machinery verified).

## What's left for the desktop (Linux/Blackwell)

1. `bash scripts/download_datasets.sh /data/PaperRepro` — or carry the
   external drive over (everything already on it).
2. **BDD100k videos (1.8TB)**: register free at bdd-data.berkeley.edu and
   download `bdd100k_videos.zip` → `datasets/gama/videos/{train,val}/`.
   The only public mirror serves ~0 B/s; there is no other source.
3. Precompute MSLS features (`vidtag.train.precompute`, both splits), then
   Phase I (600 epochs) → Phase II (100) → eval, per README. GAMa/CityGuessr
   train in frames mode with their configs.
4. Open question to verify when the CityGuessr video archives finish
   extracting: whether videos are flat files or per-video folders
   (loader currently expects flat; one-line change if not).
