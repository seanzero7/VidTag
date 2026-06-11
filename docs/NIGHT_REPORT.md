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

## Scale-up validation (second pass, 03:30–08:00)

Pushed beyond "it runs" toward "it will reproduce":

- **Full-index audit (all 24 cities):** 1,245,504 train frames / 16,588
  sequences, zero NaN/out-of-range coords, frame numbers strictly monotonic,
  every train sequence ≥16 frames. Sequence lengths 16–1093 (median 47).
- **Paper-scale gallery measured:** 0.1 km grid over all train coords =
  **2,020,621 points ≈ 4.1 GB fp32 embeddings** — comfortably fits Blackwell
  memory; retrieval uses the chunked argmax. Building + embedding it is
  minutes, not hours.
- **Paper batch geometry on MPS:** batch 128×16 Phase-I step = **0.18 s,
  1.2 GB** allocated (2048×2048 contrastive matrix). Extrapolation: full
  MSLS Phase I (600 epochs × 129 steps) ≈ 3.9 h *on this Mac* in cached-
  feature mode; the Blackwell will be far faster — consistent with the
  paper's 2.75 h precomputed-feature figure on an A6000.
- **Resume equivalence:** straight 5-epoch run vs 3+resume+2 → identical
  per-epoch losses pre-resume, final weights within 2.4e-3 (dataloader
  shuffle RNG is not checkpointed — documented, standard).
- **GAMa through the real trainer** (decode→GPS-interpolate→loss→ckpt): PASS.
- **CityGuessr real-frame training:** 124 genuine release videos, Phase-I
  loss 5.26 → 4.44 over 30 epochs (`runs/cityguessr_mini`).
- **6-city long MSLS run** (300-epoch Phase I → 60-epoch Phase II → eval,
  `runs/msls_long`) — **the paper's central claim reproduces in direction**:

  | stage | @0.5km | @1km | @5km | @25km | median |
  |---|---|---|---|---|---|
  | initial retrieval | 10.5 | 22.3 | 64.0 | 83.0 | 3.47 km |
  | after GeoRefiner | **14.9** | **26.5** | **69.0** | **91.5** | **2.76 km** |

  GeoRefiner *hurt* at 15 Phase-II epochs (first smoke) and *helps on every
  metric* at 60 — the refinement gains the paper reports emerge with
  training scale, as designed. Phase-I loss reached 2.25 (vs 3.09 at 30
  epochs); Phase-II reached 0.87 (vs 1.24).

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
