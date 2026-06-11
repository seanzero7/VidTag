# VidTAG Reproduction — Implementation Specification

Source: *VidTAG: Temporally Aligned Video to GPS Geolocalization with Denoising
Sequence Prediction at a Global Scale* (Kulkarni, Gupta, Chhipa, Shah —
arXiv:2604.12159v1). Code was not published; this spec is extracted from the
paper + supplementary. Anything the paper does not state is recorded in
[GUESSES.md](GUESSES.md) with rationale.

## 1. Task

Frame-to-GPS retrieval. Given a video of `T` frames, predict a GPS coordinate
for every frame so the sequence forms a temporally consistent trajectory.
Retrieval happens against a *gallery* of GPS coordinates embedded by a
location encoder — no image gallery.

## 2. Architecture

### 2.1 Dual Frame Encoder (paper §3.1, suppl. B.2)

- **CLIP ViT-L/14** (frozen): per-frame *projected* class embedding,
  `f_clip ∈ R^768`.
- **DINOv2 ViT-L/14** (frozen, `facebook/dinov2-large`): class token,
  `f_dino ∈ R^1024`.
- Fusion: concatenation `z_t = [f_clip ‖ f_dino] ∈ R^1792`, then
  **L2-normalized** (paper: "fused, unit-normalised embeddings").
- Input frames resized to **224×224** (suppl. A).

### 2.2 TempGeo (paper §3.2, suppl. A & B.2)

- Add temporal positional embedding: `ẑ_t = z_t + p_t` (type unspecified →
  GUESSES).
- **2-layer** transformer *encoder*, full (non-causal) self-attention across
  the T frames of one video, `d_model = 1792` (preserves dimensionality),
  **pre-norm** + dropout ("standard pre-normalization and dropout").
- Output: attended embeddings `z*_t ∈ R^1792`.
- **MLP head** (suppl. B.2): 3 layers, `1792 → 1024 → 768 → 512`, **Mish**
  activations between layers. Output = final 512-d frame embedding.
  (Paper counts the MLP as part of the frame-encoder stack; ablation table 8
  confirms 3 layers, ~56.3M trainable params total for the trained parts.)
- Trained in Phase I (jointly with location encoder); **frozen in Phase II**.

### 2.3 Location Encoder (paper §3.3; GeoCLIP [54])

GeoCLIP-style:
- Standardize (lat, lon) via **Equal Earth Projection (EEP)**.
- **Random Fourier Features** at 3 frequencies, `σ = [2^0, 2^3, 2^8]`
  (suppl. B.3, Table 9 best row — note this differs from GeoCLIP's default
  `[2^0, 2^4, 2^8]`).
- Each RFF branch → its own MLP; branch outputs **summed element-wise** →
  final **512-d** GPS embedding.
- Initialized from released GeoCLIP weights (`location_encoder_weights.pth`),
  then **fine-tuned contrastively in Phase I**; **frozen in Phase II**.

### 2.4 GeoRefiner (paper §3.4, suppl. A)

Encoder–decoder transformer (machine-translation style):
- **1 encoder layer, 2 decoder layers**.
- Encoder input: temporally aligned frame embeddings (the 512-d outputs of
  the frozen Phase-I frame stack) for the video.
- Decoder input (queries): GPS embeddings from the frozen Location Encoder —
  during Phase II these are embeddings of *noised* ground-truth coordinates;
  at inference they are embeddings of the Phase-I *predicted* coordinates.
- **No causal mask**: every GPS token attends to the full GPS sequence and,
  via cross-attention, to all frames.
- Width: working dim of the module is 512 (see GUESSES on the "width matches
  TempGeo" sentence).
- Output: refined GPS embeddings `g'_t ∈ R^512`.
- Only module trained in Phase II.

### 2.5 GPS Noise Model (Phase II training input; §3.4 + suppl. A)

Given a GT coordinate sequence (degrees lat/lon):
1. With probability **0.10**: collapse the entire sequence to a single point
   (use one of its points for all t → GUESSES for which point).
2. Otherwise: per-frame, per-coordinate **jitter** with magnitude
   `U(0.001, 0.02)` and a random sign, applied independently to lat and lon.
3. Plus (in the non-collapse case) a sequence-wide **shift** sampled from
   `U(-0.2, 0.2)` added to the whole sequence (per GUESSES: one draw per
   coordinate axis).
These mimic Phase-I failure modes (Fig. 4): collapse-to-point, random
scatter, shift.

## 3. Losses

### 3.1 Phase I — contrastive (Eq. 1)

For a batch: `V` = stacked attended frame embeddings (post-MLP, 512-d,
all frames of all sequences: N = batch_seqs × T rows), `G` = corresponding
GT GPS embeddings from the location encoder.

`L_contr(V, G) = CE(V Gᵀ, I)` — cross-entropy between the similarity matrix
and identity, i.e. InfoNCE with in-batch negatives; row i's positive is GPS
i. Embeddings are unit-normalized so `V Gᵀ` is cosine similarity (scaled by
a learnable temperature — GUESSES). Direction(s) of CE → GUESSES (we use the
symmetric average like CLIP/GeoCLIP).

### 3.2 Phase II — weighted Hinge loss (Eqs. 2–5)

`G'` = refined GPS embeddings (frames), `G` = GT GPS embeddings.
`G_seq, G'_seq` = the same embeddings organized at video level (per-video
aggregation → GUESSES: mean over frames, re-normalized).

- `M_f = MSE_elementwise(G' Gᵀ, I)` over all frames in batch
- `M_v = MSE_elementwise(G'_seq G_seqᵀ, I)` over videos in batch
- `L_f = α·[mean(triu(M_f)) + mean(tril(M_f))] + β·mean(diag(M_f))`
- `L_v = α·[mean(triu(M_v)) + mean(tril(M_v))] + β·mean(diag(M_v))`
- `L = L_f + L_v`, with **α = 10, β = 1** (suppl. B.1; α weights the
  negative/off-diagonal terms, β the positive/diagonal terms; triu/tril
  exclude the diagonal).

## 4. Training (suppl. A)

| Setting | Phase I | Phase II |
|---|---|---|
| Trains | location encoder, TempGeo, MLP head (backbones frozen) | GeoRefiner only |
| Epochs (MSLS) | **600** | **100** |
| Epochs (GAMa / CityGuessr68k) | 100 (lr decay 0.95) | 100 |
| LR | **5e-5** | **1e-4** |
| Optimizer | Adam | Adam |
| Scheduler | StepLR, decay **0.99** | StepLR, decay **0.95** |
| Warmup | 1000 steps | 1000 steps |
| Batch size | **128 sequences** | 128 sequences |
| Frames per sequence | **16** (sampled from longer videos) | 16 |
| Input res | 224×224 | 224×224 |
| Paper hardware | 1× RTX A6000 | same |

Efficiency note (suppl. I / Table 14): backbones are frozen, so CLIP/DINOv2
features can be **precomputed once and cached**; paper's train time drops
from 68 h to 2.75 h this way. We implement feature precompute + cached
training as the default fast path.

## 5. Datasets (§4.1, suppl. A)

### MSLS (primary)
1.6M street-level images in sequences, ~30 cities. Use **all sequences from
both query and database sets**, re-split into train/val following
CityGuessr [21]. Keep sequences with **≥16 frames** for training; val
sequences keep natural (variable) length. GPS = per-image (lat, lon) from
the MSLS metadata.

### GAMa (videos from BDD100k)
GAMa split of BDD100k: ~40 s driving videos, ~38–40 GPS points each (1 Hz).
GPS from BDD100k `info` JSONs. **16 frames sampled** per video. Train/val
per GAMa lists.

### CityGuessr68k
68k videos, 166 cities, **city-level** labels only → assign the **city-center
GPS coordinate** as GT for all frames of a video (§5.3). ~100 frames/video →
sample 16.

## 6. Gallery construction (suppl. D.1)

For evaluation (blind retrieval): per **region** (city) with sizable training
data:
1. bbox of train coordinates (drop outliers), add constant padding;
2. uniform grid at chosen resolution covering the padded bbox;
   `N_points = (dist_LAT × dist_LON) // resolution²` — grid step =
   resolution in km in both axes.
- Resolution: **0.1 km** (MSLS), **0.5 km** (GAMa). Built from *train* split
  only — model must stay blind to val coordinates.

## 7. Inference (Fig. 5, §3.4)

1. Frames → dual encoder → TempGeo → MLP → frame embeddings (512-d).
2. Gallery coords → location encoder → gallery embeddings.
3. Initial retrieval: per frame, argmax cosine similarity → predicted coords.
4. Predicted coords → location encoder → decoder queries; frame embeddings →
   GeoRefiner encoder; decoder outputs refined GPS embeddings `g'_t`.
5. **Second retrieval**: `g'_t` vs gallery embeddings (GPS-to-GPS,
   same-domain) → final per-frame coordinates.

## 8. Metrics (§4.3, suppl. E)

- **Frame-wise**: % of frames with geodesic distance to GT under
  {0.5, 1, 5, 25} km; **median distance error** (km).
- **Video-level accuracy** (suppl. E.1): centroid of GT; prediction closest
  to *prediction centroid* is the video's representative; distance →
  thresholds as above.
- **DFD** — discrete Fréchet distance between predicted and GT 2-D GPS
  sequences (classic DP implementation; distances in km).
- **MRD** — Mean Range Difference adapted to 2-D: range difference
  `|maxA−minA − (maxB−minB)|` computed separately for lat and lon, then
  averaged; mean over sequences.

## 9. Headline results to approach (Tables 1, 2)

MSLS frame-wise: 21.5 / 41.0 / 76.7 / 97.9 @ 0.5/1/5/25 km, median 1.35 km,
DFD 3.87, MRD 1.07. GAMa frame-wise: 35.4 / 53.1 / 77.8 / 94.4, median
0.88 km. (Full-scale training on the Blackwell box; Mac/MPS runs are
correctness proofs only.)

## 10. Engineering requirements (ours, not the paper's)

- Device-agnostic: `cuda` → `mps` → `cpu` auto-selection; fp32 on MPS,
  bf16 autocast + optional `torch.compile` on CUDA.
- Config-driven (YAML): all hyperparameters, paths, dataset choices.
- Precomputed-feature fast path for Phase I/II training.
- Checkpoint save/resume; deterministic seeding.
- Data roots point at `/Volumes/8TBExternal/PaperRepro/` on Mac; configurable
  for Linux.
