# GUESSES.md — Decisions the paper does not specify

Every entry: what was unspecified → what we chose → why. These are the knobs
to revisit on the Blackwell box if results fall short of the paper's.

## Model

1. **CLIP feature used** → projected class embedding (768-d) from
   `openai/clip-vit-large-patch14`. *Why:* suppl. B.2 says CLIP output is
   768-d; ViT-L/14's transformer width is 1024, so they must mean the
   projected image embedding (CLIP's retrieval-space output, standard
   practice).
2. **Order of normalize/concat** → take each backbone's class embedding raw,
   concatenate, then L2-normalize the fused 1792-d vector once. *Why:* paper
   says "fused, unit-normalised"; normalizing the fusion (not each half)
   matches that wording.
3. **Temporal positional embedding type** → learned embedding table (max T =
   512), added before TempGeo. *Why:* unspecified; learned is the default in
   modern encoders and the max-length is small. Sinusoidal is the fallback if
   sequences longer than the table appear.
4. **TempGeo heads / FFN size / dropout** → 8 heads (1792 = 8×224),
   **FFN = 2400**, dropout 0.1. *Why:* heads/dropout unspecified (transformer
   defaults), but the FFN width is pinned by the paper's own Table 8
   parameter budget: 56.3M trainable params for the 3-layer-MLP model (and
   55.7M for 2-layer — delta 0.6M matches our MLP shapes exactly). FFN=2400
   reproduces both rows to 0.1M counting TempGeo + MLP + location encoder;
   the transformer-default 4×d=7168 would give 90.5M and is excluded.
   Configurable as `model.tempgeo_ff`.
5. **Final frame embedding normalization** → L2-normalize the 512-d MLP
   output. *Why:* needed for "inner products = cosine similarity" in the
   loss; GeoCLIP does the same.
6. **GPS embedding normalization** → L2-normalize location-encoder outputs
   for the similarity matrices. *Why:* paper says inner products are cosine
   "up to the GPS embedding scale", and GeoCLIP normalizes both sides;
   keeping both normalized makes the CE temperature meaningful.
7. **Contrastive temperature** → learnable logit scale initialized to
   1/0.07, clamped at 100, exactly like CLIP/GeoCLIP (we load GeoCLIP's
   released `logit_scale_weights.pth` as init). *Why:* unspecified; GeoCLIP
   lineage.
8. **CE direction** → symmetric: `(CE(S, I) + CE(Sᵀ, I))/2`. *Why:*
   unspecified; CLIP/GeoCLIP convention. (Eq. 1 literally is one direction —
   flag to ablate: one-directional `CE(VGᵀ, I)`.)
9. **GeoRefiner width** → 512 (= GPS/frame embedding dim), 8 heads,
   FFN 2048, dropout 0.1, pre-norm. *Why:* the paper's "width matches
   TempGeo" cannot be literal (TempGeo is 1792-wide; GPS embeddings are
   512-d and the refined output must live in the 512-d gallery space for
   GPS-to-GPS retrieval; matching at 512 needs no extra projections =
   "integrates cleanly"). We read "TempGeo" there as the TempGeo *stack
   output* (post-MLP). If results disappoint: try d=1792 with in/out
   projections.
10. **GeoRefiner encoder input** → the 512-d post-MLP frame embeddings.
    *Why:* same reasoning as 9; Fig. 3 shows "Video Frame Encoder + TempGeo"
    (the full frozen stack) feeding GeoRefiner.
11. **Location encoder internals** → copied from released GeoCLIP code:
    EEP scale constants, RFF dim 512 per branch, per-branch MLP
    (Linear 1024→1024, ReLU ×4 hidden? — we mirror the actual GeoCLIP
    architecture from their repo), output 512. σ₁ changed 2⁴→2³ per Table 9;
    that branch's RFF projection is re-initialized (shape differs only in
    the random Fourier matrix scale, weights transfer).
12. **Collapse-to-point choice** → collapse to a uniformly chosen point of
    the sequence. *Why:* unspecified; sequence midpoint/first frame are
    alternatives; uniform random avoids bias.
13. **Shift noise application** → one `U(-0.2, 0.2)` draw *per coordinate
    axis* (lat and lon shifts drawn independently), same for all frames of
    the sequence. *Why:* "entire sequence is then shifted with an added
    noise sampled from a uniform distribution between -0.2 and 0.2" — a 2-D
    shift needs two numbers; independent draws are the natural reading.
14. **Are jitter and shift combined?** → yes: in the 90% non-collapse branch,
    apply per-frame jitter and the sequence-wide shift. *Why:* suppl. A
    describes them sequentially ("…jitter is added… The entire sequence is
    then shifted…").
15. **Video-level embedding for L_v** → mean over the video's frame
    embeddings, then L2-normalized. *Why:* "organized at the video level" is
    all we get; mean-pool is the standard sequence aggregate.

## Training

16. **StepLR step granularity** → decay applied once per epoch. *Why:*
    "StepLR with decay 0.99/0.95" and epoch counts (600/100) make per-epoch
    the only sensible cadence (per-step 0.99 would kill the LR in <1 epoch).
17. **Warmup shape** → linear 0→base LR over the first 1000 optimizer steps,
    then StepLR takes over. *Why:* unspecified beyond "1000 steps of warmup".
18. **Weight decay** → 0. *Why:* paper says Adam (not AdamW) with no decay
    mentioned.
19. **Adam betas/eps** → PyTorch defaults (0.9, 0.999), 1e-8.
20. **Frame sampling for 16-frame training sequences** → 16 disjoint integer
    stride cells over the sequence; train draws one uniform index per cell,
    eval takes cell centers — indices are unique by construction. *Why:*
    unspecified ("16 frames were sampled"); stride cells preserve trajectory
    coverage with per-epoch variety, and disjoint integer cells avoid
    duplicate (frame, GPS) rows acting as false negatives in the
    contrastive loss (naive fractional cells duplicate on 29% of MSLS train
    sequences).
21. **Image preprocessing** → resize 224×224 (paper), then each backbone's
    own normalization stats (CLIP mean/std for CLIP; ImageNet stats for
    DINOv2). Resize is non-aspect-preserving direct 224×224 (paper says
    "resized to 224x224"). No augmentation. *Why:* none mentioned; frozen
    backbones make augmentation less critical.
22. **Gradient clipping** → none (not mentioned). Config flag exists.
23. **Phase II "GT GPS embeddings" target** → embeddings of the *clean* GT
    coordinates from the frozen location encoder. *Why:* §3.4: refined
    embeddings "are aligned with the ground-truth embeddings g_t".

## Data

24. **MSLS train/val split "following [21]"** → WITHIN-city split: 10% of
    each city's sequences (deterministic hash) go to val, across all 24
    train_val cities. *Why:* the paper's uniform-grid gallery is built from
    train coordinates only (suppl. D.1) yet val accuracy reaches 97.9% @25km
    (Table 1) — impossible under a held-out-city split (no gallery coverage
    in val cities). A within-city sequence split is the only reading
    consistent with both, and matches CityGuessr's per-city design. The 10%
    fraction is our choice (unstated).
25. **Sequence definition in MSLS** → group images by the dataset's
    `sequence_key`, ordered by capture time. Sequences <16 frames are
    dropped from training (paper: train split has ≥16 frames per video).
26. **Gallery regions & padding** → regions = MSLS cities (train split);
    padding = 0.02° on each side; outlier drop = 0.5th/99.5th percentile
    clip per city. *Why:* suppl. D.1 leaves "padding" and "outliers"
    unquantified.
27. **CityGuessr68k city-center coordinates** → geocoded city centers for
    the 166 cities (we derive each city's center from the dataset's own
    metadata if present, else a static geocoding table committed to the
    repo). *Why:* §5.3 says "assigning the city center GPS coordinates";
    source unspecified.
28. **GAMa GPS per sampled frame** → BDD100k info JSON gives ~1 Hz
    locations; we linearly interpolate lat/lon to each sampled frame's
    timestamp. *Why:* unspecified; interpolation is standard for 1 Hz GPS +
    30 fps video.

28b. **Eq. 2 typo** → the paper prints `M_f = MSE(G'G'ᵀ, I)` (refined vs
    refined), whose diagonal is identically 1 for unit vectors (zero loss,
    nothing to learn). We implement `G'Gᵀ` (refined vs ground truth),
    matching §3.4's "aligned with the ground-truth embeddings g_t" and the
    cross-form video term in the same equation.
29. **GeoRefiner positional information** → learned positional embeddings
    added to both the encoder input (frames) and decoder queries (GPS
    tokens). *Why:* unspecified; cross-attention alignment between two
    ordered sequences needs position; learned tables match TempGeo's choice.
30. **MRD/DFD units** → kilometers (lat/lon ranges converted via local
    metric scale; DFD over haversine distances). *Why:* unspecified, but the
    paper's MRD≈1.07 / DFD≈3.87 magnitudes only make sense in km.
31. **CityGuessr city-level protocol details** → gallery = the 166 city
    centers; per-frame nearest center, per-video majority vote → predicted
    city → hierarchy accuracy via labels_list.csv. *Why:* §5.3 says the task
    is adapted "by assigning the city center GPS coordinates" but doesn't
    state the gallery or the frame→video aggregation; nearest-center +
    majority vote is the natural retrieval reading.

## Scope notes for the Mac (MPS) phase

- Partial trainings only: small city subsets, few epochs — success criterion
  is monotonically decreasing loss + working end-to-end eval, not paper
  numbers.
- fp32 on MPS (bf16/`torch.compile` reserved for CUDA configs).
