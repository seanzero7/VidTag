"""Unified-model training mix (paper suppl. H, Tables 12-13).

One model trained on all three corpora. Selection rules per suppl. H:
  1. MSLS: the entire training split (it is the smallest corpus).
  2. CityGuessr68k: (a) drop cities also present in MSLS (avoids false
     negatives in the contrastive loss); (b) cap the corpus to roughly MSLS
     size by taking all sequences from cities with < 80 sequences and
     randomly sampling 80 from the rest.
  3. GAMa: random video subset matching the other corpora's size.
Training: 200 epochs at LR decay 0.97; everything else unchanged (suppl. H).

All three sub-datasets must be in cached-feature mode (precompute each
first) so batches are homogeneous. video_id is offset per source so the
video-level loss never merges videos across datasets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .msls import TRAIN_CITIES, VAL_CITIES

# MSLS city slugs -> CityGuessr-style names (for the overlap filter).
_MSLS_TO_CITYGUESSR = {
    "sf": "San_Francisco",
    "cph": "Copenhagen",
    "saopaulo": "Sao_Paulo",
}


def msls_city_names() -> set[str]:
    """MSLS train_val cities normalized to CityGuessr naming."""
    out = set()
    for c in TRAIN_CITIES + VAL_CITIES:
        out.add(_MSLS_TO_CITYGUESSR.get(c, c.title()).lower())
    return out


def select_cityguessr_ids(
    video_ids: list[str], per_city_cap: int = 80, seed: int = 0
) -> list[str]:
    """Apply suppl. H rule 2 to a CityGuessr video-id list."""
    from .cityguessr import _video_city

    msls_cities = msls_city_names()
    by_city: dict[str, list[str]] = {}
    for vid in video_ids:
        city = _video_city(vid)
        if city.lower().replace("-", "_") in msls_cities:
            continue  # rule 2a: drop MSLS-overlapping cities
        by_city.setdefault(city, []).append(vid)
    rng = np.random.default_rng(seed)
    keep: list[str] = []
    for city in sorted(by_city):
        vids = by_city[city]
        if len(vids) < per_city_cap:
            keep.extend(vids)  # rule 2c: all sequences from small cities
        else:
            keep.extend(rng.choice(vids, per_city_cap, replace=False).tolist())
    return keep


class UnifiedSequences(Dataset):
    """Concatenation of per-dataset sequence datasets with suppl.-H mixing.

    `sources` maps a name to an already-constructed dataset; CityGuessr and
    GAMa sources are subsampled here (MSLS passes through whole). Items keep
    their schema; video_id is offset per source.
    """

    def __init__(self, sources: dict[str, Dataset], seed: int = 0,
                 per_city_cap: int = 80):
        self.parts: list[tuple[str, Dataset, np.ndarray]] = []
        sizes: dict[str, int] = {}

        msls = sources.get("msls")
        if msls is None:
            raise ValueError("unified mix requires an 'msls' source (suppl. H rule 1)")
        msls_idx = np.arange(len(msls))
        sizes["msls"] = len(msls_idx)
        self.parts.append(("msls", msls, msls_idx))

        if "cityguessr" in sources:
            cg = sources["cityguessr"]
            keep_ids = set(select_cityguessr_ids(cg.video_ids, per_city_cap, seed))
            cg_idx = np.array([i for i, v in enumerate(cg.video_ids) if v in keep_ids])
            sizes["cityguessr"] = len(cg_idx)
            self.parts.append(("cityguessr", cg, cg_idx))

        if "gama" in sources:
            ga = sources["gama"]
            target = sizes["msls"]  # rule 3: corpus size equivalent to the others
            rng = np.random.default_rng(seed + 1)
            ga_idx = (np.arange(len(ga)) if len(ga) <= target
                      else np.sort(rng.choice(len(ga), target, replace=False)))
            sizes["gama"] = len(ga_idx)
            self.parts.append(("gama", ga, ga_idx))

        self.sizes = sizes
        self._offsets = np.cumsum([0] + [len(ix) for _, _, ix in self.parts])
        # video_id namespace offset per source (dataset indexes start at 0)
        self._vid_offsets = np.cumsum([0] + [len(ds) for _, ds, _ in self.parts])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, idx: int):
        part = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        name, ds, sel = self.parts[part]
        item = ds[int(sel[idx - self._offsets[part]])]
        item["video_id"] = int(item["video_id"]) + int(self._vid_offsets[part])
        item["source"] = name
        return item
