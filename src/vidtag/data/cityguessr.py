"""CityGuessr68k (paper §4.1, §5.3; Kulkarni et al. ECCV'24).

68k videos across 166 cities with city-level labels only. Per the paper's
retrieval adaptation, every frame of a video gets the **city-center GPS
coordinate** as its ground truth (GUESSES.md #27). City centers are geocoded
once into `city_centers.csv` (committed; scripts/geocode_cityguessr.py).

The released archives (CityGuessr68k-{ac,dk,lo,pz}.sq, zstd squashfs) store
each video as a folder of numbered frame JPEGs — verified against the real
release: ``<split>/<City>/<video_id>/<n>.jpg`` with n = 1..~100, not
zero-padded. Layout expected under `cityguessr_root`:

  meta/labels_list.csv                       (from meta_files.sq)
  meta/{train,val}_labels*.txt               (video ids per split)
  meta/city_centers.csv                      (committed in data_static/)
  videos/<split>/<City>/<video_id>/<n>.jpg   (extracted video archives)

A flat-video-file fallback (videos/<video_id>.mp4 etc.) is kept for
self-prepared copies.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .msls import collate_padded  # noqa: F401  (shape-identical collate)
from .sequences import sample_indices
from .transforms import frames_to_tensor, load_image

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _video_city(video_id: str) -> str:
    """'Abu_Dhabi_0000' -> 'Abu_Dhabi' (city may contain underscores)."""
    return video_id.rsplit("_", 1)[0]


def _numeric_frame_sort(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: int(p.stem))


class CityGuessrSequences(Dataset):
    def __init__(
        self,
        cityguessr_root: str | os.PathLike,
        split: str = "train",
        frames_per_seq: int = 16,
        train: bool = True,
        mode: str = "frames",
        features_dir: str | None = None,
        image_size: int = 224,
        seed: int = 0,
    ):
        self.root = Path(cityguessr_root)
        self.split = split
        self.frames_per_seq = frames_per_seq
        self.train = train
        self.mode = mode
        self.features_dir = Path(features_dir) if features_dir else None
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

        meta = self.root / "meta"
        labels = sorted(meta.glob(f"{split}_labels*.txt"))
        if not labels:
            raise FileNotFoundError(f"no {split}_labels*.txt under {meta}")
        ids = []
        with open(labels[0]) as f:
            for line in f:
                vid = line.split(",", 1)[0].strip()
                if vid:
                    ids.append(vid)
        self.video_ids = ids

        centers = pd.read_csv(meta / "city_centers.csv")
        self.centers = {r.city: (float(r.lat), float(r.lon)) for r in centers.itertuples()}
        missing = {_video_city(v) for v in ids} - set(self.centers)
        if missing:
            raise KeyError(f"cities missing from city_centers.csv: {sorted(missing)[:5]} ...")
        if mode == "features" and self.features_dir is None:
            raise ValueError("features mode requires features_dir")
        self._frame_cache: dict[str, list[Path]] = {}

    def __len__(self) -> int:
        return len(self.video_ids)

    # ------------------------------------------------------------- lookup
    def _frame_dir(self, vid: str) -> Path | None:
        city = _video_city(vid)
        for candidate in (
            self.root / "videos" / self.split / city / vid,
            self.root / "videos" / city / vid,
        ):
            if candidate.is_dir():
                return candidate
        return None

    def _frame_paths(self, vid: str) -> list[Path]:
        if vid not in self._frame_cache:
            d = self._frame_dir(vid)
            if d is None:
                raise FileNotFoundError(
                    f"video {vid}: no frame folder under {self.root/'videos'} "
                    f"(expected <split>/<City>/<vid>/<n>.jpg) and no flat file fallback"
                )
            self._frame_cache[vid] = _numeric_frame_sort(list(d.glob("*.jpg")))
        return self._frame_cache[vid]

    def _find_video_file(self, vid: str) -> Path | None:
        for ext in VIDEO_EXTS:
            p = self.root / "videos" / f"{vid}{ext}"
            if p.exists():
                return p
        return None

    # ------------------------------------------------------------ getitem
    def __getitem__(self, idx: int):
        vid = self.video_ids[idx]
        lat, lon = self.centers[_video_city(vid)]

        if self.mode == "features":
            feats = np.load(self.features_dir / f"{vid}.npy")
            sel = sample_indices(len(feats), self.frames_per_seq, train=self.train, rng=self.rng)
            item = {"fused": torch.from_numpy(feats[sel].astype(np.float32))}
            T = len(sel)
        elif self._find_video_file(vid) is not None:
            frames, T = self._decode_video_file(vid)
            item = {"frames": frames}
        else:
            paths = self._frame_paths(vid)
            sel = sample_indices(len(paths), self.frames_per_seq, train=self.train, rng=self.rng)
            images = [load_image(paths[int(i)]) for i in sel]
            item = {"frames": frames_to_tensor(images, self.image_size)}
            T = len(sel)
        coords = torch.tensor([[lat, lon]] * T, dtype=torch.float32)
        item.update({"coords": coords, "video_id": idx, "seq_key": vid, "n_frames": T})
        return item

    def _decode_video_file(self, vid: str) -> tuple[torch.Tensor, int]:
        import av

        path = self._find_video_file(vid)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            n = stream.frames or 100
            sel = sample_indices(n, self.frames_per_seq, train=self.train, rng=self.rng)
            wanted = set(int(i) for i in sel.tolist())
            images = {}
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    images[i] = frame.to_image()
                    if len(images) == len(wanted):
                        break
        pil = [images[i] for i in sorted(images)]
        while pil and len(pil) < len(wanted):
            pil.append(pil[-1])
        return frames_to_tensor(pil, self.image_size), len(pil)

    # --- per-frame access for feature precompute (train/precompute.py) ---
    def seq_id(self, idx: int) -> str:
        return self.video_ids[idx]

    def sequence_length(self, idx: int) -> int:
        return len(self._frame_paths(self.video_ids[idx]))

    def load_frame(self, idx: int, frame_idx: int) -> torch.Tensor:
        paths = self._frame_paths(self.video_ids[idx])
        return frames_to_tensor([load_image(paths[frame_idx])], self.image_size)[0]
