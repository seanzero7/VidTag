"""CityGuessr68k (paper §4.1, §5.3; Kulkarni et al. ECCV'24).

68k videos across 166 cities with city-level labels only. Per the paper's
retrieval adaptation, every frame of a video gets the **city-center GPS
coordinate** as its ground truth (GUESSES.md #27). City centers are geocoded
once into `city_centers.csv` (committed; scripts/geocode_cityguessr.py).

Layout expected under `cityguessr_root`:
  meta/labels_list.csv                  (city hierarchy, from meta_files.sq)
  meta/train_labels_*.txt, val_labels_*.txt  (video ids per split)
  meta/city_centers.csv                 (city, lat, lon — geocoded)
  videos/<City>_<idx>.<ext>             (extracted from the .sq archives)
Videos are ~100 frames (paper suppl. A); we sample 16.
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

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _video_city(video_id: str) -> str:
    """'Abu_Dhabi_0000' -> 'Abu_Dhabi' (city may contain underscores)."""
    return video_id.rsplit("_", 1)[0]


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
        assumed_frames: int = 100,
    ):
        self.root = Path(cityguessr_root)
        self.split = split
        self.frames_per_seq = frames_per_seq
        self.train = train
        self.mode = mode
        self.features_dir = Path(features_dir) if features_dir else None
        self.image_size = image_size
        self.assumed_frames = assumed_frames
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

    def __len__(self) -> int:
        return len(self.video_ids)

    def _find_video(self, vid: str) -> Path | None:
        for ext in VIDEO_EXTS:
            p = self.root / "videos" / f"{vid}{ext}"
            if p.exists():
                return p
        return None

    def __getitem__(self, idx: int):
        vid = self.video_ids[idx]
        lat, lon = self.centers[_video_city(vid)]

        if self.mode == "features":
            feats = np.load(self.features_dir / f"{vid}.npy")
            sel = sample_indices(len(feats), self.frames_per_seq, train=self.train, rng=self.rng)
            fused = torch.from_numpy(feats[sel].astype(np.float32))
            T = len(sel)
            item = {"fused": fused}
        else:
            frames, T = self._decode(vid)
            item = {"frames": frames}
        coords = torch.tensor([[lat, lon]] * T, dtype=torch.float32)
        item.update({"coords": coords, "video_id": idx, "seq_key": vid, "n_frames": T})
        return item

    def _decode(self, vid: str) -> tuple[torch.Tensor, int]:
        import av
        from .transforms import frames_to_tensor

        path = self._find_video(vid)
        if path is None:
            raise FileNotFoundError(f"video {vid} not found under {self.root/'videos'}")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            n = stream.frames or self.assumed_frames
            sel = set(
                int(i) for i in sample_indices(
                    n, self.frames_per_seq, train=self.train, rng=self.rng
                ).tolist()
            )
            images = {}
            for i, frame in enumerate(container.decode(stream)):
                if i in sel:
                    images[i] = frame.to_image()
                    if len(images) == len(sel):
                        break
        pil = [images[i] for i in sorted(images)]
        while len(pil) < len(sel):
            pil.append(pil[-1])
        return frames_to_tensor(pil, self.image_size), len(sel)
