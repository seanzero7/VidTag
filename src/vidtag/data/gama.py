"""GAMa split of BDD100k (paper §4.1, suppl. A; GAMa = Vyas et al. ECCV'22).

Videos are BDD100k 40-second driving clips (720p, 30fps, .mov); GPS comes
from the BDD100k `info` JSONs (~1 Hz fixes with timestamps). Per the paper we
sample 16 frames per video and assign each sampled frame a GPS coordinate by
linear interpolation of the fixes at the frame's timestamp (GUESSES.md #28).

Layout expected under `gama_root`:
  info/100k/{train,val}/<video>.json     (extracted from bdd100k_info.zip)
  videos/{train,val}/<video>.mov         (BDD100k videos — desktop download)
  splits/{train,val}.txt                 (GAMa video lists; from GAMa release)
If `splits/` is absent, all videos with info JSONs are used (documented
fallback so the pipeline can be proven before the GAMa archive finishes).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .sequences import sample_indices

# collate is shape-identical to MSLS's — reuse it.
from .msls import collate_padded  # noqa: F401  (re-exported for build_dataset)


def interpolate_gps(info: dict, frame_times_ms: np.ndarray) -> np.ndarray:
    """Linearly interpolate (lat, lon) at the given absolute timestamps."""
    locs = sorted(info["locations"], key=lambda l: l["timestamp"])
    ts = np.array([l["timestamp"] for l in locs], dtype=np.float64)
    lat = np.array([l["latitude"] for l in locs], dtype=np.float64)
    lon = np.array([l["longitude"] for l in locs], dtype=np.float64)
    t = np.clip(frame_times_ms, ts[0], ts[-1])
    return np.stack([np.interp(t, ts, lat), np.interp(t, ts, lon)], axis=-1)


class GamaSequences(Dataset):
    """16-frame video sequences with interpolated per-frame GPS."""

    def __init__(
        self,
        gama_root: str | os.PathLike,
        split: str = "train",
        frames_per_seq: int = 16,
        train: bool = True,
        mode: str = "frames",
        features_dir: str | None = None,
        image_size: int = 224,
        seed: int = 0,
        fps: float = 30.0,
    ):
        self.root = Path(gama_root)
        self.split = split
        self.frames_per_seq = frames_per_seq
        self.train = train
        self.mode = mode
        self.features_dir = Path(features_dir) if features_dir else None
        self.image_size = image_size
        self.fps = fps
        self.rng = np.random.default_rng(seed)

        info_dir = self.root / "info" / "100k" / split
        split_file = self.root / "splits" / f"{split}.txt"
        if split_file.exists():
            wanted = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
            self.video_ids = [w.removesuffix(".mov").removesuffix(".json") for w in wanted]
        else:  # documented fallback: every video that has an info JSON
            self.video_ids = sorted(p.stem for p in info_dir.glob("*.json"))
        if not self.video_ids:
            raise FileNotFoundError(f"no GAMa videos found for split={split} under {self.root}")
        self.info_dir = info_dir
        if mode == "features" and self.features_dir is None:
            raise ValueError("features mode requires features_dir")

    def __len__(self) -> int:
        return len(self.video_ids)

    def _frame_count(self, info: dict) -> int:
        dur_ms = info["endTime"] - info["startTime"]
        return max(int(dur_ms / 1000.0 * self.fps), 1)

    def __getitem__(self, idx: int):
        vid = self.video_ids[idx]
        info = json.loads((self.info_dir / f"{vid}.json").read_text())
        n_frames = self._frame_count(info)
        sel = sample_indices(n_frames, self.frames_per_seq, train=self.train, rng=self.rng)
        frame_times = info["startTime"] + sel / self.fps * 1000.0
        coords = interpolate_gps(info, frame_times).astype(np.float32)

        item = {
            "coords": torch.from_numpy(coords),
            "video_id": idx,
            "seq_key": vid,
            "n_frames": len(sel),
        }
        if self.mode == "features":
            feats = np.load(self.features_dir / f"{vid}.npy")
            item["fused"] = torch.from_numpy(feats[sel].astype(np.float32))
        else:
            item["frames"] = self._decode_frames(vid, sel, n_frames)
        return item

    def _decode_frames(self, vid: str, sel: np.ndarray, n_frames: int) -> torch.Tensor:
        """Decode the selected frame indices from videos/<split>/<vid>.mov."""
        import av
        from .transforms import frames_to_tensor
        from PIL import Image

        path = self.root / "videos" / self.split / f"{vid}.mov"
        wanted = set(int(i) for i in sel.tolist())
        images: dict[int, Image.Image] = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    images[i] = frame.to_image()
                    if len(images) == len(wanted):
                        break
        pil = [images[int(i)] for i in sel.tolist() if int(i) in images]
        # Videos can be slightly shorter than endTime-startTime suggests; pad
        # by repeating the last decoded frame so shapes stay consistent.
        while len(pil) < len(sel):
            pil.append(pil[-1])
        return frames_to_tensor(pil, self.image_size)
