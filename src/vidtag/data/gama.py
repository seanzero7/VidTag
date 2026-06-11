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
        # Some BDD100k rides have no GPS fixes at all — drop them up front
        # (interpolate_gps would crash mid-epoch otherwise).
        kept = []
        for vid in self.video_ids:
            p = info_dir / f"{vid}.json"
            if p.exists() and json.loads(p.read_text()).get("locations"):
                kept.append(vid)
        if len(kept) < len(self.video_ids):
            print(f"GamaSequences: dropped {len(self.video_ids) - len(kept)} videos "
                  f"without GPS fixes ({split})", flush=True)
        self.video_ids = kept
        if mode == "features" and self.features_dir is None:
            raise ValueError("features mode requires features_dir")

    def __len__(self) -> int:
        return len(self.video_ids)

    def _frame_count(self, info: dict) -> int:
        dur_ms = info["endTime"] - info["startTime"]
        return max(int(dur_ms / 1000.0 * self.fps), 1)

    def _coords_for(self, info: dict, sel: np.ndarray) -> torch.Tensor:
        frame_times = info["startTime"] + sel / self.fps * 1000.0
        return torch.from_numpy(interpolate_gps(info, frame_times).astype(np.float32))

    def __getitem__(self, idx: int):
        vid = self.video_ids[idx]
        info = json.loads((self.info_dir / f"{vid}.json").read_text())

        if self.mode == "features":
            # Frame count comes from the feature file itself — duration-based
            # estimates over/undershoot the real decoded count (BDD videos
            # are often a few frames short of duration x fps).
            feats = np.load(self.features_dir / f"{vid}.npy")
            sel = sample_indices(len(feats), self.frames_per_seq, train=self.train, rng=self.rng)
            return {
                "fused": torch.from_numpy(feats[sel].astype(np.float32)),
                "coords": self._coords_for(info, sel),
                "video_id": idx,
                "seq_key": vid,
                "n_frames": len(sel),
            }

        n_est = self._frame_count(info)
        sel = sample_indices(n_est, self.frames_per_seq, train=self.train, rng=self.rng)
        frames, sel = self._decode_frames(vid, sel)
        return {
            "frames": frames,
            "coords": self._coords_for(info, sel),  # coords match DECODED indices
            "video_id": idx,
            "seq_key": vid,
            "n_frames": len(sel),
        }

    def _decode_frames(self, vid: str, sel: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Decode requested frame indices; returns (frames, actual_indices).

        If the container holds fewer frames than estimated, out-of-range
        requests are clamped to the last decoded frame and the returned index
        array reflects that, so GPS interpolation stays aligned with what was
        actually decoded (no frame/GPS mislabel).
        """
        import av
        from .transforms import frames_to_tensor

        path = self.root / "videos" / self.split / f"{vid}.mov"
        wanted = set(int(i) for i in sel.tolist())
        images = {}
        last_i = -1
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                last_i = i
                if i in wanted:
                    images[i] = frame.to_image()
                    if len(images) == len(wanted) and i >= max(wanted):
                        break
        if last_i < 0:
            raise RuntimeError(f"no frames decoded from {path}")
        if last_i not in images and any(i > last_i for i in wanted):
            # re-decode is wasteful; grab the last frame on a quick second pass
            with av.open(str(path)) as container:
                for i, frame in enumerate(container.decode(container.streams.video[0])):
                    if i == last_i:
                        images[last_i] = frame.to_image()
                        break
        actual = np.minimum(sel, last_i)
        pil = [images[int(min(i, last_i))] for i in sel.tolist()]
        return frames_to_tensor(pil, self.image_size), actual
