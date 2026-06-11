"""MSLS (Mapillary Street-Level Sequences) pipeline (paper §4.1, suppl. A).

Per the paper we ignore MSLS's query/database retrieval semantics and treat
*every* sequence (from both sides) as a video with per-frame GPS. Sequences
are groups of frames sharing `sequence_key` in seq_info.csv, ordered by
`frame_number`; panoramas are dropped; training keeps sequences with >= 16
frames. Train/val split: official MSLS city partition (GUESSES.md #24) —
train cities for training, val cities (cph, sf) for validation.

Index format (one CSV per split, built once by `build_index`): per-frame rows
  city, side, sequence_key, frame_number, key, lat, lon
sorted by (city, side, sequence_key, frame_number).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .sequences import sample_indices
from .transforms import frames_to_tensor, load_image

TRAIN_CITIES = [
    "trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
    "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
    "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman",
    "nairobi", "manila",
]
VAL_CITIES = ["cph", "sf"]


def _read_city_side(root: Path, city: str, side: str) -> pd.DataFrame | None:
    base = root / "train_val" / city / side
    seq_info_p = base / "seq_info.csv"
    if not seq_info_p.exists():
        return None
    seq = pd.read_csv(seq_info_p, index_col=0)
    post = pd.read_csv(base / "postprocessed.csv", index_col=0)
    raw = pd.read_csv(base / "raw.csv", index_col=0)

    df = seq.copy()
    # coords: postprocessed has unified lat/lon for both sides; fall back to raw
    for col in ("lat", "lon"):
        if col in post.columns:
            df[col] = post[col].values
        elif col in raw.columns:
            df[col] = raw[col].values
        else:
            raise KeyError(f"no '{col}' column in {base} csvs: {list(post.columns)} / {list(raw.columns)}")
    if "pano" in raw.columns:
        df = df[~raw["pano"].astype(bool).values]
    df["city"] = city
    df["side"] = side
    return df[["city", "side", "sequence_key", "frame_number", "key", "lat", "lon"]]


def build_index(
    msls_root: str | os.PathLike,
    split: str = "train",
    min_len: int | None = None,
    val_frac: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Scan extracted MSLS cities and build the per-frame index for a split.

    Split is WITHIN each city (val_frac of sequences per city go to val,
    deterministic by hash). The paper's uniform-grid gallery is built from
    train coordinates yet scores 97.9% @25km on val — only possible when
    every val sequence's city is covered by train data, i.e. a within-city
    split (GUESSES.md #24).
    """
    root = Path(msls_root)
    parts = []
    for city in TRAIN_CITIES + VAL_CITIES:
        for side in ("query", "database"):
            df = _read_city_side(root, city, side)
            if df is not None:
                parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no extracted MSLS cities found under {root}/train_val")
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["city", "side", "sequence_key", "frame_number"]).reset_index(drop=True)

    # Deterministic per-sequence split: stable hash of the sequence identity.
    seq_id = out.city + "_" + out.side + "_" + out.sequence_key
    bucket = pd.util.hash_array(seq_id.to_numpy()) % 1_000_003 / 1_000_003
    in_val = bucket < val_frac
    out = out[in_val if split == "val" else ~in_val].reset_index(drop=True)

    if min_len:
        sizes = out.groupby(["city", "side", "sequence_key"])["key"].transform("size")
        out = out[sizes >= min_len].reset_index(drop=True)
    return out


class MSLSequences(Dataset):
    """Sequence-level dataset over a prebuilt index.

    mode='frames':   returns frames (T,3,224,224) raw [0,1]
    mode='features': returns fused (T,1792) float32 from features_dir/<vid>.npy
    Always returns coords (T,2) float32 degrees, video_id int, seq_key str.
    """

    def __init__(
        self,
        msls_root: str | os.PathLike,
        index: pd.DataFrame | str,
        frames_per_seq: int = 16,
        train: bool = True,
        mode: str = "frames",
        features_dir: str | None = None,
        image_size: int = 224,
        max_len: int = 494,
        seed: int = 0,
    ):
        self.root = Path(msls_root)
        df = pd.read_csv(index) if isinstance(index, str) else index
        self.frames_per_seq = frames_per_seq
        self.train = train
        self.mode = mode
        self.features_dir = Path(features_dir) if features_dir else None
        self.image_size = image_size
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)

        if train:
            sizes = df.groupby(["city", "side", "sequence_key"])["key"].transform("size")
            df = df[sizes >= frames_per_seq]
        self.groups: list[pd.DataFrame] = [
            g.reset_index(drop=True)
            for _, g in df.groupby(["city", "side", "sequence_key"], sort=False)
        ]
        if mode == "features" and self.features_dir is None:
            raise ValueError("features mode requires features_dir")

    def __len__(self) -> int:
        return len(self.groups)

    def video_dir(self, g: pd.DataFrame) -> Path:
        subdir = "train_val"
        return self.root / subdir / g.city.iloc[0] / g.side.iloc[0] / "images"

    def seq_id(self, idx: int) -> str:
        g = self.groups[idx]
        return f"{g.city.iloc[0]}_{g.side.iloc[0]}_{g.sequence_key.iloc[0]}"

    # --- per-frame access for feature precompute (train/precompute.py) ---
    def sequence_length(self, idx: int) -> int:
        return min(len(self.groups[idx]), self.max_len)

    def load_frame(self, idx: int, frame_idx: int) -> torch.Tensor:
        g = self.groups[idx]
        img = load_image(self.video_dir(g) / f"{g.key.iloc[frame_idx]}.jpg")
        return frames_to_tensor([img], self.image_size)[0]

    def __getitem__(self, idx: int):
        g = self.groups[idx]
        n = min(len(g), self.max_len)
        if self.train:
            sel = sample_indices(n, self.frames_per_seq, train=True, rng=self.rng)
        else:
            sel = np.arange(n)
        rows = g.iloc[sel]
        coords = torch.tensor(rows[["lat", "lon"]].to_numpy(dtype=np.float32))
        item = {
            "coords": coords,
            "video_id": idx,
            "seq_key": self.seq_id(idx),
            "n_frames": len(sel),
        }
        if self.mode == "features":
            feats = np.load(self.features_dir / f"{self.seq_id(idx)}.npy")
            item["fused"] = torch.tensor(feats[sel].astype(np.float32))
        else:
            img_dir = self.video_dir(g)
            images = [load_image(img_dir / f"{k}.jpg") for k in rows["key"]]
            item["frames"] = frames_to_tensor(images, self.image_size)
        return item


def collate_padded(batch: list[dict]) -> dict:
    """Pad variable-length sequences; returns key_padding_mask (B,T) True=PAD."""
    T = max(b["n_frames"] for b in batch)
    out: dict = {
        "video_id": torch.tensor([b["video_id"] for b in batch]),
        "seq_key": [b["seq_key"] for b in batch],
        "n_frames": torch.tensor([b["n_frames"] for b in batch]),
    }
    mask = torch.ones(len(batch), T, dtype=torch.bool)
    for i, b in enumerate(batch):
        mask[i, : b["n_frames"]] = False
    out["key_padding_mask"] = mask

    def pad_stack(key: str) -> torch.Tensor:
        ts = []
        for b in batch:
            t = b[key]
            if t.shape[0] < T:
                pad_shape = (T - t.shape[0], *t.shape[1:])
                t = torch.cat([t, t.new_zeros(pad_shape)], dim=0)
            ts.append(t)
        return torch.stack(ts)

    for key in ("coords", "fused", "frames"):
        if key in batch[0]:
            out[key] = pad_stack(key)
    return out
