#!/usr/bin/env python
"""Build per-split MSLS sequence indexes (CSV) from extracted metadata.

Usage:
  PYTHONPATH=src python scripts/build_msls_index.py \
      --msls-root .../datasets/msls/extracted --out-dir .../datasets/msls/index
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vidtag.data.msls import build_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msls-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-len-train", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        min_len = args.min_len_train if split == "train" else None
        df = build_index(args.msls_root, split=split, min_len=min_len)
        n_seq = df.groupby(["city", "side", "sequence_key"]).ngroups
        path = out / f"{split}.csv"
        df.to_csv(path, index=False)
        print(f"{split}: {len(df)} frames, {n_seq} sequences -> {path}")


if __name__ == "__main__":
    main()
