#!/usr/bin/env python
"""Extract MSLS from the downloaded zips (zip64-safe, resumable).

Usage:
  python scripts/extract_msls.py --zips-dir .../datasets/msls \
      --out .../datasets/msls/extracted [--cities zurich,cph,sf] [--metadata-only]

The image volumes are big; --cities extracts only matching path prefixes so a
city subset can be ready for MPS smoke training long before the full set.
Already-extracted files (matching size) are skipped, so re-running resumes.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def extract_zip(zp: Path, out: Path, cities: list[str] | None, force: bool = False) -> int:
    n = 0
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if cities and not any(c in parts for c in cities):
                continue
            dest = out / info.filename
            if not force and dest.exists() and dest.stat().st_size == info.file_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            n += 1
            if n % 2000 == 0:
                print(f"  {zp.name}: {n} files extracted", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cities", default=None, help="comma-separated city subset")
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()

    zips_dir = Path(args.zips_dir)
    out = Path(args.out)
    cities = args.cities.split(",") if args.cities else None

    zips = [zips_dir / "metadata.zip"]
    if not args.metadata_only:
        zips += sorted(zips_dir.glob("images_vol_*.zip"))
        patch = zips_dir / "patch_v1.1.zip"
        if patch.exists():
            zips.append(patch)  # patch last: overwrites corrupt v1.0 images

    total = 0
    for zp in zips:
        if not zp.exists():
            print(f"!! missing {zp}, skipping", flush=True)
            continue
        print(f"== extracting {zp.name} ==", flush=True)
        # patch files REPLACE corrupt v1.0 images of identical-looking size;
        # the resume size-check must not skip them.
        total += extract_zip(zp, out, cities, force=zp.name.startswith("patch"))
    print(f"done, {total} files extracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
