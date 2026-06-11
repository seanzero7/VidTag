#!/usr/bin/env python
"""Geocode the 166 CityGuessr68k cities into meta/city_centers.csv.

City-center coordinates are the retrieval targets per paper §5.3 (GUESSES.md
#27). Uses the public Nominatim API (1 req/s politeness delay); the output
CSV is committed to the repo so this only ever needs to run once.

  python scripts/geocode_cityguessr.py \
      --labels .../cityguessr68k/meta/labels_list.csv \
      --out .../cityguessr68k/meta/city_centers.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "vidtag-repro/0.1 (research dataset preparation)"}


def geocode(query: str) -> tuple[float, float] | None:
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1}
    )
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        hits = json.load(r)
    if not hits:
        return None
    return float(hits[0]["lat"]), float(hits[0]["lon"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labels_list.csv (City,State,Country,Continent)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.labels)))
    out_rows = []
    for i, row in enumerate(rows):
        city = row["City"]
        # underscores -> spaces; qualify with country to disambiguate
        query = f"{city.replace('_', ' ')}, {row['Country'].replace('_', ' ')}"
        loc = geocode(query) or geocode(city.replace("_", " "))
        if loc is None:
            print(f"!! no geocode hit for {city} ({query})")
            continue
        out_rows.append({"city": city, "lat": loc[0], "lon": loc[1]})
        print(f"[{i+1}/{len(rows)}] {city}: {loc[0]:.5f}, {loc[1]:.5f}", flush=True)
        time.sleep(1.05)  # Nominatim usage policy

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["city", "lat", "lon"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)}/{len(rows)} cities -> {args.out}")


if __name__ == "__main__":
    main()
