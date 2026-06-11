#!/usr/bin/env bash
# Download all VidTAG datasets + weights that have public sources.
# Usage: bash scripts/download_datasets.sh /data/PaperRepro
# Resumable: re-running continues partial downloads (aria2c -c).
set -euo pipefail

ROOT="${1:?usage: download_datasets.sh <data-root>}"
command -v aria2c >/dev/null || { echo "install aria2 first (apt install aria2)"; exit 1; }

mkdir -p "$ROOT"/datasets/{msls,gama,cityguessr68k,bdd100k} "$ROOT"/weights/geoclip

echo "== MSLS (~57GB, public HF mirror matching official MD5s) =="
for f in msls_checksums.md5 metadata.zip patch_v1.1.zip \
         images_vol_1.zip images_vol_2.zip images_vol_3.zip \
         images_vol_4.zip images_vol_5.zip images_vol_6.zip; do
  aria2c -c -x8 -s8 --file-allocation=none -d "$ROOT/datasets/msls" -o "$f" \
    "https://huggingface.co/datasets/deansmile123/msls/resolve/main/$f"
done
(cd "$ROOT/datasets/msls" && md5sum -c <(grep -v sample.zip msls_checksums.md5)) \
  || { echo "MSLS checksum mismatch!"; exit 1; }

echo "== GeoCLIP weights =="
for f in image_encoder_mlp_weights.pth location_encoder_weights.pth logit_scale_weights.pth; do
  aria2c -c -d "$ROOT/weights/geoclip" -o "$f" \
    "https://raw.githubusercontent.com/VicenteVivan/geo-clip/main/geoclip/model/weights/$f"
done

echo "== CLIP ViT-L/14 + DINOv2-L (HF cache; needs huggingface_hub) =="
python -c "from huggingface_hub import snapshot_download as d; \
d('openai/clip-vit-large-patch14'); d('facebook/dinov2-large')"

echo "== GAMa aerial images + video lists (88GB, UCF) =="
aria2c -c -x8 -s8 --check-certificate=false --file-allocation=none \
  -d "$ROOT/datasets/gama" "https://www.crcv.ucf.edu/data1/GAMa/GAMa_dataset-zstd.sq"

echo "== CityGuessr68k (374GB total, UCF) =="
for f in CityGuessr68k-meta_files.sq CityGuessr68k-ac.sq CityGuessr68k-dk.sq \
         CityGuessr68k-lo.sq CityGuessr68k-pz.sq; do
  aria2c -c -x8 -s8 --check-certificate=false --file-allocation=none \
    -d "$ROOT/datasets/cityguessr68k" "https://www.crcv.ucf.edu/data1/CityGuessr68k/$f"
done

echo "== BDD100k GPS info (5GB; public but slow server) =="
aria2c -c -x4 -s4 --file-allocation=none -d "$ROOT/datasets/bdd100k" \
  "http://dl.yf.io/bdd100k/bdd100k_info.zip"

cat <<'EOF'

== MANUAL STEP: BDD100k videos (1.8TB, needed for GAMa training) ==
The only fast source requires a free account:
  1. Register at https://bdd-data.berkeley.edu
  2. Download "Video Parts" / bdd100k_videos.zip
  3. Unzip into <root>/datasets/gama/videos/{train,val}/  (.mov files)
Everything else in the GAMa pipeline (GPS info, splits, aerial) is automated.
EOF
echo "done."
