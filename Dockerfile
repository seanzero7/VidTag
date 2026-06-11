# VidTAG reproduction — CUDA training image.
#
# Base image by platform:
#   x86_64 + NVIDIA GPU :  pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime (default)
#   DGX Spark / Grace (aarch64):  build with
#       docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3 -t vidtag .
#     (NGC PyTorch images are multi-arch and officially supported on DGX Spark;
#      torch is preinstalled there too.)
#
# Run:    docker run --gpus all --ipc=host --shm-size=16g \
#             -v /data/PaperRepro:/data/PaperRepro -v $PWD/runs:/workspace/runs \
#             vidtag bash scripts/train_msls_full.sh /data/PaperRepro
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 squashfs-tools unzip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
# torch ships with both supported base images; install the rest
RUN grep -v "^torch" requirements.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir -r /tmp/reqs.txt

COPY . .

ENV PYTHONPATH=/workspace/src \
    HF_HOME=/data/PaperRepro/hf_cache \
    PYTHONUNBUFFERED=1

CMD ["bash"]
