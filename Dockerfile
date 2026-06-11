# VidTAG reproduction — CUDA training image (Blackwell-ready).
# torch 2.7+cu128 wheels carry sm_120 (Blackwell) kernels.
#
# Build:  docker build -t vidtag .
# Run:    docker run --gpus all --ipc=host --shm-size=16g \
#             -v /data/PaperRepro:/data/PaperRepro -v $PWD/runs:/workspace/runs \
#             vidtag bash scripts/train_msls_full.sh /data/PaperRepro
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 squashfs-tools unzip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
# torch ships with the base image; install the rest
RUN grep -v "^torch" requirements.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir -r /tmp/reqs.txt

COPY . .

ENV PYTHONPATH=/workspace/src \
    HF_HOME=/data/PaperRepro/hf_cache \
    PYTHONUNBUFFERED=1

CMD ["bash"]
