"""Dual Frame Encoder (paper §3.1): frozen CLIP ViT-L/14 + frozen DINOv2
ViT-L/14, fused by concatenation into a 1792-d unit-normalized embedding.

Inputs are raw [0, 1] RGB tensors resized to 224x224; each backbone applies
its own normalization statistics on-device (GUESSES.md #21). Because both
backbones are frozen, features can be precomputed once per dataset and
cached (suppl. I) — see vidtag.train.precompute.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, Dinov2Model

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_DIM = 768
DINO_DIM = 1024
FUSED_DIM = CLIP_DIM + DINO_DIM  # 1792


class DualFrameEncoder(nn.Module):
    def __init__(
        self,
        clip_name: str = "openai/clip-vit-large-patch14",
        dino_name: str = "facebook/dinov2-large",
    ):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_name)
        self.dino = Dinov2Model.from_pretrained(dino_name)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.register_buffer(
            "clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True):  # noqa: D401 - keep backbones in eval
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (N, 3, 224, 224) raw RGB in [0, 1].

        Returns fused, L2-normalized embeddings (N, 1792):
        [CLIP projected image embed (768) || DINOv2 CLS token (1024)].
        """
        clip_in = (frames - self.clip_mean) / self.clip_std
        dino_in = (frames - self.in_mean) / self.in_std
        f_clip = self.clip(pixel_values=clip_in).image_embeds  # (N, 768)
        f_dino = self.dino(pixel_values=dino_in).last_hidden_state[:, 0]  # (N, 1024)
        return F.normalize(torch.cat([f_clip, f_dino], dim=-1), dim=-1)
