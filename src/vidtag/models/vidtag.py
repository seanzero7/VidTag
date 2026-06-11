"""Full VidTAG assembly: Phase-I model, Phase-II model, and inference.

Phase I (paper Fig. 3a): frames -> DualFrameEncoder (frozen) -> TempGeo +
MLP (trained) vs coords -> LocationEncoder (trained, GeoCLIP init);
contrastive loss with learnable logit scale.

Phase II (Fig. 3b): everything above frozen; GeoRefiner (trained) denoises
GPS embeddings of corrupted GT sequences against frame context.

Inference (Fig. 5): initial frame->gallery retrieval, refine with GeoRefiner,
second GPS->gallery retrieval.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .frame_encoder import FUSED_DIM, DualFrameEncoder
from .georefiner import GeoRefiner
from .location_encoder import LocationEncoder
from .tempgeo import TempGeo


class VidTAG(nn.Module):
    def __init__(
        self,
        sigma: tuple[float, ...] = (2**0, 2**3, 2**8),
        tempgeo_layers: int = 2,
        tempgeo_heads: int = 8,
        tempgeo_dropout: float = 0.1,
        refiner_encoder_layers: int = 1,
        refiner_decoder_layers: int = 2,
        refiner_heads: int = 8,
        refiner_ff: int = 2048,
        refiner_dropout: float = 0.1,
        max_len: int = 512,
        embed_dim: int = 512,
        with_backbones: bool = True,
        clip_name: str = "openai/clip-vit-large-patch14",
        dino_name: str = "facebook/dinov2-large",
    ):
        super().__init__()
        # Backbones are optional so cached-feature training never loads them.
        self.frame_encoder = (
            DualFrameEncoder(clip_name, dino_name) if with_backbones else None
        )
        self.tempgeo = TempGeo(
            d_model=FUSED_DIM,
            num_layers=tempgeo_layers,
            nhead=tempgeo_heads,
            dropout=tempgeo_dropout,
            max_len=max_len,
            out_dims=(1024, 768, embed_dim),
        )
        self.location_encoder = LocationEncoder(sigma=sigma)
        self.georefiner = GeoRefiner(
            d_model=embed_dim,
            num_encoder_layers=refiner_encoder_layers,
            num_decoder_layers=refiner_decoder_layers,
            nhead=refiner_heads,
            dim_feedforward=refiner_ff,
            dropout=refiner_dropout,
            max_len=max_len,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    # ---------------------------------------------------------------- init
    def load_geoclip_init(self, location_weights: str, logit_scale_weights: str | None = None):
        self.location_encoder.load_geoclip_weights(location_weights)
        if logit_scale_weights:
            scale = torch.load(logit_scale_weights, map_location="cpu", weights_only=True)
            with torch.no_grad():
                self.logit_scale.copy_(scale.reshape(()))

    # ------------------------------------------------------------- helpers
    def encode_frames_raw(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) raw RGB -> (B, T, 1792) fused backbone features."""
        if self.frame_encoder is None:
            raise RuntimeError("model built with with_backbones=False")
        B, T = frames.shape[:2]
        feats = self.frame_encoder(frames.flatten(0, 1))
        return feats.view(B, T, -1)

    def encode_sequence(
        self,
        fused: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, 1792) fused features -> (attended (B,T,1792), final (B,T,512))."""
        return self.tempgeo(fused, key_padding_mask=key_padding_mask)

    def encode_gps(self, coords: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """(..., 2) degrees -> (..., 512) GPS embeddings."""
        flat = coords.reshape(-1, 2)
        emb = self.location_encoder(flat)
        if normalize:
            emb = nn.functional.normalize(emb, dim=-1)
        return emb.view(*coords.shape[:-1], -1)

    # ------------------------------------------------------------ phase I
    def phase1_trainable_parameters(self):
        yield from self.tempgeo.parameters()
        yield from self.location_encoder.parameters()
        yield self.logit_scale

    def forward_phase1(
        self,
        fused: torch.Tensor,
        coords: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fused: (B, T, 1792); coords: (B, T, 2) degrees.

        Returns (frame_emb (B*T, 512), gps_emb (B*T, 512)), both normalized,
        ready for the contrastive loss.
        """
        _, final = self.encode_sequence(fused, key_padding_mask)
        gps = self.encode_gps(coords)
        return final.flatten(0, 1), gps.flatten(0, 1)

    # ----------------------------------------------------------- phase II
    def phase2_trainable_parameters(self):
        yield from self.georefiner.parameters()

    def forward_phase2(
        self,
        fused: torch.Tensor,
        noisy_coords: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refined GPS embeddings (B, T, 512) for noised GT coordinates."""
        with torch.no_grad():
            _, frame_emb = self.encode_sequence(fused, key_padding_mask)
            noisy_gps = self.encode_gps(noisy_coords)
        return self.georefiner(frame_emb, noisy_gps, key_padding_mask)

    # ----------------------------------------------------------- inference
    @torch.no_grad()
    def predict(
        self,
        fused: torch.Tensor,
        gallery_coords: torch.Tensor,
        gallery_emb: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        refine: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Full retrieval pipeline for one batch of sequences.

        fused: (B, T, 1792) backbone features; gallery_coords: (G, 2);
        gallery_emb: (G, 512) normalized location-encoder outputs.
        Returns dict with 'initial_coords' and 'refined_coords' (B, T, 2).
        """
        _, frame_emb = self.encode_sequence(fused, key_padding_mask)
        sims = frame_emb @ gallery_emb.T  # (B, T, G)
        init_idx = sims.argmax(dim=-1)
        initial = gallery_coords[init_idx]  # (B, T, 2)
        out = {"initial_coords": initial}
        if refine:
            pred_gps = self.encode_gps(initial)
            refined_emb = self.georefiner(frame_emb, pred_gps, key_padding_mask)
            ref_idx = (refined_emb @ gallery_emb.T).argmax(dim=-1)
            out["refined_coords"] = gallery_coords[ref_idx]
        return out
