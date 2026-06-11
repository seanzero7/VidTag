"""GeoRefiner (paper §3.4) and the Phase-II GPS noise model (suppl. A).

Encoder-decoder transformer inspired by machine translation:
  * 1 encoder layer consuming temporally aligned frame embeddings (512-d,
    the frozen Phase-I stack output),
  * 2 decoder layers whose queries are GPS embeddings from the frozen
    location encoder (noised GT at train time, Phase-I predictions at
    inference). No causal mask: every GPS token sees the whole GPS sequence
    and all frames via cross-attention.

Width is 512 to match the frame/GPS embedding space (see GUESSES.md #9).
Learned positional embeddings are added to both streams (GUESSES.md #29).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoRefiner(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pos_embedding.weight, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

    def forward(
        self,
        frame_emb: torch.Tensor,
        gps_emb: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """frame_emb: (B, T, 512); gps_emb: (B, T, 512).

        Returns refined GPS embeddings (B, T, 512), L2-normalized.
        key_padding_mask: (B, T) True at PAD positions, applied to both streams.
        """
        T = frame_emb.shape[1]
        pos = self.pos_embedding(torch.arange(T, device=frame_emb.device))[None]
        memory = self.encoder(frame_emb + pos, src_key_padding_mask=key_padding_mask)
        refined = self.decoder(
            tgt=gps_emb + pos,
            memory=memory,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )
        return F.normalize(refined, dim=-1)


class GPSNoiser:
    """Corrupts GT coordinate sequences to simulate Phase-I failure modes
    (Fig. 4): collapse-to-point (p=0.1), per-frame jitter, sequence shift.

    Operates on (B, T, 2) tensors of degrees (lat, lon).
    """

    def __init__(
        self,
        collapse_prob: float = 0.10,
        jitter_range: tuple[float, float] = (0.001, 0.02),
        shift_range: tuple[float, float] = (-0.2, 0.2),
        generator: torch.Generator | None = None,
    ):
        self.collapse_prob = collapse_prob
        self.jitter_range = jitter_range
        self.shift_range = shift_range
        self.generator = generator

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        B, T, _ = coords.shape
        g = self.generator
        dev = coords.device

        # 90% branch: per-frame, per-coordinate jitter with random sign ...
        jit_lo, jit_hi = self.jitter_range
        mag = torch.empty(B, T, 2, device=dev).uniform_(jit_lo, jit_hi, generator=g)
        sign = torch.where(
            torch.rand(B, T, 2, device=dev, generator=g) < 0.5, -1.0, 1.0
        )
        # ... plus one shift per sequence per coordinate axis.
        sh_lo, sh_hi = self.shift_range
        shift = torch.empty(B, 1, 2, device=dev).uniform_(sh_lo, sh_hi, generator=g)
        noisy = coords + mag * sign + shift

        # 10% branch: collapse the whole sequence to one of its own points.
        collapse = torch.rand(B, device=dev, generator=g) < self.collapse_prob
        if collapse.any():
            idx = torch.randint(0, T, (B,), device=dev, generator=g)
            collapsed = coords[torch.arange(B, device=dev), idx][:, None, :].expand(
                B, T, 2
            )
            noisy = torch.where(collapse[:, None, None], collapsed, noisy)
        return noisy
