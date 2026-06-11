"""TempGeo (paper §3.2) + MLP projection head (suppl. B.2).

TempGeo: temporal positional embedding + 2-layer pre-norm transformer encoder
with full self-attention over the frames of one video; d_model = 1792
(preserves the fused CLIP+DINOv2 dimensionality).

MLP head: 1792 -> 1024 -> 768 -> 512 with Mish activations between layers,
producing the final per-frame embedding used in the contrastive loss and as
GeoRefiner context.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TempGeo(nn.Module):
    def __init__(
        self,
        d_model: int = 1792,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2400,
        dropout: float = 0.1,
        max_len: int = 512,
        out_dims: tuple[int, ...] = (1024, 768, 512),
    ):
        """dim_feedforward=2400 is reverse-engineered from the paper's Table 8
        trainable-parameter budget (56.3M total; 4x d_model would give 90.5M)
        — see GUESSES.md #4. It is exposed end-to-end as model.tempgeo_ff."""
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pos_embedding.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # "standard pre-normalization"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        mlp: list[nn.Module] = []
        dims = (d_model, *out_dims)
        for i in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mlp.append(nn.Mish())
        self.mlp = nn.Sequential(*mlp)

    def forward(
        self,
        z: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, T, 1792) fused unit-normalized frame embeddings.

        Returns (attended, final):
          attended: (B, T, 1792) TempGeo outputs z*_t
          final:    (B, T, 512) L2-normalized post-MLP frame embeddings
        key_padding_mask: (B, T) True at PAD positions (variable-length eval).
        """
        T = z.shape[1]
        if T > self.pos_embedding.num_embeddings:
            raise ValueError(
                f"sequence length {T} exceeds positional table "
                f"{self.pos_embedding.num_embeddings}"
            )
        pos = torch.arange(T, device=z.device)
        zhat = z + self.pos_embedding(pos)[None, :, :]
        attended = self.encoder(zhat, src_key_padding_mask=key_padding_mask)
        final = F.normalize(self.mlp(attended), dim=-1)
        return attended, final
