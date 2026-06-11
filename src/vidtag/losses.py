"""Training losses (paper §3.5, suppl. B.1).

Phase I:  contrastive CE between the frame/GPS similarity matrix and the
          identity (Eq. 1), with a learnable logit scale (GeoCLIP lineage)
          and optionally symmetric direction (GUESSES.md #7, #8).
Phase II: weighted Hinge loss (Eqs. 2-5) with frame-wise and video-wise
          components; alpha=10 (off-diagonal/negatives), beta=1
          (diagonal/positives).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def phase1_contrastive(
    frame_emb: torch.Tensor,
    gps_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    symmetric: bool = True,
) -> torch.Tensor:
    """frame_emb, gps_emb: (N, D), both L2-normalized; row i is a positive pair.

    L = CE(scale * V G^T, I)  [averaged with the transposed direction when
    symmetric=True, as in CLIP/GeoCLIP].
    """
    logits = logit_scale.exp() * frame_emb @ gps_emb.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.T, labels))
    return loss


def _weighted_hinge_component(
    sim: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    """mean-of-squared-error decomposition of (sim - I) into off-diagonal
    triangles (weight alpha) and diagonal (weight beta), per Eqs. 3-4."""
    n = sim.shape[0]
    eye = torch.eye(n, device=sim.device, dtype=sim.dtype)
    m = (sim - eye) ** 2  # elementwise squared error vs identity
    if n > 1:
        triu = torch.triu(torch.ones_like(m, dtype=torch.bool), diagonal=1)
        tril = torch.tril(torch.ones_like(m, dtype=torch.bool), diagonal=-1)
        neg = m[triu].mean() + m[tril].mean()
    else:
        neg = m.new_zeros(())
    pos = torch.diagonal(m).mean()
    return alpha * neg + beta * pos


def phase2_weighted_hinge(
    refined: torch.Tensor,
    gt: torch.Tensor,
    video_ids: torch.Tensor,
    alpha: float = 10.0,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """refined G' and gt G: (N, D) L2-normalized frame-level embeddings, where
    rows correspond (refined_i should match gt_i). video_ids: (N,) integer id
    of the video each frame belongs to (used for the video-wise term).

    Returns dict with 'loss', 'loss_frame', 'loss_video'.
    """
    sim_f = refined @ gt.T
    loss_f = _weighted_hinge_component(sim_f, alpha, beta)

    # Video-level: mean-pool each video's frames, re-normalize (GUESSES #15).
    uniq = video_ids.unique()
    idx = torch.searchsorted(uniq, video_ids)
    counts = torch.bincount(idx, minlength=len(uniq)).clamp(min=1).unsqueeze(1)
    agg_r = torch.zeros(len(uniq), refined.shape[1], device=refined.device, dtype=refined.dtype)
    agg_g = torch.zeros_like(agg_r)
    agg_r.index_add_(0, idx, refined)
    agg_g.index_add_(0, idx, gt)
    seq_r = F.normalize(agg_r / counts, dim=-1)
    seq_g = F.normalize(agg_g / counts, dim=-1)
    sim_v = seq_r @ seq_g.T
    loss_v = _weighted_hinge_component(sim_v, alpha, beta)

    return {"loss": loss_f + loss_v, "loss_frame": loss_f, "loss_video": loss_v}
