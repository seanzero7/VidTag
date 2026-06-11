"""Unit tests for the Phase I/II losses (SPEC §3, suppl. B.1)."""

import math

import torch
import torch.nn.functional as F

from vidtag.losses import phase1_contrastive, phase2_weighted_hinge


def _orthonormal(n: int, d: int = 512) -> torch.Tensor:
    m = torch.randn(d, d)
    q, _ = torch.linalg.qr(m)
    return q[:n]


def test_phase1_aligned_pairs_give_near_zero_loss():
    v = _orthonormal(16)
    scale = torch.tensor(math.log(100.0))
    loss = phase1_contrastive(v, v, scale)
    assert loss.item() < 1e-3


def test_phase1_random_is_near_log_n():
    torch.manual_seed(0)
    v = F.normalize(torch.randn(64, 512), dim=-1)
    g = F.normalize(torch.randn(64, 512), dim=-1)
    scale = torch.tensor(0.0)  # exp(0)=1: low-temperature random logits
    loss = phase1_contrastive(v, g, scale)
    assert abs(loss.item() - math.log(64)) < 0.2


def test_phase1_symmetric_differs_on_asymmetric_logits():
    torch.manual_seed(1)
    v = F.normalize(torch.randn(8, 512), dim=-1)
    g = F.normalize(torch.randn(8, 512), dim=-1)
    scale = torch.tensor(math.log(10.0))
    sym = phase1_contrastive(v, g, scale, symmetric=True)
    one = phase1_contrastive(v, g, scale, symmetric=False)
    assert not torch.isclose(sym, one)


def test_phase2_perfect_alignment_zero_loss():
    g = _orthonormal(12)
    vids = torch.arange(12) // 4  # 3 videos x 4 frames
    out = phase2_weighted_hinge(g, g, vids)
    assert out["loss"].item() < 1e-10


def test_phase2_hand_computed_weighting():
    # 2 frames, same video. Construct sim matrix analytically:
    # refined = [e1, e2]; gt = [e1, e2] -> sim = I -> loss 0.
    # refined = [e2, e1] (swapped) -> sim = [[0,1],[1,0]]:
    #   M = (sim - I)^2 = [[1,1],[1,1]]
    #   L_f = alpha*(mean(triu)+mean(tril)) + beta*mean(diag) = a*(1+1) + b*1
    e = torch.eye(2, 512)
    refined = e[[1, 0]]
    gt = e
    vids = torch.zeros(2, dtype=torch.long)
    out = phase2_weighted_hinge(refined, gt, vids, alpha=10.0, beta=1.0)
    expected_lf = 10.0 * (1.0 + 1.0) + 1.0 * 1.0
    assert abs(out["loss_frame"].item() - expected_lf) < 1e-6
    # video-level: both videos' mean embeddings are the same vector for
    # refined and gt (e1+e2)/sqrt(2) -> 1x1 sim matrix == 1 -> M_v = 0.
    assert out["loss_video"].item() < 1e-10


def test_phase2_video_term_separates_videos():
    torch.manual_seed(0)
    g = F.normalize(torch.randn(8, 512), dim=-1)
    vids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    bad = g[torch.randperm(8)]
    out_good = phase2_weighted_hinge(g, g, vids)
    out_bad = phase2_weighted_hinge(bad, g, vids)
    assert out_bad["loss"] > out_good["loss"]
    assert out_bad["loss_video"] > 0.0
