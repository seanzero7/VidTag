"""Unit tests for the core model modules (SPEC §2)."""

import math
from pathlib import Path

import pytest
import torch

from vidtag.models.georefiner import GeoRefiner, GPSNoiser
from vidtag.models.location_encoder import (
    GEOCLIP_DEFAULT_SIGMA,
    LocationEncoder,
    equal_earth_projection,
)
from vidtag.models.tempgeo import TempGeo
from vidtag.models.vidtag import VidTAG

GEOCLIP_WEIGHTS = Path(
    "/Volumes/8TBExternal/PaperRepro/weights/geoclip/location_encoder_weights.pth"
)


def test_equal_earth_projection_known_values():
    # At (0, 0) the projection is exactly the origin.
    out = equal_earth_projection(torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(out, torch.zeros(1, 2), atol=1e-7)

    # Hand-computed for (lat=45, lon=90) from the polynomial (A1..A4, SF).
    lat, lon = math.radians(45.0), math.radians(90.0)
    theta = math.asin(math.sqrt(3.0) / 2.0 * math.sin(lat))
    A1, A2, A3, A4, SF = 1.340264, -0.081106, 0.000893, 0.003796, 66.50336
    denom = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * math.sqrt(3.0) * lon * math.cos(theta)) / denom * SF / 180
    y = (A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta) * SF / 180
    out = equal_earth_projection(torch.tensor([[45.0, 90.0]]))
    assert torch.allclose(out, torch.tensor([[x, y]]), atol=1e-5)


def test_location_encoder_shape():
    enc = LocationEncoder()
    out = enc(torch.tensor([[47.37, 8.54], [-33.86, 151.21]]))
    assert out.shape == (2, 512)


@pytest.mark.skipif(not GEOCLIP_WEIGHTS.exists(), reason="GeoCLIP weights not downloaded")
def test_geoclip_load_rescales_sigma1_branch():
    enc = LocationEncoder(sigma=(2**0, 2**3, 2**8))
    enc.load_geoclip_weights(str(GEOCLIP_WEIGHTS))
    raw = torch.load(str(GEOCLIP_WEIGHTS), map_location="cpu", weights_only=True)
    # branch 0 and 2 match the checkpoint exactly; branch 1 is rescaled by 8/16
    assert torch.equal(enc.LocEnc0.capsule[0].b, raw["LocEnc0.capsule.0.b"])
    assert torch.allclose(enc.LocEnc1.capsule[0].b, raw["LocEnc1.capsule.0.b"] * 0.5)
    assert torch.equal(enc.LocEnc2.capsule[0].b, raw["LocEnc2.capsule.0.b"])
    # linear weights transfer untouched
    assert torch.equal(enc.LocEnc1.capsule[1].weight, raw["LocEnc1.capsule.1.weight"])


def test_tempgeo_shapes_and_norm():
    tg = TempGeo(max_len=32)
    z = torch.nn.functional.normalize(torch.randn(2, 16, 1792), dim=-1)
    attended, final = tg(z)
    assert attended.shape == (2, 16, 1792)
    assert final.shape == (2, 16, 512)
    assert torch.allclose(final.norm(dim=-1), torch.ones(2, 16), atol=1e-5)
    loss = final.sum()
    loss.backward()  # gradients flow
    assert tg.pos_embedding.weight.grad is not None


def test_tempgeo_padding_mask_isolates_pads():
    tg = TempGeo(max_len=32).eval()
    z = torch.nn.functional.normalize(torch.randn(1, 8, 1792), dim=-1)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 6:] = True
    _, with_pad = tg(z, key_padding_mask=mask)
    z2 = z.clone()
    z2[0, 6:] = 0.0  # changing PAD content must not affect unpadded outputs
    _, with_pad2 = tg(z2, key_padding_mask=mask)
    assert torch.allclose(with_pad[0, :6], with_pad2[0, :6], atol=1e-5)


def test_georefiner_non_causal():
    gr = GeoRefiner(max_len=32).eval()
    f = torch.nn.functional.normalize(torch.randn(1, 8, 512), dim=-1)
    g = torch.nn.functional.normalize(torch.randn(1, 8, 512), dim=-1)
    out1 = gr(f, g)
    g2 = g.clone()
    g2[0, -1] = -g2[0, -1]  # perturb the LAST GPS token
    out2 = gr(f, g2)
    # ... and the FIRST output must change (no causal mask)
    assert not torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
    assert torch.allclose(out1.norm(dim=-1), torch.ones(1, 8), atol=1e-5)


def test_gps_noiser_statistics():
    torch.manual_seed(0)
    noiser = GPSNoiser()
    coords = torch.randn(2000, 16, 2) * 0.01 + torch.tensor([47.0, 8.0])
    noisy = noiser(coords)
    per_seq_const = (noisy.std(dim=1).sum(-1) < 1e-9)
    frac = per_seq_const.float().mean().item()
    assert 0.07 < frac < 0.13, f"collapse fraction {frac}"
    # collapsed sequences equal one of the original points
    idx = per_seq_const.nonzero()[0, 0]
    diffs = (coords[idx] - noisy[idx, 0]).norm(dim=-1)
    assert diffs.min() < 1e-6
    # non-collapsed: per-frame deviation from the sequence-wide shift stays
    # within the jitter bounds on each axis
    nc = (~per_seq_const).nonzero()[0, 0]
    delta = noisy[nc] - coords[nc]
    shift_est = delta.median(dim=0).values
    jitter = (delta - shift_est).abs()
    assert jitter.max() <= 0.02 + 0.02 + 1e-6  # jitter +/- median estimation slack
    assert shift_est.abs().max() <= 0.2 + 0.02 + 1e-6


def test_vidtag_assembly_and_param_groups():
    m = VidTAG(with_backbones=False, max_len=32)
    fused = torch.nn.functional.normalize(torch.randn(2, 8, 1792), dim=-1)
    coords = torch.rand(2, 8, 2) * 10
    v, g = m.forward_phase1(fused, coords)
    assert v.shape == (16, 512) and g.shape == (16, 512)
    refined = m.forward_phase2(fused, coords)
    assert refined.shape == (2, 8, 512)

    p1 = {id(p) for p in m.phase1_trainable_parameters()}
    p2 = {id(p) for p in m.phase2_trainable_parameters()}
    refiner = {id(p) for p in m.georefiner.parameters()}
    assert p2 == refiner
    assert not (p1 & refiner)

    gallery_coords = torch.rand(50, 2) * 10
    gallery_emb = m.encode_gps(gallery_coords)
    out = m.predict(fused, gallery_coords, gallery_emb)
    assert out["initial_coords"].shape == (2, 8, 2)
    assert out["refined_coords"].shape == (2, 8, 2)
    # predictions are rows of the gallery
    flat = out["refined_coords"].reshape(-1, 2)
    dists = (flat[:, None, :] - gallery_coords[None]).norm(dim=-1).min(dim=1).values
    assert dists.max() < 1e-6
