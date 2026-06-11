"""GeoCLIP-style location encoder (paper §3.3, suppl. B.3).

Architecture mirrors the released GeoCLIP implementation exactly so that
`location_encoder_weights.pth` loads 1:1, with one paper-mandated change:
sigma = [2^0, 2^3, 2^8] instead of GeoCLIP's [2^0, 2^4, 2^8] (Table 9).
When a loaded RFF projection was sampled at a different sigma, we rescale it
(b ~ N(0, s^2), so b * s_new/s_old ~ N(0, s_new^2), preserving directions).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Equal Earth Projection constants (Šavrič et al.; identical to GeoCLIP).
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

GEOCLIP_DEFAULT_SIGMA = (2**0, 2**4, 2**8)


def equal_earth_projection(coords: torch.Tensor) -> torch.Tensor:
    """(..., 2) degrees (lat, lon) -> (..., 2) EEP coordinates scaled by SF/180."""
    lat = torch.deg2rad(coords[..., 0])
    lon = torch.deg2rad(coords[..., 1])
    sin_theta = (math.sqrt(3.0) / 2.0) * torch.sin(lat)
    theta = torch.asin(sin_theta)
    denom = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * math.sqrt(3.0) * lon * torch.cos(theta)) / denom
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=-1) * SF) / 180.0


class GaussianEncoding(nn.Module):
    """Random Fourier Features: gamma(v) = [cos(2*pi*B v), sin(2*pi*B v)].

    B has shape (encoded_size, input_size), sampled from N(0, sigma^2). Stored
    as a non-trainable Parameter named ``b`` to match GeoCLIP checkpoints.
    """

    def __init__(self, sigma: float, input_size: int = 2, encoded_size: int = 256):
        super().__init__()
        self.sigma = float(sigma)
        b = torch.randn(encoded_size, input_size) * sigma
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        vp = 2 * math.pi * v @ self.b.T
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class LocationEncoderCapsule(nn.Module):
    """One RFF frequency branch. Layer names/structure match GeoCLIP."""

    def __init__(self, sigma: float):
        super().__init__()
        self.km = sigma
        self.capsule = nn.Sequential(
            GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.capsule(x))


class LocationEncoder(nn.Module):
    """Sum of per-frequency capsules over EEP-projected coordinates -> (N, 512)."""

    def __init__(self, sigma: tuple[float, ...] = (2**0, 2**3, 2**8)):
        super().__init__()
        self.sigma = tuple(sigma)
        for i, s in enumerate(self.sigma):
            self.add_module(f"LocEnc{i}", LocationEncoderCapsule(sigma=s))

    def load_geoclip_weights(self, weights_path: str) -> None:
        """Load released GeoCLIP weights, rescaling RFF matrices where our
        sigma differs from the GeoCLIP default the checkpoint was sampled at."""
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        for i, (s_new, s_old) in enumerate(zip(self.sigma, GEOCLIP_DEFAULT_SIGMA)):
            if s_new != s_old:
                cap: LocationEncoderCapsule = getattr(self, f"LocEnc{i}")
                rff: GaussianEncoding = cap.capsule[0]
                with torch.no_grad():
                    rff.b.mul_(s_new / s_old)
                rff.sigma = float(s_new)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 2) degrees (lat, lon) -> (N, 512) GPS embedding (unnormalized)."""
        loc = equal_earth_projection(coords)
        out = None
        for i in range(len(self.sigma)):
            feat = getattr(self, f"LocEnc{i}")(loc)
            out = feat if out is None else out + feat
        return out
