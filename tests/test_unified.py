"""Tests for the suppl.-H unified training mix and city-level metrics."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from vidtag.data.unified import UnifiedSequences, msls_city_names, select_cityguessr_ids
from vidtag.metrics import hierarchy_accuracy


class _Stub(Dataset):
    def __init__(self, n, tag):
        self.n, self.tag = n, tag
        self.video_ids = [f"{tag}_{i:04d}" for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "fused": torch.zeros(4, 1792),
            "coords": torch.zeros(4, 2),
            "video_id": i,
            "seq_key": self.video_ids[i],
            "n_frames": 4,
        }


def test_msls_city_name_mapping():
    names = msls_city_names()
    assert {"san_francisco", "copenhagen", "sao_paulo", "amsterdam", "tokyo"} <= names


def test_select_cityguessr_drops_msls_overlap_and_caps():
    ids = (
        [f"Amsterdam_{i:04d}" for i in range(50)]          # overlaps MSLS -> dropped
        + [f"Abu_Dhabi_{i:04d}" for i in range(120)]       # capped at 80
        + [f"Adelaide_{i:04d}" for i in range(30)]         # < cap -> all kept
    )
    keep = select_cityguessr_ids(ids, per_city_cap=80, seed=0)
    cities = {k.rsplit("_", 1)[0] for k in keep}
    assert "Amsterdam" not in cities
    assert sum(k.startswith("Abu_Dhabi") for k in keep) == 80
    assert sum(k.startswith("Adelaide") for k in keep) == 30


def test_unified_sizes_and_video_id_namespacing():
    msls = _Stub(100, "msls")
    cg = _Stub(60, "Adelaide")  # one small non-MSLS city -> all kept
    gama = _Stub(500, "gama")
    uni = UnifiedSequences({"msls": msls, "cityguessr": cg, "gama": gama}, seed=0)
    # rule 1: all MSLS; rule 2: all 60 (under cap); rule 3: GAMa capped to MSLS size
    assert uni.sizes == {"msls": 100, "cityguessr": 60, "gama": 100}
    assert len(uni) == 260
    vids = {uni[i]["video_id"] for i in range(len(uni))}
    assert len(vids) == 260  # no collisions across sources


def test_hierarchy_accuracy_hand_computed():
    labels = pd.DataFrame(
        {
            "City": ["Durban", "Johannesburg", "Seoul"],
            "State": ["KZN", "Gauteng", "Seoul"],
            "Country": ["South_Africa", "South_Africa", "South_Korea"],
            "Continent": ["Africa", "Africa", "Asia"],
        }
    )
    pred = ["Durban", "Johannesburg", "Seoul", "Durban"]
    gt = ["Durban", "Durban", "Seoul", "Seoul"]
    out = hierarchy_accuracy(pred, gt, labels)
    assert out["city_acc"] == 50.0        # 2/4 exact
    assert out["state_acc"] == 50.0       # same 2
    assert out["country_acc"] == 75.0     # + Joburg/Durban same country
    assert out["continent_acc"] == 75.0
