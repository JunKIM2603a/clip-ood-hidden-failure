from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.mcm import mcm_scores
from baselines.neglabel import neglabel_scores


def test_mcm_matches_direct_formula():
    device = torch.device("cpu")
    image = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    text = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=torch.float32,
    )
    direct = torch.softmax(torch.from_numpy(image) @ text.T, dim=1).max(dim=1).values
    actual = mcm_scores(image, text, device=device, batch_size=2)
    np.testing.assert_allclose(actual, direct.numpy(), rtol=1e-6, atol=1e-7)


def test_neglabel_vectorization_matches_official_group_loop():
    device = torch.device("cpu")
    torch.manual_seed(7)
    image = torch.randn(3, 4)
    image = image / image.norm(dim=1, keepdim=True)
    pos_text = torch.randn(5, 4)
    pos_text = pos_text / pos_text.norm(dim=1, keepdim=True)
    neg_text = torch.randn(8, 4)
    neg_text = neg_text / neg_text.norm(dim=1, keepdim=True)

    actual = neglabel_scores(
        image.numpy().astype(np.float32),
        pos_text,
        neg_text,
        device=device,
        ngroup=2,
        temperature=1.0,
        logit_scale=100.0,
        batch_size=3,
    )

    pos = 100.0 * image @ pos_text.T
    neg = 100.0 * image @ neg_text.T

    torch.manual_seed(0)
    idx = torch.randperm(neg.shape[1], device=device)
    neg = neg.T[idx].T.reshape(pos.shape[0], 2, -1).contiguous()

    official_scores = []
    for i in range(2):
        full_sim = torch.cat([pos, neg[:, i, :]], dim=-1)
        full_sim = full_sim.softmax(dim=-1)
        official_scores.append(
            full_sim[:, : pos.shape[1]].sum(dim=-1).unsqueeze(-1)
        )
    expected = torch.cat(official_scores, dim=-1).mean(dim=-1)

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
