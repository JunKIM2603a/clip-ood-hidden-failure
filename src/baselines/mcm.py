from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .common import THIRD_PARTY, encode_texts


def official_imagenet_class_names() -> list[str]:
    path = THIRD_PARTY / "MCM" / "data" / "ImageNet" / "imagenet_class_clean.npy"
    if not path.exists():
        raise RuntimeError(f"Missing official MCM class names: {path}")
    names = np.load(path, allow_pickle=False).tolist()
    names = [str(x).strip() for x in names]
    if len(names) != 1000:
        raise RuntimeError(f"Expected 1000 MCM class names, got {len(names)}")
    return names


@torch.no_grad()
def prepare_mcm_text(model, device: torch.device):
    names = official_imagenet_class_names()
    prompts = [f"a photo of a {name}" for name in names]
    return encode_texts(model, prompts, device)


@torch.no_grad()
def mcm_scores(
    image_features: np.ndarray,
    text_features: torch.Tensor,
    device: torch.device,
    temperature: float = 1.0,
    batch_size: int = 2048,
) -> np.ndarray:
    scores = []
    for start in tqdm(
        range(0, len(image_features), batch_size),
        desc="MCM score",
        leave=False,
    ):
        feat = torch.from_numpy(image_features[start : start + batch_size]).to(device)
        similarity = feat @ text_features.T
        prob = torch.softmax(similarity / temperature, dim=1)
        score = prob.max(dim=1).values
        scores.append(score.cpu().numpy())
    return np.concatenate(scores).astype(np.float32, copy=False)
