from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .common import THIRD_PARTY, encode_texts, sha256_file


def _load_official_class_module():
    path = (
        THIRD_PARTY
        / "NegLabel"
        / "mmcls"
        / "models"
        / "classifiers"
        / "class_names.py"
    )
    spec = importlib.util.spec_from_file_location("neglabel_class_names_ref", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official NegLabel class names: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_positive_prompts() -> list[str]:
    module = _load_official_class_module()
    names = list(module.CLASS_NAME["imagenet"])
    template = module.prompt_templates[85]
    prompts = [template.format(name) for name in names]
    if len(prompts) != 1000:
        raise RuntimeError(f"Expected 1000 NegLabel positive prompts, got {len(prompts)}")
    return prompts


def official_selected_negative_prompts() -> tuple[list[str], str]:
    path = (
        THIRD_PARTY
        / "NegLabel"
        / "selected_neg_labels"
        / "selected_neg_labels_in1k_10k.txt"
    )
    if not path.exists():
        raise RuntimeError(f"Missing official selected negative labels: {path}")
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return prompts, sha256_file(path)


@torch.no_grad()
def prepare_neglabel_text(model, device: torch.device):
    positive = official_positive_prompts()
    negative, source_sha256 = official_selected_negative_prompts()
    pos_feat = encode_texts(model, positive, device)
    neg_feat = encode_texts(model, negative, device)
    return pos_feat, neg_feat, {
        "positive_prompt_index": 85,
        "negative_prompt_file_sha256": source_sha256,
        "negative_prompt_count": len(negative),
        "ngroup": 100,
        "temperature": 1.0,
        "logit_scale": 100.0,
    }


@torch.no_grad()
def neglabel_scores(
    image_features: np.ndarray,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    device: torch.device,
    ngroup: int = 100,
    temperature: float = 1.0,
    logit_scale: float = 100.0,
    batch_size: int = 512,
) -> np.ndarray:
    n_neg = int(negative_text.shape[0])
    drop = n_neg % ngroup
    n_used = n_neg - drop
    negative_text = negative_text[:n_used]

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed(0)
    perm = torch.randperm(n_used, device=device)

    scores = []
    group_size = n_used // ngroup
    for start in tqdm(
        range(0, len(image_features), batch_size),
        desc="NegLabel score",
        leave=False,
    ):
        feat = torch.from_numpy(image_features[start : start + batch_size]).to(device)
        pos = logit_scale * (feat @ positive_text.T)
        neg = logit_scale * (feat @ negative_text.T)
        neg = neg[:, perm].reshape(feat.shape[0], ngroup, group_size)

        # This is mathematically identical to summing the ID probability mass
        # after a softmax over [all positive labels + one negative-label group].
        pos_lse = torch.logsumexp(pos / temperature, dim=1)
        neg_lse = torch.logsumexp(neg / temperature, dim=2)
        group_scores = torch.sigmoid(pos_lse[:, None] - neg_lse)
        score = group_scores.mean(dim=1)
        scores.append(score.cpu().numpy())

    return np.concatenate(scores).astype(np.float32, copy=False)
