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


def official_positive_class_names() -> list[str]:
    module = _load_official_class_module()
    names = [
        str(name).strip()
        for name in list(module.CLASS_NAME["imagenet"])
    ]
    if len(names) != 1000:
        raise RuntimeError(
            f"Expected 1000 NegLabel positive class names, got {len(names)}"
        )
    return names


def official_positive_prompts() -> list[str]:
    module = _load_official_class_module()
    names = official_positive_class_names()
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
def prepare_neglabel_text(model, device: torch.device, backbone: str = "ViT-B/16"):
    positive = official_positive_prompts()

    if backbone == "ViT-B/16":
        negative, source_sha256 = official_selected_negative_prompts()
        negative_source = "official_provided_selected_list"
        mining_metadata = None
    else:
        from .neglabel_mining import load_mined_negative_prompts

        negative, mining_metadata = load_mined_negative_prompts(backbone)
        source_sha256 = mining_metadata["selected_prompt_file_sha256"]
        negative_source = "backbone_specific_dynamic_mining"

    pos_feat = encode_texts(model, positive, device)
    neg_feat = encode_texts(model, negative, device)
    metadata = {
        "positive_prompt_index": 85,
        "negative_prompt_file_sha256": source_sha256,
        "negative_prompt_count": len(negative),
        "negative_source": negative_source,
        "ngroup": 100,
        "temperature": 1.0,
        "logit_scale": 100.0,
    }
    if mining_metadata is not None:
        metadata["mining_metadata"] = mining_metadata
    return pos_feat, neg_feat, metadata


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


@torch.no_grad()
def neglabel_prompt_scores(
    image_features: np.ndarray,
    positive_text_by_prompt: list[torch.Tensor],
    negative_text: torch.Tensor,
    device: torch.device,
    ngroup: int = 100,
    temperature: float = 1.0,
    logit_scale: float = 100.0,
    batch_size: int = 512,
) -> np.ndarray:
    """Score one fixed NegLabel negative set under multiple positive prompts.

    H2 varies only the ImageNet positive-class prompt template. The selected
    negative prompt identities/embeddings and deterministic negative grouping
    remain exactly fixed across prompt variants. Negative logits are therefore
    computed once per image batch and reused for every positive prompt.
    """
    if not positive_text_by_prompt:
        raise ValueError("positive_text_by_prompt must not be empty")

    n_pos = int(positive_text_by_prompt[0].shape[0])
    feature_dim = int(positive_text_by_prompt[0].shape[1])
    for idx, text in enumerate(positive_text_by_prompt):
        if text.ndim != 2:
            raise ValueError(f"positive prompt {idx} must be a 2-D tensor")
        if int(text.shape[0]) != n_pos or int(text.shape[1]) != feature_dim:
            raise ValueError("All positive prompt feature matrices must match")
    if int(negative_text.shape[1]) != feature_dim:
        raise ValueError("Positive and negative text feature dimensions differ")

    n_neg = int(negative_text.shape[0])
    n_used = n_neg - (n_neg % ngroup)
    if n_used <= 0:
        raise ValueError(
            f"Need at least ngroup={ngroup} negative prompts, got {n_neg}"
        )
    negative_text = negative_text[:n_used]

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed(0)
    perm = torch.randperm(n_used, device=device)
    group_size = n_used // ngroup

    batches: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(image_features), batch_size),
        desc="NegLabel H2 prompt score",
        leave=False,
    ):
        feat = torch.from_numpy(
            image_features[start : start + batch_size]
        ).to(device)

        neg = logit_scale * (feat @ negative_text.T)
        neg = neg[:, perm].reshape(
            feat.shape[0],
            ngroup,
            group_size,
        )
        neg_lse = torch.logsumexp(
            neg / temperature,
            dim=2,
        )

        prompt_scores = []
        for positive_text in positive_text_by_prompt:
            pos = logit_scale * (feat @ positive_text.T)
            pos_lse = torch.logsumexp(
                pos / temperature,
                dim=1,
            )
            group_scores = torch.sigmoid(
                pos_lse[:, None] - neg_lse
            )
            prompt_scores.append(
                group_scores.mean(dim=1)
            )

        matrix = torch.stack(
            prompt_scores,
            dim=1,
        )
        batches.append(
            matrix.cpu().numpy()
        )

    return np.concatenate(
        batches,
        axis=0,
    ).astype(np.float32, copy=False)
