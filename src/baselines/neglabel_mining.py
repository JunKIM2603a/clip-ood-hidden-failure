from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .common import DATA_ROOT, THIRD_PARTY, backbone_slug, encode_texts, sha256_file
from .neglabel import _load_official_class_module, official_positive_prompts


@dataclass(frozen=True)
class CandidateCorpus:
    noun_prompts: list[str]
    adj_prompts: list[str]
    corpus_files: list[dict]
    raw_word_count: int
    unique_word_count: int


def _corpus_dir() -> Path:
    return THIRD_PARTY / "NegLabel" / "txtfiles"


def build_candidate_corpus() -> CandidateCorpus:
    """Build NegLabel candidate prompts with a deterministic file order.

    The official code uses os.listdir(wordnet_database), which can vary across
    filesystems. We stabilize only this enumeration step by sorting filenames.
    The original global raw-word de-duplication and noun/adjective prompt rules
    are otherwise preserved.
    """
    module = _load_official_class_module()
    noun_template = module.prompt_templates[85]
    adj_template = "This is a {} photo"

    root = _corpus_dir()
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.name.split(".")[0] in {"noun", "adj"}
    )
    if not files:
        raise RuntimeError(f"No NegLabel corpus files found under {root}")

    seen: dict[str, None] = {}
    nouns: list[str] = []
    adjs: list[str] = []
    records: list[dict] = []
    raw_count = 0

    for path in files:
        kind = path.name.split(".")[0]
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        raw_count += len(lines)
        records.append({
            "path": str(path.relative_to(THIRD_PARTY / "NegLabel")),
            "sha256": sha256_file(path),
            "raw_lines": len(lines),
            "kind": kind,
        })
        for word in lines:
            if word in seen:
                continue
            seen[word] = None
            if kind == "noun":
                nouns.append(noun_template.format(word))
            else:
                adjs.append(adj_template.format(word))

    return CandidateCorpus(
        noun_prompts=nouns,
        adj_prompts=adjs,
        corpus_files=records,
        raw_word_count=raw_count,
        unique_word_count=len(seen),
    )


@torch.no_grad()
def _negative_similarity_quantiles(
    model,
    prompts: list[str],
    positive_features: torch.Tensor,
    device: torch.device,
    *,
    quantile: float = 0.95,
    batch_size: int = 1000,
) -> np.ndarray:
    values: list[np.ndarray] = []
    import clip

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc="NegLabel mining",
        leave=True,
    ):
        chunk = prompts[start : start + batch_size]
        tokens = clip.tokenize(chunk, truncate=True).to(device)
        features = model.encode_text(tokens).float()
        features = features / features.norm(dim=-1, keepdim=True)
        similarity = features @ positive_features.T
        score = torch.quantile(similarity, q=quantile, dim=-1)
        values.append(score.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(values, axis=0)


def _selected_output(backbone: str) -> tuple[Path, Path]:
    root = DATA_ROOT / "neglabel" / backbone_slug(backbone)
    return root / "selected_negative_prompts.txt", root / "mining_metadata.json"


def selected_negative_path(backbone: str) -> Path:
    return _selected_output(backbone)[0]


def load_mined_negative_prompts(backbone: str) -> tuple[list[str], dict]:
    prompts_path, metadata_path = _selected_output(backbone)
    if not prompts_path.exists() or not metadata_path.exists():
        raise RuntimeError(
            f"Missing mined NegLabel negatives for {backbone}. Run:\n"
            f"  python scripts/baseline/mine_neglabel.py --backbone {backbone}"
        )
    prompts = [
        line.strip()
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = int(metadata["selected_total"])
    if len(prompts) != expected:
        raise RuntimeError(
            f"Selected prompt count changed: {len(prompts)} != {expected}"
        )
    actual_sha = sha256_file(prompts_path)
    if actual_sha != metadata["selected_prompt_file_sha256"]:
        raise RuntimeError(
            "Selected NegLabel prompt file hash does not match mining metadata"
        )
    return prompts, metadata


@torch.no_grad()
def mine_negative_prompts(
    model,
    backbone: str,
    device: torch.device,
    *,
    quantile: float = 0.95,
    select_fraction: float = 0.15,
    embedding_batch_size: int = 1000,
) -> dict:
    if not (0.0 < select_fraction < 1.0):
        raise ValueError("select_fraction must be between 0 and 1")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("quantile must be in [0,1]")

    corpus = build_candidate_corpus()
    positive_prompts = official_positive_prompts()
    positive_features = encode_texts(
        model,
        positive_prompts,
        device,
        batch_size=embedding_batch_size,
    )

    noun_scores = _negative_similarity_quantiles(
        model,
        corpus.noun_prompts,
        positive_features,
        device,
        quantile=quantile,
        batch_size=embedding_batch_size,
    )
    adj_scores = _negative_similarity_quantiles(
        model,
        corpus.adj_prompts,
        positive_features,
        device,
        quantile=quantile,
        batch_size=embedding_batch_size,
    )

    noun_order = np.argsort(noun_scores, kind="stable")
    adj_order = np.argsort(adj_scores, kind="stable")
    noun_keep = int(len(noun_order) * select_fraction)
    adj_keep = int(len(adj_order) * select_fraction)

    selected_nouns = [corpus.noun_prompts[int(i)] for i in noun_order[:noun_keep]]
    selected_adjs = [corpus.adj_prompts[int(i)] for i in adj_order[:adj_keep]]
    selected = selected_nouns + selected_adjs

    prompts_path, metadata_path = _selected_output(backbone)
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

    metadata = {
        "schema_version": 1,
        "method": "NegLabel",
        "backbone": backbone,
        "official_algorithm_parameters": {
            "positive_prompt_index": 85,
            "negative_prompt_index_noun": 85,
            "adjective_prompt": "This is a {} photo",
            "similarity_quantile": quantile,
            "select_fraction_per_part_of_speech": select_fraction,
            "ngroup": 100,
        },
        "determinism_amendment": {
            "official_code": "os.listdir(wordnet_database)",
            "pilot": "lexicographically sorted corpus filenames",
            "reason": "remove filesystem-order dependence before detector results",
        },
        "corpus_raw_word_count": corpus.raw_word_count,
        "corpus_unique_word_count": corpus.unique_word_count,
        "candidate_nouns": len(corpus.noun_prompts),
        "candidate_adjectives": len(corpus.adj_prompts),
        "selected_nouns": len(selected_nouns),
        "selected_adjectives": len(selected_adjs),
        "selected_total": len(selected),
        "corpus_files": corpus.corpus_files,
    }
    metadata["selected_prompt_file_sha256"] = sha256_file(prompts_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def compare_with_official_b16(selected: list[str]) -> dict:
    official = (
        THIRD_PARTY
        / "NegLabel"
        / "selected_neg_labels"
        / "selected_neg_labels_in1k_10k.txt"
    )
    if not official.exists():
        raise RuntimeError(f"Missing official B/16 selected negatives: {official}")
    ref = [
        line.strip()
        for line in official.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected_set = set(selected)
    ref_set = set(ref)
    intersection = len(selected_set & ref_set)
    union = len(selected_set | ref_set)
    return {
        "official_count": len(ref),
        "mined_count": len(selected),
        "set_intersection": intersection,
        "set_jaccard": intersection / union if union else 1.0,
        "official_file_sha256": sha256_file(official),
    }
