from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")).expanduser().resolve()
THIRD_PARTY = ROOT / "third_party"

EXPECTED_MCM_COMMIT = "ea7130f851e7d462cacd21f0e87a127705700bd9"
EXPECTED_NEGLABEL_COMMIT = "3253db684075b2db47844676eccdfda40d67a573"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_reference_repos() -> None:
    refs = [
        (THIRD_PARTY / "MCM", EXPECTED_MCM_COMMIT, "MCM"),
        (THIRD_PARTY / "NegLabel", EXPECTED_NEGLABEL_COMMIT, "NegLabel"),
    ]
    for path, expected, name in refs:
        if not (path / ".git").exists():
            raise RuntimeError(
                f"{name} reference repo missing at {path}. "
                "Run: bash scripts/env/setup_conda.sh"
            )
        actual = git_head(path)
        if actual != expected:
            raise RuntimeError(
                f"{name} commit mismatch: {actual} != {expected}. "
                "Run: bash scripts/env/setup_conda.sh"
            )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


class ManifestImageDataset(Dataset):
    def __init__(self, manifest: Path, transform):
        self.manifest = manifest
        self.transform = transform
        with manifest.open("r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise RuntimeError(f"Empty manifest: {manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = DATA_ROOT / row["relative_path"]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = int(row["label"]) if row.get("label", "") != "" else -1
        return tensor, label, row["relative_path"]


def manifest_for(dataset: str) -> Path:
    names = {
        "imagenet": "imagenet_val.csv",
        "inaturalist": "inaturalist.csv",
        "sun": "sun.csv",
        "places": "places.csv",
        "dtd": "dtd.csv",
    }
    path = DATA_ROOT / "manifests" / names[dataset]
    if not path.exists():
        raise RuntimeError(
            f"Missing manifest {path}. Install/verify datasets first."
        )
    return path


def backbone_slug(backbone: str) -> str:
    return backbone.replace("/", "-")


def feature_cache_path(backbone: str, dataset: str) -> Path:
    return (
        DATA_ROOT
        / "features"
        / "openai_clip"
        / backbone_slug(backbone)
        / f"{dataset}.npz"
    )


def load_clip(backbone: str, device: torch.device):
    import clip

    cache = DATA_ROOT / ".cache" / "clip"
    cache.mkdir(parents=True, exist_ok=True)
    model, preprocess = clip.load(
        backbone,
        device=device,
        jit=False,
        download_root=str(cache),
    )
    model.eval()
    return model, preprocess


@torch.no_grad()
def extract_image_features(
    model,
    preprocess,
    backbone: str,
    dataset: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    force: bool = False,
):
    cache_path = feature_cache_path(backbone, dataset)
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=False)
        return {
            "features": data["features"],
            "labels": data["labels"],
            "paths": data["paths"],
            "cache_path": cache_path,
        }

    ds = ManifestImageDataset(manifest_for(dataset), preprocess)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    all_features = []
    all_labels = []
    all_paths = []
    for images, labels, paths in tqdm(
        loader, desc=f"CLIP features {dataset}", leave=True
    ):
        images = images.to(device, non_blocking=True)
        feat = model.encode_image(images).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        all_features.append(feat.cpu().numpy().astype(np.float32, copy=False))
        all_labels.append(labels.numpy().astype(np.int64, copy=False))
        all_paths.extend(paths)

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    paths = np.asarray(all_paths, dtype=str)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp, features=features, labels=labels, paths=paths)
    tmp.replace(cache_path)

    return {
        "features": features,
        "labels": labels,
        "paths": paths,
        "cache_path": cache_path,
    }


def encode_texts(model, texts: list[str], device: torch.device, batch_size: int = 512):
    import clip

    outputs = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="CLIP text",
            leave=False,
        ):
            chunk = texts[start : start + batch_size]
            tokens = clip.tokenize(chunk, truncate=True).to(device)
            feat = model.encode_text(tokens).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
            outputs.append(feat)
    return torch.cat(outputs, dim=0)


def stable_cumsum(arr, rtol=1e-5, atol=1e-8):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError("unstable cumulative sum")
    return out


def fpr_at_recall(y_true, y_score, recall_level=0.95):
    y_true = y_true == 1
    desc = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc]
    y_true = y_true[desc]
    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall = np.r_[recall[sl], 1]
    fps = np.r_[fps[sl], 0]
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / np.sum(~y_true)


def ood_metrics(in_scores: np.ndarray, out_scores: np.ndarray) -> dict:
    in_scores = np.asarray(in_scores, dtype=np.float64).reshape(-1)
    out_scores = np.asarray(out_scores, dtype=np.float64).reshape(-1)
    labels = np.zeros(len(in_scores) + len(out_scores), dtype=np.int32)
    labels[: len(in_scores)] = 1
    scores = np.concatenate([in_scores, out_scores])
    auroc = sk_metrics.roc_auc_score(labels, scores)
    aupr_in = sk_metrics.average_precision_score(labels, scores)
    fpr95 = fpr_at_recall(labels, scores, 0.95)

    labels_out = np.zeros_like(labels)
    labels_out[len(in_scores) :] = 1
    aupr_out = sk_metrics.average_precision_score(labels_out, -scores)
    return {
        "auroc": float(auroc),
        "aupr_in": float(aupr_in),
        "aupr_out": float(aupr_out),
        "fpr95": float(fpr95),
    }


def write_score_csv(
    path: Path,
    dataset: str,
    sample_paths: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "relative_path", "id_label", "score"])
        for p, label, score in zip(sample_paths, labels, scores):
            writer.writerow([dataset, p, int(label), f"{float(score):.10g}"])


def write_metrics_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "backbone", "dataset", "fpr95", "auroc", "aupr_in", "aupr_out"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
