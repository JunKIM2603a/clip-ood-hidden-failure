#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "subgroups" / "text_clustering.yaml"
MAPPING_DIR = ROOT / "configs" / "subgroups" / "mappings"


def load_config():
    with CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def source_mapping_path(dataset):
    names = {
        "inaturalist": "inaturalist_mos110_taxonomy.csv",
        "sun": "sun_mos50_hierarchy.csv",
        "places": "places_mos50_hierarchy.csv",
    }
    return MAPPING_DIR / names[dataset]


def leaf_names(dataset):
    path = source_mapping_path(dataset)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values = sorted({row["leaf_concept"].strip() for row in rows})
    if not values:
        raise RuntimeError("No leaf concepts in {}".format(path))
    return values


def choose_k(n, cfg):
    rule = cfg["clustering"]
    k = int(math.floor(math.sqrt(n) + 0.5))
    k = max(int(rule["minimum_k"]), min(int(rule["maximum_k"]), k))
    return k


def stable_cluster_ids(names, labels):
    members = {}
    for name, label in zip(names, labels):
        members.setdefault(int(label), []).append(name)
    ordered = sorted(
        members,
        key=lambda label: min(x.lower() for x in members[label]),
    )
    rename = {old: "cluster_{:02d}".format(i) for i, old in enumerate(ordered)}
    return [rename[int(x)] for x in labels]


def build_one(dataset, model, cfg):
    names = leaf_names(dataset)
    expected_k = int(cfg["clustering"]["expected_k"][dataset])
    k = choose_k(len(names), cfg)
    if k != expected_k:
        raise RuntimeError(
            "{} k-rule produced {}, expected {}".format(dataset, k, expected_k)
        )

    embeddings = model.encode(
        names,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float64)
    clustering = AgglomerativeClustering(
        n_clusters=k,
        metric="cosine",
        linkage="average",
    )
    raw_labels = clustering.fit_predict(embeddings)
    cluster_ids = stable_cluster_ids(names, raw_labels)

    out = MAPPING_DIR / (
        "text_clusters_{}_minilm.csv".format(dataset)
    )
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "leaf_concept", "text_cluster"])
        for name, cluster in zip(names, cluster_ids):
            writer.writerow([dataset, name, cluster])

    counts = {}
    for cluster in cluster_ids:
        counts[cluster] = counts.get(cluster, 0) + 1
    return {
        "dataset": dataset,
        "leaf_concepts": len(names),
        "k": k,
        "cluster_leaf_counts": dict(sorted(counts.items())),
        "output": str(out.relative_to(ROOT)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["inaturalist", "sun", "places"],
        choices=["inaturalist", "sun", "places"],
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="CPU is sufficient for only 210 total concept names.",
    )
    args = parser.parse_args()

    cfg = load_config()
    emb = cfg["embedding"]
    data_root = Path(
        os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")
    ).expanduser().resolve()
    cache = data_root / ".cache" / "sentence_transformers"

    print(
        "[text-cluster] model={} revision={}".format(
            emb["model"], emb["revision"]
        )
    )
    model = SentenceTransformer(
        emb["model"],
        revision=emb["revision"],
        device=args.device,
        cache_folder=str(cache),
    )

    results = []
    for dataset in args.datasets:
        result = build_one(dataset, model, cfg)
        results.append(result)
        print(
            "[{}] leaf={} k={} -> {}".format(
                dataset,
                result["leaf_concepts"],
                result["k"],
                result["output"],
            )
        )

    manifest = {
        "schema_version": 1,
        "generated_before_detector_subgroup_results": True,
        "embedding_model": emb["model"],
        "embedding_revision": emb["revision"],
        "package_version": emb["package_version"],
        "clustering": cfg["clustering"],
        "datasets": results,
    }
    path = ROOT / "configs" / "subgroups" / "text_clustering_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[manifest]", path)
    print("Review tracked mapping files and commit them before H1 results.")


if __name__ == "__main__":
    main()
