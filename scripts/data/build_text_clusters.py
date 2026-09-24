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
from sklearn.cluster import KMeans
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
    clustering = KMeans(
        n_clusters=k,
        random_state=int(cfg["clustering"]["random_state"]),
        n_init=int(cfg["clustering"]["n_init"]),
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


def audit_image_level_clusters(dataset, mapping_result, cfg, data_root):
    semantic_path = (
        data_root
        / "semantic_labels"
        / "{}_image_semantic_labels.csv".format(dataset)
    )
    if not semantic_path.exists():
        return {
            "status": "not_audited_missing_semantic_labels",
            "semantic_path": str(semantic_path),
        }

    cluster_path = ROOT / mapping_result["output"]
    with cluster_path.open("r", newline="", encoding="utf-8") as f:
        leaf_to_cluster = {
            row["leaf_concept"].strip(): row["text_cluster"].strip()
            for row in csv.DictReader(f)
        }

    image_counts = {}
    leaf_sets = {}
    total = 0
    mapped = 0
    with semantic_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            leaf = row.get("leaf_concept", "").strip()
            cluster = leaf_to_cluster.get(leaf)
            if not cluster:
                continue
            mapped += 1
            image_counts[cluster] = image_counts.get(cluster, 0) + 1
            leaf_sets.setdefault(cluster, set()).add(leaf)

    rules = cfg["eligibility"]
    eligible = {}
    for cluster in sorted(image_counts):
        n_images = image_counts[cluster]
        n_leafs = len(leaf_sets.get(cluster, set()))
        if (
            n_images >= int(rules["minimum_images_per_group"])
            and n_leafs >= int(rules["minimum_leaf_concepts_per_group"])
        ):
            eligible[cluster] = {
                "images": n_images,
                "leaf_concepts": n_leafs,
            }

    coverage = mapped / total if total else 0.0
    primary_scope = dataset in set(cfg["scope"]["primary_sources"])
    primary_eligible = (
        coverage >= float(rules["minimum_mapping_coverage"])
        and len(eligible) >= int(rules["minimum_eligible_groups_per_source"])
    )

    return {
        "status": "audited",
        "total_images": total,
        "mapped_images": mapped,
        "mapping_coverage": coverage,
        "cluster_image_counts": dict(sorted(image_counts.items())),
        "cluster_leaf_counts": {
            k: len(v) for k, v in sorted(leaf_sets.items())
        },
        "eligible_clusters": eligible,
        "eligible_cluster_count": len(eligible),
        "primary_scope": primary_scope,
        "primary_eligible": primary_eligible if primary_scope else False,
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
        result["pre_score_image_audit"] = audit_image_level_clusters(
            dataset,
            result,
            cfg,
            data_root,
        )
        results.append(result)
        audit = result["pre_score_image_audit"]
        print(
            "[{}] leaf={} k={} eligible_clusters={} primary={}".format(
                dataset,
                result["leaf_concepts"],
                result["k"],
                audit.get("eligible_cluster_count", "NA"),
                (
                    "YES"
                    if audit.get("primary_eligible", False)
                    else ("NO" if audit.get("primary_scope", False) else "SECONDARY")
                ),
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

    failed_primary = [
        r["dataset"]
        for r in results
        if r["dataset"] in set(cfg["scope"]["primary_sources"])
        and not r["pre_score_image_audit"].get("primary_eligible", False)
    ]
    if failed_primary:
        raise SystemExit(
            "Text-cluster pre-score feasibility failed for primary source(s): "
            + ", ".join(failed_primary)
            + ". Do not open detector subgroup results; review clustering feasibility first."
        )

    print("Text-cluster pre-score feasibility PASSED for all primary sources.")
    print("Review tracked mapping files and commit them before H1 results.")


if __name__ == "__main__":
    main()
