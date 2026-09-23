#!/usr/bin/env python3
"""Reconstruct MOS image-to-concept labels from authoritative source metadata.

No detector or CLIP predictions are used to assign labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from semantic_mapping.common import (
    DATASET_SPEC,
    META_SPEC,
    ROOT,
    frozen_thresholds,
    image_files,
    load_json,
    subgroup_mapping,
)
from semantic_mapping.sources import (
    build_inaturalist,
    build_places,
    build_sun,
)


def reconstruct_one(
    dataset,
    lookup,
    dataset_spec,
    data_root,
    output_dir,
    thresholds,
    metadata_summary,
):
    root = data_root / dataset_spec["datasets"][dataset]["destination"]
    local_images = image_files(root)
    if not local_images:
        raise RuntimeError("No images found under " + str(root))

    _, leaf_groups = subgroup_mapping(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (dataset + "_image_semantic_labels.csv")

    matched = 0
    ambiguous = 0
    methods = defaultdict(int)
    group_images = defaultdict(int)
    group_leafs = defaultdict(set)
    represented_leafs = set()

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "relative_path",
            "leaf_concept",
            "primary_groups",
            "secondary_groups",
            "match_method",
            "source_identity",
        ])
        for image in local_images:
            local_relative = image.relative_to(root).as_posix()
            leaf, method, source = lookup.resolve(local_relative)
            methods[method] += 1
            if method.startswith("ambiguous"):
                ambiguous += 1

            primary = ""
            secondary = ""
            if leaf:
                matched += 1
                represented_leafs.add(leaf)
                primary, secondary = leaf_groups[leaf]
                for group in filter(None, primary.split("|")):
                    group_images[group] += 1
                    group_leafs[group].add(leaf)

            writer.writerow([
                dataset,
                image.relative_to(data_root).as_posix(),
                leaf or "",
                primary,
                secondary,
                method,
                source,
            ])

    total = len(local_images)
    coverage = matched / total if total else 0.0
    min_images = int(thresholds["minimum_images_per_group"])
    min_leafs = int(thresholds["minimum_leaf_concepts_per_group"])
    eligible = {}
    for group in sorted(group_images):
        n_images = group_images[group]
        n_leafs = len(group_leafs[group])
        if n_images >= min_images and n_leafs >= min_leafs:
            eligible[group] = {
                "images": n_images,
                "leaf_concepts": n_leafs,
            }

    primary_eligible = (
        coverage >= float(thresholds["minimum_mapping_coverage"])
        and len(eligible)
        >= int(thresholds["minimum_eligible_groups_per_source"])
    )

    summary = {
        "dataset": dataset,
        "images": total,
        "matched_images": matched,
        "ambiguous_images": ambiguous,
        "unmatched_images": total - matched - ambiguous,
        "mapping_coverage": coverage,
        "match_methods": dict(sorted(methods.items())),
        "represented_leaf_concepts": len(represented_leafs),
        "represented_primary_groups": len(group_images),
        "eligible_primary_groups": eligible,
        "eligible_primary_group_count": len(eligible),
        "primary_eligible": primary_eligible,
        "frozen_thresholds": thresholds,
        "mapping_csv": str(output),
        "metadata": metadata_summary,
    }

    state_path = data_root / "state" / (dataset + ".json")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["semantic_mapping"] = {
            "status": (
                "primary_eligible"
                if primary_eligible
                else "insufficient_for_primary"
            ),
            "coverage": coverage,
            "matched_images": matched,
            "ambiguous_images": ambiguous,
            "eligible_primary_group_count": len(eligible),
            "mapping_csv": str(output),
        }
        state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return summary


def write_audit(output_dir, summaries):
    payload = {
        "schema_version": 1,
        "generated_without_detector_scores": True,
        "sources": summaries,
    }
    (output_dir / "semantic_mapping_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# MOS image-to-concept semantic mapping audit",
        "",
        "> Generated without MCM/NegLabel detector scores.",
        "",
        "| Source | images | matched | ambiguous | coverage | eligible groups | primary? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in summaries:
        lines.append(
            "| {dataset} | {images} | {matched_images} | {ambiguous_images} | "
            "{coverage:.2f}% | {eligible_primary_group_count} | {primary} |".format(
                coverage=100 * s["mapping_coverage"],
                primary="YES" if s["primary_eligible"] else "NO",
                **s,
            )
        )
    lines += [
        "",
        "Unmatched or ambiguous images are not filled using model predictions or manual guesses.",
        "",
    ]
    for s in summaries:
        lines += [
            "## " + s["dataset"],
            "",
            "- coverage: {:.2f}%".format(100 * s["mapping_coverage"]),
            "- represented leaf concepts: {}".format(s["represented_leaf_concepts"]),
            "- represented primary groups: {}".format(s["represented_primary_groups"]),
            "- eligible primary groups: {}".format(s["eligible_primary_group_count"]),
            "- decision: {}".format(
                "PRIMARY ELIGIBLE"
                if s["primary_eligible"]
                else "INSUFFICIENT FOR PRIMARY"
            ),
            "- match methods: {}".format(
                json.dumps(s["match_methods"], sort_keys=True)
            ),
            "",
        ]
        if s["eligible_primary_groups"]:
            lines += [
                "| group | images | leaf concepts |",
                "| --- | ---: | ---: |",
            ]
            for group, values in s["eligible_primary_groups"].items():
                lines.append(
                    "| {} | {} | {} |".format(
                        group, values["images"], values["leaf_concepts"]
                    )
                )
            lines.append("")

    (output_dir / "semantic_mapping_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["inaturalist", "sun", "places"],
        default=["inaturalist", "sun", "places"],
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CLIP_OOD_DATA_ROOT", str(ROOT / "data")),
    )
    parser.add_argument("--require-primary", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    dataset_spec = load_json(DATASET_SPEC)
    metadata_spec = load_json(META_SPEC)
    thresholds = frozen_thresholds()
    cache = data_root / ".cache" / "semantic_metadata"
    output_dir = data_root / "semantic_labels"

    builders = {
        "inaturalist": build_inaturalist,
        "sun": build_sun,
        "places": build_places,
    }
    summaries = []

    for dataset in args.datasets:
        print("\n=== reconstruct {} ===".format(dataset))
        lookup, metadata_summary = builders[dataset](metadata_spec, cache)
        summary = reconstruct_one(
            dataset,
            lookup,
            dataset_spec,
            data_root,
            output_dir,
            thresholds,
            metadata_summary,
        )
        summaries.append(summary)
        print(
            "[{}] matched={}/{} coverage={:.2f}% eligible_groups={} primary={}".format(
                dataset,
                summary["matched_images"],
                summary["images"],
                100 * summary["mapping_coverage"],
                summary["eligible_primary_group_count"],
                "YES" if summary["primary_eligible"] else "NO",
            )
        )

    write_audit(output_dir, summaries)
    print("\nAudit:", output_dir / "semantic_mapping_audit.md")

    failed = [x["dataset"] for x in summaries if not x["primary_eligible"]]
    if failed:
        print("Primary criteria not met for: " + ", ".join(failed))
        print("Frozen thresholds were not relaxed.")
        return 2 if args.require_primary else 0

    print("All requested sources satisfy the frozen primary mapping criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
