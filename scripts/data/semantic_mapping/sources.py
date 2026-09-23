from __future__ import annotations

import json
from pathlib import Path

from .common import (
    Lookup,
    digest,
    download,
    git_blob_sha1,
    norm_path,
    norm_scene,
    norm_taxon,
    safe_extract,
    subgroup_mapping,
)


def build_inaturalist(meta: dict, cache: Path):
    selected, _ = subgroup_mapping("inaturalist")
    cfg = meta["inaturalist"]
    archive = download(
        cfg["url"], cache / cfg["archive_name"], cfg.get("archive_md5")
    )
    root = safe_extract(archive, cache / "inat2017_annotations")
    lookup = Lookup()
    candidate_count = 0
    selected_categories = set()

    for filename in ("train2017.json", "val2017.json"):
        found = list(root.rglob(filename))
        if not found:
            raise RuntimeError(filename + " not found in " + str(archive))
        data = json.loads(found[0].read_text(encoding="utf-8"))
        categories = {int(x["id"]): str(x["name"]) for x in data["categories"]}
        image_by_id = {int(x["id"]): x for x in data["images"]}
        for ann in data["annotations"]:
            category_name = categories[int(ann["category_id"])]
            leaf = selected.get(norm_taxon(category_name))
            if not leaf:
                continue
            relative = str(image_by_id[int(ann["image_id"])]["file_name"])
            lookup.add(relative, leaf, filename + ":" + relative)
            candidate_count += 1
            selected_categories.add(leaf)

    return lookup, {
        "metadata_archive": str(archive),
        "metadata_md5": digest(archive, "md5"),
        "selected_leafs_present_in_metadata": len(selected_categories),
        "metadata_candidate_images": candidate_count,
    }


def _sun_leaf(path: str, selected: dict):
    parts = norm_path(path).split("/")
    if len(parts) < 3:
        return None
    category = " ".join(parts[1:-1])
    return selected.get(norm_scene(category))


def build_sun(meta: dict, cache: Path):
    selected, _ = subgroup_mapping("sun")
    cfg = meta["sun"]
    lookup = Lookup()
    all_paths = set()
    selected_paths = set()
    file_records = []

    for item in cfg["files"]:
        path = download(item["url"], cache / item["name"])
        actual_blob = git_blob_sha1(path)
        if actual_blob != item["git_blob_sha1"]:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                "SUN metadata Git blob SHA mismatch for {}: {} != {}".format(
                    item["name"], actual_blob, item["git_blob_sha1"]
                )
            )

        lines = [
            line.strip()
            for line in path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
        if len(lines) != int(item["expected_lines"]):
            raise RuntimeError(
                "SUN metadata line-count mismatch for {}: {} != {}".format(
                    item["name"], len(lines), item["expected_lines"]
                )
            )

        file_records.append({
            "name": item["name"],
            "git_blob_sha1": actual_blob,
            "lines": len(lines),
        })

        for relative in lines:
            key = norm_path(relative)
            all_paths.add(key)
            leaf = _sun_leaf(relative, selected)
            if not leaf:
                continue
            selected_paths.add(key)
            lookup.add(key, leaf, item["name"] + ":" + key)

    expected_total = int(cfg["expected_total_unique_paths"])
    if len(all_paths) != expected_total:
        raise RuntimeError(
            "SUN full inventory mismatch: {} != {}".format(
                len(all_paths), expected_total
            )
        )

    return lookup, {
        "metadata_kind": cfg["metadata_kind"],
        "tfds_commit": cfg["tfds_commit"],
        "full_inventory_paths": len(all_paths),
        "selected_metadata_paths": len(selected_paths),
        "files": file_records,
    }


def _places_categories(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        category, label = line.rsplit(" ", 1)
        out[int(label)] = norm_path(category)
    return out


def _find_one(root: Path, filename: str) -> Path:
    found = list(root.rglob(filename))
    if not found:
        raise RuntimeError(filename + " not found under " + str(root))
    return found[0]


def build_places(meta: dict, cache: Path):
    selected, _ = subgroup_mapping("places")
    cfg = meta["places"]
    archive = download(
        cfg["filelist_url"],
        cache / cfg["filelist_archive_name"],
        cfg.get("filelist_md5"),
    )
    root = safe_extract(archive, cache / "places365_filelists")
    categories_file = download(
        cfg["categories_url"],
        cache / cfg["categories_name"],
        cfg.get("categories_md5"),
    )
    labels = _places_categories(categories_file)
    lookup = Lookup()
    selected_paths = 0

    for filename in ("places365_train_standard.txt", "places365_val.txt"):
        path = _find_one(root, filename)
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = line.strip()
            if not line:
                continue
            relative, label_text = line.rsplit(" ", 1)
            official_category = labels[int(label_text)]
            leaf = selected.get(official_category)
            if not leaf:
                continue
            lookup.add(relative, leaf, filename + ":" + relative)
            selected_paths += 1

    ambiguous_basenames = sum(
        1 for value in lookup.by_basename.values() if len(value) > 1
    )
    return lookup, {
        "metadata_archive": str(archive),
        "metadata_md5": digest(archive, "md5"),
        "categories_md5": digest(categories_file, "md5"),
        "selected_metadata_paths": selected_paths,
        "ambiguous_selected_basenames": ambiguous_basenames,
    }
