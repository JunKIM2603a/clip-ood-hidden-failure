from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET_SPEC = ROOT / "configs" / "datasets" / "pilot_sources.json"
META_SPEC = ROOT / "configs" / "datasets" / "semantic_metadata_sources.json"
SUBGROUP_SPEC = ROOT / "configs" / "subgroups" / "predefined.yaml"
MAPPING_DIR = ROOT / "configs" / "subgroups" / "mappings"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_thresholds() -> dict:
    values = {}
    inside = False
    for raw in SUBGROUP_SPEC.read_text(encoding="utf-8").splitlines():
        if raw.startswith("primary_eligibility:"):
            inside = True
            continue
        if inside and raw and not raw.startswith(" "):
            break
        if not inside:
            continue
        stripped = raw.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        if not value:
            continue
        try:
            values[key] = float(value) if "." in value else int(value)
        except ValueError:
            pass
    required = {
        "minimum_images_per_group",
        "minimum_leaf_concepts_per_group",
        "minimum_eligible_groups_per_source",
        "minimum_mapping_coverage",
    }
    missing = required - set(values)
    if missing:
        raise RuntimeError("Missing frozen thresholds: " + ", ".join(sorted(missing)))
    return values


def digest(path: Path, algo: str = "md5") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def norm_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./").lstrip("/")


def norm_taxon(value: str) -> str:
    value = value.replace(chr(215), "x")
    return " ".join(value.lower().split())


def norm_scene(value: str) -> str:
    for token in ("(", ")", "_", "/"):
        value = value.replace(token, " ")
    return " ".join(value.lower().split())


def download(url: str, dst: Path, expected_md5: str | None = None) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not expected_md5 or digest(dst, "md5") == expected_md5:
            print("[reuse-meta]", dst)
            return dst
        print("[meta] checksum mismatch; replacing", dst)
        dst.unlink()
    part = dst.with_suffix(dst.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "clip-ood-hidden-failure-semantic-mapper/1.0"},
    )
    print("[download-meta]", url)
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    part.replace(dst)
    if expected_md5:
        actual = digest(dst, "md5")
        if actual != expected_md5:
            dst.unlink(missing_ok=True)
            raise RuntimeError(
                "MD5 mismatch for {}: {} != {}".format(url, actual, expected_md5)
            )
    return dst


def safe_extract(archive: Path, dst: Path) -> Path:
    marker = dst / ".complete"
    archive_hash = digest(archive, "md5")
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == archive_hash:
        return dst
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    base = dst.resolve()

    def check(name: str) -> None:
        target = (dst / name).resolve()
        if os.path.commonpath([str(base), str(target)]) != str(base):
            raise RuntimeError("Unsafe archive path: " + name)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                check(item.filename)
            zf.extractall(dst)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            for item in tf.getmembers():
                check(item.name)
            try:
                tf.extractall(dst, filter="data")
            except TypeError:
                tf.extractall(dst)
    else:
        raise RuntimeError("Unsupported metadata archive: " + str(archive))
    marker.write_text(archive_hash + "\n", encoding="utf-8")
    return dst


def csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def subgroup_mapping(dataset: str):
    source_to_leaf = {}
    leaf_groups = {}
    if dataset == "inaturalist":
        path = MAPPING_DIR / "inaturalist_mos110_taxonomy.csv"
        for row in csv_rows(path):
            leaf = row["leaf_concept"].strip()
            source_to_leaf[norm_taxon(leaf)] = leaf
            leaf_groups[leaf] = (row["order"].strip(), row["family"].strip())
    elif dataset == "sun":
        path = MAPPING_DIR / "sun_mos50_hierarchy.csv"
        for row in csv_rows(path):
            leaf = row["leaf_concept"].strip()
            source_to_leaf[norm_scene(leaf)] = leaf
            leaf_groups[leaf] = (
                row["basic_groups"].strip(),
                row["s3_groups"].strip(),
            )
    elif dataset == "places":
        path = MAPPING_DIR / "places_mos50_hierarchy.csv"
        for row in csv_rows(path):
            leaf = row["leaf_concept"].strip()
            source_to_leaf[norm_path(row["official_path"])] = leaf
            leaf_groups[leaf] = (
                row["s16_groups"].strip(),
                row["s3_groups"].strip(),
            )
    else:
        raise ValueError(dataset)
    return source_to_leaf, leaf_groups


class Lookup:
    def __init__(self):
        self.by_relative = defaultdict(set)
        self.by_basename = defaultdict(set)
        self.sources = defaultdict(set)

    def add(self, relative: str, leaf: str, source: str) -> None:
        rel = norm_path(relative)
        base = Path(rel).name
        self.by_relative[rel].add(leaf)
        self.by_basename[base].add(leaf)
        self.sources[(base, leaf)].add(source)

    def resolve(self, local_relative: str):
        local_relative = norm_path(local_relative)
        candidates = [local_relative]
        if local_relative.startswith("images/"):
            candidates.append(local_relative[len("images/"):])
        for key in candidates:
            values = self.by_relative.get(key, set())
            if len(values) == 1:
                return next(iter(values)), "exact_relative_path", key
            if len(values) > 1:
                return None, "ambiguous_relative_path", "|".join(sorted(values))
        base = Path(local_relative).name
        values = self.by_basename.get(base, set())
        if len(values) == 1:
            leaf = next(iter(values))
            source = "|".join(sorted(self.sources[(base, leaf)])[:3])
            return leaf, "exact_basename_unique", source
        if len(values) > 1:
            return None, "ambiguous_basename", "|".join(sorted(values))
        return None, "unmatched", ""
