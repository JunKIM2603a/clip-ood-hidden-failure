#!/usr/bin/env python3
"""Install the frozen pilot datasets without storing image data in Git.

OOD datasets come from the original MOS/Oxford distributions. ImageNet-1K
validation is installed either from the gated Hugging Face ILSVRC mirror
(after accepting the ImageNet terms) or from official ILSVRC2012 archives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "configs" / "datasets" / "pilot_sources.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def image_count(root: Path) -> int:
    return len(image_files(root))


def digest(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def human_bytes(value: int | None) -> str:
    if value is None:
        return "?"
    n = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def download_http(urls: list[str], dst: Path, force: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"[reuse] {dst}")
        return dst

    part = dst.with_suffix(dst.suffix + ".part")
    failures = []
    for url in urls:
        try:
            existing = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "clip-ood-hidden-failure/1.0"}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            request = urllib.request.Request(url, headers=headers)
            print(f"[download] {url}")
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                append = existing > 0 and status == 206
                if not append:
                    existing = 0
                mode = "ab" if append else "wb"
                length = response.headers.get("Content-Length")
                total = existing + int(length) if length else None
                downloaded = existing
                last = time.monotonic()
                with part.open(mode) as out:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                        downloaded += len(block)
                        if time.monotonic() - last >= 5:
                            if total:
                                print(
                                    f"  {human_bytes(downloaded)} / {human_bytes(total)} "
                                    f"({100.0 * downloaded / total:.1f}%)"
                                )
                            else:
                                print(f"  {human_bytes(downloaded)}")
                            last = time.monotonic()
            part.replace(dst)
            print(f"[downloaded] {dst} ({human_bytes(dst.stat().st_size)})")
            return dst
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            print(f"[warn] failed source: {url}\n       {exc}")
    raise RuntimeError("All source URLs failed:\n" + "\n".join(failures))


def download_gdrive(file_id: str, dst: Path, force: bool = False) -> Path:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "OpenOOD fallback requires gdown. Run: pip install -r requirements-data.txt"
        ) from exc
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=file_id, output=str(dst), quiet=False)
    if not result or not dst.exists():
        raise RuntimeError(f"gdown failed for {file_id}")
    return dst


def safe_extract(archive: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    base = dst.resolve()

    def safe(name: str) -> bool:
        target = (dst / name).resolve()
        return os.path.commonpath([str(base), str(target)]) == str(base)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                if not safe(item.filename):
                    raise RuntimeError(f"Unsafe zip path: {item.filename}")
            zf.extractall(dst)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            for item in tf.getmembers():
                if not safe(item.name):
                    raise RuntimeError(f"Unsafe tar path: {item.name}")
            try:
                tf.extractall(dst, filter="data")
            except TypeError:
                tf.extractall(dst)
        return

    raise RuntimeError(f"Unsupported archive: {archive}")


def collapse_single_dir(root: Path) -> Path:
    current = root
    while True:
        children = [p for p in current.iterdir() if not p.name.startswith(".")]
        dirs = [p for p in children if p.is_dir() and p.name != "__MACOSX"]
        files = [p for p in children if p.is_file()]
        if len(dirs) == 1 and not files:
            current = dirs[0]
        else:
            return current


def move_extracted(staging: Path, destination: Path) -> None:
    source = collapse_single_dir(staging)
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != staging:
        shutil.move(str(source), str(destination))
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for child in list(staging.iterdir()):
            shutil.move(str(child), str(destination / child.name))


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    out = re.sub(r"_+", "_", out).strip("._-")
    return out or hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def canonicalize_ood(root: Path) -> None:
    """Make OOD paths safe for OpenOOD's space-delimited imglist format."""
    candidate = root / "images" if (root / "images").exists() else root
    classes = [
        p for p in sorted(candidate.iterdir())
        if p.is_dir() and image_count(p) > 0 and p.name != "images"
    ]
    if not classes:
        return

    images_root = root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    aliases = []
    used = set()

    for class_dir in classes:
        raw = class_dir.name
        path_leaf = slugify(raw)
        if path_leaf in used and raw != path_leaf:
            path_leaf += "_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        used.add(path_leaf)
        target = images_root / path_leaf
        if class_dir != target:
            if target.exists():
                raise RuntimeError(f"Class path collision: {raw} -> {path_leaf}")
            shutil.move(str(class_dir), str(target))
        aliases.append((path_leaf, raw))

        for image in image_files(target):
            if " " not in image.name:
                continue
            safe_name = slugify(image.stem) + image.suffix
            new_path = image.with_name(safe_name)
            if new_path.exists():
                suffix = hashlib.sha1(image.name.encode("utf-8")).hexdigest()[:8]
                new_path = image.with_name(f"{slugify(image.stem)}_{suffix}{image.suffix}")
            image.rename(new_path)

    with (root / "leaf_aliases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path_leaf", "raw_leaf_concept"])
        writer.writerows(aliases)


def aliases(root: Path) -> dict[str, str]:
    path = root / "leaf_aliases.csv"
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["path_leaf"]: row["raw_leaf_concept"] for row in csv.DictReader(f)}


def leaf_for(root: Path, image: Path) -> str:
    parts = image.relative_to(root).parts
    if not parts:
        return ""
    mapping = aliases(root)
    if parts[0].lower() == "images":
        key = parts[1] if len(parts) >= 3 else ""
    else:
        key = parts[0] if len(parts) >= 2 else ""
    return mapping.get(key, key)


def build_ood_manifest(name: str, root: Path, data_root: Path) -> tuple[Path, int]:
    out = data_root / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    leaves = set()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "relative_path", "leaf_concept"])
        for image in image_files(root):
            leaf = leaf_for(root, image)
            if leaf:
                leaves.add(leaf)
            writer.writerow([name, image.relative_to(data_root).as_posix(), leaf])
    return path, len(leaves)


def build_ood_imglist(name: str, root: Path, data_root: Path) -> Path:
    out = data_root / "benchmark_imglist_local" / "imagenet"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"test_{name}_local.txt"
    with path.open("w", encoding="utf-8") as f:
        for image in image_files(root):
            rel = image.relative_to(data_root / "images_largescale").as_posix()
            if " " in rel:
                raise RuntimeError(f"OpenOOD-unsafe path still contains a space: {rel}")
            f.write(f"{rel} -1\n")
    return path


def write_state(data_root: Path, name: str, payload: dict) -> None:
    out = data_root / "state"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_ood(name: str, cfg: dict, data_root: Path) -> dict:
    root = data_root / cfg["destination"]
    files = image_files(root)
    leaves = {leaf_for(root, p) for p in files}
    leaves.discard("")
    result = {
        "dataset": name,
        "root": str(root),
        "images": len(files),
        "expected_images": int(cfg["expected_images"]),
        "leaf_concepts": len(leaves),
        "expected_leaf_concepts": int(cfg["expected_leaf_concepts"]),
    }
    result["ok"] = (
        result["images"] == result["expected_images"]
        and result["leaf_concepts"] == result["expected_leaf_concepts"]
    )
    return result


def setup_ood(
    name: str,
    cfg: dict,
    data_root: Path,
    force: bool,
    allow_fallback: bool,
) -> dict:
    current = verify_ood(name, cfg, data_root)
    if current["ok"] and not force:
        print(f"[ok] {name}: already installed")
        return current

    cache = data_root / ".cache" / "downloads"
    archive = cache / cfg["archive_name"]
    source_kind = "official"
    try:
        download_http(cfg["official_urls"], archive, force=force)
    except Exception:
        if not allow_fallback:
            raise
        source_kind = "openood_mirror"
        archive = cache / f"{name}.openood.zip"
        download_gdrive(cfg["openood_google_drive_id"], archive, force=force)

    destination = data_root / cfg["destination"]
    extract_root = data_root / ".cache" / "extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=extract_root) as temp:
        staging = Path(temp)
        safe_extract(archive, staging)
        move_extracted(staging, destination)

    canonicalize_ood(destination)
    manifest, _ = build_ood_manifest(name, destination, data_root)
    imglist = build_ood_imglist(name, destination, data_root)
    result = verify_ood(name, cfg, data_root)
    state = {
        **result,
        "installed_at": now(),
        "source_kind": source_kind,
        "source_archive_sha256": digest(archive),
        "manifest_sha256": digest(manifest),
        "openood_local_imglist": str(imglist),
    }
    write_state(data_root, name, state)
    if not result["ok"]:
        raise RuntimeError(
            f"{name} verification failed:\n{json.dumps(result, indent=2)}"
        )
    print(
        f"[verified] {name}: {result['images']} images, "
        f"{result['leaf_concepts']} leaf concepts"
    )
    return state


def verify_imagenet(cfg: dict, data_root: Path) -> dict:
    root = data_root / cfg["destination"]
    val = root / "val"
    classes = [p for p in val.iterdir() if p.is_dir()] if val.exists() else []
    result = {
        "dataset": "imagenet",
        "root": str(root),
        "images": image_count(val),
        "expected_images": int(cfg["expected_images"]),
        "classes": len(classes),
        "expected_classes": int(cfg["expected_classes"]),
    }
    result["ok"] = (
        result["images"] == result["expected_images"]
        and result["classes"] == result["expected_classes"]
    )
    return result


def imagenet_outputs(
    root: Path,
    data_root: Path,
    label_names: list[str],
) -> tuple[Path, Path]:
    manifests = data_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = manifests / "imagenet_val.csv"
    imglists = data_root / "benchmark_imglist_local" / "imagenet"
    imglists.mkdir(parents=True, exist_ok=True)
    imglist = imglists / "test_imagenet_local.txt"

    with manifest.open("w", newline="", encoding="utf-8") as mf, imglist.open(
        "w", encoding="utf-8"
    ) as of:
        writer = csv.writer(mf)
        writer.writerow(["dataset", "relative_path", "label", "label_name"])
        for class_dir in sorted(p for p in (root / "val").iterdir() if p.is_dir()):
            label = int(class_dir.name)
            label_name = label_names[label] if label < len(label_names) else class_dir.name
            for image in image_files(class_dir):
                writer.writerow(
                    ["imagenet", image.relative_to(data_root).as_posix(), label, label_name]
                )
                rel = image.relative_to(data_root / "images_largescale").as_posix()
                of.write(f"{rel} {label}\n")
    return manifest, imglist


def setup_imagenet_hf(cfg: dict, data_root: Path, force: bool) -> dict:
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "HF ImageNet mode needs datasets + huggingface_hub. "
            "Run: pip install -r requirements-data.txt"
        ) from exc

    root = data_root / cfg["destination"]
    current = verify_imagenet(cfg, data_root)
    if current["ok"] and not force:
        print("[ok] imagenet: already installed")
        return current
    if force and root.exists():
        shutil.rmtree(root)
    val_root = root / "val"
    val_root.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or True
    try:
        info = HfApi().dataset_info(cfg["hf_repo"], token=token)
        revision = info.sha
    except Exception as exc:
        raise RuntimeError(
            "ImageNet access is gated. Accept the terms at "
            "https://huggingface.co/datasets/ILSVRC/imagenet-1k and run "
            "hf auth login (or set HF_TOKEN)."
        ) from exc

    print(f"[imagenet:hf] {cfg['hf_repo']} revision={revision}")
    ds = load_dataset(
        cfg["hf_repo"],
        split="validation",
        revision=revision,
        token=token,
    )
    ds = ds.cast_column("image", HFImage(decode=False))
    label_names = list(getattr(ds.features["label"], "names", []) or [])

    for i, row in enumerate(ds):
        label = int(row["label"])
        obj = row["image"]
        source_path = obj.get("path") if isinstance(obj, dict) else None
        suffix = Path(source_path).suffix if source_path and Path(source_path).suffix else ".JPEG"
        out = val_root / f"{label:04d}" / f"{i:08d}{suffix}"
        if out.exists() and out.stat().st_size > 0:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        raw = obj.get("bytes") if isinstance(obj, dict) else None
        if raw is not None:
            tmp.write_bytes(raw)
        elif source_path:
            shutil.copy2(source_path, tmp)
        else:
            raise RuntimeError(f"ImageNet row {i} has neither bytes nor path")
        tmp.replace(out)
        if (i + 1) % 1000 == 0:
            print(f"  extracted {i + 1}/{len(ds)}")

    (root / "labels.json").write_text(
        json.dumps(label_names, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest, imglist = imagenet_outputs(root, data_root, label_names)
    result = verify_imagenet(cfg, data_root)
    state = {
        **result,
        "installed_at": now(),
        "source_kind": "huggingface_gated",
        "hf_repo": cfg["hf_repo"],
        "hf_revision": revision,
        "manifest_sha256": digest(manifest),
        "openood_local_imglist": str(imglist),
    }
    write_state(data_root, "imagenet", state)
    if not result["ok"]:
        raise RuntimeError(f"ImageNet verification failed:\n{json.dumps(result, indent=2)}")
    return state


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def setup_imagenet_official(
    cfg: dict,
    data_root: Path,
    val_archive: Path,
    devkit_archive: Path,
    force: bool,
) -> dict:
    try:
        from torchvision.datasets import ImageNet
    except ImportError as exc:
        raise RuntimeError(
            "Official ImageNet archive mode needs torchvision from the experiment environment."
        ) from exc

    for path, expected in [
        (val_archive, cfg["official_val_md5"]),
        (devkit_archive, cfg["official_devkit_md5"]),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
        actual = digest(path, "md5")
        if actual != expected:
            raise RuntimeError(f"MD5 mismatch for {path}: {actual} != {expected}")

    root = data_root / cfg["destination"]
    current = verify_imagenet(cfg, data_root)
    if current["ok"] and not force:
        print("[ok] imagenet: already installed")
        return current
    if force and root.exists():
        shutil.rmtree(root)

    work = data_root / ".cache" / "imagenet_official"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    link_or_copy(val_archive, work / "ILSVRC2012_img_val.tar")
    link_or_copy(devkit_archive, work / "ILSVRC2012_devkit_t12.tar.gz")

    print("[imagenet:official] torchvision is extracting the validation split")
    ds = ImageNet(str(work), split="val")
    label_names = [
        ", ".join(names) if isinstance(names, (tuple, list)) else str(names)
        for names in ds.classes
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / "val").mkdir(parents=True, exist_ok=True)
    for wnid_dir in sorted((work / "val").iterdir()):
        if wnid_dir.is_dir():
            label = ds.wnid_to_idx[wnid_dir.name]
            shutil.move(str(wnid_dir), str(root / "val" / f"{label:04d}"))

    (root / "labels.json").write_text(
        json.dumps(label_names, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest, imglist = imagenet_outputs(root, data_root, label_names)
    result = verify_imagenet(cfg, data_root)
    state = {
        **result,
        "installed_at": now(),
        "source_kind": "official_imagenet_archives",
        "official_val_archive_md5": cfg["official_val_md5"],
        "official_devkit_archive_md5": cfg["official_devkit_md5"],
        "manifest_sha256": digest(manifest),
        "openood_local_imglist": str(imglist),
    }
    write_state(data_root, "imagenet", state)
    shutil.rmtree(work, ignore_errors=True)
    if not result["ok"]:
        raise RuntimeError(f"ImageNet verification failed:\n{json.dumps(result, indent=2)}")
    return state


def targets(values: list[str]) -> list[str]:
    out = []
    for value in values:
        if value == "all":
            out += ["imagenet", "inaturalist", "sun", "places", "dtd"]
        elif value == "ood":
            out += ["inaturalist", "sun", "places", "dtd"]
        else:
            out.append(value)
    return list(dict.fromkeys(out))


def print_results(results: list[dict]) -> None:
    print("\nDataset verification")
    print("-" * 80)
    for r in results:
        detail = ""
        if "classes" in r:
            detail = f" classes={r['classes']}/{r['expected_classes']}"
        if "leaf_concepts" in r:
            detail = f" leaf={r['leaf_concepts']}/{r['expected_leaf_concepts']}"
        print(
            f"{r['dataset']:<14} images={r['images']}/{r['expected_images']}"
            f"{detail} => {'OK' if r['ok'] else 'FAIL'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "ood", "imagenet", "inaturalist", "sun", "places", "dtd"],
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CLIP_OOD_DATA_ROOT", str(REPO_ROOT / "data")),
    )
    parser.add_argument(
        "--imagenet-source",
        choices=["auto", "hf", "official"],
        default="auto",
    )
    parser.add_argument("--imagenet-val-archive")
    parser.add_argument("--imagenet-devkit-archive")
    parser.add_argument("--allow-openood-fallback", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = spec()
    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(data_root).free / 1024**3
    if free_gb < cfg.get("recommended_free_gb", 35):
        print(f"[warn] only {free_gb:.1f} GiB free under {data_root}")

    selected = targets(args.datasets)
    print(f"[data-root] {data_root}")
    print(f"[targets] {', '.join(selected)}")
    results = []

    if args.verify_only:
        for name in selected:
            ds = cfg["datasets"][name]
            results.append(
                verify_imagenet(ds, data_root)
                if name == "imagenet"
                else verify_ood(name, ds, data_root)
            )
        print_results(results)
        return 0 if all(r["ok"] for r in results) else 2

    for name in selected:
        ds = cfg["datasets"][name]
        if name != "imagenet":
            results.append(
                setup_ood(
                    name,
                    ds,
                    data_root,
                    args.force,
                    args.allow_openood_fallback,
                )
            )
            continue

        val_arg = args.imagenet_val_archive or os.environ.get("IMAGENET_VAL_ARCHIVE")
        devkit_arg = args.imagenet_devkit_archive or os.environ.get("IMAGENET_DEVKIT_ARCHIVE")
        source = args.imagenet_source
        if source == "auto":
            source = (
                "official"
                if val_arg and devkit_arg and Path(val_arg).exists() and Path(devkit_arg).exists()
                else "hf"
            )
        print(f"[imagenet-source] {source}")
        if source == "hf":
            results.append(setup_imagenet_hf(ds, data_root, args.force))
        else:
            if not val_arg or not devkit_arg:
                raise RuntimeError(
                    "Official mode needs --imagenet-val-archive and --imagenet-devkit-archive "
                    "or IMAGENET_VAL_ARCHIVE / IMAGENET_DEVKIT_ARCHIVE."
                )
            results.append(
                setup_imagenet_official(
                    ds,
                    data_root,
                    Path(val_arg),
                    Path(devkit_arg),
                    args.force,
                )
            )

    print_results(results)
    print("\nDone. Images stay outside Git; manifests and provenance are under the data root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
