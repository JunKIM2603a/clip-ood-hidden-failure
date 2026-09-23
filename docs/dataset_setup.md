# Dataset Setup and Multi-PC Reproducibility

This project **does not store dataset images in Git**. Git stores only the installer, source specification, subgroup mappings, and a small dataset fingerprint lock.

The intended workflow is:

```text
git pull
   ↓
run dataset installer
   ↓
images are downloaded/extracted under ./data
   ↓
image/class counts are verified
   ↓
local manifests and OpenOOD-compatible imglists are generated
   ↓
dataset fingerprint can be frozen/checked across PCs
```

The repository already ignores `data/`.

## Quick start

Install the optional downloader dependencies:

```bash
python3 -m pip install -r requirements-data.txt
```

### OOD datasets only

No account is required for the primary OOD downloads:

```bash
python3 scripts/data/setup_datasets.py --datasets ood
```

Expected contents:

| Dataset | Images | Leaf concepts |
| --- | ---: | ---: |
| iNaturalist MOS | 10,000 | 110 |
| SUN MOS | 10,000 | 50 |
| Places MOS | 10,000 | 50 |
| DTD | 5,640 | 47 |

The three 10k OOD datasets are the **MOS-curated subsets** used in the traditional ImageNet OOD benchmark. DTD is the complete Oxford DTD release.

## ImageNet-1K validation

ImageNet is treated differently because its images are gated by the ImageNet Terms of Access. The installer does not bypass that gate.

Only the **50,000-image validation split** is required for this pilot. The 1.28M-image training split is not needed for zero-shot MCM/NegLabel evaluation.

### Recommended automatic route: gated Hugging Face mirror

1. Accept the ImageNet access conditions once on:
   - https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. Authenticate the current machine:

```bash
hf auth login
```

or provide a token through the environment:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

Never commit the token.

Then run:

```bash
python3 scripts/data/setup_datasets.py --datasets all --imagenet-source hf
```

The installer records the exact Hugging Face dataset revision returned by the Hub.

### Alternative: official ImageNet archives

If the official files have already been downloaded from ImageNet:

```bash
export IMAGENET_VAL_ARCHIVE=/path/to/ILSVRC2012_img_val.tar
export IMAGENET_DEVKIT_ARCHIVE=/path/to/ILSVRC2012_devkit_t12.tar.gz

python3 scripts/data/setup_datasets.py \
  --datasets imagenet \
  --imagenet-source official
```

This route verifies the standard MD5 values before extraction:

- `ILSVRC2012_img_val.tar`: `29b22e2961454d5413ddabcf34fc5622`
- `ILSVRC2012_devkit_t12.tar.gz`: `fa75699e90414af021442c21a62c3abf`

The official-archive route uses `torchvision.datasets.ImageNet` to arrange the validation images.

## Dataset-specific policy

| Dataset | Primary route | Auth | Optional fallback |
| --- | --- | --- | --- |
| ImageNet-1K val | gated `ILSVRC/imagenet-1k` | required | official ImageNet val + devkit |
| iNaturalist MOS-10k | original MOS Wisconsin URL | none | OpenOOD Google Drive mirror |
| SUN MOS-10k | original MOS Wisconsin URL | none | OpenOOD Google Drive mirror |
| Places MOS-10k | original MOS Wisconsin URL | none | OpenOOD Google Drive mirror |
| DTD | Oxford VGG official archive | none | OpenOOD Google Drive mirror |

The OpenOOD mirror is not used silently. If an original OOD host is temporarily unavailable:

```bash
python3 scripts/data/setup_datasets.py \
  --datasets ood \
  --allow-openood-fallback
```

For ImageNet, a mirror that bypasses the normal access agreement is deliberately not the default path.

## Local directory layout

After installation:

```text
data/
├── images_largescale/
│   ├── imagenet_1k/
│   │   └── val/
│   │       ├── 0000/
│   │       ├── 0001/
│   │       └── ... 1000 label directories
│   ├── inaturalist/
│   │   ├── images/
│   │   └── leaf_aliases.csv
│   ├── sun/
│   │   ├── images/
│   │   └── leaf_aliases.csv
│   ├── places/
│   │   ├── images/
│   │   └── leaf_aliases.csv
│   └── DTD/
│       └── images/
├── manifests/
├── benchmark_imglist_local/
│   └── imagenet/
├── state/
└── .cache/
    ├── downloads/
    └── extract/
```

### Why OOD directory names are normalized

MOS concepts contain spaces and punctuation, e.g. `Coprosma lucida` or `desert (sand)`. OpenOOD's `ImglistDataset` splits each image-list line at the first space, so a filesystem path containing spaces is unsafe.

The installer therefore changes only **path names**:

```text
Coprosma lucida  →  Coprosma_lucida
```

The original concept is preserved in:

- `leaf_aliases.csv`
- `data/manifests/<dataset>.csv`

Image bytes are not modified.

This gives us both semantic subgroup names and OpenOOD-safe paths.

## Verify an installation

Run:

```bash
python3 scripts/data/setup_datasets.py --datasets all --verify-only
```

The frozen count checks are:

- ImageNet: 50,000 validation images / 1,000 classes
- iNaturalist: 10,000 / 110 leaf concepts
- SUN: 10,000 / 50
- Places: 10,000 / 50
- DTD: 5,640 / 47

These checks happen before MCM/NegLabel subgroup results are inspected.

The next image-level subgroup audit joins the generated manifests with the already committed mappings under `configs/subgroups/mappings/` and applies the pre-fixed rules:

- at least 200 OOD images per headline group
- at least 2 leaf concepts
- at least 90% mapping coverage
- at least 3 eligible groups per primary source

## Freeze the exact dataset fingerprint

The installer records local provenance in `data/state/*.json`, including:

- source kind
- OOD archive SHA-256
- Hugging Face ImageNet revision
- generated manifest SHA-256
- image/class counts

After the **first fully verified installation**, freeze that fingerprint into Git:

```bash
python3 scripts/data/dataset_lock.py freeze

git add configs/datasets/pilot_data_lock.json
git commit -m "Freeze pilot dataset fingerprint"
git push
```

Then another PC can use:

```bash
git pull
python3 -m pip install -r requirements-data.txt
python3 scripts/data/setup_datasets.py --datasets all
python3 scripts/data/dataset_lock.py check
```

The large images never enter Git, but the exact source/archive/revision and manifest fingerprints can still be compared.

## Multiple-PC deployment choices

### A. Each PC downloads independently

This is the default reproducibility path.

```text
Git source spec
   ├── PC A → download → verify → fingerprint check
   ├── PC B → download → verify → fingerprint check
   └── PC C → download → verify → fingerprint check
```

### B. Shared NAS/NFS/SSD

If several experiment machines can access the same authorized storage:

```bash
export CLIP_OOD_DATA_ROOT=/mnt/research-data/clip-ood-hidden-failure
python3 scripts/data/setup_datasets.py --datasets all
```

Use the same environment variable on the other machines. This avoids duplicate downloads.

### C. Restricted-internet machines

Copy the original public OOD archives or official ImageNet archives through an approved internal storage channel, then point the installer at the local ImageNet archives. Keeping `data/.cache/downloads/` also makes reinstallations faster.

## What not to do

### Do not store the datasets in normal Git or Git LFS

Reasons:

- datasets are tens of gigabytes;
- clone/pull behavior becomes poor;
- binary history is expensive;
- ImageNet has explicit access terms;
- reproducibility is better represented by source URLs + hashes + counts.

### Do not recreate the MOS subsets from full upstream datasets

The pilot uses the exact MOS-curated iNaturalist/SUN/Places subsets. Independently downloading the full datasets and resampling 10,000 images would define a different experiment.

## Useful commands

```bash
# Public OOD datasets only
python3 scripts/data/setup_datasets.py --datasets ood

# Full setup with gated HF ImageNet
python3 scripts/data/setup_datasets.py --datasets all --imagenet-source hf

# One dataset
python3 scripts/data/setup_datasets.py --datasets sun

# Verify only
python3 scripts/data/setup_datasets.py --datasets all --verify-only

# Reinstall one damaged dataset
python3 scripts/data/setup_datasets.py --datasets places --force

# Use OpenOOD mirror only as fallback
python3 scripts/data/setup_datasets.py --datasets ood --allow-openood-fallback

# Shared data disk
CLIP_OOD_DATA_ROOT=/mnt/nas/clip-ood-data \
  python3 scripts/data/setup_datasets.py --datasets all

# Freeze/check local dataset fingerprint
python3 scripts/data/dataset_lock.py freeze
python3 scripts/data/dataset_lock.py check
```


## Troubleshooting: Hugging Face token not found

If the installer reports `LocalTokenNotFoundError` or says that no Hugging Face token is available:

```bash
conda activate clip-ood
hf auth login
hf auth whoami
```

The ImageNet access agreement must also be accepted in a browser for the same Hugging Face account:

- https://huggingface.co/datasets/ILSVRC/imagenet-1k

If `hf auth whoami` succeeds but ImageNet still returns an access error, the most likely cause is that the gated dataset terms were not accepted by that same account. Accept the terms in the browser and rerun only ImageNet:

```bash
python scripts/data/setup_datasets.py \
  --datasets imagenet \
  --imagenet-source hf
```

Public OOD data can be installed independently while ImageNet access is being resolved:

```bash
python scripts/data/setup_datasets.py --datasets ood
```

Never paste a Hugging Face access token into this repository, an issue, a log committed to Git, or a chat message.
