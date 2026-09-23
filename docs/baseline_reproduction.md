# Baseline Reproduction Protocol

> Stage begins only after the semantic mapping feasibility gate.
>
> Current primary H1 sources after the pre-score gate: **iNaturalist + SUN**.
> Places remains in aggregate Traditional Four reproduction but is not primary-eligible for predefined semantic subgroup H1.

## Why reproduce ViT-B/16 first

The research pilot ultimately tests CLIP ViT-B/32 and then ViT-B/16, but the baseline reproduction stage begins with **ViT-B/16** because:

- the published MCM benchmark reports a canonical ViT-B/16 result;
- official NegLabel uses CLIP ViT-B/16 by default;
- NegLabel's repository provides a selected ImageNet-1K negative-label file generated for its official B/16 setup;
- a matched shared backbone lets us validate both methods before changing prompt/backbone conditions.

This reproduction stage is a calibration step. It does not change the pre-registered pilot backbone.

## Reference implementations

The environment pins immutable copies of:

- MCM: `deeplearning-wisc/MCM@ea7130f851e7d462cacd21f0e87a127705700bd9`
- NegLabel: `XueJiang16/NegLabel@3253db684075b2db47844676eccdfda40d67a573`

The pilot runner reads original reference assets from these repositories:

### MCM

- ImageNet cleaned class names: `data/ImageNet/imagenet_class_clean.npy`
- positive prompt: `a photo of a {label}`
- score:
  [
  S_{MCM}(x)=max_c operatorname{softmax}(z(x)/T)_c
  ]
- reproduction temperature: (T=1)

### NegLabel

- positive ImageNet class list and prompt definitions from the original code;
- positive prompt index 85: `the nice {label}`;
- official provided selected negative prompt file:
  `selected_neg_labels/selected_neg_labels_in1k_10k.txt`;
- 100 negative-label groups;
- CLIP cosine logits multiplied by 100 as in the original implementation;
- deterministic negative-label permutation with seed 0.

The optimized implementation uses log-sum-exp to compute the positive probability mass per negative group. A unit test checks numerical agreement against the original explicit softmax loop.

## Why not run current OpenOOD-VLM official scripts directly

OpenOOD-VLM remains a useful comparison ecosystem, but the pinned current repository contains development/debug artifacts, including a `pdb.set_trace()` inside one OOD evaluator path.

For the decisive pilot we therefore use a thin evaluator whose scoring formulas are checked directly against the original MCM/NegLabel implementations. OpenOOD-VLM results remain an external reproduction comparison, not the sole executable source of truth.

## Shared image feature cache

Image encoding is the expensive common operation. It is performed once per backbone and stored under:

```text
data/features/openai_clip/ViT-B-16/
├── imagenet.npz
├── inaturalist.npz
├── sun.npz
├── places.npz
└── dtd.npz
```

Each cache contains:

- normalized CLIP image feature;
- ID label when available;
- exact relative image path.

These files remain outside Git.

## Prerequisites

The following must pass first:

```bash
conda activate clip-ood

python scripts/env/verify_environment.py
python scripts/data/setup_datasets.py --datasets all --verify-only
pytest -q tests/test_baseline_scores.py
```

ImageNet-1K validation must be present before reproduction.

## Refresh immutable reference repositories

After pulling the version that introduced baseline reproduction support:

```bash
git pull
bash scripts/env/setup_conda.sh
```

This clones/checks the exact MCM and NegLabel commits under `third_party/`.

## Step 1 — feature cache

```bash
python scripts/baseline/prepare_features.py \
  --backbone ViT-B/16 \
  --gpu 0
```

Do not start two feature-extraction processes against the same missing cache simultaneously.

## Step 2 — two-GPU reproduction

After the feature cache exists:

```bash
bash scripts/baseline/run_dual_gpu.sh
```

The workflow uses:

```text
GPU 0 -> MCM
GPU 1 -> NegLabel
```

and both consume the same cached image embeddings.

Equivalent individual commands:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/baseline/run_reproduction.py \
  --method mcm --backbone ViT-B/16 --gpu 0

CUDA_VISIBLE_DEVICES=1 python scripts/baseline/run_reproduction.py \
  --method neglabel --backbone ViT-B/16 --gpu 0
```

## Outputs

Per-sample confidence scores:

```text
results/raw/reproduction/ViT-B-16/
├── mcm/
│   ├── imagenet.csv
│   ├── inaturalist.csv
│   ├── sun.csv
│   ├── places.csv
│   └── dtd.csv
└── neglabel/
    └── ...
```

Aggregate metric tables:

```text
results/tables/
├── reproduction_ViT-B-16_mcm.csv
└── reproduction_ViT-B-16_neglabel.csv
```

Each OOD dataset reports:

- FPR95
- AUROC
- AUPR-IN
- AUPR-OUT

plus the arithmetic mean across the Traditional Four sources.

## Published reference points

These are **reference points, not automatic pass thresholds**:

| Method | Backbone | Mean FPR95 ↓ | Mean AUROC ↑ |
| --- | --- | ---: | ---: |
| MCM | CLIP ViT-B/16 | about 42.7 | about 90.8 |
| NegLabel | CLIP ViT-B/16 | about 25.4 | about 94.2 |

Exact equality is not expected automatically because implementation details such as CLIP checkpoint packaging can differ. If reproduction diverges materially, stop before H1 and inspect:

1. ImageNet class-name ordering;
2. CLIP checkpoint / preprocessing;
3. score orientation;
4. prompt string;
5. negative-label ordering/grouping;
6. dataset identity/counts.

Do not compensate by tuning on the final OOD test sets.

## Gate into the hypothesis experiment

Proceed from reproduction into H1/H2 only after:

1. all five feature caches have the expected sample counts;
2. MCM and NegLabel score unit tests pass;
3. aggregate Traditional Four metrics are qualitatively consistent with the official references;
4. any meaningful discrepancy is explained and logged.

Then start the Minimum Decisive Experiment on ViT-B/32 with the already-frozen prompt and subgroup protocol.
