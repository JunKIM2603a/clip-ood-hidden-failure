# ViT-B/32 Minimum Decisive Experiment — Stage 1

## Entry condition

Stage-0 CLIP ViT-B/16 aggregate reproduction passed before any detector subgroup result was inspected.

Stage-0 mean results:

| Method | Ours FPR95 | Reference FPR95 | Ours AUROC | Reference AUROC |
| --- | ---: | ---: | ---: | ---: |
| MCM | 43.04 | 42.74 | 90.74 | 90.77 |
| NegLabel | 25.63 | 25.40 | 94.16 | 94.21 |

## Why NegLabel must be re-mined on ViT-B/32

NegLabel selects negative labels by comparing their CLIP text embeddings with the ID positive-label embeddings. Because the text encoder changes with the CLIP backbone, the selected negative-label set is backbone-dependent.

Therefore the B/16 selected negative labels are **not reused** for the B/32 Minimum Decisive Experiment.

Frozen B/32 mining rule:

- positive prompt index: 85 (`the nice {label}`)
- noun negative prompt index: 85
- adjective negative prompt: `This is a {word} photo`
- similarity statistic: 0.95 quantile across 1,000 ImageNet positive labels
- select lowest 15% separately from nouns and adjectives
- NegLabel score groups: 100
- corpus files: exact pinned NegLabel repository
- corpus file enumeration: lexical path order for deterministic reproduction

The final item is a deterministic stabilization of the original implementation, whose `os.listdir()` order is filesystem-dependent. It was frozen before B/32 detector results.

## Stage-1A — B/32 aggregate sanity

Run:

```bash
git pull
conda activate clip-ood
bash scripts/baseline/run_stage1a_b32_aggregate.sh
```

The script uses both RTX 4090 GPUs and performs:

```text
GPU0: ImageNet B/32 feature extraction
GPU1: B/32 NegLabel negative-label mining
              ↓
GPU1: OOD B/32 feature extraction
              ↓
GPU0: MCM aggregate score
GPU1: NegLabel aggregate score
```

Published sanity anchors for CLIP ViT-B/32 ImageNet Traditional Four:

| Method | Mean AUROC | Mean FPR95 |
| --- | ---: | ---: |
| MCM | about 89.96 | about 49.96 |
| NegLabel | 93.67 | 27.92 |

These values are comparison anchors only. They are not used for tuning prompts, negative labels, temperature, subgroup definitions, or test thresholds.

After Stage-1A inspect:

```bash
cat results/tables/reproduction_ViT-B-32_mcm.csv
cat results/tables/reproduction_ViT-B-32_neglabel.csv
```

Do not open H1 subgroup outputs before the aggregate sanity check is reviewed.

## Stage-1B — primary H1 predefined semantic groups

After Stage-1A is accepted as a sane B/32 implementation:

```bash
bash scripts/analysis/run_stage1b_h1_predefined.sh
```

Primary sources only:

- iNaturalist → taxonomic order
- SUN → official basic-level scene hierarchy

Places cannot determine the predefined H1 GO/KILL result because it failed the frozen image-to-concept mapping gate before detector scoring.

Stage-1B produces aggregate-vs-worst tables, all eligible subgroup metrics, confidence intervals, and subgroup plots. It intentionally does not assign GO/KILL automatically.
