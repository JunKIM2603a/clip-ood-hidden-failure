# Experimental Protocol

## Minimum Decisive Experiment

### Goal

Determine within 1–2 days whether hidden semantic subgroup failure exists strongly enough to justify the full study.

### Baselines

- MCM
- NegLabel

### Backbone

Pilot:

- CLIP ViT-B/32

Expansion:

- CLIP ViT-B/16

### Data

- One ImageNet/OpenOOD-style ID dataset
- Two to three OOD sources
- Capped pilot subsets with thousands of samples per source
- Expand to official full splits after the hypothesis shows signal

### Prompts

- 8–16 semantically equivalent prompt templates for pilot
- Expand to 16–32 templates for robustness experiments

### Semantic subgroup definitions

1. Dataset-provided superclass/category
2. Class-name or text-embedding clustering

Both definitions are required to test whether the effect depends on one grouping scheme.

## Primary metrics

- Aggregate AUROC
- Aggregate AUPR
- Aggregate FPR95
- Subgroup AUROC
- Subgroup FPR95
- Worst-group FPR95
- Between-group dispersion

## Prompt sensitivity analysis

For every sample:

- prompt score mean
- prompt score standard deviation
- optional lower quantiles

Test association between sensitivity and:

- misdetection indicator
- OOD score error margin
- subgroup-level failure

Report correlation and predictive ability where appropriate.

## Statistical robustness

- bootstrap confidence intervals for aggregate and subgroup metrics
- uncertainty for worst-group gaps
- repeat across OOD sources
- repeat across both subgroup definitions
- avoid tuning thresholds or \(\lambda\) on final test OOD

## Decision rule

### GO

Aggregate performance is good, but large subgroup degradation repeatedly appears and survives bootstrap uncertainty.

### CONDITIONAL GO

Subgroup failure is reproducible, but prompt sensitivity is not predictive. Narrow the contribution to semantic subgroup hidden failure.

### KILL

Meaningful failure concentration does not appear across reasonable subgroup definitions and performance closely tracks aggregate metrics.
