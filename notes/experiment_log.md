# Experiment Log

Use one section per experimental run.

## Template

### Date

YYYY-MM-DD

### Commit / environment

- commit:
- GPU:
- CUDA:
- PyTorch:
- CLIP implementation:

### Configuration

- method:
- backbone:
- ID dataset:
- OOD source:
- subset size:
- prompt set:
- subgroup definition:

### Result

- AUROC:
- AUPR:
- FPR95:
- worst-group FPR95:
- subgroup dispersion:
- prompt sensitivity association:

### Interpretation

State only what is supported by the current result.

### Decision impact

- GO / CONDITIONAL GO / KILL / no decision yet

### Notes

Record anomalies, failed runs, implementation deviations, and reasons for reruns.

## 2026-09-24 — MOS semantic mapping feasibility (pre-score)

- Detector scores inspected: **No**
- Stage: dataset / semantic-label feasibility gate before H1
- iNaturalist MOS-10k:
  - images: 10,000
  - mapped: 10,000 / 10,000
  - mapping coverage: **100.00%**
  - eligible predefined primary groups: **16**
  - primary eligibility: **PASS**
- SUN: metadata download failed before mapping because the historical Princeton `Partitions.zip` URL returned HTTP 404.
- Action taken before detector scoring:
  - replaced the broken SUN metadata dependency with the complete SUN397 image-path inventory bundled by TensorFlow Datasets;
  - pinned TFDS commit `764eb13ad271629b32eca17d96b12bc13b509db4` and per-file Git blob SHA;
  - frozen subgroup thresholds remain unchanged.

Interpretation: iNaturalist is usable as a primary predefined-semantic H1 source. This is **not** evidence for or against H1; no MCM/NegLabel subgroup performance has been examined.


## 2026-09-24 — Completed MOS semantic mapping feasibility gate

- Detector scores inspected: **No**
- Frozen mapping threshold: **>=90%**
- Frozen group eligibility: **>=200 images, >=2 leaf concepts, >=3 eligible groups/source**

### iNaturalist MOS-10k
- mapped: 10,000 / 10,000
- coverage: 100.00%
- eligible predefined groups: 16
- primary eligibility: **PASS**

### SUN MOS-10k
- mapped: 10,000 / 10,000
- coverage: 100.00%
- eligible predefined groups: 3
- primary eligibility: **PASS**

### Places MOS-10k
- mapped: 178 / 10,000
- coverage: 1.78%
- eligible predefined groups: 0
- primary eligibility: **FAIL / DEMOTED**
- frozen thresholds were not relaxed

### Protocol impact
- proceed with primary H1 on **iNaturalist + SUN**;
- retain Places for standard aggregate Traditional Four reproduction and secondary analysis only;
- this decision was made before MCM/NegLabel subgroup scores were inspected.

Decision impact: **no H1/H2 decision yet; feasibility gate passed with two primary OOD sources.**

## 2026-09-24 — Pre-score H1 metric and alternate-grouping freeze

- Detector subgroup scores inspected: **No**
- Primary predefined H1 sources after mapping gate: **iNaturalist + SUN**
- Places primary status: **demoted before detector scoring** (1.78% mapping coverage)
- H1 FPR95 convention frozen:
  - larger score = more ID-like
  - one ImageNet-derived 95%-TPR threshold shared across all OOD subgroups
  - threshold = ID 5th percentile with order-statistic rule `higher`
  - subgroup-specific threshold retuning forbidden
- H1 pilot bootstrap convention frozen:
  - full ImageNet ID reference held fixed
  - OOD samples bootstrapped
  - aggregate-to-group gap bootstrap preserves source/group nesting
- Alternate text-clustering definition frozen:
  - model: `sentence-transformers/all-MiniLM-L6-v2`
  - model revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
  - package: `sentence-transformers==5.1.2`
  - normalized raw leaf-concept embeddings
  - agglomerative clustering / average linkage / cosine metric
  - k rule: `floor(sqrt(N)+0.5)`, bounded [3,12]
  - expected k: iNaturalist=10, SUN=7, Places=7
- These rules are frozen before MCM/NegLabel subgroup outcomes are inspected.

Decision impact: no H1/H2 decision yet.

## 2026-09-24 — Dependency compatibility amendment before H1 scoring

- Detector subgroup scores inspected: **No**
- Previous text-clustering runtime pin: `sentence-transformers==6.1.0`
- Conflict observed during environment reconstruction:
  - existing `huggingface-hub==0.35.3`
  - `sentence-transformers==6.1.0` requires `huggingface-hub>=1.3.0,<2.0.0`
- Resolution:
  - keep the existing tested Hugging Face stack (`huggingface-hub==0.35.3`, `datasets==4.1.1`);
  - use `sentence-transformers==5.1.2`, `transformers==4.57.1`, `tokenizers==0.22.1`;
  - keep the exact same embedding model, model revision, clustering algorithm, k rule, and subgroup eligibility rules.
- Research-definition impact: **none**; runtime dependency compatibility only.

## 2026-09-24 — Stage-0 aggregate reproduction PASS

- Subgroup detector scores inspected: **No**
- Backbone: CLIP ViT-B/16
- Traditional Four aggregate reproduction:
  - MCM ours: mean FPR95 **43.04**, mean AUROC **90.74**
  - MCM reference: FPR95 **42.74**, AUROC **90.77**
  - NegLabel ours: mean FPR95 **25.63**, mean AUROC **94.16**
  - NegLabel reference: FPR95 **25.40**, AUROC **94.21**
- Gate decision: **PASS**
- No formal reproduction tolerance had been preregistered; PASS is based on close aggregate and per-source agreement rather than a newly invented cutoff.
- Next: ViT-B/32 backbone-specific NegLabel mining → B/32 aggregate sanity check → frozen H1 predefined subgroup analysis.
