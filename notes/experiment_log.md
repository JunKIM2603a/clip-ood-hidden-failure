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
