# CLIP-OOD-Hidden-Failure

CLIP/VLM 기반 zero-shot OOD detector의 높은 aggregate 성능이 특정 semantic subgroup 및 prompt 조건에서의 심각한 실패를 숨기는지 검증하는 연구 저장소입니다.

## Research Question

> 높은 aggregate AUROC를 보이는 CLIP 기반 OOD detector도 특정 semantic subgroup 또는 prompt 조건에서는 체계적으로 실패하는가?

본 연구의 목적은 새로운 architecture를 제안하는 것이 아니라, 기존 평균 중심 OOD 평가가 숨길 수 있는 failure mode를 규명하고 그 발생 조건을 분석하는 것입니다.

## Core Hypotheses

- **H1 — Semantic subgroup hidden failure**  
  CLIP/VLM OOD detector의 오류는 semantic subgroup에 균일하게 분포하지 않고 특정 subgroup에 집중된다.

- **H2 — Prompt sensitivity as a failure signal**  
  prompt에 따른 OOD score 변동성이 높은 sample/group일수록 OOD detection failure가 증가한다.

- **H3 — Robust prompt aggregation**  
  prompt mean만 사용하는 것보다 variance penalty 또는 lower quantile을 고려한 aggregation이 worst-group failure를 줄일 수 있다.

## Baselines

- MCM — Ming et al., NeurIPS 2022
- NegLabel — Jiang et al., ICLR 2024

## Pilot Backbones

- CLIP ViT-B/32
- CLIP ViT-B/16

## Evaluation

- Aggregate AUROC
- AUPR
- FPR95
- Subgroup AUROC / FPR95
- Worst-group FPR95
- Between-group dispersion
- Prompt sensitivity
- Failure–sensitivity association / predictive ability

For prompt-specific score \(s_p(x)\):

\[
\mu(x)=\operatorname{mean}_p s_p(x), \qquad
\sigma(x)=\operatorname{std}_p s_p(x)
\]

Candidate robust score:

\[
S_{robust}(x)=\mu(x)-\lambda\sigma(x)
\]

Lower prompt-score quantiles will also be evaluated.


## Environment Setup

Create and verify the experiment environment **before downloading datasets**.

```bash
git pull
bash scripts/env/setup_conda.sh
conda activate clip-ood
python scripts/env/verify_environment.py
```

The pilot environment is pinned around:

- Python 3.10
- PyTorch 2.9.1
- torchvision 0.24.1
- CUDA 12.6 PyTorch wheel
- NumPy 1.26.4
- exact OpenOOD-VLM / OpenAI CLIP Git commits

After the first trusted installation:

```bash
python scripts/env/environment_lock.py freeze
```

Commit the generated `configs/environment/pilot_environment_lock.json` so other PCs can verify the same critical software environment.

See [docs/environment_setup.md](docs/environment_setup.md) for the rationale, GPU checks, recreation procedure, and multi-PC workflow.

## Dataset Setup

Pilot datasets are installed reproducibly **without committing image data to Git**.

Public OOD datasets:

```bash
python3 -m pip install -r requirements-data.txt
python3 scripts/data/setup_datasets.py --datasets ood
```

Full pilot including ImageNet-1K validation:

```bash
# Accept ImageNet access terms on ILSVRC/imagenet-1k first, then:
hf auth login
python3 scripts/data/setup_datasets.py --datasets all --imagenet-source hf
```

ImageNet can alternatively be installed from already-downloaded official
`ILSVRC2012_img_val.tar` + devkit archives. The installer verifies image/class
counts, creates semantic manifests and path-safe local OpenOOD imglists, and records
dataset fingerprints under `data/state/`.

After the first trusted full installation:

```bash
python3 scripts/data/dataset_lock.py freeze
```

Commit the resulting small `configs/datasets/pilot_data_lock.json` so other PCs can
verify that they use the same dataset revision/archive and generated manifest.

See [docs/dataset_setup.md](docs/dataset_setup.md) for dataset-specific download
rules, ImageNet access constraints, shared-NAS setup, and recovery/fallback options.

## Semantic Subgroup Definitions

Two definitions are evaluated in parallel:

1. Dataset-provided superclass/category
2. Class-name/text-embedding clustering

The claim should not depend on a single subgroup construction.

## Minimum Decisive Experiment

- CLIP ViT-B/32
- MCM + NegLabel
- 1 ID dataset
- 2–3 OOD sources
- Thousands of samples per OOD source
- 8–16 semantically equivalent prompts

### Decision Gate

**GO**  
Aggregate metric remains strong while large subgroup dispersion in AUROC/FPR95 repeatedly appears and bootstrap confidence intervals preserve the gap.

**CONDITIONAL GO**  
Subgroup failure exists but prompt sensitivity is not predictive. Narrow the paper to semantic subgroup hidden failure.

**KILL**  
Across reasonable subgroup definitions, subgroup performance is close to aggregate performance and failure concentration is not reproducible.

## Main Figure Target

Aggregate AUROC is high  
→ certain semantic groups show sharply degraded FPR95/AUROC  
→ errors concentrate in prompt-sensitive samples/groups  
→ robust aggregation changes worst-group failure

## Repository Structure

```text
.
├── README.md
├── environment.yml
├── requirements/
│   └── experiment.txt
├── docs/
│   ├── problem_definition.md
│   ├── hypotheses.md
│   ├── literature_review.md
│   ├── novelty_collision_search.md
│   ├── experimental_protocol.md
│   ├── pilot_dataset_and_subgroups.md
│   ├── environment_setup.md
│   └── dataset_setup.md
├── configs/
│   ├── pilot.yaml
│   ├── environment/
│   │   ├── versions.env
│   │   └── pilot_environment_lock.json
│   ├── datasets/
│   │   ├── pilot_sources.json
│   │   └── pilot_data_lock.json
│   └── subgroups/
├── src/
│   ├── methods/
│   ├── prompts/
│   ├── subgroups/
│   ├── metrics/
│   └── analysis/
├── scripts/
│   ├── env/
│   │   ├── setup_conda.sh
│   │   ├── verify_environment.py
│   │   └── environment_lock.py
│   └── data/
│       ├── setup_datasets.py
│       └── dataset_lock.py
├── tests/
├── results/
│   ├── raw/
│   ├── tables/
│   └── figures/
└── notes/
    └── experiment_log.md
```

Empty code/result directories can be created locally as implementation begins.

## Timeline

- **~2026-10-02**: literature validation, novelty collision search, professor proposal
- **PASS + 1–2 days**: reproduce MCM / NegLabel
- **next 1–2 days**: test subgroup / prompt hypotheses
- **~2026-10-12**: secure main figure and main finding
- **2026-10-13 ~ 2026-11-20**: expand backbone / OOD source / prompt family / statistical robustness
- **~2026-12-14**: complete experiments and thesis

## Research Principle

The study follows this order:

**Problem definition → related work → limitation → hypothesis → falsification experiment → results → interpretation → contribution**

Primary endpoints and subgroup definitions should be fixed before examining final test results whenever possible.
