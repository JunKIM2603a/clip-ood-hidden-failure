# Literature Review

> Evidence review status: **2026-09-23**
>
> Principle: primary proceedings/PDFs are preferred. Later work is used only to assess whether the proposed gap has already been closed. “No direct collision found” means only that no direct match was identified in the reviewed search set; it is not proof that no such work exists.

## Executive synthesis

The four core papers support the project, but with an important qualification.

- **MCM (NeurIPS 2022)** establishes strong zero-shot VLM OOD detection and reports aggregate OOD metrics. It also studies prompt ensembling, but only at aggregate dataset level.
- **NegLabel (ICLR 2024)** improves aggregate OOD detection substantially and, critically, already shows that prompt wording can change OOD performance dramatically. Therefore, “VLM OOD is prompt-sensitive” is **not** a novel claim by itself.
- **OpenOOD (NeurIPS 2022)** motivates standardized near/far OOD evaluation and explicitly shows that performance varies across OOD datasets, but it does not provide semantic subgroup / worst-group evaluation inside an OOD source.
- **Salaudeen et al. (NeurIPS 2025)** demonstrates that aggregation can hide semantically coherent OOD-generalization failures. This is the strongest conceptual bridge, but its target is **OOD generalization accuracy across models**, not zero-shot VLM OOD detection scores.

The novelty-safe intersection that remains to be tested is:

> **Whether strong aggregate VLM OOD performance hides semantic worst-group failures, and whether within-sample / within-group dispersion of OOD scores across meaning-preserving prompts predicts those failures.**

---

## 1. Ming et al. — MCM

**Paper:** *Delving into Out-of-Distribution Detection with Vision-Language Representations*  
**Venue:** NeurIPS 2022  
**Method:** Maximum Concept Matching (MCM)  
**Primary source:** https://proceedings.neurips.cc/paper_files/paper/2022/hash/e43a33994a28f746dcfd53eb51ed3c2d-Abstract-Conference.html

### 1) Problem defined by the authors

Most previous OOD detection methods use a single modality. The paper asks how joint vision-language representations, particularly CLIP, can be used directly for zero-shot OOD detection without task-specific ID training.

MCM represents each ID class with a textual concept prototype and scores an image by its maximum normalized similarity to the ID concepts.

### 2) Dataset / model / method / metric

- **VLM:** primarily CLIP; experiments include different CLIP backbones.
- **ID datasets:** ImageNet-1K and smaller semantic-class benchmarks such as CUB, Stanford Cars, Food-101, Oxford-Pet, plus ImageNet subset tasks.
- **Traditional ImageNet OOD sources:** iNaturalist, SUN, Places, Textures.
- **Hard OOD settings:** semantically similar ImageNet class splits and Waterbirds-style spurious-correlation evaluation.
- **Score:** maximum concept-matching probability over ID textual concepts.
- **Main metrics:** AUROC and FPR95.

### 3) Main experimental results

The paper reports that multimodal MCM outperforms a visual-only baseline by **13.1 AUROC points** on a hard task with semantically similar ID/OOD classes.

On the standard ImageNet-1K four-OOD benchmark, the reported aggregate results are approximately:

| Setting | Avg. FPR95 ↓ | Avg. AUROC ↑ |
| --- | ---: | ---: |
| MCM, CLIP ViT-B/16 | 42.74 | 90.77 |
| MCM, stronger CLIP backbone reported in paper | 38.17 | 91.49 |

The paper also tests **prompt ensembling**. Averaging several prompt-derived textual features can improve aggregate performance; the custom 5-prompt ensemble is reported to slightly outperform the larger 80-prompt ensemble in that experiment.

### 4) What remains unresolved

MCM does **not** answer the present project’s core questions:

- Results are primarily aggregated over complete OOD datasets/sources.
- It does not report semantic subgroup AUROC/FPR95 or worst-group FPR95.
- Prompt ensembling averages text features globally; it does not measure per-sample score dispersion across prompts.
- It does not test whether prompt dispersion predicts false positives / false negatives.
- It does not ask whether aggregate strength hides coherent semantic failure concentrations.

### 5) Direct connection to this project

MCM is an ideal baseline because:

1. it is a canonical zero-shot CLIP OOD method;
2. it reports strong aggregate performance;
3. it already exposes a prompt interface, allowing controlled prompt perturbation without changing the visual backbone;
4. the proposed study changes the **evaluation lens**, not the architecture.

### 6) Has later work already closed the gap?

Later VLM-OOD work substantially improves aggregate performance and explores prompt construction, negative labels, tuning, feature adaptation, and new scoring functions. However, in the reviewed literature, no work was found that jointly evaluates:

- MCM-style VLM OOD,
- semantic subgroup / worst-group failure,
- **within-sample prompt-score dispersion**, and
- its predictive relationship with detection failure.

This is a provisional gap, subject to continued collision search.

---

## 2. Jiang et al. — NegLabel

**Paper:** *Negative Label Guided OOD Detection with Pretrained Vision-Language Models*  
**Venue:** ICLR 2024  
**Method:** NegLabel  
**Primary source:** https://proceedings.iclr.cc/paper_files/paper/2024/hash/40eff1670d6b08bb1bda48b0c5f30110-Abstract-Conference.html

### 1) Problem defined by the authors

Although VLMs provide both visual and textual information, existing OOD detectors underuse the language modality. NegLabel introduces a large set of mined **negative labels** and constructs an OOD score using competition between ID labels and negative labels.

### 2) Dataset / model / method / metric

- **Default VLM:** CLIP ViT-B/16 in major experiments; multiple VLM architectures are also evaluated.
- **ID:** ImageNet-1K in the principal large-scale benchmark.
- **OOD:** iNaturalist, SUN, Places, Textures.
- **Additional evaluations:** hard semantic OOD settings, Waterbirds-style spurious setting, and domain-shifted ID variants.
- **Negative-label source:** large lexical corpus / WordNet-based candidate pool.
- **Metrics:** AUROC and FPR95, with additional standard OOD metrics in experiments.

### 3) Main experimental results

For ImageNet-1K against the four traditional OOD datasets, NegLabel reports approximately:

| Method | Avg. FPR95 ↓ | Avg. AUROC ↑ |
| --- | ---: | ---: |
| MCM baseline as reported by NegLabel | 43.93 | 90.82 |
| NegLabel | **25.40** | **94.21** |

The paper also reports improvements in hard semantic OOD tasks and under domain shift.

#### Critical prompt-sensitivity evidence already in NegLabel

Appendix A.5.3 explicitly studies prompt engineering. The reported aggregate ImageNet-1K OOD result changes substantially with wording:

| Prompt | Avg. FPR95 ↓ | Avg. AUROC ↑ |
| --- | ---: | ---: |
| “A dark photo of a <label>.” | 54.00 | 86.50 |
| “A blurry photo of a <label>.” | 46.64 | 88.68 |
| “A low resolution photo of a <label>.” | 42.73 | 89.89 |
| “A cropped photo of a <label>.” | 47.41 | 88.28 |
| “A photo of a <label>.” | 36.78 | 91.34 |
| “A good photo of a <label>.” | 32.65 | 92.33 |
| “<label>.” | 28.34 | 93.65 |
| “The nice <label>.” | **25.40** | **94.21** |

This is a major prior-art constraint for the current project.

### 4) What remains unresolved

NegLabel establishes **global prompt wording sensitivity**, but it does not test the proposed mechanism:

- no semantic subgroup/worst-group evaluation of prompt sensitivity;
- no per-sample prompt score vector (s_p(x)) across meaning-preserving templates;
- no test that (sigma_p(s_p(x))) predicts detection error;
- no control showing whether that relationship remains after accounting for mean score / generic sample difficulty;
- no robust inference-time aggregation designed specifically to reduce worst-group failure.

### 5) Direct connection to this project

NegLabel is important for two reasons.

First, it is a strong baseline that can make hidden-failure analysis harder and therefore more informative. Second, its appendix proves that prompt wording is already a recognized source of aggregate performance variation.

Therefore the project must **not** claim:

> “We discover that VLM-based OOD detection is sensitive to prompts.”

The defensible question is narrower:

> “Does within-sample / within-semantic-group prompt-score instability identify where the detector will fail, even when aggregate performance looks strong?”

### 6) Has later work already closed the gap?

LAPT (ECCV 2024), later prompt-tuning methods, and a 2025 empirical VLM-OOD analysis explicitly address prompt sensitivity or prompt optimization. These substantially collide with a generic prompt-sensitivity contribution.

No reviewed paper was found that closes the **joint semantic-worst-group + within-sample prompt-dispersion prediction** question.

---

## 3. Yang et al. — OpenOOD

**Paper:** *OpenOOD: Benchmarking Generalized Out-of-Distribution Detection*  
**Venue:** NeurIPS 2022, Datasets and Benchmarks  
**Primary source:** https://proceedings.neurips.cc/paper_files/paper/2022/hash/d201587e3a84fc4761eadc743e9b3f35-Abstract-Datasets_and_Benchmarks.html

### 1) Problem defined by the authors

The OOD literature lacked a unified, strict, comprehensive benchmark, making cross-paper comparisons difficult and sometimes unfair. OpenOOD unifies related settings and implementations under a generalized OOD framework.

### 2) Dataset / model / method / metric

The original benchmark covers multiple settings including anomaly detection, open-set recognition, and OOD detection. For OOD detection it includes benchmarks such as MNIST, CIFAR-10/100, and ImageNet, with a distinction between **Near-OOD** and **Far-OOD**.

The codebase implements **30+ methods** in the published paper.

- **Primary reported metric:** AUROC.
- **Also reported:** FPR95 and AUPR.

### 3) Main experimental results

The paper’s primary contribution is benchmarking rather than a single detector. It shows that:

- method ranking can vary across OOD datasets/settings;
- preprocessing/data-augmentation approaches can be highly competitive;
- post-hoc methods remain strong and complementary;
- standardized evaluation materially changes how progress should be interpreted.

### 4) What remains unresolved

OpenOOD aggregates at benchmark / OOD-source level. It does not establish a protocol for:

- semantic subgroup AUROC/FPR95 inside an OOD source;
- worst-semantic-group FPR95;
- prompt-conditioned variability, because the original benchmark is not VLM/prompt-centered.

### 5) Direct connection to this project

OpenOOD provides the evaluation backbone and the near/far OOD framing. The present project can be viewed as adding a **reliability axis** to conventional OOD evaluation:

[
	ext{aggregate metric} ightarrow 	ext{semantic-group distribution} ightarrow 	ext{worst group}.
]

### 6) Has later work already closed the gap?

The current **OpenOOD-VLM** codebase extends the ecosystem to VLM methods and includes MCM, NegLabel, LAPT, AdaNeg and newer methods, with near/far OOD and covariate-shift settings:

https://github.com/PolyU-VCLab/OpenOOD-VLM

This is highly relevant for implementation/reproduction efficiency. However, the reviewed OpenOOD-VLM materials do not present semantic subgroup/worst-group prompt-dispersion evaluation as the central protocol proposed here.

---

## 4. Salaudeen et al. — aggregation-hidden OOD failures

**Paper:** *Aggregation Hides Out-of-Distribution Generalization Failures from Spurious Correlations*  
**Venue:** NeurIPS 2025  
**Primary source:** https://papers.nips.cc/paper_files/paper/2025/hash/87654b1ef6dd2412c71a538944bafe07-Abstract-Conference.html

### 1) Problem defined by the authors

Across many distribution-shift benchmarks, higher ID accuracy often correlates with higher OOD accuracy (“accuracy-on-the-line”). The paper asks whether this apparent robustness can be an artifact of averaging heterogeneous OOD examples.

The authors introduce **OODSelect**, which identifies semantically coherent subsets for which the aggregate positive trend can break or reverse.

### 2) Dataset / model / method / metric

The study covers multiple OOD-generalization datasets including:

- PACS
- VLCS
- TerraIncognita
- WILDS Camelyon variants
- CivilComments
- chest X-ray data

The analysis considers large collections of models and relationships between ID and OOD **task accuracy**, not OOD detector AUROC/FPR95.

### 3) Main experimental results

The paper finds semantically coherent OOD subsets—sometimes more than half of the nominal OOD set—where aggregate “accuracy-on-the-line” behavior breaks down.

A concrete example reported in the paper is TerraIncognita: the full-set ID/OOD accuracy correlation is strongly positive, while a selected coherent subset exhibits a strong negative relationship.

The paper also examines zero-shot VLM trends. It reports positive ID-versus-selected-subset trends for the evaluated VLMs and explicitly cautions that this should **not** be interpreted as proof that VLMs are immune to spurious correlations.

### 4) What remains unresolved

This work is conceptually central but methodologically different from the current project.

It studies:

- **OOD generalization**: task accuracy under distribution shift,
- across a population of trained models.

The current project studies:

- **OOD detection**: ID-vs-OOD discrimination,
- with zero-shot VLM detectors,
- using AUROC/FPR95 and prompt-dependent detection scores.

Therefore Salaudeen et al. motivates the aggregation concern but does not directly answer whether MCM/NegLabel hide semantic worst-group detector failures.

### 5) Direct connection to this project

It provides the strongest evidence for the general proposition:

> Aggregate OOD evaluation can hide coherent, consequential failures.

The proposed project asks whether the same evaluation pathology transfers to a distinct regime:

> **zero-shot CLIP/VLM OOD detection**.

### 6) Has later work already closed the gap?

No direct follow-up identified in the reviewed set jointly transfers OODSelect-style hidden-failure characterization to MCM/NegLabel while also incorporating prompt-conditioned score dispersion.

---

# Later work that constrains the research gap

## LAPT — ECCV 2024

**Paper:** *Label-driven Automated Prompt Tuning for OOD Detection with Vision-Language Models*  
Source: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09117.pdf

LAPT explicitly states that manual prompt engineering is sensitive to linguistic nuances and replaces hand-crafted prompts with automated label-driven prompt tuning.

**Implication:** prompt sensitivity / better prompt selection alone cannot be the contribution.

## CSP — NeurIPS 2024

**Paper:** *Conjugated Semantic Pool Improves OOD Detection with Pre-trained Vision-Language Models*  
Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/967017dbe801dc95f5a2587c6d6a1ef3-Abstract-Conference.html

CSP expands the negative semantic pool using modified superclass concepts and reports a sizable FPR95 improvement.

**Implication:** semantic organization of negative labels is already an active design direction, but this is different from evaluating failures by semantic subgroup.

## Self-Calibrated Tuning — NeurIPS 2024

**Paper:** *Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection*  
Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/666e5e1df2d04dbe2b545ea3a3e3f7d3-Abstract-Conference.html

SCT targets unreliable/spurious OOD regularization in prompt-tuned VLM detectors.

**Implication:** the project should emphasize **evaluation and failure characterization**, rather than broadly claiming to be the first reliability analysis of VLM OOD.

## OSPCoOp — CVPR 2025

**Paper:** *Overcoming Shortcut Problem in VLM for Robust Out-of-Distribution Detection*  
Source: https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Overcoming_Shortcut_Problem_in_VLM_for_Robust_Out-of-Distribution_Detection_CVPR_2025_paper.html

OSPCoOp identifies a concrete VLM-OOD shortcut failure involving foreground/background entanglement and introduces ImageNet-Bg.

**Implication:** systematic VLM-OOD failure modes are already known. Our contribution must be about **aggregate-hidden semantic group failure and prompt-conditioned predictability**, not “VLM OOD can fail systematically” in general.

## Information-theoretical VLM OOD framework — NeurIPS 2025

**Paper:** *An Information-theoretical Framework for Understanding Out-of-distribution Detection with Pretrained Vision-Language Models*  
Source: https://proceedings.neurips.cc/paper_files/paper/2025/hash/02d965b818b0567a3dab507dddbfb9ec-Abstract-Conference.html

This work gives a theoretical PMI interpretation of CLIP-based post-hoc OOD detection and develops a stronger method.

**Implication:** mechanistic understanding of VLM OOD is progressing, but this does not replace subgroup/worst-group reliability evaluation.

## Lee et al. — VLM-OOD sensitivity analysis, 2025

**Paper:** *An Empirical Analysis of VLM-based OOD Detection: Mechanisms, Advantages, and Sensitivity*  
Source: https://arxiv.org/abs/2509.13375

This is the **closest collision** found. The paper systematically analyzes VLM-OOD behavior and concludes that methods are relatively robust to common image noise but highly sensitive to prompt phrasing.

**Implication:** H2 must not be framed as discovery of prompt sensitivity. The novel target must be the relationship between **within-sample prompt-score dispersion and actual detection failure**, ideally conditional on mean score / visual similarity / dataset difficulty.

## Promise — ICLR 2026

**Paper:** *PROMISE: Prompt-Robust Vision-Language Models via Meta-Finetuning*  
Source: https://openreview.net/forum?id=3wZ6IIwPJq

Promise studies prompt robustness broadly across VLM tasks and reduces cross-prompt sensitivity through meta-finetuning.

**Implication:** generic “robustness to prompt variations” is not novel. H3 should stay narrowly scoped to an **inference-only fixed-prompt-set aggregation rule evaluated on worst-group OOD detection**, not broad prompt-robust VLM training.

---

# Current evidence-based conclusion

## What is already established

1. CLIP/VLMs can perform strong zero-shot OOD detection.
2. Negative labels and learned/tuned textual spaces can strongly improve aggregate results.
3. VLM-OOD performance is sensitive to prompt wording.
4. VLM-OOD has known systematic failure modes such as background shortcuts.
5. Aggregating heterogeneous OOD examples can hide failures in OOD generalization.

## What remains a defensible question

The reviewed literature does **not yet directly answer**:

> When MCM/NegLabel look strong in aggregate, are their errors disproportionately concentrated in semantically coherent OOD subgroups, and can the **same sample’s score dispersion across meaning-preserving prompts** predict those hidden failures?

That is the research question to preserve.
