# Novelty Collision Search

> Search status: **2026-09-23**
>
> This document is a collision audit, not a proof of novelty. The conclusion should be revised whenever a closer paper is found.

## Target claim under test

Original broad idea:

> Strong aggregate VLM OOD performance may hide semantic subgroup and prompt-conditioned failures.

The search shows that parts of this statement are already established independently. The proposed novelty must therefore lie in their **intersection**, not in either component alone.

---

## Search queries used

Core queries:

- vision-language OOD worst-group
- CLIP OOD subgroup
- semantic subgroup OOD
- prompt sensitivity OOD detection
- prompt ensemble CLIP OOD
- VLM OOD reliability

Additional targeted queries:

- VLM OOD prompt phrasing sensitivity
- CLIP OOD shortcut failure
- VLM OOD background shortcut
- prompt robust vision-language model OOD
- VLM OOD benchmark near far OOD
- worst-group CLIP OOD FPR95
- subgroup FPR95 vision-language OOD
- per-sample prompt variance OOD detection
- prompt score variance CLIP OOD

---

# Collision map

| Work | VLM OOD? | Prompt sensitivity / tuning? | Semantic failure / subgroup? | Worst-group OOD metric? | Per-sample prompt dispersion predicts error? | Collision level |
| --- | --- | --- | --- | --- | --- | --- |
| MCM, NeurIPS 2022 | Yes | Prompt ensemble | No | No | No | Baseline |
| NegLabel, ICLR 2024 | Yes | **Yes; wording changes aggregate performance strongly** | No | No | No | **High partial** |
| LAPT, ECCV 2024 | Yes | **Yes; automated prompt tuning** | No | No | No | **High partial** |
| CSP, NeurIPS 2024 | Yes | Text/negative semantic pool design | Semantic concepts used for method design | No | No | Medium |
| Self-Calibrated Tuning, NeurIPS 2024 | Yes | Prompt tuning | Spurious context discussed | No | No | Medium |
| OSPCoOp, CVPR 2025 | Yes | Prompt-learning framework | **Background shortcut failure** | Specialized dataset, not semantic worst-group protocol | No | Medium-high |
| Salaudeen et al., NeurIPS 2025 | No: OOD generalization rather than detector scoring | No | **Yes; coherent hidden subsets** | Not detector FPR95 | No | **High conceptual** |
| Lee et al., arXiv 2025 | **Yes** | **Yes; direct prompt-phrasing sensitivity analysis** | No direct semantic worst-group analysis found | No | No direct test found | **Closest direct collision** |
| Information-theoretical VLM OOD, NeurIPS 2025 | Yes | Not central | No | No | No | Medium |
| Promise, ICLR 2026 | General VLM, not OOD-specific | **Yes; prompt robustness** | No OOD semantic worst-group focus | No | No OOD-error test | Medium |
| OpenOOD-VLM, current codebase | **Yes** | Supports multiple prompt/VLM methods | Near/Far/covariate settings | No semantic worst-group prompt-dispersion protocol identified | No | Implementation overlap |

---

# Closest collisions

## C1. NegLabel prompt-engineering ablation — HIGH PARTIAL COLLISION

NegLabel’s Appendix A.5.3 already shows that changing the textual template can move aggregate ImageNet OOD performance from roughly:

- FPR95 **54.00 / AUROC 86.50** for a poor prompt,
- to FPR95 **25.40 / AUROC 94.21** for the best reported prompt.

### Consequence

The following claim is **invalid as a novelty claim**:

> “We show for the first time that VLM OOD detection is sensitive to prompts.”

### Still open

NegLabel does not test whether the **dispersion of scores for the same input across semantically equivalent prompts** predicts which inputs/groups fail.

---

## C2. LAPT — HIGH PARTIAL COLLISION

LAPT explicitly motivates automated prompt tuning by noting that manual prompt engineering is sensitive to linguistic nuances.

### Consequence

A paper centered only on finding a better prompt or learning an OOD-specific prompt would collide heavily with existing work.

### Still open

Our planned prompt set is not primarily an optimization search space. It is intended as a **measurement instrument for reliability**:

[
{s_p(x)}_{p=1}^P
ightarrow
mu(x),sigma(x),q_alpha(x)
ightarrow
	ext{failure risk}.
]

---

## C3. Lee et al. 2025 — CLOSEST DIRECT COLLISION

*An Empirical Analysis of VLM-based OOD Detection: Mechanisms, Advantages, and Sensitivity* directly reports that VLM-based OOD detectors are highly sensitive to prompt phrasing.

This paper most strongly narrows H2.

### Consequence

The project must distinguish:

**Already studied**
- change prompt set / phrasing;
- observe changes in aggregate OOD AUROC/performance;
- analyze VLM-OOD text sensitivity globally.

**Proposed here**
- hold a set of meaning-preserving prompts;
- obtain a prompt score distribution for each sample;
- quantify **within-sample** sensitivity;
- test whether it predicts actual detection errors;
- aggregate the result by semantic subgroup;
- ask whether high-sensitivity groups are the same groups hidden by aggregate metrics.

### Required analysis to survive this collision

At minimum, H2 should test whether (sigma(x)) contributes information **beyond the mean OOD score**:

[
Pr(	ext{error}mid mu(x),sigma(x)).
]

Preferably add one or more difficulty controls:

- ID/OOD text-embedding or image-text similarity margin;
- nearest-ID concept similarity;
- OOD source fixed effects;
- semantic subgroup fixed effects / stratification.

If (sigma(x)) loses all association after such controls, H2 should fail.

---

## C4. Salaudeen et al. 2025 — HIGH CONCEPTUAL COLLISION, DIFFERENT TASK

This work already establishes that aggregation can hide semantically coherent failures.

### Difference that must be stated precisely

Salaudeen et al.:

[
	ext{ID task accuracy across models}
leftrightarrow
	ext{OOD task accuracy across models}.
]

Current project:

[
	ext{ID-vs-OOD detector score per sample}
ightarrow
	ext{AUROC/FPR95 by semantic subgroup}.
]

### Consequence

Do **not** claim to discover the general aggregation-hides-failure phenomenon. Claim only to test whether it transfers to the distinct regime of zero-shot VLM OOD detection.

---

## C5. OSPCoOp — SYSTEMATIC FAILURE ALREADY EXISTS

OSPCoOp shows that VLM OOD detection can fail through foreground/background shortcuts and creates ImageNet-Bg to expose this.

### Consequence

Avoid a broad claim such as:

> “Existing VLM OOD benchmarks hide all systematic failures.”

The project instead studies a different axis:

- semantic subgroup concentration;
- prompt-conditioned uncertainty;
- worst-group metrics.

Background/context can also become a competing explanation to analyze if subgroup failures appear.

---

## C6. Promise — GENERIC PROMPT ROBUSTNESS COLLISION

Promise (ICLR 2026) explicitly targets instability to natural-language prompt variations across VLM tasks.

### Consequence

H3 should not be framed as a general new prompt-robustness method. Its defensible scope is:

> a lightweight, inference-only aggregation rule over a pre-fixed set of meaning-preserving prompts, evaluated specifically for its effect on **worst-group OOD detection**.

---

# Claims that should NOT appear in the proposal

Do not write:

1. “Prompt sensitivity in VLM OOD has not been studied.”
2. “No prior work has shown systematic VLM OOD failures.”
3. “No prior work has shown aggregation can hide OOD failures.”
4. “This is the first work to make VLMs robust to prompt variation.”
5. “Worst-group failure must be caused by semantic structure.”

All five are either contradicted by prior work or too strong for the current evidence.

---

# Novelty-safe positioning

## Recommended primary research gap

> Existing VLM OOD studies report strong aggregate performance and have separately documented sensitivity to prompt phrasing. Separately, OOD-generalization research has shown that aggregation can hide semantically coherent failures. It remains unclear whether these phenomena intersect in zero-shot VLM OOD detection: specifically, whether aggregate MCM/NegLabel performance hides semantic worst-group failures and whether within-sample score dispersion across meaning-preserving prompts predicts those failures.

This is narrower than the original wording, but materially stronger against prior-art criticism.

---

# Operational novelty conditions

For the paper to make a defensible contribution, the study should satisfy all of the following.

## A. Subgroup result

Report, for each detector / OOD source:

- aggregate AUROC/FPR95;
- subgroup AUROC/FPR95;
- worst-group FPR95;
- between-group dispersion;
- bootstrap confidence intervals.

The main finding cannot rely on a single grouping rule.

## B. Prompt sensitivity result

For each sample (x):

[
mu(x)=rac{1}{P}sum_p s_p(x)
]

[
sigma(x)=operatorname{std}_p s_p(x).
]

Test whether (sigma(x)) predicts detection failure beyond (mu(x)).

Useful endpoints:

- error rate by prompt-sensitivity quantile;
- AUROC of (sigma(x)) as an error-risk score;
- logistic regression or equivalent:
  [
  	ext{error}simmu(x)+sigma(x)+	ext{OOD source}+	ext{difficulty controls};
  ]
- group-level correlation between mean/upper-quantile (sigma) and subgroup FPR95.

## C. Competing explanation tests

If H1/H2 are positive, check whether the effect is explained by:

- generic OOD source difficulty;
- ID–OOD visual/text similarity;
- background/context shortcut;
- subgroup size;
- one arbitrary clustering definition.

## D. Robust aggregation result

Only after H1/H2 characterization:

[
S_{mathrm{robust}}(x)=mu(x)-lambdasigma(x)
]

or a pre-specified lower quantile.

Requirements:

- fixed prompt family;
- (lambda) selected only on validation conditions;
- no final-test OOD tuning;
- report aggregate and worst-group trade-off;
- a negative method result is acceptable if characterization remains strong.

---

# Decision after collision search

## Current decision: **PROCEED, WITH NARROWED POSITIONING**

Reasoning:

- The broad “prompt sensitivity” component is already occupied.
- The broad “aggregation hides failures” component is already occupied.
- The reviewed literature contains related systematic VLM-OOD failures.
- **The intersection of semantic worst-group VLM-OOD evaluation and within-sample prompt-score dispersion as a failure predictor remains unclosed in the reviewed set.**

This makes the **Minimum Decisive Experiment** meaningful: the first priority is now H1 plus the narrowed H2, not proposing a new scoring rule.

## Priority order

1. Reproduce MCM + NegLabel.
2. Freeze subgroup definitions and prompt family.
3. Test aggregate-vs-subgroup gap (H1).
4. Test conditional predictive value of prompt dispersion (narrowed H2).
5. Only if useful, test robust aggregation (H3).

If H1 fails robustly, stop.  
If H1 succeeds and H2 fails, proceed as a semantic hidden-failure paper.  
If H1 and H2 both succeed, H3 becomes a secondary mitigation experiment.

---

# Sources / papers to keep in the active related-work set

- MCM, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/e43a33994a28f746dcfd53eb51ed3c2d-Abstract-Conference.html
- OpenOOD, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/d201587e3a84fc4761eadc743e9b3f35-Abstract-Datasets_and_Benchmarks.html
- NegLabel, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/40eff1670d6b08bb1bda48b0c5f30110-Abstract-Conference.html
- LAPT, ECCV 2024: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09117.pdf
- CSP, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/967017dbe801dc95f5a2587c6d6a1ef3-Abstract-Conference.html
- Self-Calibrated Tuning, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/666e5e1df2d04dbe2b545ea3a3e3f7d3-Abstract-Conference.html
- OSPCoOp, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Overcoming_Shortcut_Problem_in_VLM_for_Robust_Out-of-Distribution_Detection_CVPR_2025_paper.html
- Salaudeen et al., NeurIPS 2025: https://papers.nips.cc/paper_files/paper/2025/hash/87654b1ef6dd2412c71a538944bafe07-Abstract-Conference.html
- Information-theoretical VLM OOD, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/02d965b818b0567a3dab507dddbfb9ec-Abstract-Conference.html
- Lee et al., 2025: https://arxiv.org/abs/2509.13375
- Promise, ICLR 2026: https://openreview.net/forum?id=3wZ6IIwPJq
- OpenOOD-VLM implementation ecosystem: https://github.com/PolyU-VCLab/OpenOOD-VLM
