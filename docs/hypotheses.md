# Hypotheses

> Updated after literature/collision review on **2026-09-23**, before the main hypothesis experiments.
>
> The original scientific direction is preserved. H2/H3 are operationally narrowed because prior work already establishes generic prompt sensitivity in VLM OOD detection. This prevents a post-result novelty shift.

## H1 — Semantic subgroup hidden failure

OOD errors are not uniformly distributed across semantic subgroups. Some coherent groups exhibit substantially worse AUROC and/or FPR95 than the aggregate result suggests.

### Primary H1 endpoints

- subgroup FPR95;
- worst-group FPR95;
- subgroup AUROC;
- aggregate-to-worst-group gap;
- between-group dispersion;
- bootstrap confidence intervals.

### Robustness requirement

H1 should not depend on only one subgroup construction.

Evaluate both:

1. predefined dataset superclass/category;
2. class-name/text-embedding clustering.

---

## H2 — Within-sample prompt sensitivity predicts failure

Prior work already shows that **aggregate VLM-OOD performance changes with prompt wording**. Therefore H2 is deliberately narrower.

For a fixed set of meaning-preserving prompt templates and sample (x), let (s_p(x)) be the OOD score under prompt (p):

[
mu(x)=operatorname{mean}_p s_p(x)
]

[
sigma(x)=operatorname{std}_p s_p(x).
]

### H2 claim

Samples or semantic groups with larger **within-sample prompt-score dispersion** (sigma(x)) have higher OOD detection failure, and (sigma(x)) provides predictive information beyond the mean score (mu(x)).

### Minimum H2 test

At minimum compare:

[
Pr(	ext{error}mid mu(x))
]

against

[
Pr(	ext{error}mid mu(x),sigma(x)).
]

Also report:

- failure rate by (sigma(x)) quantile;
- ability of (sigma(x)) to discriminate error vs correct detection;
- subgroup-level relationship between prompt sensitivity and FPR95/AUROC.

### Preferred controls

Where feasible include:

- OOD source;
- nearest-ID concept / image-text similarity;
- score margin or another generic difficulty proxy;
- subgroup size.

This directly addresses the competing explanation that (sigma(x)) merely tracks generally difficult examples.

---

## H3 — Inference-only robust prompt aggregation can reduce worst-group failure

Only after H1/H2 characterization, test whether a pre-specified aggregation rule changes worst-group reliability.

Candidate:

[
S_{robust}(x)=mu(x)-lambdasigma(x)
]

Alternative:

- lower prompt-score quantile.

### H3 scope

H3 is **not** a claim to introduce general prompt-robust VLM training. It is a lightweight, inference-only aggregation experiment over a fixed prompt family.

### Anti-leakage requirement

(lambda) must be selected using a limited validation condition and never tuned on final test OOD data.

Report both:

- aggregate metric change;
- worst-group metric change.

A worst-group improvement that substantially destroys aggregate performance is not considered a clean success.

---

# Null / competing hypotheses

## H0-1 — No hidden subgroup gap

Aggregate and subgroup performance are not meaningfully different once sampling uncertainty is considered.

## H0-2 — Prompt dispersion adds no predictive information

After conditioning on mean OOD score and reasonable difficulty controls, prompt dispersion has no reproducible relationship with detection failure.

## Alternative A — Generic difficulty

Failure concentration is explained by generic dataset/sample difficulty rather than semantic structure.

## Alternative B — ID–OOD similarity

Failure is mainly explained by visual/textual similarity to ID classes.

## Alternative C — Context / background shortcut

Observed subgroup patterns are caused mainly by background/context shortcuts rather than the semantic grouping itself.

## Alternative D — Group-definition artifact

The effect appears only for one arbitrary subgroup definition and disappears under reasonable alternatives.

---

# Pre-experiment decision rules

## GO

Proceed with the full paper when:

- aggregate metrics are strong;
- meaningful subgroup dispersion / worst-group degradation repeats across OOD sources;
- the gap survives bootstrap uncertainty;
- and the result is not confined to one arbitrary subgroup definition.

If narrowed H2 also succeeds, retain prompt sensitivity as a second main finding.

## CONDITIONAL GO

H1 succeeds but narrowed H2 does not.

Then the project becomes primarily:

> **Semantic subgroup hidden failures in VLM OOD evaluation.**

Do not change H2 after inspecting results.

## KILL

Stop or redesign if:

- subgroup performance closely tracks aggregate performance across reasonable definitions/sources;
- observed extremes are explainable by small-group sampling noise;
- no reproducible failure concentration remains after robustness checks.

---

# Falsification principle

The hypotheses and primary endpoints are fixed before examining final test outcomes.

If a hypothesis fails, preserve the negative result rather than redefining the hypothesis after seeing the data.
