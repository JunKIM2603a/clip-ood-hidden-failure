# Experimental Protocol

> Protocol revision date: **2026-09-23**  
> This revision incorporates the literature/collision search before the main hypothesis experiments.

## Minimum Decisive Experiment

### Goal

Determine within 1–2 days whether hidden semantic subgroup failure exists strongly enough to justify the full study, and whether **within-sample prompt-score dispersion** contains failure information beyond the mean OOD score.

### Baselines

- MCM
- NegLabel

Implementation should preferentially reuse/validate against official code or the current OpenOOD-VLM ecosystem before making local modifications.

### Backbone

Pilot:

- CLIP ViT-B/32

Expansion:

- CLIP ViT-B/16

### Data

- One ImageNet/OpenOOD-style ID dataset
- Two to three OOD sources
- Capped pilot subsets with thousands of samples per source
- Expand to official full splits only after the pilot establishes a reproducible signal

Pilot subset sampling must be deterministic and seed-recorded.

---

## Prompt protocol

### Pilot size

- 8 meaning-preserving templates
- expand to 16–32 only after the pilot

### Prompt-design rule

The pilot prompt family should vary **surface phrasing without intentionally changing visual conditions or sentiment**.

Avoid using modifiers such as:

- dark
- blurry
- cropped
- low-resolution
- good / nice

as the primary sensitivity family, because prior work already shows those globally change OOD performance and some alter the implied image distribution.

Prefer neutral paraphrases such as:

- “a photo of a {label}”
- “a photograph of a {label}”
- “an image of a {label}”
- “a picture of a {label}”
- “this is a photo of a {label}”
- “this is an image of a {label}”
- “a photo depicting a {label}”
- “an image depicting a {label}”

Exact templates must be frozen before reading final subgroup results.

### Prompt score storage

Do **not** save only the averaged score.

For every sample save:

- sample ID/path;
- ID/OOD source;
- class/concept metadata where available;
- semantic subgroup assignments;
- score for every prompt (s_p(x));
- prompt mean (mu(x));
- prompt standard deviation (sigma(x));
- optional lower quantiles.

This enables later analysis without re-running CLIP inference.

---

## Semantic subgroup definitions

Use both:

1. dataset-provided superclass/category;
2. class-name/text-embedding clustering.

Both definitions are required to test whether the effect depends on one grouping scheme.

### Anti-artifact rules

- freeze clustering algorithm, embedding model, number of clusters or selection rule before final test analysis;
- report group sizes;
- impose a pre-specified minimum group size for primary worst-group claims;
- retain small groups in descriptive appendices rather than letting tiny groups define the headline worst case.

---

## Metric definitions

### Aggregate

- AUROC
- AUPR
- FPR95

### Subgroup

For each OOD subgroup (g):

- compare that subgroup against the **same ID reference set**;
- compute subgroup AUROC;
- compute subgroup FPR95 using the same score orientation/convention as the aggregate benchmark.

### Worst group

- worst-group FPR95 = maximum valid subgroup FPR95;
- worst-group AUROC = minimum valid subgroup AUROC;
- aggregate-to-worst-group gap;
- between-group dispersion.

The score orientation and exact FPR95 convention must be unit-tested before experiments.

---

## H1 analysis — hidden semantic failure

For every method × OOD source × subgroup definition:

1. report aggregate metrics;
2. report the full subgroup metric distribution;
3. report worst-group metrics;
4. report aggregate-to-worst-group gap;
5. bootstrap confidence intervals;
6. check whether the same or related groups recur across methods/prompts/backbones.

### H1 success pattern

A convincing signal is not merely one bad small group. It should show:

- sizeable degradation;
- sufficient group sample size;
- uncertainty that does not trivially overlap the aggregate expectation;
- recurrence across OOD sources, grouping definitions, methods, or backbones.

---

## H2 analysis — prompt dispersion as an error predictor

For sample (x) and prompt (p):

[
mu(x)=operatorname{mean}_p s_p(x),
qquad
sigma(x)=operatorname{std}_p s_p(x).
]

The key question is **incremental information**, not simple correlation.

### Required comparison

Fit/evaluate an error-risk model using only the mean score:

[
M_0:quad 	ext{error}sim mu(x)
]

and compare against:

[
M_1:quad 	ext{error}sim mu(x)+sigma(x).
]

If (M_1) does not reproducibly improve prediction/association beyond (M_0), H2 is not supported.

### Required descriptive analyses

- detection error rate by (sigma(x)) quantile;
- distribution of (sigma(x)) for correct vs failed detections;
- subgroup mean/upper-quantile (sigma) versus subgroup FPR95;
- confidence intervals for the association.

### Preferred competing-difficulty controls

Where available add:

- OOD-source indicator;
- nearest-ID concept similarity;
- image–ID-text similarity / score margin;
- subgroup size;
- subgroup indicator or stratified analysis.

A positive H2 finding is strongest if prompt dispersion remains informative after these controls.

---

## H3 analysis — robust prompt aggregation

Run only after H1/H2 characterization.

Candidate:

[
S_{mathrm{robust}}(x)=mu(x)-lambdasigma(x)
]

Alternative:

- pre-specified lower prompt-score quantile.

### Rules

- prompt family fixed;
- (lambda) selected on validation conditions only;
- no final-test OOD tuning;
- compare against ordinary prompt mean;
- report both aggregate and worst-group changes;
- report whether improvements transfer across OOD sources.

H3 is secondary: failure characterization remains publishable even if robust aggregation does not improve results.

---

## Statistical robustness

- bootstrap confidence intervals for aggregate and subgroup metrics;
- bootstrap uncertainty for aggregate-to-worst-group gaps;
- repeat across OOD sources;
- repeat across both subgroup definitions;
- record random seeds;
- do not tune subgroup definitions, prompts, thresholds, or (lambda) by inspecting final test OOD outcomes.

For later full experiments, consider bootstrap resampling at the natural independent unit (sample/class as appropriate) and document the choice.

---

## Competing explanations

If H1/H2 are positive, explicitly test:

1. **generic dataset difficulty**;
2. **ID–OOD semantic/visual similarity**;
3. **background/context shortcut**;
4. **subgroup-size noise**;
5. **one arbitrary clustering definition**.

---

## Decision rule

### GO

Aggregate performance is good, but substantial subgroup degradation repeatedly appears and survives uncertainty analysis.

If (sigma(x)) also adds predictive information beyond (mu(x)), retain prompt sensitivity as a second main finding.

### CONDITIONAL GO

H1 is reproducible, but narrowed H2 is not.

Narrow the contribution to:

> semantic subgroup hidden failures in VLM OOD evaluation.

### KILL

Meaningful failure concentration does not appear across reasonable subgroup definitions/sources, or apparent worst groups disappear after accounting for sample size and uncertainty.

---

## Minimum output of the pilot

The pilot is not complete until it produces:

1. one aggregate-vs-subgroup metric table;
2. one subgroup FPR95/AUROC distribution plot;
3. one worst-group table with bootstrap CI;
4. one (sigma)-quantile vs error plot;
5. (M_0) vs (M_1) comparison;
6. a GO / CONDITIONAL GO / KILL decision recorded in the experiment log.
