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

### Data — frozen pilot selection

The Minimum Decisive Experiment uses the same benchmark lineage as MCM/NegLabel:

- **ID:** ImageNet-1K, ILSVRC2012 validation set, full 50,000 images.
- **OOD-1:** iNaturalist MOS curated split, 10,000 images.
- **OOD-2:** SUN MOS curated split, 10,000 images.
- **OOD-3:** Places365 MOS curated split, 10,000 images.
- **Aggregate reproduction control:** DTD / Textures, 5,640 images.

No additional OOD subsampling is applied in the pilot. The MOS 10k sources are already curated/capped benchmark subsets.

The primary predefined subgroup schemes are frozen in configs/subgroups/predefined.yaml and documented in docs/pilot_dataset_and_subgroups.md:

- iNaturalist → official taxonomic **order**;
- SUN → official hierarchy **basic-level (15-node level)**;
- Places365 → official hierarchy **S16**.

For primary worst-group reporting, groups require at least **200 images** and **2 leaf concepts**. A source must retain at least **3 eligible groups** and at least **90% mapping coverage** to define the pilot GO/KILL decision.

The exact class-to-group lookup tables must be generated from official taxonomy/hierarchy sources and committed before detector subgroup scores are inspected.

---



## Pre-score feasibility amendment — 2026-09-24

This amendment was made **before any MCM/NegLabel subgroup detector scores were inspected**.

Image-level semantic mapping feasibility produced:

| Source | Mapping coverage | Eligible predefined groups | Primary status |
| --- | ---: | ---: | --- |
| iNaturalist MOS-10k | 100.00% | 16 | PASS |
| SUN MOS-10k | 100.00% | 3 | PASS |
| Places MOS-10k | 1.78% | 0 | DEMOTED |

The frozen rule required at least 90% image-to-concept mapping coverage and at least 3 eligible groups. Those thresholds were **not changed**.

Accordingly:

- primary predefined-semantic H1 sources: **iNaturalist + SUN**;
- Places remains in the Traditional Four aggregate reproduction and may be used for secondary/non-primary analyses;
- DTD remains aggregate reproduction / later text-clustering analysis.

This is a dataset-feasibility amendment, not a response to H1/H2 outcomes. The Minimum Decisive Experiment originally allowed 2–3 OOD sources; two primary sources remain, so the pilot proceeds without redefining H1.

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

### Frozen text-embedding clustering

The second grouping definition is frozen before detector subgroup results:

- text encoder: `sentence-transformers/all-MiniLM-L6-v2`;
- model revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- package: `sentence-transformers==5.1.2`;
- input: raw leaf-concept/class name only;
- embeddings: L2 normalized;
- clustering: KMeans on L2-normalized MiniLM embeddings;
- distance interpretation: squared Euclidean distance on unit vectors is monotonic with cosine distance;
- deterministic settings: `random_state=42`, `n_init=50`;
- cluster-count rule: `floor(sqrt(N) + 0.5)`, bounded to [3, 12];
- expected pilot k: iNaturalist=10, SUN=7, Places=7;
- feasibility amendment: the earlier average-linkage agglomerative partition was rejected **before any subgroup detector result was inspected** because iNaturalist collapsed 92/110 leaf concepts into one cluster with multiple singleton clusters; the embedding model/revision, k rule, and eligibility thresholds were unchanged.
- same group eligibility rules as predefined semantic groups.

This encoder is deliberately independent of the CLIP detector so the robustness
analysis does not define semantic groups using the same VLM representation being
evaluated. The model revision and constructor `revision` parameter are pinned
before H1 scores are inspected.

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

### Fixed-ID FPR95 convention

Scores are oriented so larger means **more ID-like**.

For each method/backbone run, derive **one** threshold from the full ImageNet ID
reference:

- threshold = 5th percentile of ID scores;
- NumPy order-statistic rule = `method="higher"`;
- this gives 95% ID TPR when scores are unique; ties may make realized TPR
  slightly larger.

The exact same threshold is then used for:

- aggregate OOD FPR95;
- every predefined semantic subgroup;
- every text-cluster subgroup.

**Do not recompute a 95%-TPR threshold separately for each subgroup.**

AUROC for each subgroup uses the same full ID reference set.

Pilot confidence intervals use OOD-sample bootstrap conditional on the fixed
full ID reference. Aggregate-to-group gap bootstrap resamples the OOD source
with group membership preserved.

### Worst group

- worst-group FPR95 = maximum valid subgroup FPR95;
- worst-group AUROC = minimum valid subgroup AUROC;
- aggregate-to-worst-group gap;
- between-group dispersion.

The score orientation and exact FPR95 convention are unit-tested before experiments.

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

## H2 operationalization freeze — 2026-09-25

After H1 PASS and before reading any H2 prompt-score outcomes, the H2 execution
and decision details were frozen in
`docs/h2_prompt_sensitivity_protocol.md`.

Key clarification: the primary pilot remains the **8 prompt templates already
committed in configs/pilot.yaml**. Expanding to 16 prompts after H1 would change
a frozen primary condition, so any future 16-template run must be labeled
secondary robustness and cannot replace the 8-template primary result.

The H2 error label is the original H1 single-prompt detector error at the
original method-specific ImageNet ID95 threshold. The prompt ensemble therefore
does not redefine the outcome it is asked to predict.

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

## Alternative-explanation control freeze — 2026-09-25

H2 failed its frozen primary rule (0/4 supported conditions), so the project
moves to CONDITIONAL GO and H3 is not started.

Before inspecting alternative-control outcomes, the visual/semantic near-ID
control protocol was frozen in docs/alternative_controls_protocol.md.

The control carries forward the already-selected H1 worst-FPR95 groups without
reselection, uses only near-ID visual and independent MiniLM semantic
similarity as difficulty proxies, and reports residual subgroup gaps after a
5-fold OOF similarity-only risk model.

This stage diagnoses the pre-specified competing explanation; it does not
create a new global PASS/FAIL gate and does not redefine H1.

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
