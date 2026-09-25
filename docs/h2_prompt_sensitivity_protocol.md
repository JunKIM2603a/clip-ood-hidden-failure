# H2 Prompt-Sensitivity Protocol

> **Frozen before H2 outcomes: 2026-09-25**  
> This file operationalizes H2 after H1 PASS. H1 results are not redefined here.

## 1. Question

H2 is not the generic claim that changing a CLIP prompt changes aggregate OOD
performance. Prior work already establishes that phenomenon.

The primary H2 question is narrower:

> Does within-sample prompt-score dispersion provide reproducible information
> about OOD detection failure beyond the prompt-score mean?

For sample x and the frozen prompt family p:

[
mu(x)=operatorname{mean}_p s_p(x),
qquad
sigma(x)=operatorname{std}_p s_p(x).
]

## 2. Frozen primary conditions

- Backbone: **CLIP ViT-B/32**
- Methods: **MCM, NegLabel**
- ID reference: **ImageNet-1K validation**
- Primary OOD: **iNaturalist MOS-10k, SUN MOS-10k**
- Conditions: **2 methods x 2 sources = 4**
- Prompt family: the **8 meaning-preserving templates already committed in
  configs/pilot.yaml**

The primary H2 prompt count remains 8. It is not expanded to 16 after observing
H1. A later 16-template experiment, if performed, must be labeled secondary
prompt-family robustness and may not replace this primary result.

## 3. Prompt intervention

### MCM

For every ImageNet class, replace only the positive class prompt template using
the same frozen 8-template family and compute one MCM score per prompt.

### NegLabel

Vary only the **positive ImageNet class prompt template**.

The following H1 components stay fixed across all H2 prompt variants:

- selected negative-label identities;
- already-templated negative-label strings / CLIP embeddings;
- B/32 negative-mining result and file hash;
- deterministic negative-label permutation/grouping;
- ngroup, temperature, and logit scale.

This prevents prompt wording sensitivity from being confounded with re-mining a
different negative-label set.

## 4. Error label is frozen from H1

H2 does **not** create a new operating threshold from the prompt ensemble.

For each method, use the original H1 ImageNet score file and its original
fixed-ID95 threshold:

[
t_{95}=Q_{0.05}^{	ext{higher}}(s_{	ext{H1}}(X_{ID})).
]

For an OOD sample x:

[
E(x)=1[s_{	ext{H1}}(x)ge t_{95}].
]

Thus H2 asks whether prompt dispersion predicts an already-defined H1 failure.

## 5. Primary statistical test

Per method x source condition, first standardize predictors within that
condition and fit:

[
M_{mathrm{assoc}}:
operatorname{logit}P(E=1)
=
alpha+eta_mu z(mu)+eta_sigma z(sigma).
]

Primary association target:

[
eta_sigma.
]

Association support requires the OOD-sample bootstrap 95% CI lower bound for
(eta_sigma) to be strictly greater than zero.

The required predictive comparison is:

[
M_0:Esim z(mu)
]

versus

[
M_1:Esim z(mu)+z(sigma).
]

Use stratified 5-fold out-of-fold predictions with seed 42. The primary
predictive metric is:

[
Delta L
=
operatorname{LogLoss}(M_0)
-
operatorname{LogLoss}(M_1).
]

Positive values favor M1. Predictive support requires the paired OOD-sample
bootstrap 95% CI lower bound of (Delta L) to be strictly greater than zero.

OOF AUROC change is reported as a secondary metric, not substituted for the
primary log-loss test after results are seen.

## 6. Eligibility and decision rule

A condition is primary-test eligible only when it contains:

- at least **50 H1 failures**; and
- at least **50 correct OOD detections**.

A condition supports H2 only if **both** are true:

1. 95% CI lower bound for (eta_sigma > 0);
2. 95% CI lower bound for OOF log-loss improvement (Delta L > 0).

Decision:

- **H2 PASS:** at least **3 of the 4** primary eligible conditions support H2.
- **H2 FAIL:** all four are eligible, but fewer than 3 support H2.
- **INSUFFICIENT ELIGIBLE CONDITIONS:** any primary condition fails the frozen
  count requirement; do not silently convert this to PASS or FAIL.

Given H1 is already PASS:

- H2 PASS -> retain prompt sensitivity as the second main finding; H3 may be
  considered next.
- H2 FAIL -> **CONDITIONAL GO** focused on semantic subgroup hidden failure.
- insufficient eligibility -> H2 remains unresolved until the pre-specified
  eligibility issue is handled transparently.

## 7. Required descriptive / secondary outputs

For every condition report:

- failure rate by sigma quartile with Wilson 95% CI;
- mean sigma for correct vs failed OOD samples;
- AUROC of sigma alone for discriminating H1 failure;
- M0/M1 OOF log-loss and AUROC;
- beta_sigma and bootstrap CI.

Using the already-frozen H1 subgroup definitions, also report descriptively:

- predefined-group mean / q75 / q90 sigma vs group FPR95;
- MiniLM-KMeans-group mean / q75 / q90 sigma vs group FPR95;
- Spearman association across eligible groups.

The subgroup correlation is secondary because some grouping/source
combinations contain few eligible groups.

## 8. Stored scores

Do not save only prompt averages. For ImageNet, iNaturalist, and SUN store:

- relative path / sample identifier;
- all 8 prompt-specific scores;
- mu;
- sigma with ddof=0;
- q10 and q25 for future robustness/H3 analyses.

q10/q25 are stored now to avoid re-scoring but are **not** H2 primary
endpoints.

## 9. Execution

```bash
cd /home/junkim2603a/clip-ood-hidden-failure
git fetch origin
git checkout h2-prompt-sensitivity
conda activate clip-ood

pytest -q tests/test_h2_prompt.py
bash scripts/analysis/run_stage2_h2_prompt_sensitivity.sh
```

The launcher uses:

- GPU 0 -> MCM prompt scoring
- GPU 1 -> NegLabel prompt scoring
- CPU -> frozen H2 statistical analysis after both scorers complete

## 10. Outputs

Primary outputs are written under:

```text
results/h2_prompt/ViT-B-32/
├── condition_summary.csv
├── sigma_quartiles.csv
├── subgroup_sigma_vs_fpr95.csv
├── subgroup_correlations.csv
├── h2_decision.json
├── README.md
└── *_sigma_quartiles.png
```

Raw per-prompt scores are stored under:

```text
results/raw/h2_prompt/ViT-B-32/{mcm,neglabel}/
```

## 11. Anti-posthoc rule

Do not change after reading H2 outcomes:

- the 8 primary templates;
- H1 baseline error definition or ID95 threshold;
- primary method/source set;
- 50/50 eligibility counts;
- M0/M1 definitions;
- 5-fold OOF comparison;
- beta_sigma and log-loss CI support rules;
- 3-of-4 H2 PASS threshold.

H3 is outside this stage and must not be run by the H2 launcher.
