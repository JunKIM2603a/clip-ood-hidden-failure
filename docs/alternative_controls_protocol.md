# Alternative-Explanation Controls

> Frozen: **2026-09-25**, after H1 PASS and H2 FAIL, before inspecting these control outcomes.

## Purpose

The project is now **CONDITIONAL GO** because the narrowed H2 prompt-dispersion hypothesis failed its frozen 0/4 support rule.

This stage does not rescue or redefine H2. It addresses the main pre-specified competing explanation for H1:

> Are the observed semantic subgroup failures mostly a consequence of generic hard-OOD / near-ID similarity rather than residual subgroup structure?

## Frozen scope

- Backbone: CLIP ViT-B/32
- Methods: MCM, NegLabel
- Sources: iNaturalist, SUN
- H1 grouping definitions: predefined semantic groups and frozen MiniLM KMeans groups
- Error label: original H1 detector error at the original method-specific fixed ImageNet ID95 threshold

No H3 experiment is part of this stage.

## Difficulty / similarity proxies

### Visual ID proximity

For each OOD image x:

V(x) = max cosine similarity between its frozen CLIP ViT-B/32 image feature and any ImageNet validation image feature.

This reuses the normalized image-feature cache and does not use class prompts or MCM/NegLabel OOD scores.

### Semantic ID proximity

For each OOD leaf concept c:

T(c) = max cosine similarity between the frozen MiniLM embedding of c and the MiniLM embedding of any ImageNet class name.

Frozen encoder:

- sentence-transformers/all-MiniLM-L6-v2
- revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
- normalized raw concept/class names

## Similarity-only risk model

Within each method x source condition, estimate OOD failure risk with stratified 5-fold out-of-fold logistic regression.

Allowed predictors:

- V
- T
- V^2
- V*T
- T^2

Forbidden predictors:

- detector score
- prompt mean
- prompt dispersion
- subgroup identity

Let the OOF probability be p_hat(x) and define residual r(x) = observed_error(x) - p_hat(x).

For each frozen H1 group report:

- raw gap = group failure rate - source failure rate
- adjusted residual gap = group mean residual - source mean residual

## Fixed H1 worst-group rule

The control analysis reads the worst-FPR95 group already recorded by H1 and must not select a new worst group after seeing similarity controls.

For that fixed H1 worst group:

- residual_failure_persists = true when the paired OOD-sample bootstrap 95% CI lower bound of the adjusted residual gap is > 0
- otherwise residual_failure_persists = false

The analysis reconstructs the raw H1 gap and stops if it does not match the stored H1 gap within tolerance.

This persistence flag is a diagnostic, not a new global project PASS/FAIL rule.

## Execution

    cd /home/junkim2603a/clip-ood-hidden-failure
    git fetch origin
    git switch -c stage3-alternative-controls --track origin/stage3-alternative-controls

    conda activate clip-ood
    pytest -q tests/test_alternative_controls.py
    bash scripts/analysis/run_stage3_alternative_controls.sh

The launcher uses GPU 0 for iNaturalist proximity, GPU 1 for SUN proximity, then runs the final residual analysis.

## Outputs

    results/alternative_controls/ViT-B-32/
    ├── fixed_h1_worst_groups.csv
    ├── all_group_adjusted_gaps.csv
    ├── similarity_risk_summary.csv
    ├── analysis_metadata.json
    └── README.md

## Viewing the two first-result files

Direct view:

    cat results/alternative_controls/ViT-B-32/fixed_h1_worst_groups.csv
    cat results/alternative_controls/ViT-B-32/similarity_risk_summary.csv

Recommended readable view with pandas:

    python - <<'PY'
    import pandas as pd

    for path in [
        'results/alternative_controls/ViT-B-32/fixed_h1_worst_groups.csv',
        'results/alternative_controls/ViT-B-32/similarity_risk_summary.csv',
    ]:
        print('\n###', path)
        df = pd.read_csv(path)
        print(df.to_string(index=False))
    PY

For a wide CSV in the terminal:

    column -s, -t < results/alternative_controls/ViT-B-32/fixed_h1_worst_groups.csv | less -S

## Interpretation

- adjusted gap stays positive with its 95% CI above zero: the two frozen similarity proxies do not fully explain the H1 worst-group excess
- adjusted gap moves near zero and its CI includes zero: near-ID similarity is a plausible substantial explanation
- adjusted gap becomes negative: the similarity model expected at least as much failure as was actually observed in that group

These are explanatory controls, not causal identification.

After reviewing this stage, the next pre-planned robustness experiment is ViT-B/16 H1 replication.
