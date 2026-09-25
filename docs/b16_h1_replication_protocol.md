# ViT-B/16 H1 Replication Protocol

> Frozen before inspecting ViT-B/16 subgroup outcomes: 2026-09-25.

## Purpose

H1 already passed on ViT-B/32. This stage asks whether the same hidden-failure
phenomenon transfers to another CLIP backbone.

This stage does not change the project gate. The project remains CONDITIONAL GO
after H2 FAIL.

## Frozen scope

- backbone: CLIP ViT-B/16
- methods: MCM, NegLabel
- primary OOD sources: iNaturalist, SUN
- subgroup definitions: predefined + frozen MiniLM KMeans
- same ID95 threshold convention
- same subgroup eligibility rules
- no regrouping or k changes after B/16 results

## Uncertainty

- 5000 OOD-sample bootstrap resamples
- full ImageNet ID reference fixed
- selection-aware worst-group bootstrap
- FPR95 gap is the primary replication endpoint
- AUROC gap is secondary corroboration

Each method x source x grouping condition is supported when the selection-aware
FPR95 gap 95% CI lower bound is strictly above zero.

## Overall interpretation

Strong replication requires:

1. at least 3 of 4 MiniLM method x source conditions supported;
2. both MCM and NegLabel represented among those supported MiniLM conditions;
3. predefined iNaturalist supported for both MCM and NegLabel.

Partial replication means the strong rule is not met, but at least 2 of 4
MiniLM conditions support or at least one predefined iNaturalist method supports.

Otherwise the result is limited replication.

This strong/partial/limited label is only a cross-backbone robustness summary.
It does not overwrite the original H1 PASS.

## Execution

    cd /home/junkim2603a/clip-ood-hidden-failure
    git fetch origin
    git switch --track origin/stage4-b16-h1-replication

    conda activate clip-ood
    pytest -q tests/test_selection_aware.py
    bash scripts/analysis/run_stage4_b16_h1_replication.sh

If the launcher reports missing ViT-B/16 raw scores, generate them first:

    python scripts/baseline/run_reproduction.py --method mcm --backbone ViT-B/16 --gpu 0 --ood inaturalist sun
    python scripts/baseline/run_reproduction.py --method neglabel --backbone ViT-B/16 --gpu 1 --ood inaturalist sun

## First two files to inspect

    results/h1_replication/ViT-B-16/replication_interpretation.json
    results/h1_replication/ViT-B-16/selection_aware_summary.csv

### View the JSON

    cat results/h1_replication/ViT-B-16/replication_interpretation.json

### Recommended readable CSV view

    python - <<'PY'
    import pandas as pd

    p = 'results/h1_replication/ViT-B-16/selection_aware_summary.csv'
    df = pd.read_csv(p)

    cols = [
        'method',
        'source',
        'grouping',
        'metric',
        'point_aggregate',
        'point_worst',
        'point_gap',
        'point_worst_group',
        'selection_aware_gap_median',
        'selection_aware_ci_low',
        'selection_aware_ci_high',
        'empirical_p_gap_le_zero',
        'condition_supported',
    ]

    print(df[df['metric'] == 'fpr95'][cols].to_string(index=False))
    PY

To inspect AUROC as well, remove the metric filter or replace fpr95 with auroc.
