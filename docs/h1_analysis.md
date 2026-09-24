# H1 Analysis Protocol

This stage begins only after baseline reproduction produces per-sample ID/OOD scores.

## Frozen primary sources

- iNaturalist MOS-10k
- SUN MOS-10k

Places failed the pre-score 90% image-to-concept mapping gate and cannot define the primary predefined-semantic GO/KILL result. It remains part of Traditional Four aggregate reproduction.

## Frozen FPR95 convention

All scores are oriented so larger means more ID-like.

For each method/backbone run:

1. use the full ImageNet ID reference;
2. set one threshold at the 5th percentile of ID scores;
3. use NumPy quantile method `higher`;
4. reuse exactly that threshold for aggregate OOD and every subgroup.

Subgroup-specific threshold selection is forbidden.

## Frozen uncertainty convention

Pilot bootstrap:

- hold the full ID reference fixed;
- bootstrap OOD samples;
- compute subgroup FPR95 and AUROC CIs;
- for aggregate-to-subgroup gaps, bootstrap the OOD source while preserving group membership.

## Second grouping definition

Before viewing detector subgroup results, generate the frozen independent text-cluster mapping:

```bash
python scripts/data/build_text_clusters.py
```

The fixed configuration is in `configs/subgroups/text_clustering.yaml`.

It uses:

- `sentence-transformers/all-MiniLM-L6-v2`
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- normalized raw class/leaf names
- agglomerative clustering / average linkage / cosine distance
- `k=floor(sqrt(N)+0.5)`, clamped to [3,12]

Expected k: iNaturalist=10, SUN=7, Places=7.

The generated mapping files and manifest are small tracked artifacts. Review and commit them before H1 detector results.

## Tests before scoring

```bash
pytest -q tests/test_semantic_mapping.py tests/test_baseline_scores.py tests/test_h1_metrics.py
```

## H1 predefined semantic analysis

After reproduction score CSVs exist:

```bash
python scripts/analysis/run_h1_predefined.py --method mcm --backbone ViT-B/32
python scripts/analysis/run_h1_predefined.py --method neglabel --backbone ViT-B/32
```

The runner defaults to the primary sources from `configs/pilot.yaml`, currently iNaturalist and SUN.

The analysis runner intentionally does not assign GO / CONDITIONAL GO / KILL automatically. The pre-registered decision rule must be applied after inspecting effect size, confidence intervals, recurrence across methods, and the alternate grouping definition.
