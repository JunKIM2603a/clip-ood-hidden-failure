# Stage-1A ViT-B/32 Aggregate Sanity Result

> Date: 2026-09-24
> Subgroup detector results inspected before this gate: **No**

## MCM — CLIP ViT-B/32

| Dataset | FPR95 | AUROC |
| --- | ---: | ---: |
| iNaturalist | 33.98 | 93.53 |
| SUN | 40.59 | 91.53 |
| Places | 45.62 | 89.47 |
| DTD | 60.82 | 84.96 |
| Mean | **45.25** | **89.87** |

Published/follow-up sanity range used before opening subgroup results: mean AUROC about 89.82–89.96 and mean FPR95 about 45.75–49.96.

## NegLabel — CLIP ViT-B/32

| Dataset | FPR95 | AUROC |
| --- | ---: | ---: |
| iNaturalist | 3.39 | 99.20 |
| SUN | 22.72 | 95.28 |
| Places | 34.17 | 91.80 |
| DTD | 52.41 | 88.12 |
| Mean | **28.17** | **93.60** |

Official sanity anchor used before opening subgroup results: mean FPR95 27.92 and mean AUROC 93.67.

Difference from the NegLabel anchor:

- FPR95: +0.25 percentage points
- AUROC: -0.07 percentage points

## Gate decision

**PASS.** Both B/32 methods exhibit aggregate behavior consistent with the pre-specified external sanity anchors. No prompt, temperature, negative-label fraction, subgroup definition, or test threshold was tuned in response to these results.

Next: open the frozen primary H1 predefined-semantic analysis for iNaturalist and SUN.
