# Stage-0 Baseline Reproduction Result

> Date: 2026-09-24
> Stage: aggregate reproduction only; no subgroup detector result was inspected.

## MCM — CLIP ViT-B/16

| Dataset | Ours FPR95 | Ref FPR95 | Delta pp | Ours AUROC | Ref AUROC | Delta pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iNaturalist | 31.98 | 30.91 | +1.07 | 94.43 | 94.61 | -0.18 |
| SUN | 38.59 | 37.59 | +1.00 | 92.37 | 92.57 | -0.20 |
| Places | 43.71 | 44.69 | -0.98 | 90.03 | 89.77 | +0.26 |
| DTD | 57.87 | 57.77 | +0.10 | 86.14 | 86.11 | +0.03 |
| Mean | **43.04** | **42.74** | **+0.30** | **90.74** | **90.77** | **-0.03** |

## NegLabel — CLIP ViT-B/16

| Dataset | Ours FPR95 | Ref FPR95 | Delta pp | Ours AUROC | Ref AUROC | Delta pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iNaturalist | 1.92 | 1.91 | +0.01 | 99.50 | 99.49 | +0.01 |
| SUN | 21.13 | 20.53 | +0.60 | 95.33 | 95.49 | -0.16 |
| Places | 34.38 | 35.59 | -1.21 | 91.99 | 91.64 | +0.35 |
| DTD | 45.11 | 43.56 | +1.55 | 89.84 | 90.22 | -0.38 |
| Mean | **25.63** | **25.40** | **+0.23** | **94.16** | **94.21** | **-0.05** |

## Gate decision

**PASS.** No formal numerical tolerance was preregistered, so this decision is not based on a post-hoc hard threshold. The aggregate means are extremely close to the published references, and per-dataset deviations are small enough to treat the implementation, score orientation, class-name ordering, preprocessing, and dataset identity as successfully reproduced for the pilot.

Important implementation note: MCM uses OpenAI CLIP rather than the Hugging Face packaged checkpoint. The original MCM repository states that similar results can be obtained with the OpenAI CLIP checkpoint.

Next stage: generate backbone-specific NegLabel negative labels for ViT-B/32, reproduce ViT-B/32 aggregate behavior, then run the frozen H1 predefined subgroup analysis on iNaturalist and SUN.
