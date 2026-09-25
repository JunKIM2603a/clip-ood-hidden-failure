# Alternative-Control Result — ViT-B/32

> Recorded after the frozen alternative-explanation control was executed.

## Project state

- H1: PASS
- H2: FAIL under the frozen 0/4 support rule
- Project gate: CONDITIONAL GO
- H3: not evaluated

## Main control result

The near-ID control does not uniformly explain the H1 hidden subgroup gaps.
Similarity-only OOF risk prediction is modest (AUROC approximately 0.60-0.63),
and several H1-positive groups retain a positive residual gap after adjustment.

| Method | Source | Grouping | Fixed H1 group | Raw gap | Adjusted gap | 95% CI | Attenuation |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| MCM | iNaturalist | predefined | Solanales | +28.90pp | +28.45pp | [+24.12,+32.65]pp | 1.6% |
| MCM | iNaturalist | MiniLM | cluster_08 | +13.61pp | +11.40pp | [+7.66,+15.16]pp | 16.2% |
| MCM | SUN | MiniLM | cluster_04 | +9.17pp | +1.99pp | [-0.61,+4.57]pp | 78.3% |
| NegLabel | iNaturalist | predefined | Rosales | +3.46pp | +3.19pp | [+1.40,+5.01]pp | 7.7% |
| NegLabel | iNaturalist | MiniLM | cluster_02 | +2.69pp | +2.44pp | [+1.39,+3.57]pp | 9.4% |
| NegLabel | SUN | MiniLM | cluster_01 | +7.34pp | +7.01pp | [+5.35,+8.67]pp | 4.5% |

## Interpretation

The strongest MCM/iNaturalist predefined failure is essentially unchanged by
the frozen visual and semantic near-ID controls. MCM/iNaturalist MiniLM and
both NegLabel/iNaturalist analyses also retain most of their raw gaps.
NegLabel/SUN MiniLM likewise remains strong after adjustment.

In contrast, the MCM/SUN MiniLM gap is attenuated by about 78% and its adjusted
95% CI includes zero. Near-ID similarity is therefore a plausible substantial
explanation for that condition.

The predefined SUN rows are not counted as hidden-failure evidence because
their original raw H1 gaps were null or negative. A positive residual after
adjustment in such a row must not be re-labeled as a new H1-positive result.

## Safe claim

> Semantic subgroup hidden failures in ViT-B/32 are not generally reducible to
> the frozen visual/semantic near-ID similarity proxies, although the degree of
> explanation is detector/source/grouping dependent and one MCM/SUN MiniLM
> condition is largely attenuated.

This is an explanatory robustness result, not causal identification.

## Next step

Run the pre-planned ViT-B/16 H1 replication with the same subgroup definitions,
ID95 convention, and selection-aware bootstrap.
