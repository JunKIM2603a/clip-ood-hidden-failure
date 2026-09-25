# Alternative-explanation controls

This stage follows H1 PASS and H2 FAIL/CONDITIONAL GO.

It does not redefine H1 and has no new global PASS/FAIL gate.
The fixed H1 worst-FPR95 group is carried forward unchanged.

Difficulty model: 5-fold OOF logistic regression using only:

- nearest ImageNet-val CLIP image similarity;
- independent MiniLM leaf-concept-to-ImageNet-class similarity;
- frozen degree-2 polynomial terms of those two proxies.

Detector score, prompt mean/std, and subgroup labels are forbidden predictors.

| Method | Source | Grouping | Fixed H1 worst | Raw gap | Adjusted residual gap | 95% CI | Persists? |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| mcm | inaturalist | predefined | Solanales | 28.90pp | 28.45pp | [24.12,32.65]pp | yes |
| mcm | inaturalist | minilm_kmeans | cluster_08 | 13.61pp | 11.40pp | [7.66,15.16]pp | yes |
| mcm | sun | predefined | mountains, hills, desert, sky | 0.51pp | -2.97pp | [-5.25,-0.67]pp | no |
| mcm | sun | minilm_kmeans | cluster_04 | 9.17pp | 1.99pp | [-0.61,4.57]pp | no |
| neglabel | inaturalist | predefined | Rosales | 3.46pp | 3.19pp | [1.40,5.01]pp | yes |
| neglabel | inaturalist | minilm_kmeans | cluster_02 | 2.69pp | 2.44pp | [1.39,3.57]pp | yes |
| neglabel | sun | predefined | forest, field, jungle | -0.25pp | 1.63pp | [0.73,2.54]pp | yes |
| neglabel | sun | minilm_kmeans | cluster_01 | 7.34pp | 7.01pp | [5.35,8.67]pp | yes |

Interpretation:

- persists=yes: the frozen similarity proxies do not fully explain the fixed H1 worst-group excess;
- persists=no: the fixed H1 excess is no longer distinguishable from zero after adjustment, supporting the competing explanation as a plausible account;
- neither result is a causal attribution.

Similarity-only OOF risk-model diagnostics:

| Method | Source | Failure rate | OOF AUROC | OOF log-loss |
| --- | --- | ---: | ---: | ---: |
| mcm | inaturalist | 33.96% | 0.6147 | 0.6177 |
| mcm | sun | 40.56% | 0.6282 | 0.6391 |
| neglabel | inaturalist | 3.39% | 0.5980 | 0.1464 |
| neglabel | sun | 22.71% | 0.6100 | 0.5123 |

After this control, the next pre-planned robustness stage is ViT-B/16 H1 replication.
