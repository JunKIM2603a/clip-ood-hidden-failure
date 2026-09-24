# Text-cluster pre-score feasibility result

> Evaluated before opening any detector semantic-subgroup result.

## iNaturalist

- leaf concepts: 110
- k: 10
- image mapping coverage: 100%
- eligible clusters: **9 / 10**
- primary eligibility: **PASS**
- ineligible cluster: one cluster with 144 images (<200 threshold)

## SUN

- leaf concepts: 50
- k: 7
- image mapping coverage: 100%
- eligible clusters: **7 / 7**
- primary eligibility: **PASS**

## Places

- image mapping coverage: 1.78%
- eligible clusters: 0
- primary scope: false
- status: secondary only, unchanged from the predefined mapping feasibility gate

## Decision

The independent MiniLM KMeans grouping is feasible for both primary H1 sources. No detector subgroup scores were used to choose or repair individual clusters. The fixed eligibility rules remain unchanged.
