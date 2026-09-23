# Pilot Dataset and Predefined Semantic Subgroups

> **Frozen for the Minimum Decisive Experiment: 2026-09-23**
>
> This document fixes the pilot ID/OOD datasets and the **predefined semantic subgroup schemes before detector subgroup results are inspected**. The exact class-to-group lookup tables for iNaturalist (110/110), SUN (50/50), and Places365 (50/50) were generated and committed on 2026-09-23 before H1/H2 detector subgroup-score analysis.

## Final pilot selection

| Role | Dataset / split | Size used | Why selected |
| --- | --- | ---: | --- |
| ID reference | ImageNet-1K (ILSVRC2012 validation / OpenOOD-VLM traditional benchmark test list) | 50,000 | Canonical ID used by MCM/NegLabel and the Traditional Four benchmark |
| OOD-1 | iNaturalist, MOS curated split | 10,000 | Different semantic domain; 110 plant concepts with external biological taxonomy |
| OOD-2 | SUN, MOS curated split | 10,000 | Official three-level scene hierarchy gives predefined semantic groups |
| OOD-3 | Places365, MOS curated split | 10,000 | Independent scene dataset with official coarse/fine scene hierarchy; replication source |
| Aggregate reproduction only | DTD / Textures | 5,640 | Completes the MCM/NegLabel Traditional Four benchmark; no strong official superclass hierarchy, so not a primary H1 source |

**No additional OOD subsampling is performed in the pilot.** The MOS splits are already curated/capped benchmark subsets. Using all 10,000 images per main source avoids adding avoidable sampling noise.

## Why these exact OOD sources

MCM/NegLabel evaluation follows the large-scale OOD benchmark lineage in which ImageNet-1K is ID and iNaturalist, SUN, Places, and Textures are the traditional far-OOD sources. OpenOOD-VLM exposes the same ImageNet Traditional Four configuration.

This gives two benefits:

1. **baseline comparability** — aggregate results can be checked against published / reproduced MCM and NegLabel behavior;
2. **new evaluation layer** — we add semantic subgroup and prompt-dispersion analysis without first changing the benchmark.

Primary references:

- OpenOOD-VLM traditional ImageNet configuration: https://github.com/PolyU-VCLab/OpenOOD-VLM
- MOS supplementary category list: https://openaccess.thecvf.com/content/CVPR2021/supplemental/Huang_MOS_Towards_Scaling_CVPR_2021_supplemental.pdf
- SUN official hierarchy: https://vision.princeton.edu/projects/2010/SUN/
- Places365 official resources / hierarchy: https://github.com/CSAILVision/places365
- iNaturalist taxonomy: https://www.inaturalist.org/pages/taxonomy

---

# Predefined subgroup mapping

## OOD-1 — iNaturalist MOS-10k

### Important benchmark fact

The curated iNaturalist OOD split is **not a representative sample of the entire iNaturalist dataset**. MOS selected **110 concepts, all plant taxa**, to avoid semantic overlap with ImageNet-1K.

Therefore the common high-level iNaturalist supercategories (Animalia, Plantae, etc.) are useless here: essentially the entire curated split would collapse into Plantae.

### Primary predefined grouping

**Taxonomic order**

~~~text
species
  ↓
genus
  ↓
family
  ↓
ORDER       ← primary subgroup
  ↓
class / phylum / kingdom
~~~

Each of the 110 MOS plant concepts will be mapped to its official iNaturalist backbone taxonomic order.

Why order:

- externally defined before our experiment;
- semantically meaningful;
- coarser than species/family, reducing tiny-group noise;
- still provides multiple distinct botanical groups inside an all-plant split.

### Secondary predefined grouping

**Taxonomic family**

Family-level results are reported as secondary/finer analysis. They do not define the primary pilot GO decision if most families fail the minimum-size rule.

### Mapping source / freeze rule

- source: iNaturalist Backbone Taxonomy;
- rank: order (primary), family (secondary);
- mapping resolution date is recorded;
- exact mapping file: `configs/subgroups/mappings/inaturalist_mos110_taxonomy.csv`;
- status: **110/110 taxa mapped** before detector subgroup-score analysis;
- once committed for the pilot, taxonomy mappings are not changed because of observed performance.

---

## OOD-2 — SUN MOS-10k

MOS selected 50 SUN scene concepts. SUN provides an official **three-level semantic scene hierarchy**.

Official hierarchy structure:

~~~text
root
 └── 3 superordinate scene groups
      └── 15 basic-level parent groups
           └── SUN leaf categories
~~~

The hierarchy is a DAG rather than a strict tree; some categories may have more than one valid semantic parent.

### Primary predefined grouping

**SUN official basic-level group (15-node hierarchy level)**

Each selected MOS SUN leaf category inherits membership in its official basic-level parent(s).

Why this level:

- the 3-way top level is likely too coarse for a worst-group study;
- leaf-class analysis is too fine and closer to ordinary per-class failure analysis;
- the middle level provides an independently authored semantic grouping.

### Secondary predefined grouping

**SUN official 3 superordinate groups**

- indoor
- outdoor natural
- outdoor man-made

Use this as a coarse sanity check, not as the only semantic definition.

### Multi-parent policy

Do **not** invent an arbitrary tie-breaker.

If a SUN category has multiple official parents:

- retain all official memberships;
- compute each group metric on all samples belonging to that group;
- mark the predefined analysis as **overlapping groups**;
- report membership counts.

This policy is frozen before score analysis.

---

## OOD-3 — Places365 MOS-10k

MOS selected 50 Places365 categories. Places365 provides an official scene hierarchy in addition to the 365 leaf scene categories.

The published hierarchy can be summarized at:

- **S365**: original 365 scene categories;
- **S16**: 16 intermediate semantic scene groups;
- **S3**: 3 superordinate scene groups.

The official Places365 repository also provides indoor/outdoor labels. For this curated MOS subset, the selected concepts are heavily oriented toward outdoor/natural scenes, so the binary indoor/outdoor split is **not** chosen as the primary subgroup definition.

### Primary predefined grouping

**Places365 S16 hierarchy group**

This is the main predefined Places subgroup scheme.

### Secondary predefined grouping

**Places365 S3 hierarchy group**

- indoor
- outdoor natural
- outdoor man-made

Use only if at least two represented groups satisfy the minimum-size rule.

### Diagnostic-only grouping

Places365 official indoor/outdoor binary labels may be reported descriptively, but they cannot define the primary H1 GO/KILL result.

### Multi-parent policy

As with SUN, preserve official multi-membership rather than imposing an outcome-driven single parent.

---

# Primary subgroup eligibility rule

For a predefined group to enter the **headline worst-group statistic**, it must satisfy both:

1. **at least 200 OOD images**, and
2. **at least 2 distinct leaf concepts/classes**.

Groups below either threshold remain visible in descriptive tables/appendices but cannot define the primary worst-group FPR95/AUROC.

Why 200:

- the main OOD sources each contain 10,000 images;
- very small groups can create unstable extreme worst-group estimates;
- 200 keeps the pilot sensitive to meaningful failures while excluding obvious small-n extremes.

This threshold is fixed before detector subgroup results are inspected.

---

# Mapping audit before H1 analysis

Before any worst-group result is opened, generate a mapping audit table:

| Dataset | Mapping scheme | # represented groups | min n | median n | max n | % samples mapped | multi-parent % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

A dataset can define the primary GO/KILL decision only if:

- at least **3 eligible groups** remain after the frozen size rule;
- at least **90% of samples** receive a predefined semantic mapping.

If a source fails these criteria for reasons inherent to its official hierarchy, it becomes a **secondary descriptive source**. The thresholds and mapping rule are not altered to rescue the hypothesis.

---

# What is deliberately excluded from the primary pilot

## DTD / Textures

Keep DTD in the aggregate Traditional Four reproduction, but exclude it from the primary predefined-supergroup H1 test.

Reason:

- its 47 official texture categories are useful leaf classes;
- it does not provide a comparably strong official semantic superclass hierarchy;
- letting a hand-built texture taxonomy define the pilot would weaken the predefined-group argument.

DTD can later be analyzed with the **text-embedding clustering** definition.

## NINCO / SSB-hard

Reserve these near-OOD datasets for the expansion phase.

Reason:

- the first pilot should reproduce the classic MCM/NegLabel ImageNet benchmark before mixing benchmark protocols;
- NINCO already emphasizes detailed per-class OOD difficulty, so using it as the headline H1 source would create a closer prior-art overlap;
- after H1 is established or rejected on the frozen far-OOD pilot, near-OOD is a strong robustness extension.

---

# Final pilot structure

~~~text
ID: ImageNet-1K
│
├── OOD: iNaturalist MOS-10k
│    └── predefined groups: taxonomic ORDER
│
├── OOD: SUN MOS-10k
│    └── predefined groups: official 15 basic-level scene groups
│
└── OOD: Places365 MOS-10k
     └── predefined groups: official S16 scene groups

aggregate reproduction additionally:
└── DTD / Textures
~~~

This setup intentionally combines:

- one biological/taxonomic domain;
- two independent scene datasets with official hierarchies;
- direct compatibility with the MCM/NegLabel traditional aggregate benchmark.

---

# Freeze statement

The following are fixed before pilot detector subgroup analysis:

- ID dataset: ImageNet-1K;
- primary OOD sources: iNaturalist, SUN, Places MOS curated splits;
- primary predefined hierarchy level for each source;
- minimum primary group size: 200 images;
- minimum leaf concepts per group: 2;
- multi-parent handling: official multi-membership retained;
- mapping coverage requirement: 90%;
- minimum eligible groups per source: 3.

The **text-embedding clustering** subgroup definition remains the second, independent grouping strategy specified elsewhere in the protocol. It must also be frozen before final test analysis.


---

# Pre-score mapping audit status

The concept-level audit is complete and stored at `configs/subgroups/pre_score_audit.md`.

| Source | Primary scheme | Leaf concepts mapped | Represented primary groups | Groups with >=2 leaf concepts |
| --- | --- | ---: | ---: | ---: |
| iNaturalist | taxonomic order | 110 / 110 | 30 | 22 |
| SUN | official 15 basic-level hierarchy | 50 / 50 | 6 | 3 |
| Places365 | official S16 hierarchy | 50 / 50 | 8 | 7 |

This audit uses no MCM/NegLabel subgroup scores.

The remaining audit is **image-level only** after the curated datasets are materialized: verify actual image counts, >=90% mapping coverage, and >=3 groups with >=200 images. Thresholds remain frozen regardless of detector results.
