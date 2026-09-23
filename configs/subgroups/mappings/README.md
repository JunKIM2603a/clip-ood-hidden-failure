# Predefined mapping provenance

These lookup tables are generated **before detector subgroup-score analysis**.

## Places MOS-50

- benchmark leaf list: MOS supplementary material;
- hierarchy authority: Places365 official scene hierarchy linked by the official Places365 repository;
- machine-readable hierarchy used to generate the committed table: `MatteoCamillo-code/GeoLoc-CVCS/scene_hierarchy_places365.csv`;
- that file is downloaded by the mirror project from the official Places365 Google Sheet;
- primary column: S16;
- secondary column: S3;
- official multi-memberships are retained.

## SUN MOS-50

- benchmark leaf list: MOS supplementary material;
- hierarchy authority: SUN official three-level hierarchy;
- machine-readable 15-group copy used to generate the committed table: `amberyzheng/LTO/data/sun397/superclass-15.json`, blob SHA `6869a425096b4e1aaf36651550e6472a91d7f3b0`;
- primary column: basic-level group;
- secondary S3 group is derived from the official 15→3 hierarchy;
- all 50 MOS SUN leaves resolved successfully.

## iNaturalist MOS-110

- benchmark leaf list: MOS supplementary material (110 selected plant taxa);
- primary hierarchy rank: taxonomic **order**;
- secondary hierarchy rank: **family**;
- exact mapping file: `inaturalist_mos110_taxonomy.csv`;
- 50 taxa resolve as exact species in a public iNaturalist 2018 competition taxonomy metadata snapshot;
- 42 additional taxa resolve through an unambiguous same-genus family/order in that snapshot;
- the remaining 18 taxa were cross-checked against Kew Plants of the World Online (POWO) or NCBI Taxonomy on 2026-09-23;
- all 110 benchmark taxa are mapped; each CSV row records its resolution method/source.

The genus-level fallback is used only for family/order assignment: a botanical genus belongs to the same family/order for this grouping purpose. Taxonomic names/synonyms are not silently substituted in detector prompts; this mapping is only subgroup metadata.
