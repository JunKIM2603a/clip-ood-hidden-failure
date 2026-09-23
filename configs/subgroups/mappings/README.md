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

The design is frozen to **taxonomic order** (primary) and **family** (secondary), but the exact 110-species lookup table must be generated from an authoritative taxonomy snapshot before H1/H2 score analysis.

Do not hand-label obscure taxa from memory. Record the taxonomy source and resolution date in the generated CSV. If a taxon cannot be resolved, keep it explicit as unresolved rather than guessing.
