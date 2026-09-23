# Predefined Subgroup Pre-Score Audit

> Audit date: **2026-09-23**  
> This audit uses only benchmark concept lists and external taxonomy/hierarchy metadata. No MCM/NegLabel subgroup scores were inspected.

## Concept-level coverage

| Source | Primary predefined scheme | MOS leaf concepts | Mapped | Represented primary groups | Groups with >=2 leaf concepts |
| --- | --- | ---: | ---: | ---: | ---: |
| iNaturalist | taxonomic order | 110 | 110 | 30 | 22 |
| SUN | official 15 basic-level scene hierarchy | 50 | 50 | 6 | 3 |
| Places365 | official S16 scene hierarchy | 50 | 50 | 8 | 7 |

Image-level eligibility (>=200 images/group) will be checked after the curated benchmark images/image lists are materialized. The MOS sources are random 10,000-image samples over their selected concepts, so concept count alone must **not** be treated as an image-count guarantee.

## iNaturalist order distribution

- Caryophyllales: 9 leaf concepts
- Asterales: 9 leaf concepts
- Rosales: 8 leaf concepts
- Lamiales: 7 leaf concepts
- Asparagales: 6 leaf concepts
- Ericales: 6 leaf concepts
- Sapindales: 5 leaf concepts
- Liliales: 5 leaf concepts
- Poales: 5 leaf concepts
- Brassicales: 5 leaf concepts
- Polypodiales: 4 leaf concepts
- Malpighiales: 4 leaf concepts
- Fabales: 4 leaf concepts
- Alismatales: 4 leaf concepts
- Gentianales: 3 leaf concepts
- Myrtales: 3 leaf concepts
- Pinales: 3 leaf concepts
- Solanales: 3 leaf concepts
- Ranunculales: 3 leaf concepts
- Apiales: 2 leaf concepts
- Osmundales: 2 leaf concepts
- Malvales: 2 leaf concepts
- Cucurbitales: 1 leaf concepts
- Saxifragales: 1 leaf concepts
- Selaginellales: 1 leaf concepts
- Ceratophyllales: 1 leaf concepts
- Gigartinales: 1 leaf concepts
- Dipsacales: 1 leaf concepts
- Fagales: 1 leaf concepts
- Laurales: 1 leaf concepts

## SUN basic-level distribution

- forest, field, jungle: 21 leaf concepts
- water, ice, snow: 20
- mountains, hills, desert, sky: 6
- cultural (art, education, religion, etc.): 1
- transportation (roads, parking, bridges, boats, airports, etc.): 1
- industrial and construction: 1

Under the frozen >=2-leaf rule, three SUN groups remain concept-level eligible.

## Places S16 distribution

- water_ice_snow: 22 leaf concepts
- forest_field_jungle: 18
- man_made_elements: 10
- mountains_hills_desert_sky: 7
- houses_cabins_gardens_farms: 7
- transportation_outdoor: 4
- sports_fields_parks_leisure: 4
- industrial_construction: 1

Under the frozen >=2-leaf rule, seven Places S16 groups remain concept-level eligible.

## Decision

The frozen predefined grouping schemes pass the **concept-level** pre-score audit.

The next required check is image-level mapping/count coverage after the MOS-curated image lists are installed:

1. >=90% images mapped;
2. >=3 primary groups with >=200 images;
3. each headline group contains >=2 leaf concepts.

If a source fails those rules, it is demoted to secondary analysis; thresholds are not relaxed after observing detector performance.
