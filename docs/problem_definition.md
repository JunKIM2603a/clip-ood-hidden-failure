# Problem Definition

## Core problem

Recent CLIP/VLM-based OOD detectors often report strong aggregate AUROC, AUPR, and FPR95. However, aggregate metrics can hide severe failures concentrated in semantically coherent subsets.

This project asks whether the same hidden-failure phenomenon appears in zero-shot VLM OOD detection.

## Primary research question

> Can a CLIP/VLM OOD detector with strong aggregate performance systematically fail on particular semantic subgroups or prompt conditions?

## Why this matters

A detector can look reliable at dataset level while being unreliable for a coherent semantic subset. If such failures exist, aggregate evaluation alone is insufficient for characterizing OOD reliability.

## Scope

This project focuses on:

- existing CLIP/VLM OOD methods rather than a new architecture;
- semantic subgroup analysis;
- prompt-conditioned score sensitivity;
- worst-group evaluation;
- lightweight robust aggregation only as a secondary intervention.

## Exclusions

The project does not begin with the claim that prior work ignored this issue entirely. A collision search must be performed before novelty claims are made.
