---
name: multimodal-fashion
description: Use for fashion article text embeddings, image embeddings, CLIP-style retrieval, dense content features, and cold-start signals.
license: MIT
compatibility: opencode
metadata:
  domain: recommender-systems
  workflow: multimodal
---

# Multimodal Fashion Skill

Use this skill when article metadata, descriptions, or images can improve retrieval or ranking.

## Feature Sources

- Text: product name, detail description, product type, department, garment group, color, perceived color, graphical appearance.
- Image: article images when present; not every article is guaranteed to have one.
- Structured metadata: categorical fields and price or availability proxies derived only from allowed training-period data.

## Mechanisms

- Create cached text embeddings and image embeddings with versioned encoders.
- Use CLIP-style embeddings for item-item similarity and cold-start candidate generation.
- Project dense content embeddings into the ID embedding space when training sequential or two-tower models.
- Use dimensionality reduction or ANN indexing only with exact article ID maps.

## Guardrails

- Missing images are normal and must not break the pipeline.
- Do not commit images or generated embedding tables.
- Evaluate content features as candidate sources and ranker features; do not assume they improve MAP@12.
