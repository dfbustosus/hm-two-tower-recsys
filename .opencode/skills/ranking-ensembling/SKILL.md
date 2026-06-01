---
name: ranking-ensembling
description: Use for reranking, LambdaMART/GBDT rankers, neural rankers, blending, ensembling, and MAP@12 optimization.
license: MIT
compatibility: opencode
metadata:
  domain: recommender-systems
  workflow: ranking
---

# Ranking And Ensembling Skill

Use this skill after candidate generation exists.

## Ranker Inputs

Strong H&M rankers should consider:

- Candidate source indicators and source ranks.
- User history features: recency, frequency, spend, channel, category preferences.
- Article features: metadata, age of article, popularity trends, content embeddings.
- User-article crossing features: repeat flag, category affinity, price affinity, color or garment affinity.
- Sequence features: last purchased items, transition counts, and sequential model scores.
- Retrieval scores from item-item, graph, two-tower, and content models.

## Ranking Strategies

- Start with a transparent baseline ranker and compare against source-rank blending.
- Use LambdaMART/GBDT-style ranking when tabular feature quality is high.
- Consider neural rerankers only after a strong tabular baseline exists.
- Blend model families when their error patterns differ.
- Backfill every row to 12 predictions with valid non-duplicate articles.

## Validation

- Validate with a temporal holdout and MAP@12.
- Track ablations by candidate source and feature family.
- Report failure slices for cold customers, sparse customers, dense customers, and new or low-history articles.
