---
name: two-tower-recsys
description: Use for two-tower recommender architecture, negative sampling, embedding export, ANN retrieval, and blending strategies.
license: MIT
compatibility: opencode
metadata:
  domain: recommender-systems
  model: two-tower
---

# Two-Tower Recommender Skill

Use this skill when implementing or reviewing retrieval models.

## Architecture Decisions

Make these explicit before coding:

- Customer tower inputs: customer ID embedding, age buckets, club status, fashion news flags, recent purchases, and aggregated behavior.
- Item tower inputs: article ID embedding, product group/type, color, department, garment group, descriptive text, and optional image embeddings.
- Shared embedding dimension and normalization strategy.
- Loss: sampled softmax, in-batch negatives, BPR, or contrastive loss.
- Negative sampling: in-batch, cross-batch, random, popularity-weighted, same-category hard negatives, or mixed.
- Logit correction or sampling-bias correction when negatives are not uniformly sampled.

## Training Guardrails

- Build examples only from transactions before the split cutoff.
- Sample negatives from items known before the cutoff.
- Keep ID-to-index mappings stable and saved with the model.
- Test on a tiny sample before full training.
- Log split dates, row counts, customer/item counts, and random seeds.

## Retrieval

- Export item embeddings with an index-to-article-ID mapping.
- Retrieve more than 12 candidates, then blend or rerank.
- Backfill with recent popularity and customer repeat candidates.
- Evaluate retrieval recall separately from final MAP@12.
- Compare against graph, co-visitation, and sequence retrieval; do not assume two-tower is the retrieval champion.

## Practical Baselines

Do not skip simple baselines:

- Recent global popularity.
- Customer repeat purchases ranked by recency/frequency.
- Segment popularity by age, product group, or sales channel.
- Blend baseline candidates with model retrieval before reranking.
