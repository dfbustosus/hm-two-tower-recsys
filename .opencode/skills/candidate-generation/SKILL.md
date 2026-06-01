---
name: candidate-generation
description: Use for high-recall candidate generation, source blending, co-visitation, graph retrieval, and retrieval diagnostics.
license: MIT
compatibility: opencode
metadata:
  domain: recommender-systems
  workflow: retrieval
---

# Candidate Generation Skill

Use this skill when designing or reviewing retrieval before ranking.

## Candidate Source Ladder

Build from simple to complex:

- Repeat purchases: customer history, recency, frequency, and seasonality.
- Recent popularity: global, channel-specific, product-group-specific, and age or segment popularity.
- Co-visitation and item-item: articles bought near each other in time or by similar customers.
- Collaborative filtering and graph: matrix factorization, nearest neighbors, LightGCN-style graph propagation.
- Sequence retrieval: next-item models such as SASRec, BERT4Rec, or newer time-aware sequence encoders.
- Two-tower retrieval: user and item towers with in-batch, cross-batch, and hard-negative sampling.
- Content retrieval: text/image embeddings for metadata and cold-start support.

## Required Diagnostics

- Recall@12, recall@50, recall@100, and recall@K by source and combined sources.
- Candidate count distribution per customer.
- Coverage for customers with no history, sparse history, and dense history.
- Article coverage and duplicate rate.
- Temporal split date and excluded target-window rows.

## Promotion Rule

Do not promote a retrieval source only because it is modern. Promote it when it increases combined recall or downstream MAP@12 without unacceptable cost or leakage risk.
