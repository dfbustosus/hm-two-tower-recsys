# H&M Recommender Architecture

This repository follows a multi-stage recommendation architecture, but each stage must earn its place through leakage-safe validation. The Kaggle objective remains MAP@12 over the next 7-day purchase window.

## Stage Funnel

1. **Data and contracts**: validate raw H&M files, preserve string IDs, and keep artifacts local.
2. **Retrieval / candidate generation**: generate high-recall candidates from repeat purchases, popularity, co-visitation, graph, two-tower, and content/multimodal sources.
3. **Refinement**: apply hard constraints such as valid article IDs, max-12 output, no duplicates, and future stock/eligibility rules if such data is available without leakage.
4. **Ranking**: score candidates with transparent source-rank features first, then stronger tabular or neural rankers when justified.
5. **Re-ranking / list optimization**: optimize the final list for diversity, novelty, business constraints, and fairness only after relevance is measured correctly.
6. **Submission validation**: enforce exact Kaggle shape and ID preservation before any CSV is trusted.

## Package Layout

- `hm_recsys.core`: shared ID validators and cross-cutting primitives.
- `hm_recsys.data`: H&M schema contracts and safe streaming readers.
- `hm_recsys.evaluation`: temporal split semantics, MAP@12, recall@K, and submission checks.
- `hm_recsys.retrieval`: candidate generators and baseline retrieval methods.
- `hm_recsys.embeddings`: provider contracts/factories for article text, image, and multimodal embeddings.
- `hm_recsys.indexing`: vector index contracts/factories for exact or ANN retrieval pipelines.
- `hm_recsys.training`: training configuration contracts, including two-tower retrieval.
- `hm_recsys.infrastructure`: path and local artifact management.
- `hm_recsys.tools`: repository hygiene and local automation helpers.

## Metric Contract

Offline MAP@12 must mirror the competition contract:

- use unique actual target articles per customer,
- normalize by `min(number_of_unique_actual_articles, 12)`,
- ignore predictions beyond rank 12,
- give no extra credit for duplicate predictions,
- still let duplicate predictions consume rank slots,
- exclude validation customers with no target purchases when mirroring Kaggle scoring.

Recall@K is used for candidate diagnostics and preserves top-K slots before deduplication, so duplicate candidates can harm measured recall.

## Two-Tower and Multimodal Position

Two-tower retrieval is a first-class advanced path, not a shortcut around validation. Before implementing the model, the project must define:

- cutoff-safe training examples,
- negative sampling strategy,
- customer and article tower inputs,
- text/image embedding providers,
- article ID to vector/index mappings,
- ANN/index provider,
- recall@K diagnostics against simple retrieval sources,
- downstream MAP@12 comparison after blending/ranking.

The codebase now has provider/factory contracts for embeddings and indexing so CLIP-style image/text encoders, local models, or external services can be added without changing downstream retrieval code. No fake provider is registered by default.

## Governance Rule

The architecture can support SOTA methods, but no method is accepted because it is fashionable. A challenger must improve leakage-safe candidate recall or MAP@12 against the current champion and must preserve submission validity.
