# H&M Two-Tower Recommender Project Instructions

This repository targets the Kaggle H&M Personalized Fashion Recommendations task: recommend up to 12 `article_id` values for each `customer_id` in `sample_submission.csv`, predicting purchases in the 7-day period immediately after the training data ends.

## Project Priorities

- Preserve Kaggle ID formats exactly. Treat `customer_id` and `article_id` as strings; never let leading zeroes disappear.
- Optimize for MAP@12 with a realistic temporal validation split. Never train on data from the validation target window.
- Build in layers: reproducible data loading, baseline/popularity candidates, offline MAP@12, feature generation, multi-source retrieval, ranking/ensembling, optional multimodal retrieval, and submission checks.
- Keep raw Kaggle data, article images, checkpoints, experiment logs, and submissions out of git.
- Prefer deterministic scripts and small tests over notebook-only workflows. Notebooks can explore, but production logic belongs in importable Python modules.
- Before changing modeling logic, identify the evaluation contract and the expected input/output schemas.
- Treat "latest model" claims as hypotheses. Promote a model only when it beats strong baselines on leakage-safe temporal validation and survives ablations.
- Keep OpenCode provider credentials and model preferences in global user config unless the user explicitly wants a project-pinned model. Project config should remain portable across collaborators.
- Keep [docs/spec-driven-development.md](../../docs/spec-driven-development.md) aligned with major scope or architecture changes.

## Dataset Contract

Expected Kaggle files:

- `transactions_train.csv`: purchase rows over time, including `t_dat`, `customer_id`, `article_id`, `price`, and `sales_channel_id`.
- `articles.csv`: article metadata, text fields, categorical product descriptors, and color/garment fields.
- `customers.csv`: customer metadata such as age, club status, fashion news settings, and postal code.
- `sample_submission.csv`: authoritative customer list for submission.
- `images/`: optional article images stored under subfolders named by the first three digits of the article id.

Recommended local layout:

- `data/raw/` for Kaggle downloads.
- `data/interim/` for typed, cleaned, or filtered tables.
- `data/processed/` for model-ready features.
- `artifacts/` for metrics, reports, and candidate dumps.
- `models/` for checkpoints and indexes.
- `submissions/` for generated Kaggle CSVs.

## Modeling Guardrails

- Use a last-week temporal holdout for fast validation, and consider multiple rolling windows before trusting leaderboard expectations.
- Include a popularity baseline and a repeat-purchase/customer-history baseline before deep modeling.
- Candidate generation should be evaluated separately from ranking. Track recall@12, recall@50, recall@100, and MAP@12 where possible.
- Candidate sources should include repeat purchases, recent global popularity, segment popularity, co-visitation/item-item signals, graph retrieval, two-tower retrieval, and content or multimodal retrieval when useful.
- Ranking should be treated as a first-class stage. Prefer a strong tabular ranker or LambdaMART-style approach over assuming retrieval scores alone are enough.
- Two-tower models should define negative sampling clearly, include enough item metadata to handle cold-ish items, save embeddings with the exact mapping from row index to ID, and compare in-batch, cross-batch, and hard-negative strategies.
- Sequential models can be valuable, but they must be judged against simpler recency/candidate baselines. Transformer, Mamba/SSM, graph, LLM-assisted, and multimodal methods are research candidates, not automatic winners.
- LLM-assisted recommendation ideas should be used cautiously: semantic feature generation, product-text normalization, explanation, or offline analysis are safer first uses than direct top-12 generation.
- Multimodal features from article text and images should be cached, versioned, and evaluated as retrieval/ranking signals rather than blindly added.
- Every experiment should log split dates, source commit, config, seed, row counts, feature versions, candidate-source recall, MAP@12, and output artifact paths.
- Final prediction rows must contain no duplicate article IDs per customer and should backfill to 12 items with valid popular candidates.

## Spec-Driven Development Contract

Before broad implementation work, create or update a short specification with:

1. Outcomes: exact successful end-state for users and submissions.
2. Scope boundaries: explicitly in-scope and out-of-scope work.
3. Constraints and assumptions: data, compute, licensing, leakage, and reproducibility constraints.
4. Prior decisions: architectural commitments already made.
5. Task breakdown: modular work units with owners or agents.
6. Verification criteria: acceptance checks, edge cases, metrics, and failure modes.

## Competition And License Notes

Use the competition data only for non-commercial and academic/competition purposes. Do not redistribute raw data, derived private data dumps, or article images through the repository. External data must be publicly available and equally accessible to all competitors at no cost if used for competition submissions.
