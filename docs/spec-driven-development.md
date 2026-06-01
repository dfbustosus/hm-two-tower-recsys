# H&M Recommender Spec-Driven Development Plan

## 1. Outcomes

The project succeeds when it can reproducibly generate a valid H&M Kaggle-style recommendation file for every `customer_id` in `sample_submission.csv`, with up to 12 non-duplicate `article_id` predictions per customer, string IDs preserved exactly, and offline MAP@12 measured on leakage-safe temporal validation splits.

The engineering outcome is a maintainable recommendation pipeline with typed data loading, temporal validation, baseline recommenders, multi-source candidate generation, ranking/ensembling, experiment tracking, submission validation, and clear artifact governance.

## 2. Scope Boundaries

In scope:

- Data contracts for `transactions_train.csv`, `articles.csv`, `customers.csv`, `sample_submission.csv`, and optional `images/`.
- Popularity and repeat-purchase baselines.
- Temporal MAP@12 validation and candidate recall diagnostics.
- Candidate sources: popularity, segment popularity, repurchase, item-item/co-visitation, graph retrieval, sequential retrieval, two-tower retrieval, and multimodal/content retrieval.
- Ranking and ensembling with ablations.
- Experiment metadata, artifact versioning, and reproducibility checks.
- Kaggle submission formatting and validation.

Out of scope for configuration alone:

- Downloading or redistributing Kaggle data.
- Training production models without the raw data present.
- Committing raw CSVs, images, checkpoints, embeddings, experiment logs, or submissions.
- Hardcoding private OpenCode provider credentials or collaborator-specific model preferences.

## 3. Constraints And Assumptions

- H&M competition data is for non-commercial and academic/competition use.
- IDs must remain strings; leading zeroes are part of the data contract.
- Validation must be temporal, usually predicting the next 7 days from past transactions.
- Hidden test data is unavailable and must not be inferred through leakage.
- Candidate generation and ranking must be evaluated separately.
- Modern neural methods are challengers, not automatic winners. They must beat simple recency, popularity, repurchase, and GBDT/Learning-to-Rank baselines.
- Large data, images, generated features, embeddings, model indexes, and submissions must remain outside git history.

## 4. Prior Decisions

- OpenCode project configuration lives in `opencode.json` and `.opencode/`.
- The default OpenCode agent is `hm-recsys-orchestrator`.
- Provider/model selection remains in global user config unless explicitly project-pinned later.
- Raw data paths follow `data/raw/`, with derived files under `data/interim/` and `data/processed/`.
- Generated artifacts use `artifacts/`, `models/`, and `submissions/`.
- OpenCode snapshots stay enabled for rollback safety, while heavy ML paths are ignored by file watching and git.

## 5. Task Breakdown

1. Data contract: validate required files, columns, dtypes, date ranges, and ID formatting.
2. Metrics: implement and test MAP@12, recall@K, duplicate handling, empty-label behavior, and submission shape checks.
3. Baselines: build recent popularity, segment popularity, and customer repeat-purchase recommenders.
4. Candidate generation: add item-item/co-visitation, graph, sequence, two-tower, and multimodal/content retrieval sources with per-source recall.
5. Feature engineering: create user, article, user-article, time, source-rank, content, and sequence features.
6. Ranking: train and evaluate tabular rankers first, then challenger neural rerankers if justified.
7. Ensembling: blend diverse candidate/ranker families with ablation evidence.
8. Experiment governance: log split, seed, config, git state, feature versions, metrics, runtime, and artifacts for every run.
9. Submission: generate one row per required customer, enforce max 12 predictions, preserve IDs, remove duplicates, and backfill safely.

## 6. Verification Criteria

- `opencode.json` validates as JSON and resolves through `opencode debug config`.
- All custom agents and skills are discoverable by OpenCode.
- Unit tests cover MAP@12 edge cases, duplicate predictions, empty actuals, ID formatting, and submission validation.
- Temporal split tests prove validation target rows are excluded from training features.
- Candidate reports include recall@12, recall@50, recall@100, coverage, candidate counts, and duplicate rate by source.
- Ranking reports include MAP@12, ablations, failure slices, and champion/challenger comparison.
- Final submission has exactly the expected customer set and no more than 12 valid article IDs per row.
