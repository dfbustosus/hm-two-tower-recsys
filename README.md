# hm-two-tower-recsys

Spec-driven H&M personalized fashion recommendation project for the Kaggle MAP@12 task.

The immediate goal is not to jump directly to a two-tower model. The repository is being set up as a reusable recommendation-system pipeline with clear contracts for data loading, temporal validation, metrics, candidate generation, ranking, experiment governance, and submission validation. Two-tower retrieval is treated as a challenger component that must beat strong recency, repeat-purchase, popularity, co-visitation, and ranker baselines on leakage-safe validation.

## Current project layer

This repo is in the SDD and governance bootstrap layer. Production modeling code should come only after the data contract, validation contract, metrics, artifact layout, and baseline acceptance checks are explicit.

## Local data and artifact policy

Keep Kaggle data and generated outputs local only:

- `data/raw/h-and-m-personalized-fashion-recommendations/` for raw competition files.
- `data/interim/` and `data/processed/` for derived tables.
- `artifacts/` for metrics, diagnostics, and reports.
- `models/` for checkpoints, embeddings, and indexes.
- `submissions/` for generated Kaggle CSV files.

Raw CSVs, images, derived feature tables, checkpoints, experiment logs, and submissions must not be committed.

The expected local raw-data directory contains:

- `articles.csv`
- `customers.csv`
- `sample_submission.csv`
- `transactions_train.csv`
- `images/`

## Planning document

The active specification lives in [`docs/spec-driven-development.md`](docs/spec-driven-development.md). It defines outcomes, scope, constraints, prior decisions, modular task breakdown, and verification criteria for the project.

## Repository governance

- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) defines collaboration standards.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) defines PR and engineering expectations.
- [`SECURITY.md`](SECURITY.md) defines vulnerability reporting and secret/data handling.
- [`.github/pull_request_template.md`](.github/pull_request_template.md) requires validation, data, leakage, ID-format, and security checks for each PR.
- GitHub Actions run quality, typing, security, secret scanning, dependency audit, and CodeQL checks.
- Dependabot monitors GitHub Actions and Python development tooling; see [`docs/dependency-management.md`](docs/dependency-management.md).

## Next implementation gate

The next code milestone is foundation work only: data-contract validation, safe string-ID loading, exact temporal split semantics, MAP@12 tests, and submission validation. Recommender models come after those checks pass.
