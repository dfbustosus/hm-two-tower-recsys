# hm-two-tower-recsys

Spec-driven H&M personalized fashion recommendation project for the Kaggle MAP@12 task.

The immediate goal is not to jump directly to a two-tower model. The repository is being set up as a reusable recommendation-system pipeline with clear contracts for data loading, temporal validation, metrics, candidate generation, ranking, experiment governance, and submission validation. Two-tower retrieval is treated as a challenger component that must beat strong recency, repeat-purchase, popularity, co-visitation, and ranker baselines on leakage-safe validation.

## Current project layer

This repo is in the foundation implementation layer. Governance, artifact policy, CI, and the initial `src/` package layout are in place. Production foundation components now cover H&M data-contract validation, safe string-ID CSV loading, temporal split summaries, MAP@12/recall metrics, and submission validation. Recommender models should still come only after these checks pass and baseline acceptance criteria are explicit.

## Project structure

- `src/hm_recsys/` contains importable production code.
- `tests/` contains synthetic unit tests for contracts and edge cases.
- `scripts/` contains repository-maintenance utilities used by local checks and CI.
- `docs/` contains the active specification and dependency-management policy.
- `data/`, `artifacts/`, `models/`, and `submissions/` are local-only ignored paths.

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

## Local development

The project targets Python 3.11 for CI and local development. A local virtual environment belongs in `.venv/`; it is intentionally ignored by git.

Create or refresh the environment:

```bash
make venv
```

If your default `python3` is not Python 3.11, use:

```bash
make clean-venv
make venv PYTHON=python3.11
```

Run the full local gate before opening a PR:

```bash
make check
```

This repository is not using Poetry at the bootstrap stage. The current dependency policy is pinned Python development tools in `requirements-dev.txt`, tool configuration in `pyproject.toml`, and a Makefile command surface. Runtime dependencies should be added only when production package code exists.

Validate the local H&M raw data contract:

```bash
make data-contract
```

This writes an ignored JSON report to:

```text
artifacts/data-contract/data_contract_report.json
```

Summarize a leakage-safe last-week validation split:

```bash
make temporal-split CUTOFF=2020-09-16
```

This writes an ignored JSON report under:

```text
artifacts/validation/
```

Validate a generated Kaggle submission when one exists:

```bash
make validate-submission SUBMISSION=submissions/example.csv
```

## Next implementation gate

The next code milestone is the first recommender baseline layer: recent global popularity, repeat-purchase recommendations, deterministic blending, popularity backfill to 12, and offline MAP@12 reporting on the temporal split.
