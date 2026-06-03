# hm-two-tower-recsys

Spec-driven H&M personalized fashion recommendation project for the Kaggle MAP@12 task.

The immediate goal is not to jump directly to a two-tower model. The repository is being set up as a reusable recommendation-system pipeline with clear contracts for data loading, temporal validation, metrics, candidate generation, ranking, experiment governance, and submission validation. Two-tower retrieval is treated as a challenger component that must beat strong recency, repeat-purchase, popularity, co-visitation, and ranker baselines on leakage-safe validation.

## Current project layer

This repo is in the ranking baseline implementation layer. Governance, artifact
policy, CI, and the layered `src/` package layout are in place. Production
components now cover H&M data-contract validation, safe string-ID CSV loading,
temporal split summaries, MAP@12/recall metrics, submission validation,
repeat-plus-popularity baselines, candidate-source diagnostics, co-visitation
retrieval, ranker-ready candidate exports, deterministic ranking, learned linear
ranking, rolling-window ranker validation, and learned-ranker submission
generation. The first two-tower challenger foundation now exports cutoff-safe
training examples, stable ID mappings, and deterministic random negatives. The
multimodal foundation now inventories local article images without loading image
pixels so image/text retrieval sources can be added with measured coverage.

## Project structure

- `src/hm_recsys/` contains importable production code organized by layer:
  - `core/`: ID and shared validation primitives.
  - `data/`: data contracts and safe CSV/string-ID loading.
  - `evaluation/`: temporal splits, MAP@12/recall metrics, and submission validation.
  - `retrieval/`: candidate generation and retrieval baselines.
  - `ranking/`: deterministic, learned linear, rolling-window evaluation, and ranker submission generation.
  - `embeddings/`: provider contracts and factories for text, image, and multimodal embeddings.
  - `indexing/`: vector-index contracts and factories for retrieval pipelines.
  - `training/`: training configuration contracts and two-tower example export.
  - `infrastructure/`: path resolution and local artifact locations.
  - `tools/`: repository-maintenance utilities used by local checks and CI.
- `tests/` contains synthetic unit tests for contracts and edge cases.
- `docs/` contains the active specification and dependency-management policy.
- `data/`, `artifacts/`, `models/`, and `submissions/` are local-only ignored paths.

The multi-stage architecture plan is documented in [`docs/architecture.md`](docs/architecture.md).
The image-aware/two-tower challenger design is documented in
[`docs/multimodal-two-tower-architecture.md`](docs/multimodal-two-tower-architecture.md).
The mathematical methodology for temporal validation, MAP@12, retrieval,
ranking, multimodal embeddings, and two-tower challengers is documented in
[`docs/methodology.md`](docs/methodology.md).

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
- GitHub Actions run quality, typing, pre-commit, unit-test coverage, docs build, security, secret scanning, dependency audit, and CodeQL checks.
- Dependabot monitors GitHub Actions, Python development tooling, and Python documentation tooling; see [`docs/dependency-management.md`](docs/dependency-management.md).

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

Run pre-commit hooks locally before pushing when you want the same fast static
checks used by CI:

```bash
make pre-commit
```

Build the Sphinx documentation locally:

```bash
make docs
```

This repository is not using Poetry at the foundation stage. The current dependency policy is pinned Python development tools in `requirements-dev.txt`, pinned documentation tools in `docs/requirements.txt`, tool configuration in `pyproject.toml`, and a Makefile command surface. Runtime dependencies should be added only when production package code actually requires them.

Validate the local H&M raw data contract:

```bash
make data-contract
```

This writes an ignored JSON report to:

```text
artifacts/data-contract/data_contract_report.json
```

Inventory local article images before building multimodal retrieval sources:

```bash
make image-inventory
```

This maps each `articles.csv` `article_id` to the expected local Kaggle image
path, reports missing images and malformed/extra image files, and does not load
image pixels. It writes ignored artifacts under:

```text
artifacts/multimodal/image-inventory/
```

Export encoder-ready article content records for open-source text/image models:

```bash
make article-content-export
```

This writes normalized article text fields, combined text prompts, and canonical
local image-path availability for each `article_id` without loading image pixels
or requiring heavy ML dependencies. The intended next encoder providers are
open-source FashionCLIP first, with OpenCLIP and SigLIP/SigLIP2 as challengers.
Generated records and reports are local-only under:

```text
artifacts/multimodal/article-content/
```

For bounded embedding experiments, prefer a leakage-safe transaction-prioritized
content subset instead of the first rows from `articles.csv`. For example, this
exports the 5,000 articles with strongest pre-cutoff popularity in the 30 days
before the validation cutoff:

```bash
make article-content-export \
  ARTICLE_CONTENT_PRIORITY_CUTOFF=2020-09-16 \
  ARTICLE_CONTENT_PRIORITY_LOOKBACK_DAYS=30 \
  ARTICLE_CONTENT_MAX_ARTICLES=5000
```

When the content cache will be used to train a learned ranker on a previous
label window, build the bounded cache with a priority cutoff no later than the
ranker training cutoff, or use a full article cache. A partial cache selected
using future transactions can leak through candidate availability.

The package also includes lightweight contracts for loading versioned cached
article embeddings and exact cosine retrieval. Provider jobs should write
embeddings under ignored `models/embeddings/articles/` with manifests that record
provider name, model ID, revision, dimension, preprocessing, license notes, and
the exact article-ID mapping. Cached embedding retrieval must be evaluated as a
candidate source before it is blended into ranker submissions.

Generate a bounded smoke cache with an optional open-source HuggingFace
CLIP-style provider, defaulting to FashionCLIP:

```bash
python -m pip install torch transformers pillow
make article-embeddings ARTICLE_EMBEDDING_MAX_ARTICLES=100
```

To embed a prioritized subset, point the embedding job at the exported content
CSV. The default cache directory includes the content-subset stem and optional
article cap to avoid overwriting unrelated caches:

```bash
make article-embeddings \
  ARTICLE_EMBEDDING_ARTICLE_CONTENT_PATH=artifacts/multimodal/article-content/article_content_priority_cutoff_2020-09-16_lookback_30_first_5000_articles.csv \
  ARTICLE_EMBEDDING_MAX_ARTICLES=1000 \
  ARTICLE_EMBEDDING_BATCH_SIZE=64
```

Set `ARTICLE_EMBEDDING_KIND=image`, `text`, or `multimodal`; override
`ARTICLE_EMBEDDING_MODEL_ID` for OpenCLIP/SigLIP-style HuggingFace checkpoints
that expose `get_image_features` and/or `get_text_features`. Full caches belong
under ignored local storage:

```text
models/embeddings/articles/
```

Evaluate a cached embedding source as leakage-safe content-similarity retrieval
before blending it into ranker candidates:

```bash
make content-similarity-diagnostics \
  CUTOFF=2020-09-16 \
  CONTENT_SIMILARITY_MAX_TARGET_CUSTOMERS=1000
```

This computes MAP@12 and Recall@12/50/100 from pre-cutoff customer history
vectors and writes ignored diagnostics under:

```text
artifacts/multimodal/content-similarity/
```

Raw content similarity can be too semantic and not purchase-calibrated. To test
a leakage-safe popularity-calibrated content source, rerank an oversampled
content-neighbor pool with a pre-cutoff recent-popularity prior:

```bash
make content-similarity-diagnostics \
  CUTOFF=2020-09-16 \
  CONTENT_SIMILARITY_MANIFEST=models/embeddings/articles/hf-clip_patrickjohncyh_fashion-clip_main_article_content_priority_cutoff_2020-09-09_lookback_30_first_5000_articles/multimodal_manifest.json \
  CONTENT_SIMILARITY_POPULARITY_PRIOR_WEIGHT=0.3 \
  CONTENT_SIMILARITY_POPULARITY_LOOKBACK_DAYS=7 \
  CONTENT_SIMILARITY_CANDIDATE_POOL_SIZE=200
```

When enabled, the default source name becomes
`multimodal_similarity_popularity_prior` unless explicitly overridden. Treat this
as a candidate-recall experiment until a same-split ranker validation improves
MAP@12.

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

Generate and validate a full repeat-plus-popularity baseline submission:

```bash
make baseline-submission
```

This writes an ignored CSV under:

```text
submissions/
```

Submit a locally validated CSV to Kaggle when you intentionally want to use a
submission slot:

```bash
make kaggle-submit \
  SUBMISSION=submissions/repeat_popularity_baseline_lookback_7_k_12.csv \
  KAGGLE_MESSAGE="repeat popularity baseline smoke test"
```

The submit target validates the CSV before upload and supports either Kaggle's
standard `KAGGLE_USERNAME`/`KAGGLE_KEY` variables or this repo's local
`KAGGLE_USER_NAME`/`KAGGLE_API_TOKEN` names from `.env`. It does not print the
credential values.

Evaluate the first leakage-safe baseline on a temporal split:

```bash
make baseline CUTOFF=2020-09-16
```

The baseline command generates predictions for every customer in
`sample_submission.csv` and reports full-length coverage for that full target
universe. Offline MAP@12 and recall@12 are computed only on target-universe
customers with purchases in the validation window, matching Kaggle's scoring
behavior.

This evaluates repeat purchases plus recent global popularity with deterministic backfill to 12 and writes an ignored report under:

```text
artifacts/baselines/
```

Evaluate baseline candidate sources before adding more complex retrieval:

```bash
make candidate-diagnostics CUTOFF=2020-09-16
```

This compares repeat-only, recent popularity, all-time popularity, co-visitation,
and deterministic blends with MAP@12, recall@12/50/100, candidate coverage,
article coverage, duplicate rows, candidate-count distributions, and customer
history slices. Reports are written under:

```text
artifacts/candidate-diagnostics/
```

Export ranker-ready candidate rows for validation-label customers:

```bash
make candidate-export CUTOFF=2020-09-16
```

For a bounded smoke export, cap the deterministic target customer count:

```bash
make candidate-export CUTOFF=2020-09-16 CANDIDATE_EXPORT_MAX_CUSTOMERS=1000
```

The CSV schema is:

```text
customer_id,article_id,source,source_rank,source_score
```

Exports are written under ignored local artifacts:

```text
artifacts/candidate-exports/
```

Evaluate the first transparent deterministic ranker baseline:

```bash
make ranker-baseline CUTOFF=2020-09-16
```

For a bounded smoke evaluation:

```bash
make ranker-baseline CUTOFF=2020-09-16 RANKER_MAX_TARGET_CUSTOMERS=1000
```

The ranker aggregates multiple source rows per `(customer_id, article_id)` into
source-indicator, source-rank, and source-score features, then compares MAP@12
against the same-scope repeat→recent-popularity→all-time-popularity baseline.
Reports are written under:

```text
artifacts/ranker-baselines/
```

Train a leakage-safe learned linear ranker on the previous 7-day window and
evaluate it on the requested validation cutoff:

```bash
make learned-ranker-baseline CUTOFF=2020-09-16
```

For a bounded smoke evaluation:

```bash
make learned-ranker-baseline CUTOFF=2020-09-16 LEARNED_RANKER_MAX_TARGET_CUSTOMERS=1000
```

By default, `CUTOFF=2020-09-16` trains on labels from `2020-09-09` through
`2020-09-16` exclusive, then evaluates on labels from `2020-09-16` through
`2020-09-23` exclusive. This avoids fitting on the same labels used for the
reported MAP@12.

Validate ranker improvements across rolling temporal windows before promotion:

```bash
make rolling-ranker-validation
```

The default rolling cutoffs are `2020-09-02`, `2020-09-09`, and `2020-09-16`.
Each window trains the learned linear ranker on the previous non-overlapping
7-day label window, evaluates on the requested cutoff, and reports source-order,
deterministic-ranker, and learned-ranker MAP@12/recall side by side.

For a bounded smoke run:

```bash
make rolling-ranker-validation ROLLING_RANKER_MAX_TARGET_CUSTOMERS=1000
```

To override the evaluated windows:

```bash
make rolling-ranker-validation \
  ROLLING_RANKER_CUTOFFS="2020-09-02 2020-09-09 2020-09-16"
```

Reports are written under:

```text
artifacts/ranker-baselines/
```

Generate and validate a full learned-linear-ranker submission after rolling
validation promotes it:

```bash
make learned-ranker-submission
```

This trains on the latest available 7-day label window, uses all official
training transactions for final candidate features, writes the CSV under ignored
`submissions/`, validates it against `sample_submission.csv` and `articles.csv`,
and writes a reproducibility report under:

```text
artifacts/ranker-submissions/
```

Export cutoff-safe two-tower training examples and stable ID mappings for a
bounded smoke run:

```bash
make two-tower-example-export CUTOFF=2020-09-16
```

By default this exports the first `100000` unique pre-cutoff positive pairs, one
deterministic random negative per positive, and article/customer mapping CSVs.
Negatives are sampled only from articles known before the cutoff and exclude all
pre-cutoff positives for the selected customer. To change the cap or intentionally
run a full export:

```bash
make two-tower-example-export \
  CUTOFF=2020-09-16 \
  TWO_TOWER_MAX_POSITIVE_EXAMPLES=10000

make two-tower-example-export \
  CUTOFF=2020-09-16 \
  TWO_TOWER_MAX_POSITIVE_EXAMPLES=
```

Artifacts are written under ignored local storage:

```text
artifacts/two-tower/
```

## Next implementation gate

The next code milestone is a minimal two-tower training/evaluation smoke run
that consumes the exported examples, retrieves candidates, and must improve
candidate recall or downstream MAP@12 before promotion.
