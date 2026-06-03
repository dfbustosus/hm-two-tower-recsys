# H&M Recommender Spec-Driven Development Plan

## Current Project Layer

The repository is in the **ranking baseline implementation** layer. Governance and CI are in place, and production code now uses a layered `src/hm_recsys/` package with tests. Implemented foundation components include H&M data-contract validation, safe CSV/string-ID loading, temporal split summaries, MAP@12/recall metrics, submission validation, repeat-plus-popularity baseline evaluation/submission generation, baseline candidate-source diagnostics, a first leakage-safe co-visitation challenger, ranker-ready candidate-table export, a transparent deterministic ranker baseline, a leakage-safe learned linear ranker baseline trained on a previous temporal window, rolling-window ranker validation for promotion checks, learned-ranker submission generation, cutoff-safe two-tower example export with stable ID mappings and deterministic random negatives, lightweight article-image inventory, encoder-ready article-content export, and versioned embedding-cache manifest contracts for multimodal coverage diagnostics. The package now has explicit contracts for embeddings, indexing, and two-tower training so advanced retrieval can be added without turning the package into a monolith. The repo name mentions two-tower retrieval, but the architecture must not assume a two-tower model is the best solution. For this competition, two-tower retrieval is a challenger that must prove candidate-recall or MAP@12 gains over simpler recency, repeat-purchase, popularity, co-visitation, and ranker baselines.

## Architectural Posture

- **Kaggle objective first:** recommend up to 12 `article_id` values per `customer_id`, optimized for leakage-safe MAP@12 over the next 7-day target window.
- **Evidence-gated complexity:** add graph, sequential, neural, and multimodal systems only after baselines, validation, and diagnostics can measure their value.
- **Reusable by contract:** isolate dataset-specific schemas from generic recommender stages so the same pipeline shape can serve other retail recommendation projects.
- **DRY/SOLID/KISS/YAGNI/SRP:** keep one responsibility per component, share parsing and validation utilities, avoid notebook-only production logic, and reject model families without an evaluation path.
- **Retrieval and ranking separation:** candidate generation must be evaluated independently from reranking and ensembling.

## Generic Component Contracts

The implementation should keep dataset-specific parsing separate from reusable recommender stages. Each component has one responsibility and communicates through explicit records or tables rather than hidden global state:

- **Data locator:** resolves local raw, interim, processed, artifact, model, and submission paths.
- **Schema validator:** checks required files, columns, date parseability, null behavior, and ID formats without changing data.
- **Safe loader:** reads IDs as strings and exposes typed transaction, customer, article, and submission frames.
- **Temporal splitter:** creates cutoff-based train and target windows with auditable row counts.
- **Metric evaluator:** computes MAP@12 and recall@K with deterministic duplicate handling.
- **Candidate generator:** emits `(customer_id, article_id, source, source_rank, source_score)` records.
- **Candidate diagnostics:** evaluates source-specific and blended candidate recall, coverage, duplicate rate, count distributions, and history slices before ranking.
- **Feature builder:** joins cutoff-safe customer, article, interaction, time, and source features.
- **Ranker:** consumes candidates/features and emits ordered article lists per customer.
- **Submission validator:** checks final CSV shape, customer universe, ID validity, duplicate predictions, and max-12 length.
- **Experiment reporter:** records command, config, seed, split, source version, metrics, runtime, and artifact paths.

## 1. Outcomes

The project succeeds when it can reproducibly generate a valid H&M Kaggle-style submission for every `customer_id` in `sample_submission.csv`, with exactly the expected customer set, no more than 12 non-duplicate `article_id` predictions per row, and string IDs preserved exactly.

The ML outcome is a champion/challenger recommendation pipeline that reports leakage-safe offline MAP@12 and candidate diagnostics before any submission is trusted. The initial champion should be a strong, transparent blend of recent popularity, customer repeat purchases, and segment popularity; advanced retrieval and ranking models must beat this champion on comparable temporal splits.

The engineering outcome is a maintainable, stack-agnostic recommender architecture with explicit contracts for:

- data location and schema validation,
- temporal train/validation splitting,
- MAP@12 and recall@K metrics,
- baseline recommenders,
- candidate generation,
- feature generation,
- ranking and ensembling,
- experiment metadata,
- submission generation and validation,
- local-only artifact storage.

The research outcome is disciplined experimentation: every modeling claim is tied to a split date, configuration, seed, source version, row counts, feature versions, candidate-source recall, MAP@12, runtime, and artifact path.

## 2. Scope Boundaries

### In scope now

- Repository governance and local artifact policy.
- SDD plan, architecture boundaries, and acceptance gates.
- H&M data contract for `transactions_train.csv`, `articles.csv`, `customers.csv`, `sample_submission.csv`, and optional `images/`.
- Canonical local layout: `data/raw/`, `data/interim/`, `data/processed/`, `artifacts/`, `models/`, and `submissions/`.
- Temporal validation design for next-7-day purchase prediction.
- Metric contracts for MAP@12, recall@12, recall@50, recall@100, coverage, duplicate rate, and candidate counts.
- Baseline plan: recent global popularity, repeat-purchase recommendations, customer-segment popularity, and safe backfill.
- Experiment metadata and promotion criteria.
- Submission-format contract and validation rules.

### In scope after foundation gates pass

- Item-item and co-visitation candidate generation.
- Tabular or learning-to-rank reranker with source-rank and behavioral features.
- Two-tower retrieval with explicit negative sampling, embedding export, ANN retrieval, and recall diagnostics.
- Content and multimodal retrieval using article text and optional images.
- Graph, sequential, neural reranking, and ensembling methods when ablations justify the complexity.

### Strictly out of scope now

- Redistributing Kaggle raw data or article images.
- Committing raw CSVs, images, generated feature tables, checkpoints, embeddings, ANN indexes, experiment logs, or submissions.
- Training production-scale models before data contracts, splits, metrics, and baselines exist.
- Using public leaderboard feedback as a replacement for offline validation.
- Real-time serving APIs, production infrastructure, or UI work.
- Direct LLM-generated top-12 recommendations.
- Project-pinned private credentials or collaborator-specific provider/model settings.
- Tech-stack-specific implementation choices that are not required by the current contract.

## 3. Constraints And Assumptions

### Hard constraints

- The target is purchases in the 7-day period immediately after the available training window.
- Predictions must be generated for all customers in `sample_submission.csv`.
- Customers with no purchases in the hidden test period are excluded by Kaggle scoring, but the repository must still output predictions for them.
- `customer_id` and `article_id` are strings from ingestion through submission; leading zeroes must never be lost.
- Offline validation must be temporal. Training features, popularity windows, candidate sources, negative sampling, and ranker labels must not use validation target rows.
- The default temporal split convention is `train.t_dat < cutoff` and `validation.t_dat >= cutoff` with `validation.t_dat < cutoff + 7 days`, using dates parsed from `t_dat` without timezone assumptions.
- Offline MAP@12 should be computed on validation customers with at least one target purchase, matching Kaggle's exclusion of customers with no purchases in the scoring period.
- Baseline and challenger evaluators should still generate predictions for the full `sample_submission.csv` customer universe, then compute offline MAP@12 on the labeled subset with validation purchases.
- Submission upload commands must validate the CSV first and must not print Kaggle credentials or commit generated submission files.
- Duplicate transaction rows can represent multiple purchases and must not be silently deduplicated unless the component contract explicitly requires unique items.
- Raw data and generated artifacts stay outside git.
- Runs must be reproducible for a fixed split, seed, config, and source version.
- External data for competition submissions must be public, free, and equally available to competitors.

### Assumptions to test, not believe

- A last-week holdout is a useful proxy for the hidden test week; this should be validated with rolling windows before major model promotion.
- Recency, repeat purchases, and segment popularity are strong initial baselines for H&M; they still need measured MAP@12 and recall reports.
- Two-tower retrieval will add value only if it improves candidate recall, sparse-user coverage, or downstream MAP@12 beyond simpler retrieval sources.
- Multimodal features will help mainly on cold-ish items, sparse users, or content-similarity slices; they should not be added blindly.
- Full-data runs may exceed local memory or runtime; pipelines should support deterministic sampled checks without changing contracts.

## 4. Prior Decisions

- The competition objective is MAP@12 with up to 12 ranked predictions per customer.
- The authoritative submission customer universe comes from `sample_submission.csv`.
- OpenCode project configuration lives in `opencode.json` and `.opencode/`.
- The default OpenCode agent is `hm-recsys-orchestrator`.
- Provider/model selection remains in global user config unless explicitly project-pinned later.
- Canonical raw data should live under `data/raw/h-and-m-personalized-fashion-recommendations/`; if Kaggle extracts to a root-level `h-and-m-personalized-fashion-recommendations/` folder, move it under `data/raw/` and keep both locations ignored.
- Derived data belongs under `data/interim/` or `data/processed/`.
- Metrics, diagnostics, and reports belong under `artifacts/`.
- Checkpoints, embeddings, and indexes belong under `models/`.
- Generated Kaggle files belong under `submissions/`.
- Production logic should live in importable modules once implementation begins; notebooks may explore but must not become the only source of truth.
- Production logic uses the layered `src/hm_recsys/` package layout with synthetic tests under `tests/`.
- Ranking is a first-class stage. Retrieval scores alone are not assumed sufficient.
- Submission validation is a hard gate before any CSV is considered usable.

## 5. Task Breakdown

### Stage 0: Governance and structure

- Protect raw Kaggle extraction paths and generated artifact directories from git.
- Document the canonical local layout and current project layer.
- Maintain community and security files: code of conduct, contributing guide, security policy, and PR template.
- Maintain GitHub Actions for quality checks, typing, pre-commit hooks, unit-test coverage, docs builds, security scanning, dependency auditing, secret scanning, and CodeQL.
- Maintain Dependabot for GitHub Actions, Python development tooling, and Python documentation tooling.
- Maintain a Makefile command surface for local setup, validation, linting, typing, testing, security checks, and artifact hygiene.
- Define reusable component boundaries before implementation.
- Keep the specification aligned with major architecture changes.

### Stage 1: Data contract

- Validate presence of required files.
- Validate required columns, parseable dates, ID string preservation, duplicate semantics, and expected date range.
- Produce a lightweight data-contract report with row counts, customer counts, article counts, date min/max, and null summaries.
- Fail fast on missing files, missing columns, or corrupted ID formats.
- Expose the local check through `make data-contract`; write reports under ignored `artifacts/data-contract/`.

### Stage 2: Temporal validation

- Implement a split contract that trains on `t_dat < cutoff` and validates on `cutoff <= t_dat < cutoff + 7 days`.
- Match Kaggle scoring behavior by evaluating MAP@12 on validation customers with at least one target purchase.
- Support rolling validation windows for robustness checks.
- Log split dates, transaction counts, customer counts, article counts, and excluded rows.
- Expose split summaries through `make temporal-split CUTOFF=YYYY-MM-DD`; write reports under ignored `artifacts/validation/`.

### Stage 3: Metrics and submission validation

- Implement MAP@12 with duplicate-prediction handling, empty-label behavior, fewer-than-12 predictions, and deterministic ordering.
- Implement recall@K and coverage diagnostics for candidates.
- Validate submission header, exact customer set, max-12 predictions, no duplicate article IDs per row, valid article IDs, and preserved string formatting.
- Expose submission checks through `make validate-submission SUBMISSION=path/to/file.csv`; write reports under ignored `artifacts/submission-validation/`.

### Stage 4: Baseline recommenders

- Build recent global popularity with configurable lookback windows.
- Build customer repeat-purchase recommendations ordered by recency and frequency.
- Build customer-segment popularity using safe pre-cutoff customer/article metadata.
- Blend baseline sources with deterministic tie-breaking and popularity backfill to 12 items.
- Report MAP@12, recall, coverage, duplicate rate, and runtime.
- Expose the first blended repeat-plus-popularity baseline through `make baseline CUTOFF=YYYY-MM-DD`; write reports under ignored `artifacts/baselines/`.
- Expose final-data repeat-plus-popularity CSV generation through `make baseline-submission`; write generated files under ignored `submissions/` and validate them before any upload.
- Expose baseline candidate-source diagnostics through `make candidate-diagnostics CUTOFF=YYYY-MM-DD`; write reports under ignored `artifacts/candidate-diagnostics/`.

### Stage 5: Candidate generation diagnostics

- Add source-specific candidate sets with source name, source rank, and source score.
- Evaluate per-source and combined recall@12/50/100, coverage, candidates per customer, duplicate rate, and sparse-user performance.
- Add item-item/co-visitation before more complex graph or neural retrieval, and ablate source ordering before promotion.
- Export ranker-ready candidate records with `customer_id,article_id,source,source_rank,source_score` under ignored `artifacts/candidate-exports/`.

### Stage 6: Feature engineering

- Create reusable feature families: customer, article, interaction, time, source-rank, price, popularity, repeat, and content features.
- Enforce cutoff-aware feature computation.
- Version feature definitions and persist only generated artifacts outside git.

### Stage 7: Ranking and ensembling

- Train a transparent tabular or learning-to-rank baseline before neural rerankers.
- Use candidate-source features and time-aware labels.
- Compare rankers against the blended baseline with ablations.
- Promote only when MAP@12 improves on comparable splits without harming candidate coverage or submission validity.
- Expose the first transparent deterministic ranker through `make ranker-baseline CUTOFF=YYYY-MM-DD`; write reports under ignored `artifacts/ranker-baselines/`.
- Expose the first leakage-safe learned linear ranker through `make learned-ranker-baseline CUTOFF=YYYY-MM-DD`; train on a previous non-overlapping target window and write reports under ignored `artifacts/ranker-baselines/`.
- Expose rolling-window ranker promotion checks through `make rolling-ranker-validation`; train each learned ranker on the previous non-overlapping label window and write aggregate reports under ignored `artifacts/ranker-baselines/`.
- Expose promoted learned-ranker submission generation through `make learned-ranker-submission`; train on the latest available non-overlapping label window, generate predictions for the full `sample_submission.csv` universe, validate the CSV, and write reports under ignored `artifacts/ranker-submissions/`.

### Stage 8: Two-tower and advanced challengers

- Define customer/item features, negative sampling, loss, batch strategy, embedding export, ID mapping, ANN retrieval, and backfill behavior.
- Expose cutoff-safe two-tower example export through `make two-tower-example-export CUTOFF=YYYY-MM-DD`; write examples, stable ID mappings, and reports under ignored `artifacts/two-tower/`.
- Expose local article-image inventory through `make image-inventory`; write coverage reports and article-to-image manifests under ignored `artifacts/multimodal/image-inventory/` without loading image pixels.
- Expose encoder-ready article content export through `make article-content-export`; write normalized text fields and local image-path availability under ignored `artifacts/multimodal/article-content/` without adding heavyweight ML dependencies.
- Cache future FashionCLIP/OpenCLIP/SigLIP article embeddings under ignored `models/embeddings/articles/` with provider/model/revision/dimension/preprocessing manifests.
- Compare two-tower candidates against popularity, repeat, segment, and co-visitation candidates on recall and downstream MAP@12.
- Add multimodal, graph, sequential, or neural reranking only when they have a narrow hypothesis and measurable acceptance criteria.

### Stage 9: Experiment governance and release gates

- Log command, config, seed, source version, split, row counts, feature versions, metrics, runtime, and artifact paths.
- Maintain champion/challenger comparison tables.
- Require data-contract, validation, metric, baseline, candidate, ranker, and submission checks before any final submission.

## 6. Verification Criteria

### Repository and data governance

- `.gitignore` excludes raw Kaggle files, article images, common extraction folders, generated features, checkpoints, experiment logs, and submissions.
- `git status --short` must not expose raw Kaggle CSVs or images as untracked commit candidates.
- Pull requests use the repository PR template and explicitly state validation, data/artifact impact, leakage risk, ID-format impact, and security impact.
- GitHub Actions cover repository hygiene, YAML/JSON validation, Python linting, formatting, typing, tests when present, secret scanning, dependency audit, Bandit, and CodeQL when Python source exists.
- Dependabot is configured for GitHub Actions and Python development tooling with dependency-only PRs.
- `make venv` creates an ignored local `.venv/`, and `make check` runs the local equivalent of the repository quality gates.
- Lightweight docs and source files are allowed in git; real data and generated artifacts are not.

### Data contract acceptance

- Required files are present in the configured local raw-data location.
- Required columns exist with expected semantics.
- `customer_id` and `article_id` load and save as strings.
- Date parsing is deterministic and date ranges are reported.
- Duplicate purchase rows remain available to components that need purchase frequency.

### Temporal validation acceptance

- Validation labels cover exactly the next 7 days after each cutoff.
- No validation target transaction contributes to training features, candidate popularity, co-visitation counts, negative samples, or ranker labels.
- Rolling windows produce comparable reports before major model promotion.

### Metric acceptance

- MAP@12 treats each customer's actual target articles as a unique set for relevance, uses the denominator `min(number_of_unique_actual_articles, 12)`, ignores ranks beyond 12, and gives no extra credit for duplicate predictions while still letting duplicates consume rank slots.
- Unit tests cover MAP@12 known examples, duplicate predictions, repeated actual purchases, empty actuals, fewer than 12 predictions, customers with no labels, and deterministic ranking ties.
- Candidate diagnostics report recall@12, recall@50, recall@100, coverage, candidate counts, duplicate rate, and source contribution.

### Baseline and model acceptance

- Recent popularity, repeat-purchase, segment popularity, and blended baselines run before advanced models.
- Any challenger reports deltas against the current champion on the same split and seed.
- Ablations identify whether each added source or feature improves MAP@12, recall, coverage, or an explicit failure slice.

### Submission acceptance

- CSV header is exactly `customer_id,prediction`.
- There is one row per `sample_submission.csv` customer and no extra customers.
- Each prediction is a space-separated string of valid article IDs.
- Each row has no more than 12 article IDs and no duplicates.
- Article and customer IDs preserve leading zeroes exactly.
- Backfill guarantees 12 predictions whenever enough valid articles are available.
