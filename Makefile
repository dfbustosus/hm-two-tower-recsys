SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
VENV_KAGGLE := $(VENV)/bin/kaggle
VENV_STAMP := $(VENV)/.requirements-dev.stamp
EXCLUDED_SCAN_DIRS := .git,.venv,artifacts,data,env,models,outputs,submissions,venv

.DEFAULT_GOAL := help

BASELINE_LOOKBACK_DAYS ?= 7
BASELINE_K ?= 12
BASELINE_SUBMISSION ?= submissions/repeat_popularity_baseline_lookback_$(BASELINE_LOOKBACK_DAYS)_k_$(BASELINE_K).csv
CANDIDATE_K ?= 12
CANDIDATE_EXPORT_MAX_CUSTOMERS ?=
RANKER_K ?= 12
RANKER_CANDIDATE_K ?= 12
RANKER_MAX_TARGET_CUSTOMERS ?=
LEARNED_RANKER_K ?= 12
LEARNED_RANKER_CANDIDATE_K ?= 12
LEARNED_RANKER_MAX_TARGET_CUSTOMERS ?=
LEARNED_RANKER_EPOCHS ?= 3
LEARNED_RANKER_LEARNING_RATE ?= 0.01
LEARNED_RANKER_L2 ?= 0.001
LEARNED_RANKER_SUBMISSION ?=
LEARNED_RANKER_SUBMISSION_NO_CO_VISITATION ?=
ROLLING_RANKER_CUTOFFS ?= 2020-09-02 2020-09-09 2020-09-16
ROLLING_RANKER_K ?= 12
ROLLING_RANKER_CANDIDATE_K ?= 12
ROLLING_RANKER_MAX_TARGET_CUSTOMERS ?=
ROLLING_RANKER_EPOCHS ?= 3
ROLLING_RANKER_LEARNING_RATE ?= 0.01
ROLLING_RANKER_L2 ?= 0.001
ROLLING_RANKER_NO_CO_VISITATION ?=
TWO_TOWER_NEGATIVES_PER_POSITIVE ?= 1
TWO_TOWER_NEGATIVE_SAMPLING ?= random
TWO_TOWER_SEED ?= 42
TWO_TOWER_MAX_POSITIVE_EXAMPLES ?= 100000
TWO_TOWER_POSITIVE_SELECTION ?= latest
TWO_TOWER_EMBEDDING_DIM ?= 16
TWO_TOWER_EPOCHS ?= 3
TWO_TOWER_LEARNING_RATE ?= 0.05
TWO_TOWER_L2 ?= 0.0
TWO_TOWER_LOSS ?= logistic
TWO_TOWER_LOGQ_CORRECTION_ALPHA ?= 0.0
TWO_TOWER_POSITIVE_RECENCY_HALF_LIFE_DAYS ?=
TWO_TOWER_MAX_TRAINING_EXAMPLES ?=
TWO_TOWER_MAX_EVAL_CUSTOMERS ?= 1000
TWO_TOWER_MAX_RETRIEVAL_ARTICLES ?= 5000
TWO_TOWER_EVALUATION_KS ?= 12 50 100
TWO_TOWER_POPULARITY_PRIOR_WEIGHT ?=
TWO_TOWER_POPULARITY_PRIOR_LOOKBACK_DAYS ?= 7
INCLUDE_TWO_TOWER_RETRIEVAL ?=
TWO_TOWER_RANKER_PRESENCE_WEIGHT ?=
TWO_TOWER_RANKER_SCORE_WEIGHT ?=
ARTICLE_CONTENT_OUTPUT ?=
ARTICLE_CONTENT_REPORT ?=
ARTICLE_CONTENT_MAX_ARTICLES ?=
ARTICLE_CONTENT_PRIORITY_CUTOFF ?=
ARTICLE_CONTENT_PRIORITY_LOOKBACK_DAYS ?=
ARTICLE_EMBEDDING_PROVIDER ?= hf-clip
ARTICLE_EMBEDDING_MODEL_ID ?= patrickjohncyh/fashion-clip
ARTICLE_EMBEDDING_MODEL_REVISION ?= main
ARTICLE_EMBEDDING_KIND ?= multimodal
ARTICLE_EMBEDDING_BATCH_SIZE ?= 32
ARTICLE_EMBEDDING_MAX_ARTICLES ?= 100
ARTICLE_EMBEDDING_ARTICLE_CONTENT_PATH ?=
CONTENT_SIMILARITY_MANIFEST ?= models/embeddings/articles/hf-clip_patrickjohncyh_fashion-clip_main/multimodal_manifest.json
CONTENT_SIMILARITY_SOURCE ?= multimodal_similarity
CONTENT_SIMILARITY_MAX_TARGET_CUSTOMERS ?= 1000
CONTENT_SIMILARITY_POPULARITY_PRIOR_WEIGHT ?=
CONTENT_SIMILARITY_POPULARITY_LOOKBACK_DAYS ?=
CONTENT_SIMILARITY_CANDIDATE_POOL_SIZE ?=
LEARNED_RANKER_CONTENT_SIMILARITY_MANIFEST ?=
LEARNED_RANKER_CONTENT_SIMILARITY_SOURCE ?= multimodal_similarity
INCLUDE_AGE_SEGMENT_POPULARITY ?=
AGE_SEGMENT_BUCKET_SIZE ?= 10
AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS ?=
INCLUDE_GARMENT_GROUP_POPULARITY ?=
GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS ?=
GARMENT_GROUP_MAX_HISTORY_ITEMS ?= 8
DETERMINISTIC_TUNING_TOP_TRIALS ?= 10
DETERMINISTIC_TUNING_RESEARCH_GRID ?=
KAGGLE_COMPETITION ?= h-and-m-personalized-fashion-recommendations
KAGGLE_MESSAGE ?= repeat popularity baseline smoke test

EDA_ROLLING_CUTOFFS ?= 2020-09-02 2020-09-09 2020-09-16
EDA_COLD_MAX_TRANSACTIONS ?= 0
EDA_SPARSE_MAX_TRANSACTIONS ?= 4
EDA_TOP_HIERARCHY_VALUES ?= 20
EDA_TOP_BUSY_DAYS ?= 30
EDA_REPORT_PATH ?=
EDA_MARKDOWN_PATH ?=

.PHONY: help venv install-dev check validate lint type test security audit pre-commit docs data-contract eda-report image-inventory article-content-export article-embeddings content-similarity-diagnostics temporal-split validate-submission baseline baseline-submission candidate-diagnostics candidate-export ranker-baseline deterministic-ranker-tuning learned-ranker-baseline rolling-ranker-validation deterministic-ranker-submission learned-ranker-submission two-tower-example-export two-tower-retrieval-smoke kaggle-submit format clean clean-venv

help:
	@printf "H&M recommender development commands\n\n"
	@printf "Setup:\n"
	@printf "  make venv          Create/update the ignored local virtual environment\n"
	@printf "  make install-dev   Alias for make venv\n\n"
	@printf "Quality gates:\n"
	@printf "  make check         Run validation, lint, typing, security, and tests\n"
	@printf "  make validate      Validate JSON/TOML/YAML configuration\n"
	@printf "  make lint          Run ruff, black, isort, and flake8\n"
	@printf "  make type          Run mypy when Python files exist\n"
	@printf "  make test          Run pytest when tests exist\n"
	@printf "  make security      Run pip-audit and Bandit when Python files exist\n"
	@printf "  make audit         Check tracked files for forbidden data/artifacts\n\n"
	@printf "  make pre-commit    Run pre-commit hooks across all files\n"
	@printf "  make docs          Build Sphinx documentation locally\n\n"
	@printf "Data:\n"
	@printf "  make data-contract Validate local H&M raw data and write an ignored report\n\n"
	@printf "  make eda-report    Run Phase -1 EDA and write JSON+Markdown reports\n\n"
	@printf "  make image-inventory  Map articles to local images and write ignored reports\n\n"
	@printf "  make article-content-export  Export article text/image paths for encoders\n\n"
	@printf "  make article-embeddings  Generate optional open-source article embeddings\n\n"
	@printf "  make content-similarity-diagnostics CUTOFF=YYYY-MM-DD  Evaluate cached embeddings\n\n"
	@printf "Validation/submission:\n"
	@printf "  make temporal-split CUTOFF=YYYY-MM-DD  Summarize a temporal split\n"
	@printf "  make validate-submission SUBMISSION=path/to.csv  Validate a submission CSV\n"
	@printf "  make kaggle-submit SUBMISSION=path/to.csv  Submit a validated CSV to Kaggle\n\n"
	@printf "Baselines:\n"
	@printf "  make baseline CUTOFF=YYYY-MM-DD  Evaluate repeat+popularity baseline\n"
	@printf "  make baseline-submission  Generate and validate repeat+popularity CSV\n"
	@printf "  make candidate-diagnostics CUTOFF=YYYY-MM-DD  Evaluate candidate sources\n\n"
	@printf "  make candidate-export CUTOFF=YYYY-MM-DD  Export ranker-ready candidates\n\n"
	@printf "  make ranker-baseline CUTOFF=YYYY-MM-DD  Evaluate deterministic ranker\n\n"
	@printf "  make deterministic-ranker-tuning CUTOFF=YYYY-MM-DD  Tune deterministic weights\n\n"
	@printf "  make learned-ranker-baseline CUTOFF=YYYY-MM-DD  Train/evaluate linear ranker\n\n"
	@printf "  make rolling-ranker-validation  Validate rankers across rolling windows\n\n"
	@printf "  make deterministic-ranker-submission  Generate tuned deterministic CSV\n\n"
	@printf "  make learned-ranker-submission  Generate validated learned-ranker CSV\n\n"
	@printf "  make two-tower-example-export CUTOFF=YYYY-MM-DD  Export two-tower examples\n\n"
	@printf "  make two-tower-retrieval-smoke CUTOFF=YYYY-MM-DD  Train/evaluate two-tower smoke\n\n"
	@printf "Maintenance:\n"
	@printf "  make format        Auto-format Python files when present\n"
	@printf "  make clean         Remove local caches, not data or the virtualenv\n"
	@printf "  make clean-venv    Remove the local virtualenv\n"

$(VENV_PYTHON):
	$(PYTHON) -m venv "$(VENV)"
	"$(VENV_PYTHON)" -m pip install --upgrade pip

$(VENV_STAMP): requirements-dev.txt pyproject.toml $(VENV_PYTHON)
	"$(VENV_PIP)" install -r requirements-dev.txt
	"$(VENV_PIP)" install -e .
	@touch "$(VENV_STAMP)"

venv: $(VENV_STAMP)
	@printf "Virtual environment ready at $(VENV). Activate with: source $(VENV)/bin/activate\n"

install-dev: venv

check: validate audit lint type security test

validate: venv
	"$(VENV_PYTHON)" -m json.tool opencode.json > /tmp/opencode-json-validated.json
	"$(VENV_PYTHON)" -c 'import tomllib; tomllib.load(open("pyproject.toml", "rb")); print("pyproject.toml: valid TOML")'
	"$(VENV)/bin/yamllint" .github .yamllint

audit:
	PYTHONPATH=src $(PYTHON) -m hm_recsys.tools.check_repo_hygiene

lint: venv
	"$(VENV)/bin/ruff" check .
	"$(VENV)/bin/black" --check .
	"$(VENV)/bin/isort" --check-only .
	"$(VENV)/bin/flake8" .

type: venv
	@files="$$(git ls-files --cached --others --exclude-standard -- '*.py' | grep -Ev '^(data|artifacts|models|outputs|submissions|\.venv|venv|env)/' | while IFS= read -r path; do [[ -f "$$path" ]] && printf '%s\n' "$$path"; done || true)"; \
	if [[ -n "$$files" ]]; then \
		"$(VENV_PYTHON)" -m mypy $$files; \
	else \
		printf "No Python files detected; skipping mypy.\n"; \
	fi

test: venv
	@tests="$$(git ls-files --cached --others --exclude-standard -- 'test_*.py' '*_test.py' 'tests/*.py' 'tests/**/*.py' | grep -Ev '^(data|artifacts|models|outputs|submissions|\.venv|venv|env)/' | while IFS= read -r path; do [[ -f "$$path" ]] && printf '%s\n' "$$path"; done || true)"; \
	if [[ -n "$$tests" ]]; then \
		"$(VENV_PYTHON)" -m pytest; \
	else \
		printf "No pytest tests detected; skipping pytest.\n"; \
	fi

security: venv
	"$(VENV)/bin/pip-audit" -r requirements-dev.txt --progress-spinner off
	@files="$$(git ls-files --cached --others --exclude-standard -- '*.py' | grep -Ev '^(data|artifacts|models|outputs|submissions|\.venv|venv|env)/' | while IFS= read -r path; do [[ -f "$$path" ]] && printf '%s\n' "$$path"; done || true)"; \
	if [[ -n "$$files" ]]; then \
		"$(VENV)/bin/bandit" -r . --exclude "$(EXCLUDED_SCAN_DIRS)" -ll; \
	else \
		printf "No Python files detected; skipping Bandit.\n"; \
	fi

pre-commit: venv
	"$(VENV)/bin/pre-commit" run --all-files --show-diff-on-failure --color=always

docs: venv
	"$(VENV_PIP)" install -r docs/requirements.txt
	"$(VENV_PYTHON)" -m sphinx -b html docs docs/_build/html

data-contract: venv
	"$(VENV_PYTHON)" -m hm_recsys.cli validate-data-contract

eda-report: venv
	@extra_args=""; \
	if [[ -n "$(EDA_REPORT_PATH)" ]]; then \
		extra_args="$$extra_args --report-path $(EDA_REPORT_PATH)"; \
	fi; \
	if [[ -n "$(EDA_MARKDOWN_PATH)" ]]; then \
		extra_args="$$extra_args --markdown-path $(EDA_MARKDOWN_PATH)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli eda-report --rolling-cutoffs $(EDA_ROLLING_CUTOFFS) --cold-max-transactions "$(EDA_COLD_MAX_TRANSACTIONS)" --sparse-max-transactions "$(EDA_SPARSE_MAX_TRANSACTIONS)" --top-hierarchy-values "$(EDA_TOP_HIERARCHY_VALUES)" --top-busy-days "$(EDA_TOP_BUSY_DAYS)" $$extra_args

image-inventory: venv
	"$(VENV_PYTHON)" -m hm_recsys.cli inventory-article-images

article-content-export: venv
	@extra_args=""; \
	if [[ -n "$(ARTICLE_CONTENT_OUTPUT)" ]]; then \
		extra_args="$$extra_args --output-path $(ARTICLE_CONTENT_OUTPUT)"; \
	fi; \
	if [[ -n "$(ARTICLE_CONTENT_REPORT)" ]]; then \
		extra_args="$$extra_args --report-path $(ARTICLE_CONTENT_REPORT)"; \
	fi; \
	if [[ -n "$(ARTICLE_CONTENT_MAX_ARTICLES)" ]]; then \
		extra_args="$$extra_args --max-articles $(ARTICLE_CONTENT_MAX_ARTICLES)"; \
	fi; \
	if [[ -n "$(ARTICLE_CONTENT_PRIORITY_CUTOFF)" ]]; then \
		extra_args="$$extra_args --priority-cutoff $(ARTICLE_CONTENT_PRIORITY_CUTOFF)"; \
	fi; \
	if [[ -n "$(ARTICLE_CONTENT_PRIORITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --priority-lookback-days $(ARTICLE_CONTENT_PRIORITY_LOOKBACK_DAYS)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli export-article-content $$extra_args

article-embeddings: venv
	@extra_args=""; \
	if [[ -n "$(ARTICLE_EMBEDDING_MAX_ARTICLES)" ]]; then \
		extra_args="$$extra_args --max-articles $(ARTICLE_EMBEDDING_MAX_ARTICLES)"; \
	fi; \
	if [[ -n "$(ARTICLE_EMBEDDING_ARTICLE_CONTENT_PATH)" ]]; then \
		extra_args="$$extra_args --article-content-path $(ARTICLE_EMBEDDING_ARTICLE_CONTENT_PATH)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli generate-article-embeddings --provider "$(ARTICLE_EMBEDDING_PROVIDER)" --model-id "$(ARTICLE_EMBEDDING_MODEL_ID)" --model-revision "$(ARTICLE_EMBEDDING_MODEL_REVISION)" --embedding-kind "$(ARTICLE_EMBEDDING_KIND)" --batch-size "$(ARTICLE_EMBEDDING_BATCH_SIZE)" $$extra_args

content-similarity-diagnostics: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make content-similarity-diagnostics CUTOFF=2020-09-16\n"; exit 2; fi
	@extra_args=""; \
	if [[ -n "$(CONTENT_SIMILARITY_MAX_TARGET_CUSTOMERS)" ]]; then \
		extra_args="$$extra_args --max-target-customers $(CONTENT_SIMILARITY_MAX_TARGET_CUSTOMERS)"; \
	fi; \
	if [[ -n "$(CONTENT_SIMILARITY_POPULARITY_PRIOR_WEIGHT)" ]]; then \
		extra_args="$$extra_args --popularity-prior-weight $(CONTENT_SIMILARITY_POPULARITY_PRIOR_WEIGHT)"; \
	fi; \
	if [[ -n "$(CONTENT_SIMILARITY_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --popularity-lookback-days $(CONTENT_SIMILARITY_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(CONTENT_SIMILARITY_CANDIDATE_POOL_SIZE)" ]]; then \
		extra_args="$$extra_args --candidate-pool-size $(CONTENT_SIMILARITY_CANDIDATE_POOL_SIZE)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli content-similarity-diagnostics --cutoff "$(CUTOFF)" --manifest-path "$(CONTENT_SIMILARITY_MANIFEST)" --source-name "$(CONTENT_SIMILARITY_SOURCE)" --evaluation-ks 12 50 100 $$extra_args

temporal-split: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make temporal-split CUTOFF=2020-09-16\n"; exit 2; fi
	"$(VENV_PYTHON)" -m hm_recsys.cli summarize-temporal-split --cutoff "$(CUTOFF)"

validate-submission: venv
	@if [[ -z "$(SUBMISSION)" ]]; then printf "SUBMISSION is required, e.g. make validate-submission SUBMISSION=submissions/file.csv\n"; exit 2; fi
	"$(VENV_PYTHON)" -m hm_recsys.cli validate-submission --submission-path "$(SUBMISSION)"

baseline: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make baseline CUTOFF=2020-09-16\n"; exit 2; fi
	"$(VENV_PYTHON)" -m hm_recsys.cli evaluate-baseline --cutoff "$(CUTOFF)" --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --k "$(BASELINE_K)"

baseline-submission: venv
	"$(VENV_PYTHON)" -m hm_recsys.cli generate-baseline-submission --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --k "$(BASELINE_K)" --output-path "$(BASELINE_SUBMISSION)"

candidate-diagnostics: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make candidate-diagnostics CUTOFF=2020-09-16\n"; exit 2; fi
	"$(VENV_PYTHON)" -m hm_recsys.cli candidate-diagnostics --cutoff "$(CUTOFF)" --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --evaluation-ks 12 50 100

candidate-export: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make candidate-export CUTOFF=2020-09-16\n"; exit 2; fi
	@extra_args=""; \
	if [[ -n "$(CANDIDATE_EXPORT_MAX_CUSTOMERS)" ]]; then \
		extra_args="$$extra_args --max-target-customers $(CANDIDATE_EXPORT_MAX_CUSTOMERS)"; \
	fi; \
	if [[ -n "$(INCLUDE_AGE_SEGMENT_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-age-segment-popularity --age-segment-bucket-size $(AGE_SEGMENT_BUCKET_SIZE)"; \
	fi; \
	if [[ -n "$(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --age-segment-popularity-lookback-days $(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_GARMENT_GROUP_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-garment-group-popularity --garment-group-max-history-items $(GARMENT_GROUP_MAX_HISTORY_ITEMS)"; \
	fi; \
	if [[ -n "$(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --garment-group-popularity-lookback-days $(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_TWO_TOWER_RETRIEVAL)" ]]; then \
		extra_args="$$extra_args --include-two-tower-retrieval --two-tower-negatives-per-positive $(TWO_TOWER_NEGATIVES_PER_POSITIVE) --two-tower-negative-sampling $(TWO_TOWER_NEGATIVE_SAMPLING) --two-tower-seed $(TWO_TOWER_SEED) --two-tower-positive-selection $(TWO_TOWER_POSITIVE_SELECTION) --two-tower-max-positive-examples $(TWO_TOWER_MAX_POSITIVE_EXAMPLES) --two-tower-embedding-dim $(TWO_TOWER_EMBEDDING_DIM) --two-tower-epochs $(TWO_TOWER_EPOCHS) --two-tower-learning-rate $(TWO_TOWER_LEARNING_RATE) --two-tower-l2 $(TWO_TOWER_L2) --two-tower-loss $(TWO_TOWER_LOSS) --two-tower-logq-correction-alpha $(TWO_TOWER_LOGQ_CORRECTION_ALPHA) --two-tower-max-retrieval-articles $(TWO_TOWER_MAX_RETRIEVAL_ARTICLES)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli export-candidates --cutoff "$(CUTOFF)" --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --k "$(CANDIDATE_K)" $$extra_args

ranker-baseline: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make ranker-baseline CUTOFF=2020-09-16\n"; exit 2; fi
	@extra_args=""; \
	if [[ -n "$(RANKER_MAX_TARGET_CUSTOMERS)" ]]; then \
		extra_args="$$extra_args --max-target-customers $(RANKER_MAX_TARGET_CUSTOMERS)"; \
	fi; \
	if [[ -n "$(INCLUDE_AGE_SEGMENT_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-age-segment-popularity --age-segment-bucket-size $(AGE_SEGMENT_BUCKET_SIZE)"; \
	fi; \
	if [[ -n "$(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --age-segment-popularity-lookback-days $(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_GARMENT_GROUP_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-garment-group-popularity --garment-group-max-history-items $(GARMENT_GROUP_MAX_HISTORY_ITEMS)"; \
	fi; \
	if [[ -n "$(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --garment-group-popularity-lookback-days $(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_TWO_TOWER_RETRIEVAL)" ]]; then \
		extra_args="$$extra_args --include-two-tower-retrieval --two-tower-negatives-per-positive $(TWO_TOWER_NEGATIVES_PER_POSITIVE) --two-tower-negative-sampling $(TWO_TOWER_NEGATIVE_SAMPLING) --two-tower-seed $(TWO_TOWER_SEED) --two-tower-positive-selection $(TWO_TOWER_POSITIVE_SELECTION) --two-tower-max-positive-examples $(TWO_TOWER_MAX_POSITIVE_EXAMPLES) --two-tower-embedding-dim $(TWO_TOWER_EMBEDDING_DIM) --two-tower-epochs $(TWO_TOWER_EPOCHS) --two-tower-learning-rate $(TWO_TOWER_LEARNING_RATE) --two-tower-l2 $(TWO_TOWER_L2) --two-tower-loss $(TWO_TOWER_LOSS) --two-tower-logq-correction-alpha $(TWO_TOWER_LOGQ_CORRECTION_ALPHA) --two-tower-max-retrieval-articles $(TWO_TOWER_MAX_RETRIEVAL_ARTICLES)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_RANKER_PRESENCE_WEIGHT)" ]]; then \
		extra_args="$$extra_args --two-tower-ranker-presence-weight $(TWO_TOWER_RANKER_PRESENCE_WEIGHT)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_RANKER_SCORE_WEIGHT)" ]]; then \
		extra_args="$$extra_args --two-tower-ranker-score-weight $(TWO_TOWER_RANKER_SCORE_WEIGHT)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli evaluate-ranker-baseline --cutoff "$(CUTOFF)" --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --candidate-k "$(RANKER_CANDIDATE_K)" --k "$(RANKER_K)" $$extra_args

deterministic-ranker-tuning: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make deterministic-ranker-tuning CUTOFF=2020-09-16\n"; exit 2; fi
	@extra_args=""; \
	if [[ -n "$(RANKER_MAX_TARGET_CUSTOMERS)" ]]; then \
		extra_args="$$extra_args --max-target-customers $(RANKER_MAX_TARGET_CUSTOMERS)"; \
	fi; \
	if [[ -n "$(INCLUDE_AGE_SEGMENT_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-age-segment-popularity --age-segment-bucket-size $(AGE_SEGMENT_BUCKET_SIZE)"; \
	fi; \
	if [[ -n "$(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --age-segment-popularity-lookback-days $(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_GARMENT_GROUP_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-garment-group-popularity --garment-group-max-history-items $(GARMENT_GROUP_MAX_HISTORY_ITEMS)"; \
	fi; \
	if [[ -n "$(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --garment-group-popularity-lookback-days $(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_TWO_TOWER_RETRIEVAL)" ]]; then \
		extra_args="$$extra_args --include-two-tower-retrieval --two-tower-negatives-per-positive $(TWO_TOWER_NEGATIVES_PER_POSITIVE) --two-tower-negative-sampling $(TWO_TOWER_NEGATIVE_SAMPLING) --two-tower-seed $(TWO_TOWER_SEED) --two-tower-positive-selection $(TWO_TOWER_POSITIVE_SELECTION) --two-tower-max-positive-examples $(TWO_TOWER_MAX_POSITIVE_EXAMPLES) --two-tower-embedding-dim $(TWO_TOWER_EMBEDDING_DIM) --two-tower-epochs $(TWO_TOWER_EPOCHS) --two-tower-learning-rate $(TWO_TOWER_LEARNING_RATE) --two-tower-l2 $(TWO_TOWER_L2) --two-tower-loss $(TWO_TOWER_LOSS) --two-tower-logq-correction-alpha $(TWO_TOWER_LOGQ_CORRECTION_ALPHA) --two-tower-max-retrieval-articles $(TWO_TOWER_MAX_RETRIEVAL_ARTICLES)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_RANKER_PRESENCE_WEIGHT)" ]]; then \
		extra_args="$$extra_args --two-tower-ranker-presence-weight $(TWO_TOWER_RANKER_PRESENCE_WEIGHT)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_RANKER_SCORE_WEIGHT)" ]]; then \
		extra_args="$$extra_args --two-tower-ranker-score-weight $(TWO_TOWER_RANKER_SCORE_WEIGHT)"; \
	fi; \
	if [[ -n "$(DETERMINISTIC_TUNING_RESEARCH_GRID)" ]]; then \
		extra_args="$$extra_args --research-weight-grid"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli tune-deterministic-ranker --cutoff "$(CUTOFF)" --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --candidate-k "$(RANKER_CANDIDATE_K)" --k "$(RANKER_K)" --top-trials "$(DETERMINISTIC_TUNING_TOP_TRIALS)" $$extra_args

learned-ranker-baseline: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make learned-ranker-baseline CUTOFF=2020-09-16\n"; exit 2; fi
	@extra_args=""; \
	if [[ -n "$(LEARNED_RANKER_MAX_TARGET_CUSTOMERS)" ]]; then \
		extra_args="$$extra_args --max-target-customers $(LEARNED_RANKER_MAX_TARGET_CUSTOMERS)"; \
	fi; \
	if [[ -n "$(INCLUDE_AGE_SEGMENT_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-age-segment-popularity --age-segment-bucket-size $(AGE_SEGMENT_BUCKET_SIZE)"; \
	fi; \
	if [[ -n "$(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --age-segment-popularity-lookback-days $(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_GARMENT_GROUP_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-garment-group-popularity --garment-group-max-history-items $(GARMENT_GROUP_MAX_HISTORY_ITEMS)"; \
	fi; \
	if [[ -n "$(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --garment-group-popularity-lookback-days $(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(LEARNED_RANKER_CONTENT_SIMILARITY_MANIFEST)" ]]; then \
		extra_args="$$extra_args --content-similarity-manifest-path $(LEARNED_RANKER_CONTENT_SIMILARITY_MANIFEST) --content-similarity-source-name $(LEARNED_RANKER_CONTENT_SIMILARITY_SOURCE)"; \
	fi; \
	if [[ -n "$(CONTENT_SIMILARITY_POPULARITY_PRIOR_WEIGHT)" ]]; then \
		extra_args="$$extra_args --content-similarity-popularity-prior-weight $(CONTENT_SIMILARITY_POPULARITY_PRIOR_WEIGHT)"; \
	fi; \
	if [[ -n "$(CONTENT_SIMILARITY_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --content-similarity-popularity-lookback-days $(CONTENT_SIMILARITY_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(CONTENT_SIMILARITY_CANDIDATE_POOL_SIZE)" ]]; then \
		extra_args="$$extra_args --content-similarity-candidate-pool-size $(CONTENT_SIMILARITY_CANDIDATE_POOL_SIZE)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli evaluate-learned-ranker-baseline --cutoff "$(CUTOFF)" --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --candidate-k "$(LEARNED_RANKER_CANDIDATE_K)" --k "$(LEARNED_RANKER_K)" --epochs "$(LEARNED_RANKER_EPOCHS)" --learning-rate "$(LEARNED_RANKER_LEARNING_RATE)" --l2 "$(LEARNED_RANKER_L2)" $$extra_args

rolling-ranker-validation: venv
	@extra_args=""; \
	if [[ -n "$(ROLLING_RANKER_MAX_TARGET_CUSTOMERS)" ]]; then \
		extra_args="$$extra_args --max-target-customers $(ROLLING_RANKER_MAX_TARGET_CUSTOMERS)"; \
	fi; \
	if [[ -n "$(ROLLING_RANKER_NO_CO_VISITATION)" ]]; then \
		extra_args="$$extra_args --no-co-visitation"; \
	fi; \
	if [[ -n "$(INCLUDE_AGE_SEGMENT_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-age-segment-popularity --age-segment-bucket-size $(AGE_SEGMENT_BUCKET_SIZE)"; \
	fi; \
	if [[ -n "$(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --age-segment-popularity-lookback-days $(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_GARMENT_GROUP_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-garment-group-popularity --garment-group-max-history-items $(GARMENT_GROUP_MAX_HISTORY_ITEMS)"; \
	fi; \
	if [[ -n "$(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --garment-group-popularity-lookback-days $(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli rolling-ranker-validation --cutoffs $(ROLLING_RANKER_CUTOFFS) --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --candidate-k "$(ROLLING_RANKER_CANDIDATE_K)" --k "$(ROLLING_RANKER_K)" --epochs "$(ROLLING_RANKER_EPOCHS)" --learning-rate "$(ROLLING_RANKER_LEARNING_RATE)" --l2 "$(ROLLING_RANKER_L2)" $$extra_args

deterministic-ranker-submission: venv
	@extra_args=""; \
	if [[ -n "$(RANKER_MAX_TARGET_CUSTOMERS)" ]]; then \
		extra_args="$$extra_args --max-target-customers $(RANKER_MAX_TARGET_CUSTOMERS)"; \
	fi; \
	if [[ -n "$(INCLUDE_AGE_SEGMENT_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-age-segment-popularity --age-segment-bucket-size $(AGE_SEGMENT_BUCKET_SIZE)"; \
	fi; \
	if [[ -n "$(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --age-segment-popularity-lookback-days $(AGE_SEGMENT_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_GARMENT_GROUP_POPULARITY)" ]]; then \
		extra_args="$$extra_args --include-garment-group-popularity --garment-group-max-history-items $(GARMENT_GROUP_MAX_HISTORY_ITEMS)"; \
	fi; \
	if [[ -n "$(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)" ]]; then \
		extra_args="$$extra_args --garment-group-popularity-lookback-days $(GARMENT_GROUP_POPULARITY_LOOKBACK_DAYS)"; \
	fi; \
	if [[ -n "$(INCLUDE_TWO_TOWER_RETRIEVAL)" ]]; then \
		extra_args="$$extra_args --include-two-tower-retrieval --two-tower-negatives-per-positive $(TWO_TOWER_NEGATIVES_PER_POSITIVE) --two-tower-negative-sampling $(TWO_TOWER_NEGATIVE_SAMPLING) --two-tower-seed $(TWO_TOWER_SEED) --two-tower-positive-selection $(TWO_TOWER_POSITIVE_SELECTION) --two-tower-max-positive-examples $(TWO_TOWER_MAX_POSITIVE_EXAMPLES) --two-tower-embedding-dim $(TWO_TOWER_EMBEDDING_DIM) --two-tower-epochs $(TWO_TOWER_EPOCHS) --two-tower-learning-rate $(TWO_TOWER_LEARNING_RATE) --two-tower-l2 $(TWO_TOWER_L2) --two-tower-loss $(TWO_TOWER_LOSS) --two-tower-logq-correction-alpha $(TWO_TOWER_LOGQ_CORRECTION_ALPHA) --two-tower-max-retrieval-articles $(TWO_TOWER_MAX_RETRIEVAL_ARTICLES)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_RANKER_PRESENCE_WEIGHT)" ]]; then \
		extra_args="$$extra_args --two-tower-ranker-presence-weight $(TWO_TOWER_RANKER_PRESENCE_WEIGHT)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_RANKER_SCORE_WEIGHT)" ]]; then \
		extra_args="$$extra_args --two-tower-ranker-score-weight $(TWO_TOWER_RANKER_SCORE_WEIGHT)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli generate-deterministic-ranker-submission --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --candidate-k "$(RANKER_CANDIDATE_K)" --k "$(RANKER_K)" --top-trials "$(DETERMINISTIC_TUNING_TOP_TRIALS)" $$extra_args

learned-ranker-submission: venv
	@extra_args=""; \
	if [[ -n "$(LEARNED_RANKER_SUBMISSION)" ]]; then \
		extra_args="$$extra_args --output-path $(LEARNED_RANKER_SUBMISSION)"; \
	fi; \
	if [[ -n "$(LEARNED_RANKER_SUBMISSION_NO_CO_VISITATION)" ]]; then \
		extra_args="$$extra_args --no-co-visitation"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli generate-learned-ranker-submission --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --candidate-k "$(LEARNED_RANKER_CANDIDATE_K)" --k "$(LEARNED_RANKER_K)" --epochs "$(LEARNED_RANKER_EPOCHS)" --learning-rate "$(LEARNED_RANKER_LEARNING_RATE)" --l2 "$(LEARNED_RANKER_L2)" $$extra_args

two-tower-example-export: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make two-tower-example-export CUTOFF=2020-09-16\n"; exit 2; fi
	@extra_args=""; \
	if [[ -n "$(TWO_TOWER_MAX_POSITIVE_EXAMPLES)" ]]; then \
		extra_args="$$extra_args --max-positive-examples $(TWO_TOWER_MAX_POSITIVE_EXAMPLES)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli export-two-tower-examples --cutoff "$(CUTOFF)" --negatives-per-positive "$(TWO_TOWER_NEGATIVES_PER_POSITIVE)" --negative-sampling "$(TWO_TOWER_NEGATIVE_SAMPLING)" --seed "$(TWO_TOWER_SEED)" --positive-selection "$(TWO_TOWER_POSITIVE_SELECTION)" $$extra_args

two-tower-retrieval-smoke: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make two-tower-retrieval-smoke CUTOFF=2020-09-16\n"; exit 2; fi
	@export_args=""; eval_args=""; \
	if [[ -n "$(TWO_TOWER_MAX_POSITIVE_EXAMPLES)" ]]; then \
		export_args="$$export_args --max-positive-examples $(TWO_TOWER_MAX_POSITIVE_EXAMPLES)"; \
		eval_args="$$eval_args --max-positive-examples $(TWO_TOWER_MAX_POSITIVE_EXAMPLES)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_MAX_TRAINING_EXAMPLES)" ]]; then \
		eval_args="$$eval_args --max-training-examples $(TWO_TOWER_MAX_TRAINING_EXAMPLES)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_POSITIVE_RECENCY_HALF_LIFE_DAYS)" ]]; then \
		eval_args="$$eval_args --positive-recency-half-life-days $(TWO_TOWER_POSITIVE_RECENCY_HALF_LIFE_DAYS)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_MAX_EVAL_CUSTOMERS)" ]]; then \
		eval_args="$$eval_args --max-eval-customers $(TWO_TOWER_MAX_EVAL_CUSTOMERS)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_MAX_RETRIEVAL_ARTICLES)" ]]; then \
		eval_args="$$eval_args --max-retrieval-articles $(TWO_TOWER_MAX_RETRIEVAL_ARTICLES)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_EVALUATION_KS)" ]]; then \
		eval_args="$$eval_args --evaluation-ks $(TWO_TOWER_EVALUATION_KS)"; \
	fi; \
	if [[ -n "$(TWO_TOWER_POPULARITY_PRIOR_WEIGHT)" ]]; then \
		eval_args="$$eval_args --popularity-prior-weight $(TWO_TOWER_POPULARITY_PRIOR_WEIGHT) --popularity-prior-lookback-days $(TWO_TOWER_POPULARITY_PRIOR_LOOKBACK_DAYS)"; \
	fi; \
	"$(VENV_PYTHON)" -m hm_recsys.cli export-two-tower-examples --cutoff "$(CUTOFF)" --negatives-per-positive "$(TWO_TOWER_NEGATIVES_PER_POSITIVE)" --negative-sampling "$(TWO_TOWER_NEGATIVE_SAMPLING)" --seed "$(TWO_TOWER_SEED)" --positive-selection "$(TWO_TOWER_POSITIVE_SELECTION)" $$export_args; \
	"$(VENV_PYTHON)" -m hm_recsys.cli evaluate-two-tower-retrieval --cutoff "$(CUTOFF)" --negatives-per-positive "$(TWO_TOWER_NEGATIVES_PER_POSITIVE)" --negative-sampling "$(TWO_TOWER_NEGATIVE_SAMPLING)" --seed "$(TWO_TOWER_SEED)" --positive-selection "$(TWO_TOWER_POSITIVE_SELECTION)" --embedding-dim "$(TWO_TOWER_EMBEDDING_DIM)" --epochs "$(TWO_TOWER_EPOCHS)" --learning-rate "$(TWO_TOWER_LEARNING_RATE)" --l2 "$(TWO_TOWER_L2)" --loss "$(TWO_TOWER_LOSS)" --logq-correction-alpha "$(TWO_TOWER_LOGQ_CORRECTION_ALPHA)" $$eval_args

kaggle-submit: venv
	@if [[ -z "$(SUBMISSION)" ]]; then printf "SUBMISSION is required, e.g. make kaggle-submit SUBMISSION=submissions/file.csv\n"; exit 2; fi
	"$(VENV_PYTHON)" -m hm_recsys.cli validate-submission --submission-path "$(SUBMISSION)"
	@if [[ ! -x "$(VENV_KAGGLE)" ]]; then printf "kaggle CLI not found in $(VENV). Run make venv to install pinned development tools.\n"; exit 127; fi; \
	if [[ -f ".env" ]]; then set -a; source ".env"; set +a; fi; \
	username="$${KAGGLE_USERNAME:-$${KAGGLE_USER_NAME:-}}"; \
	key="$${KAGGLE_KEY:-$${KAGGLE_API_TOKEN:-}}"; \
	if [[ -z "$$username" || -z "$$key" ]]; then \
		printf "Kaggle credentials are missing. Set KAGGLE_USERNAME/KAGGLE_KEY or KAGGLE_USER_NAME/KAGGLE_API_TOKEN.\n"; \
		exit 2; \
	fi; \
	KAGGLE_USERNAME="$$username" KAGGLE_KEY="$$key" "$(VENV_KAGGLE)" competitions submit -c "$(KAGGLE_COMPETITION)" -f "$(SUBMISSION)" -m "$(KAGGLE_MESSAGE)"

format: venv
	"$(VENV)/bin/black" .
	"$(VENV)/bin/isort" .
	"$(VENV)/bin/ruff" check . --fix

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache .python-files htmlcov coverage.xml .coverage .coverage.* docs/_build src/*.egg-info
	find . \( -path './.git' -o -path './.venv' -o -path './artifacts' -o -path './data' -o -path './models' -o -path './outputs' -o -path './submissions' \) -prune -o -type d -name __pycache__ -exec rm -rf {} +

clean-venv:
	rm -rf "$(VENV)"
