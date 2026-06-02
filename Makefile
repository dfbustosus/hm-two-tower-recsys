SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
VENV_STAMP := $(VENV)/.requirements-dev.stamp
EXCLUDED_SCAN_DIRS := .git,.venv,artifacts,data,env,models,outputs,submissions,venv

.DEFAULT_GOAL := help

BASELINE_LOOKBACK_DAYS ?= 7
BASELINE_K ?= 12

.PHONY: help venv install-dev check validate lint type test security audit data-contract temporal-split validate-submission baseline format clean clean-venv

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
	@printf "Data:\n"
	@printf "  make data-contract Validate local H&M raw data and write an ignored report\n\n"
	@printf "Validation/submission:\n"
	@printf "  make temporal-split CUTOFF=YYYY-MM-DD  Summarize a temporal split\n"
	@printf "  make validate-submission SUBMISSION=path/to.csv  Validate a submission CSV\n\n"
	@printf "Baselines:\n"
	@printf "  make baseline CUTOFF=YYYY-MM-DD  Evaluate repeat+popularity baseline\n\n"
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

data-contract: venv
	"$(VENV_PYTHON)" -m hm_recsys.cli validate-data-contract

temporal-split: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make temporal-split CUTOFF=2020-09-16\n"; exit 2; fi
	"$(VENV_PYTHON)" -m hm_recsys.cli summarize-temporal-split --cutoff "$(CUTOFF)"

validate-submission: venv
	@if [[ -z "$(SUBMISSION)" ]]; then printf "SUBMISSION is required, e.g. make validate-submission SUBMISSION=submissions/file.csv\n"; exit 2; fi
	"$(VENV_PYTHON)" -m hm_recsys.cli validate-submission --submission-path "$(SUBMISSION)"

baseline: venv
	@if [[ -z "$(CUTOFF)" ]]; then printf "CUTOFF is required, e.g. make baseline CUTOFF=2020-09-16\n"; exit 2; fi
	"$(VENV_PYTHON)" -m hm_recsys.cli evaluate-baseline --cutoff "$(CUTOFF)" --popularity-lookback-days "$(BASELINE_LOOKBACK_DAYS)" --k "$(BASELINE_K)"

format: venv
	"$(VENV)/bin/black" .
	"$(VENV)/bin/isort" .
	"$(VENV)/bin/ruff" check . --fix

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache .python-files htmlcov .coverage .coverage.* src/*.egg-info
	find . \( -path './.git' -o -path './.venv' -o -path './artifacts' -o -path './data' -o -path './models' -o -path './outputs' -o -path './submissions' \) -prune -o -type d -name __pycache__ -exec rm -rf {} +

clean-venv:
	rm -rf "$(VENV)"
