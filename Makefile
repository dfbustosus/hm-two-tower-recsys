SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
VENV_STAMP := $(VENV)/.requirements-dev.stamp
EXCLUDED_SCAN_DIRS := .git,.venv,artifacts,data,env,models,outputs,submissions,venv

.DEFAULT_GOAL := help

.PHONY: help venv install-dev check validate lint type test security audit format clean clean-venv

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
	@printf "Maintenance:\n"
	@printf "  make format        Auto-format Python files when present\n"
	@printf "  make clean         Remove local caches, not data or the virtualenv\n"
	@printf "  make clean-venv    Remove the local virtualenv\n"

$(VENV_PYTHON):
	$(PYTHON) -m venv "$(VENV)"
	"$(VENV_PYTHON)" -m pip install --upgrade pip

$(VENV_STAMP): requirements-dev.txt $(VENV_PYTHON)
	"$(VENV_PIP)" install -r requirements-dev.txt
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
	$(PYTHON) scripts/check_repo_hygiene.py

lint: venv
	"$(VENV)/bin/ruff" check .
	"$(VENV)/bin/black" --check .
	"$(VENV)/bin/isort" --check-only .
	"$(VENV)/bin/flake8" .

type: venv
	@files="$$(git ls-files --cached --others --exclude-standard -- '*.py' | grep -Ev '^(data|artifacts|models|outputs|submissions|\.venv|venv|env)/' || true)"; \
	if [[ -n "$$files" ]]; then \
		"$(VENV_PYTHON)" -m mypy $$files; \
	else \
		printf "No Python files detected; skipping mypy.\n"; \
	fi

test: venv
	@tests="$$(git ls-files --cached --others --exclude-standard -- 'test_*.py' '*_test.py' 'tests/*.py' 'tests/**/*.py' | grep -Ev '^(data|artifacts|models|outputs|submissions|\.venv|venv|env)/' || true)"; \
	if [[ -n "$$tests" ]]; then \
		"$(VENV_PYTHON)" -m pytest; \
	else \
		printf "No pytest tests detected; skipping pytest.\n"; \
	fi

security: venv
	"$(VENV)/bin/pip-audit" -r requirements-dev.txt --progress-spinner off
	@files="$$(git ls-files --cached --others --exclude-standard -- '*.py' | grep -Ev '^(data|artifacts|models|outputs|submissions|\.venv|venv|env)/' || true)"; \
	if [[ -n "$$files" ]]; then \
		"$(VENV)/bin/bandit" -r . --exclude "$(EXCLUDED_SCAN_DIRS)" -ll; \
	else \
		printf "No Python files detected; skipping Bandit.\n"; \
	fi

format: venv
	"$(VENV)/bin/black" .
	"$(VENV)/bin/isort" .
	"$(VENV)/bin/ruff" check . --fix

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache .python-files htmlcov .coverage .coverage.*
	find . \( -path './.git' -o -path './.venv' -o -path './artifacts' -o -path './data' -o -path './models' -o -path './outputs' -o -path './submissions' \) -prune -o -type d -name __pycache__ -exec rm -rf {} +

clean-venv:
	rm -rf "$(VENV)"
