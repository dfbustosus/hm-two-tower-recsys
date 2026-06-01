# Contributing

Contributions must preserve the repository's core contract: H&M recommendations are evaluated with leakage-safe MAP@12, IDs remain strings, and raw Kaggle data or generated artifacts never enter git.

## Before Opening a Pull Request

1. Read `docs/spec-driven-development.md`.
2. Identify the project layer affected by the change: governance, data contract, validation, metrics, baselines, candidate generation, ranking, experiment tracking, or submission validation.
3. Keep changes small and reviewable.
4. Do not commit raw CSVs, article images, derived feature tables, model checkpoints, experiment logs, or submissions.
5. Update documentation when contracts, paths, metrics, or workflows change.

## Engineering Standards

- Prefer simple, composable modules with one responsibility each.
- Avoid duplicated ID parsing, date parsing, split logic, metric logic, or submission validation logic.
- Treat two-tower retrieval and other advanced models as challengers until validation proves value.
- Add tests for edge cases before trusting model or data-pipeline changes.
- Keep production logic in importable modules; notebooks may explore but must not be the only source of truth.

## Local Quality Checks

The GitHub Actions workflows use the tools listed in `requirements-dev.txt`. Install them locally when you start implementing Python code:

```bash
python -m pip install -r requirements-dev.txt
```

Run the relevant checks before requesting review:

```bash
ruff check .
black --check .
isort --check-only .
flake8 .
mypy <changed-python-files>
bandit -r . --exclude data,artifacts,models,outputs,submissions
```

If the repository has no Python source yet, the CI workflows intentionally skip Python-specific linting and typing while still checking repository governance.

## Pull Request Expectations

- Explain the objective and the project layer touched.
- State the validation performed.
- Call out data, artifact, leakage, ID-format, and security implications.
- Link related issues or experiments when applicable.
- Follow `CODE_OF_CONDUCT.md`.
