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

The GitHub Actions workflows use the tools pinned in `requirements-dev.txt`. Create the ignored local virtual environment with:

```bash
make venv
```

If `python3` does not resolve to Python 3.11 locally, use:

```bash
make clean-venv
make venv PYTHON=python3.11
```

Run the full gate before requesting review:

```bash
make check
```

Useful narrower commands are:

```bash
make validate
make lint
make type
make security
make test
```

The CI workflows discover Python files and tests automatically. Data-dependent commands such as `make data-contract` require local Kaggle files and are not run in public CI.

## Pull Request Expectations

- Explain the objective and the project layer touched.
- State the validation performed.
- Call out data, artifact, leakage, ID-format, and security implications.
- Link related issues or experiments when applicable.
- Follow `CODE_OF_CONDUCT.md`.
