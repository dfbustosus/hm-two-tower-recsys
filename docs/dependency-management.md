# Dependency Management

This repository uses Dependabot to keep automation and Python development tooling current without mixing dependency updates into feature work.

## Current Strategy

- Python version target: `3.11`, recorded in `.python-version` and CI workflows.
- Local environment: `.venv/`, created with `make venv`, ignored by git.
- Development tooling: exact pins in `requirements-dev.txt`.
- Tool configuration: `pyproject.toml`, `.flake8`, and `.yamllint`.
- Command surface: `Makefile` targets such as `make check`, `make lint`, `make security`, and `make test`.

The repository is not using Poetry at this stage because there is not yet a production Python package or runtime dependency graph. Introducing Poetry now would create ceremony without solving an actual packaging problem. Re-evaluate Poetry, uv, or another lockfile-based manager when Stage 1 creates importable package code and runtime dependencies.

## What Dependabot Monitors

- GitHub Actions in `.github/workflows/`.
- Python development tooling in `requirements-dev.txt`.

## Review Rules

For each Dependabot pull request:

1. Read the linked changelog or release notes.
2. Check for breaking changes, deprecations, and security advisories.
3. Let CI run all quality and security workflows.
4. Do not combine dependency updates with modeling or data-pipeline changes.
5. Confirm no raw Kaggle data, generated artifacts, secrets, or local credentials are included.

## Update Cadence

Dependabot runs weekly. Security updates should be reviewed before routine version bumps when both are open.

## Pinning Policy

The current repo is in the bootstrap layer and pins development tools exactly in `requirements-dev.txt` for deterministic CI. When production Python packaging begins, add an application dependency lock strategy and update this document.
