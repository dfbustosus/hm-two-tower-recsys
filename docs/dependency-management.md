# Dependency Management

This repository uses Dependabot to keep automation and Python development tooling current without mixing dependency updates into feature work.

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

## Locking Policy

The current repo is in the bootstrap layer and uses bounded development-tool ranges rather than an application lockfile. When production Python packaging begins, add a deterministic lock strategy and update this document.
