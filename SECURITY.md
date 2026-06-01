# Security Policy

## Supported Branches

Security fixes target the active default branch, currently `main`.

## Reporting a Vulnerability

Use GitHub private vulnerability reporting if it is enabled for this repository. If it is not enabled, open a minimal public issue asking for a private maintainer contact; do not include exploit details, secrets, private data, or credentials in a public issue.

Include enough non-sensitive information for maintainers to reproduce the concern:

- affected file, workflow, or command,
- expected vs. observed behavior,
- security impact,
- safe reproduction steps,
- dependency or action versions if relevant.

## Secret and Data Handling

Never commit:

- Kaggle raw CSVs or article images,
- generated features, embeddings, checkpoints, indexes, experiment logs, or submissions,
- `.env` files, API keys, tokens, cookies, service-account material, or private credentials,
- external data that is not public, free, and equally available to Kaggle competitors.

The repository uses `.gitignore`, GitHub Actions, Gitleaks, CodeQL, Bandit, pip-audit, and Dependabot to reduce security risk. These checks do not replace human review.

## Dependency Updates

Dependabot is configured for GitHub Actions and Python development tooling. Review dependency PRs for changelogs, breaking changes, security advisories, and workflow behavior before merging.
