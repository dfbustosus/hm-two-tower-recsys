---
name: data-governance
description: Use for Kaggle data handling, local artifact layout, git hygiene, and large-file safety.
license: MIT
compatibility: opencode
metadata:
  domain: mlops
  workflow: data-management
---

# Data Governance Skill

Use this skill whenever data paths, artifacts, generated files, or external assets are involved.

## Local Layout

Recommended directories:

- `data/raw/`: raw Kaggle files.
- `data/interim/`: cleaned and typed intermediate files.
- `data/processed/`: model-ready feature tables.
- `artifacts/`: metrics, plots, candidate dumps, reports.
- `models/`: checkpoints, embedding indexes, ANN indexes.
- `submissions/`: generated Kaggle CSV files.

## Git Hygiene

Do not commit:

- Raw Kaggle CSVs.
- Article images.
- Generated parquet/feather/arrow tables.
- Model checkpoints or embedding indexes.
- Experiment tracker directories.
- Submission CSVs unless the user explicitly wants a small example fixture.

Commit:

- Source code.
- Lightweight config.
- Tests.
- Schema documentation.
- Tiny synthetic fixtures when needed.

## External Data And Code

For competition work, external data must be public, free, and equally accessible. External code should have a permissive license before incorporation.

## Operational Safety

- Prefer sampled reads for exploration.
- Avoid loading all images unless the task requires it.
- Store reproducibility metadata beside artifacts: command, git commit, split date, seed, and config.
