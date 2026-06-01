---
name: experiment-governance
description: Use for ML experiment tracking, ablation design, reproducibility, champion/challenger comparison, and artifact promotion.
license: MIT
compatibility: opencode
metadata:
  domain: mlops
  workflow: experiments
---

# Experiment Governance Skill

Use this skill for any model, feature, candidate, ranking, or submission experiment.

## Required Run Record

Each experiment should record:

- Run ID and timestamp.
- Git commit or diff summary.
- Split name, train dates, validation target dates, and excluded rows.
- Input file versions or hashes when available.
- Config, random seeds, package environment, and hardware notes.
- Candidate sources and counts.
- Metrics: MAP@12, candidate recall, coverage, duplicate rate, and runtime.
- Artifact paths for metrics, predictions, embeddings, indexes, and generated submissions.

## Ablation Discipline

- Change one meaningful component at a time where possible.
- Keep a current champion and compare challengers against it.
- Record negative results; they prevent repeated dead ends.
- Promote only if the improvement is reproducible across the agreed split protocol.

## Reproducibility Checks

- Same config and seed should recreate the same small-sample outputs.
- Output IDs must remain strings.
- Artifacts must include enough metadata to trace data, split, code, and config.
