---
name: sdd-recsys
description: Use for spec-driven development plans, architecture reviews, and acceptance criteria for recommender-system projects.
license: MIT
compatibility: opencode
metadata:
  domain: recommender-systems
  workflow: specification
---

# SDD Recommender Skill

Use this skill before implementing large changes or when reviewing whether the project plan is coherent.

## Six-Part Specification

1. Outcomes: the exact successful end-state for the user, model, pipeline, and submission.
2. Scope boundaries: what is in scope now and what is deliberately excluded.
3. Constraints and assumptions: data availability, compute, licensing, leakage, reproducibility, memory, and runtime.
4. Prior decisions: architectural commitments already made, such as offline MAP@12, Kaggle submission shape, and string ID handling.
5. Task breakdown: modular work units that can be implemented, reviewed, and tested independently.
6. Verification criteria: tests, metrics, edge cases, artifacts, and promotion thresholds.

## Review Posture

- Challenge "state of the art" claims unless backed by a validation plan.
- Prefer evidence-producing milestones over broad architecture diagrams.
- Demand a baseline and champion/challenger protocol before deep modeling.
- Keep raw competition data and generated artifacts out of version control.
