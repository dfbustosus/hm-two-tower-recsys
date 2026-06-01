# OpenCode Project Setup

This directory defines project-local OpenCode behavior for the H&M two-tower recommender.

## Components

- `../opencode.json`: root project config, permissions, watcher ignores, shared commands, and default agent.
- `../docs/spec-driven-development.md`: six-part SDD plan for the recommender project.
- `instructions/project.md`: always-loaded project instructions.
- `agents/`: specialized primary and subagents for orchestration, data audit, feature engineering, two-tower modeling, evaluation, submission checks, research, and review.
- `skills/`: reusable on-demand instructions for the H&M competition, SDD, validation, candidate generation, ranking, two-tower modeling, multimodal content, experiment governance, and data governance.
- `commands/`: slash commands for common workflows.

## Useful Commands

- `/repo-audit`: inspect repo state and next steps.
- `/bootstrap-ml-repo`: plan the initial codebase.
- `/data-contract` or `/audit-data`: check expected files and leakage risks.
- `/validation-design`: design temporal validation and MAP@12.
- `/two-tower-plan` or `/plan-two-tower`: plan retrieval modeling.
- `/submission-check` or `/check-submission`: verify Kaggle output shape.
- `/review-ml`: review current changes.
- `/sdd-audit`: produce the six-part spec-driven development audit.
- `/candidate-plan` or `/plan-candidates`: design high-recall retrieval sources.
- `/ranking-plan` or `/plan-ranking`: design reranking, blending, and ensembling.
- `/experiment-plan` or `/plan-experiments`: design experiment governance and promotion criteria.
- `/multimodal-plan` or `/plan-multimodal`: design text/image/content embedding strategy.
- `/model-policy`: review whether model/provider choices belong in global OpenCode config or project config.

The config intentionally does not set a model or provider. Keep provider credentials and model preferences in your global OpenCode config or environment.
